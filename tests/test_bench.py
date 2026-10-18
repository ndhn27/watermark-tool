"""Tests for the benchmark framework itself (bench/). The framework reports
rates and confidence intervals, so what needs pinning down is that its
statistics are right and that its plumbing (oracle BER, plan, records,
summaries, baseline gate) measures what it claims to - not the watermark's
own performance, which is what `python3 bench/run_benchmark.py` is for.
"""

import json
import math

import numpy as np
import pytest

import wmcore
import bench_attacks as A
import bench_corpus as C
import bench_metrics as M
import bench_schemes as S
import run_benchmark as B  # bench/ on sys.path via conftest.py


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_wilson_interval_known_value():
    lo, hi = M.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_interval_stays_in_unit_range():
    assert M.wilson_interval(0, 5)[0] == 0.0
    assert M.wilson_interval(5, 5)[1] == 1.0
    assert M.wilson_interval(0, 0) == (0.0, 1.0)


def test_clopper_pearson_known_values():
    # Zero events in 100 trials: the "rule of three" territory, exact upper bound 3.62%.
    lo, hi = M.clopper_pearson(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.0362, abs=5e-4)
    lo, hi = M.clopper_pearson(3, 10)
    assert (lo, hi) == (pytest.approx(0.0667, abs=1e-3), pytest.approx(0.6525, abs=1e-3))
    assert M.clopper_pearson(10, 10)[1] == 1.0


def test_clopper_pearson_is_wider_than_wilson_at_zero():
    # Why FPR uses the exact interval: it must not understate the upper bound.
    assert M.clopper_pearson(0, 30)[1] > M.wilson_interval(0, 30)[1]


# ---------------------------------------------------------------------------
# BER / PSNR
# ---------------------------------------------------------------------------

def test_ber_zero_and_half_on_constructed_signals():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    idx = np.arange(64)
    truth = bits[idx % len(bits)]
    perfect = np.where(truth == 1, 3.0, -3.0)
    assert M.ber_raw(perfect, idx, bits) == 0.0
    assert M.ber_combined(perfect, idx, bits) == 0.0
    assert M.ber_raw(-perfect, idx, bits) == 1.0


def test_combining_repeats_beats_raw_ber():
    # The claim spread.py's docstring rests on: a per-repeat error rate in the
    # tens of percent still combines to a clean decision.
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 120)
    idx = np.arange(120 * 40)
    signal = np.where(bits[idx % 120] == 1, 1.0, -1.0)
    soft = signal + rng.normal(0, 1.2, signal.shape)
    raw, comb = M.ber_raw(soft, idx, bits), M.ber_combined(soft, idx, bits)
    assert raw > 0.15
    assert comb == 0.0


def test_ber_uses_original_index_not_position():
    # A crop drops the first observations: the alignment has to come from orig_idx.
    bits = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    idx_full = np.arange(100)
    soft_full = np.where(bits[idx_full % 10] == 1, 1.0, -1.0)
    assert M.ber_combined(soft_full[3:], idx_full[3:], bits) == 0.0
    assert M.ber_combined(soft_full[3:], np.arange(97), bits) > 0.0  # naive positions would misalign


def test_psnr():
    a = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert math.isinf(M.psnr(a, a))
    b = a.copy()
    b[...] = 101  # mse = 1
    assert M.psnr(a, b) == pytest.approx(10 * math.log10(255 ** 2), abs=1e-6)


# ---------------------------------------------------------------------------
# Oracle BER through the real embedder
# ---------------------------------------------------------------------------

SIZE = (320, 240)
PAYLOAD = S.PAYLOADS["short"]


def _stamped(scheme="ss", key=4242):
    base = C.load_source({"kind": "pink", "w": SIZE[0], "h": SIZE[1], "seed": 3})
    return S.embed(scheme, base, PAYLOAD, key, wmcore.DEFAULT_SS_ALPHA).convert("RGB"), key


def test_oracle_ber_is_zero_on_clean_ss_image():
    img, key = _stamped()
    soft, idx = S.oracle_soft("ss", img, key, A.Geometry("identity"), SIZE)
    bits = wmcore.text_to_bits(PAYLOAD)
    assert M.ber_combined(soft, idx, bits) == 0.0
    assert M.ber_raw(soft, idx, bits) < 0.1


def test_oracle_ber_averages_half_over_wrong_keys():
    # A single wrong key is NOT 0.5: its spreading sequence overlaps the real one by some
    # random correlation rho, and the payload comes out right (rho > 0) or bit-inverted
    # (rho < 0) - the mechanism behind the README's "1 in 4 wrong keys still decode".
    # Only the average over many keys is chance level.
    img, key = _stamped()
    bits = wmcore.text_to_bits(PAYLOAD)
    bers = []
    for i in range(1, 41):
        soft, idx = S.oracle_soft("ss", img, key + i * 7919, A.Geometry("identity"), SIZE)
        bers.append(M.ber_raw(soft, idx, bits))
    assert np.mean(bers) == pytest.approx(0.5, abs=0.1)
    assert min(bers) < 0.35 < max(bers)  # both regimes show up


@pytest.mark.parametrize("scheme", ["ss", "lsb"])
def test_oracle_follows_a_block_aligned_crop(scheme):
    img, key = _stamped(scheme)
    attacked, geom = A.REGISTRY["crop_top_64"].fn(img, np.random.default_rng(0))
    assert geom.kind == "crop_blocks" and geom.crop_blocks == (8, 0)
    soft, idx = S.oracle_soft(scheme, attacked, key, geom, SIZE)
    assert M.ber_combined(soft, idx, wmcore.text_to_bits(PAYLOAD)) == 0.0


def test_oracle_follows_a_resize_back_to_original_grid():
    img, key = _stamped()
    attacked, geom = A.REGISTRY["resize_0.75"].fn(img, np.random.default_rng(0))
    assert attacked.size != img.size and geom.kind == "resize"
    soft, idx = S.oracle_soft("ss", attacked, key, geom, SIZE)
    assert M.ber_combined(soft, idx, wmcore.text_to_bits(PAYLOAD)) == 0.0  # signal survived; only sync is missing


def test_no_oracle_for_unaligned_geometry():
    img, key = _stamped()
    attacked, geom = A.REGISTRY["crop_unaligned_3"].fn(img, np.random.default_rng(0))
    assert geom.kind == "none"
    assert S.oracle_soft("ss", attacked, key, geom, SIZE) is None


# ---------------------------------------------------------------------------
# Attacks / corpus
# ---------------------------------------------------------------------------

def test_attacks_are_deterministic_given_the_rng():
    img, _ = _stamped()
    for name in ("noise_5", "jpeg_75", "blur_1.0", "resize_0.75"):
        a1, _ = A.REGISTRY[name].fn(img, np.random.default_rng([1, 2]))
        a2, _ = A.REGISTRY[name].fn(img, np.random.default_rng([1, 2]))
        assert np.array_equal(np.asarray(a1), np.asarray(a2)), name


def test_documented_limits_are_not_gated():
    for name in ("crop_unaligned_3", "rotate_1deg", "resize_0.5", "crop_topleft_64"):
        assert not A.REGISTRY[name].gate
    assert A.REGISTRY["jpeg_75"].gate and A.REGISTRY["identity"].gate


def test_attack_selection_by_name_family_and_glob():
    assert A.select(["jpeg"]) == [n for n, a in A.REGISTRY.items() if a.family == "jpeg"]
    assert "resize_0.75" in A.select(["resize_0.7*"])
    assert A.select(None) == list(A.REGISTRY)


def test_corpus_is_seed_deterministic_and_varied():
    spec = {"kind": "pink", "w": 128, "h": 96, "seed": 7}
    a, b = C.load_source(spec), C.load_source(dict(spec))
    assert np.array_equal(np.asarray(a), np.asarray(b))
    c = C.load_source(dict(spec, seed=8))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


# ---------------------------------------------------------------------------
# End to end: plan -> records -> summary -> report -> gate
# ---------------------------------------------------------------------------

ARGV = ["--suite", "quick", "--kinds", "gradnoise", "--sizes", "320x240", "--n-images", "2",
        "--n-neg", "4", "--n-neg-rescale", "1", "--n-wrongkey-images", "3", "--n-wrong-keys", "2",
        "--rescale-images", "0", "--attacks", "identity,jpeg_75,crop_top_64", "--decoders", "lsb,ss",
        "--no-progress"]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    assert B.main(ARGV + ["--out", str(out)]) == 0
    return out


def _rows(run_dir):
    return json.load(open(run_dir / "summary.json"))["rows"]


def test_run_writes_all_artifacts(run_dir):
    for f in ("records.jsonl", "summary.json", "summary.csv", "report.md", "meta.json"):
        assert (run_dir / f).exists(), f
    meta = json.load(open(run_dir / "meta.json"))
    assert meta["wmcore_fingerprint"] and meta["env"]["numpy"]


def test_spread_spectrum_rows_report_full_detection_and_zero_combined_ber(run_dir):
    pos = [r for r in _rows(run_dir) if r["kind"] == "pos" and r["decoder"] == "ss"]
    assert {r["attack"] for r in pos} == {"identity", "jpeg_75", "crop_top_64"}
    for r in pos:
        assert r["det_rate"] == 1.0, r["attack"]
        assert r["ber_comb_mean"] == 0.0
        assert r["det_ci"][0] < 1.0 <= r["det_ci"][1]
        assert r["psnr_mean"] > 25


def test_lsb_out_of_scope_rows_are_ungated_and_show_destroyed_signal(run_dir):
    lsb = {r["attack"]: r for r in _rows(run_dir) if r["kind"] == "pos" and r["decoder"] == "lsb"}
    assert lsb["identity"]["gate"] and lsb["identity"]["det_rate"] == 1.0
    assert not lsb["jpeg_75"]["gate"]
    assert lsb["jpeg_75"]["det_rate"] == 0.0 and lsb["jpeg_75"]["ber_raw_mean"] > 0.4


def test_negative_rows_carry_fpr_with_an_exact_upper_bound(run_dir):
    neg = [r for r in _rows(run_dir) if r["kind"] == "neg_unmarked" and r["decoder"] == "ss"]
    assert neg and all(r["fp"] == 0 and r["fpr_ci"][1] > 0 for r in neg)
    wk = [r for r in _rows(run_dir) if r["kind"] == "neg_wrongkey"]
    assert wk and all(r["n"] == 3 * 2 for r in wk)  # images x wrong keys
    assert all(r["leak"] <= r["fp"] for r in wk)


def test_runtime_is_recorded(run_dir):
    rt = json.load(open(run_dir / "summary.json"))["runtime"]
    ops = {r["op"] for r in rt}
    assert {"embed:ss", "decode:ss", "decode:lsb"} <= ops
    assert all(r["p50_s"] > 0 for r in rt)


def test_same_seed_reproduces_outcomes(run_dir, tmp_path):
    B.main(ARGV + ["--out", str(tmp_path)])
    def outcomes(d):
        keep = ("kind", "decoder", "attack", "image_id", "outcome", "ber_raw", "ber_comb", "psnr")
        return [tuple(json.loads(l).get(k) for k in keep) for l in open(d / "records.jsonl")]
    assert outcomes(run_dir) == outcomes(tmp_path)


def test_report_mentions_the_requested_metrics(run_dir):
    text = (run_dir / "report.md").read_text()
    for needle in ("detect [95% CI]", "BER raw", "BER comb", "FPR (95% upper)", "## Runtime", "wmcore fingerprint"):
        assert needle in text, needle


def test_report_only_rerenders_without_running(run_dir, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(run_dir, d)
    (d / "report.md").unlink()
    assert B.main(["--report-only", str(d)]) == 0
    assert (d / "report.md").exists()


def test_baseline_gate_passes_against_itself_and_fails_on_a_drop(run_dir, tmp_path):
    base = run_dir / "summary.json"
    assert B.main(ARGV + ["--out", str(tmp_path / "again"), "--baseline", str(base)]) == 0

    doctored = json.load(open(base))
    for r in doctored["rows"]:  # pretend the baseline was better than what we just measured
        if r["kind"] == "pos" and r["decoder"] == "lsb" and r["attack"] == "identity":
            r["det_rate"] = 1.0
    cur = json.load(open(base))["rows"]
    for r in cur:
        if r["kind"] == "pos" and r["decoder"] == "lsb" and r["attack"] == "identity":
            r["det_rate"] = 0.5
    problems = M.compare_summaries(doctored["rows"], cur)
    assert len(problems) == 1 and problems[0].startswith("detection lsb/identity"), problems
    assert "100.0% -> 50.0%" in problems[0]


def test_compare_ignores_ungated_rows_and_runtime_unless_asked():
    def row(rate, gate, p50=0.01, kind="pos"):
        return {"kind": kind, "decoder": "ss", "attack": "rotate_1deg", "image_kind": "pink", "size": [8, 8],
                "alpha": 12.0, "payload_id": "short", "gate": gate, "det_rate": rate, "ber_comb_mean": 0.0,
                "decode_s_p50": p50}
    assert M.compare_summaries([row(1.0, False)], [row(0.0, False)]) == []
    assert M.compare_summaries([row(1.0, True)], [row(0.0, True)]) != []
    assert M.compare_summaries([row(1.0, True, 0.01)], [row(1.0, True, 0.5)]) == []
    assert M.compare_summaries([row(1.0, True, 0.01)], [row(1.0, True, 0.5)], runtime_factor=2.0) != []


# ---------------------------------------------------------------------------
# --resume
# ---------------------------------------------------------------------------

def test_resume_skips_units_that_already_have_records_and_drops_a_torn_line(tmp_path, monkeypatch):
    units = [dict(type="neg_unmarked", spec=dict(kind="pink", w=64, h=48, seed=s)) for s in range(4)]
    calls = []

    def fake_run_unit(u):
        calls.append(u["spec"]["seed"])
        return [dict(kind="neg_unmarked", image_id=C.source_id(u["spec"]), scheme=None, alpha=None,
                     payload_id=None, decoder="ss", attack="identity", outcome="none")]

    monkeypatch.setattr(B, "run_unit", fake_run_unit)
    monkeypatch.setattr(B, "_warmup", lambda: None)
    path = str(tmp_path / "records.jsonl")

    assert B.execute(units[:2], 1, path, progress=False) == 2
    with open(path, "a") as fh:                       # a run killed mid-write: half a line, unit 2 never completed
        fh.write('{"kind": "neg_unmarked", "image_id": "tru')
    calls.clear()

    total = B.execute(units, 1, path, progress=False, resume=True)

    assert calls == [2, 3]                            # 0 and 1 were kept, the torn line's unit was re-run
    assert total == 4
    with open(path) as fh:
        assert len([json.loads(line) for line in fh]) == 4     # every line parses; nothing duplicated


def test_without_resume_the_records_file_is_started_afresh(tmp_path, monkeypatch):
    unit = dict(type="neg_unmarked", spec=dict(kind="pink", w=64, h=48, seed=0))
    monkeypatch.setattr(B, "run_unit", lambda u: [dict(kind="neg_unmarked", image_id=C.source_id(u["spec"]),
                                                       scheme=None, alpha=None, payload_id=None, decoder="ss",
                                                       attack="identity", outcome="none")])
    monkeypatch.setattr(B, "_warmup", lambda: None)
    path = str(tmp_path / "records.jsonl")
    B.execute([unit], 1, path, progress=False)
    assert B.execute([unit], 1, path, progress=False) == 1
