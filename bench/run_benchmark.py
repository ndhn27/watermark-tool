#!/usr/bin/env python3
"""
run_benchmark.py - measure how well the invisible marks are actually
recovered, instead of only asserting that a handful of cases pass.

pytest answers "does this specific case still work?" (binary, one image, one
seed). This answers the questions a release actually depends on:

  detection rate   marked -> attacked -> decoded: how often is the exact
                   payload recovered, per attack, with a 95% interval?
  BER              how much of the SIGNAL survived (raw per-block error and
                   error after combining repeats), measured with oracle
                   alignment so a decoder that fails to re-synchronise can be
                   told apart from a mark that is really gone.
  FPR              how often is text "found" where there is nothing to find:
                   unmarked images (through the same attacks) and marked
                   images decoded with a wrong key (README: ~1 in 4 leak).
  runtime          wall-clock embed / decode / rescale-search cost, p50 and
                   p95, per image size and per megapixel.

Usage (from the repo root):
    python3 bench/run_benchmark.py --suite quick
    python3 bench/run_benchmark.py --suite standard --out bench/results/nightly
    python3 bench/run_benchmark.py --suite standard --images ~/photos --n-images 50
    python3 bench/run_benchmark.py --suite full --alphas 8,12,16 --jobs 1
    python3 bench/run_benchmark.py --suite quick --baseline bench/baselines/quick.json   # exit 1 on regression
    python3 bench/run_benchmark.py --report-only bench/results/nightly                   # re-render, no re-run
    python3 bench/run_benchmark.py --list-attacks

Outputs (in --out, default bench/results/<timestamp>/): records.jsonl (one
line per trial), summary.json, summary.csv, report.md, meta.json. See
bench/README.md for how to read them.

Timing note: numbers are only comparable between runs made with --jobs 1 on
the same machine; with --jobs > 1 workers compete for cores and inflate every
runtime figure (detection/BER/FPR are unaffected).
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import bench_attacks as A  # noqa: E402
import bench_corpus as C  # noqa: E402
import bench_metrics as M  # noqa: E402
import bench_report as R  # noqa: E402
import bench_schemes as S  # noqa: E402
import wmcore  # noqa: E402  (cli/ is put on sys.path by bench_schemes)

SUITES = {
    # ~1-2 min on one core. For CI / a pre-commit sanity check.
    "quick": dict(kinds=["pink"], sizes=[(640, 480)], n_images=3, payloads=["short"],
                  alphas=[wmcore.DEFAULT_SS_ALPHA], attacks=list(A.QUICK_ATTACKS),
                  rescale_images=2, n_neg=12, n_neg_rescale=2, n_wrongkey_images=6, n_wrong_keys=4),
    # Tens of minutes on one core. The default for a real measurement.
    "standard": dict(kinds=["pink", "gradnoise"], sizes=[(640, 480)], n_images=10, payloads=["short"],
                     alphas=[wmcore.DEFAULT_SS_ALPHA], attacks=None,
                     rescale_images=4, n_neg=100, n_neg_rescale=6, n_wrongkey_images=20, n_wrong_keys=8),
    # Payload-length x strength x size sweep. An hour or more.
    "full": dict(kinds=["pink", "gradnoise"], sizes=[(640, 480), (1280, 960)], n_images=12,
                 payloads=["short", "medium", "long"], alphas=[8.0, 12.0, 16.0], attacks=None,
                 rescale_images=3, n_neg=300, n_neg_rescale=8, n_wrongkey_images=30, n_wrong_keys=10),
}

# Which attack families each decoder is run on. lsb/ss are cheap and see
# everything; the resize search is expensive, so it only sees the families it
# exists for (an unresized image takes its fast path anyway).
RESCALE_FAMILIES = ("identity", "resize", "resize_jpeg", "pipeline")

# Attack families each decoder is NOT designed to survive (README: "LSB ... does
# not survive recompression, resizing, or screenshots"; plain `ss` needs
# --rescale for resizes). They are still measured - the numbers show exactly why
# the rescale search exists - but are excluded from headlines and the regression
# gate for that decoder, which would otherwise be dragged down by design limits.
OUT_OF_SCOPE = {
    "lsb": {"jpeg", "resize", "resize_jpeg", "noise", "blur", "tone", "geometric", "flip"},
    "ss": {"resize", "resize_jpeg", "flip"},        # a mirror is read only by the mirror-aware decoders below
    "ss_id": {"resize", "resize_jpeg", "flip"},     # same reach as `ss`: no resize search
    "ss_flip": {"resize", "resize_jpeg"}, "ss_id_flip": {"resize", "resize_jpeg"},   # ...and no resize search either
    "ss_rescale": set(), "ss_id_rescale": set(), "ss_or_id_rescale": set(),
}

# (decoder, attack family) pairs that are in scope only for the listed payloads. `ss_flip` reads a MIRRORED copy
# only when the payload is a watermark ID (wmcore/orient.py: mirrored orientations are searched for IDs only; a
# mirrored text payload is not recovered - bench/results/p3-flip-text-probe.txt), so the flip family is gated for
# it on `wmid` and merely measured (dagger) on free-text payloads.
IN_SCOPE_ONLY_FOR_PAYLOAD = {("ss_flip", "flip"): {"wmid"}}


def _in_scope(decoder, family, payload_id):
    if family in OUT_OF_SCOPE[decoder]:
        return False
    only = IN_SCOPE_ONLY_FOR_PAYLOAD.get((decoder, family))
    return only is None or payload_id in only


# (decoder -> attack names) used for the unmarked-image FPR trials. LSB has
# no notion of a resize; the rescale search is the slow one, hence the short list.
NEG_DECODER_ATTACKS = {
    "lsb": ("identity", "jpeg_75"),
    "ss": A.NEGATIVE_ATTACKS,
    "ss_id": A.NEGATIVE_ATTACKS,
    "ss_flip": A.NEGATIVE_ATTACKS,
    "ss_id_flip": A.NEGATIVE_ATTACKS,
    "ss_rescale": ("identity", "resize_0.75", "resize_1.25"),
    "ss_id_rescale": ("identity", "resize_0.75", "resize_1.25"),
    "ss_or_id_rescale": ("identity", "resize_0.75", "resize_1.25"),
}
WRONGKEY_ATTACKS = ("identity", "jpeg_75")


# ---------------------------------------------------------------------------
# One unit of work = one source image taken through embed -> all attacks -> decoders.
# Top-level functions taking/returning plain dicts, so ProcessPoolExecutor can
# pickle them under the "spawn" start method (same rule the rest of the tool follows).
# ---------------------------------------------------------------------------

_WARM = False


def _warmup():
    """Pay first-call costs (einsum path setup, cv2 thread pool, imports)
    before anything is timed."""
    global _WARM
    if _WARM:
        return
    img = C.load_source({"kind": "gradnoise", "w": 320, "h": 240, "seed": 0})
    st = S.embed("ss", img, "warm", 1, 12.0).convert("RGB")
    S.decode_ss(st, 1)
    S.decode_lsb(S.embed("lsb", img, "warm", 1, None).convert("RGB"), 1)
    y = np.random.default_rng(0).random((200, 200), dtype=np.float32)
    wmcore.rescale._ss_patch_bimodality(y, 1.0, wmcore._ss_pn_vector(1, len(wmcore.SS_COEFF_POSITIONS)), 160)
    _WARM = True


def _crc(s):
    return zlib.crc32(s.encode("utf-8"))


def _clean(x):
    """JSON-safe: numpy scalars -> python, non-finite floats -> None."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _outcome(text, payload):
    if text is None:
        return "none"
    return "correct" if text == payload else "wrong"


def _decode_timed(decoder, img, key, budget):
    t0 = time.perf_counter()
    out = S.DECODERS[decoder](img, key, budget)
    return out, time.perf_counter() - t0


def _base_fields(u, W, H):
    return {"scheme": u.get("scheme"), "image_kind": u["spec"]["kind"], "image_id": C.source_id(u["spec"]),
            "size": [W, H], "alpha": u.get("alpha"), "payload_id": u.get("payload_id")}


def run_unit(u):
    _warmup()
    return {"pos": _run_pos, "neg_unmarked": _run_neg_unmarked, "neg_wrongkey": _run_neg_wrongkey}[u["type"]](u)


def _stamp(u):
    """Load the source, embed, and return everything the decode loops need."""
    base = C.load_source(u["spec"])
    uid = f"{u['type']}|{C.source_id(u['spec'])}|{u['payload_id']}|{u['scheme']}|{u.get('alpha')}"
    payload = S.resolve_payload(u["payload_id"], uid, u["seed"])
    key = wmcore.derive_seed(u["seed"], uid)
    t0 = time.perf_counter()
    stamped = S.embed(u["scheme"], base, payload, key, u.get("alpha"))
    embed_s = time.perf_counter() - t0
    stamped_rgb = stamped.convert("RGB")
    psnr = M.psnr(np.asarray(base.convert("RGB")), np.asarray(stamped_rgb))
    return payload, uid, key, stamped_rgb, embed_s, psnr


def _run_pos(u):
    payload, uid, key, stamped, embed_s, psnr = _stamp(u)
    W, H = stamped.size
    bits = wmcore.text_to_bits(payload)
    base = _base_fields(u, W, H)
    base.update(payload_bytes=len(payload.encode("utf-8")), psnr=psnr, embed_s=embed_s)
    recs = [dict(base, kind="embed", decoder=None, mp=W * H / 1e6)]

    for name in u["attacks"]:
        atk = A.REGISTRY[name]
        rng = np.random.default_rng([u["seed"], _crc(uid + "|" + name)])
        attacked, geom = atk.fn(stamped, rng)
        aw, ah = attacked.size

        ber_r = ber_c = None
        oracle = S.oracle_soft(u["scheme"], attacked, key, geom, (W, H))
        if oracle is not None:
            soft, idx = oracle
            ber_r, ber_c = M.ber_raw(soft, idx, bits), M.ber_combined(soft, idx, bits)

        for dec in u["decoders"]:
            if dec in S.RESCALE_DECODERS and (not u["rescale"] or atk.family not in RESCALE_FAMILIES):
                continue
            out, secs = _decode_timed(dec, attacked, key, u["budget"])
            recs.append(dict(
                base, kind="pos", decoder=dec, attack=name, family=atk.family,
                gate=atk.gate and _in_scope(dec, atk.family, u["payload_id"]),
                outcome=_outcome(out.text, payload), ber_raw=ber_r, ber_comb=ber_c,
                confidence=out.confidence, factor=out.factor, repeats=out.repeats,
                factor_rel_err=S.factor_rel_err(geom, out.factor) if out.text == payload else None,
                budget_exhausted=out.budget_exhausted, decode_s=secs, mp=aw * ah / 1e6))
    return recs


def _run_neg_unmarked(u):
    """Never-stamped image, decoded with the key a real marked image would use."""
    base_img = C.load_source(u["spec"]).convert("RGB")
    W, H = base_img.size
    uid = f"neg|{C.source_id(u['spec'])}"
    key = wmcore.derive_seed(u["seed"], uid)
    base = _base_fields(dict(u, scheme=None, alpha=None, payload_id=None), W, H)
    recs = []
    for dec in u["decoders"]:
        if dec in S.RESCALE_DECODERS and not u["rescale"]:
            continue
        for name in NEG_DECODER_ATTACKS[dec]:
            atk = A.REGISTRY[name]
            rng = np.random.default_rng([u["seed"], _crc(uid + "|" + name)])
            attacked, _ = atk.fn(base_img, rng)
            out, secs = _decode_timed(dec, attacked, key, u["budget"])
            # No payload exists, so any returned text is a false positive: mark
            # it "wrong" (there is nothing it could be "correct" against).
            recs.append(dict(base, kind="neg_unmarked", decoder=dec, attack=name, family=atk.family,
                             gate=True, outcome="none" if out.text is None else "wrong",
                             confidence=out.confidence, budget_exhausted=out.budget_exhausted,
                             decode_s=secs, mp=attacked.size[0] * attacked.size[1] / 1e6))
    return recs


def _run_neg_wrongkey(u):
    payload, uid, key, stamped, _, _ = _stamp(u)
    W, H = stamped.size
    base = _base_fields(u, W, H)
    rng = np.random.default_rng([u["seed"], _crc(uid + "|wrongkeys")])
    wrong = []
    while len(wrong) < u["n_wrong_keys"]:
        k = int(rng.integers(1, 2 ** 31))
        if k != key:
            wrong.append(k)
    recs = []
    for name in WRONGKEY_ATTACKS:
        atk = A.REGISTRY[name]
        attacked, _ = atk.fn(stamped, np.random.default_rng([u["seed"], _crc(uid + "|" + name)]))
        for dec in u["decoders"]:
            for k in wrong:
                out, secs = _decode_timed(dec, attacked, k, u["budget"])
                recs.append(dict(base, kind="neg_wrongkey", decoder=dec, attack=name, family=atk.family,
                                 gate=True, outcome=_outcome(out.text, payload),
                                 confidence=out.confidence, budget_exhausted=out.budget_exhausted,
                                 decode_s=secs, mp=attacked.size[0] * attacked.size[1] / 1e6))
    return recs


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def build_plan(cfg):
    seed = cfg["seed"]
    decoders = cfg["decoders"]
    schemes = sorted({S.DECODER_SCHEME[d] for d in decoders})
    attacks = cfg["attacks"] or list(A.REGISTRY)
    common = {"seed": seed, "budget": cfg["rescale_budget"]}
    units = []

    cells = max(1, len(cfg["kinds"]) * len(cfg["sizes"])) if not cfg["images_dir"] else 1

    def per_cell(total):
        return max(1, math.ceil(total / cells))

    def rank(spec, specs, seed0):
        """Position of `spec` within its (kind, size) cell - so "the first K images
        of every cell run the slow decoder" spreads it across cells."""
        return specs.index(spec) if spec["kind"] == "file" else spec["seed"] - seed0

    pos_specs = C.make_specs(cfg["kinds"], cfg["sizes"], cfg["n_images"], seed, cfg["images_dir"], cfg["max_side"])
    for spec in pos_specs:
        for payload_id in cfg["payloads"]:
            for scheme in schemes:
                unit_decoders = [d for d in decoders if S.DECODER_SCHEME[d] == scheme and S.decoder_applies(d, payload_id)]
                if not unit_decoders:
                    continue
                for alpha in (cfg["alphas"] if scheme == "ss" else [None]):
                    units.append(dict(common, type="pos", spec=spec, payload_id=payload_id, scheme=scheme,
                                      alpha=alpha, attacks=attacks, decoders=unit_decoders,
                                      rescale=rank(spec, pos_specs, seed) < cfg["rescale_images"]))

    # Negatives use seeds disjoint from the positives'.
    neg_seed0 = seed + 1_000_000
    neg_specs = C.make_specs(cfg["kinds"], cfg["sizes"], per_cell(cfg["n_neg"]), neg_seed0,
                             cfg["images_dir"], cfg["max_side"])
    for spec in neg_specs:
        units.append(dict(common, type="neg_unmarked", spec=spec, decoders=list(decoders),
                          rescale=rank(spec, neg_specs, neg_seed0) < per_cell(cfg["n_neg_rescale"])))

    wk_payload = cfg["payloads"][0]
    wk_fast = [d for d in ("ss", "ss_id", "ss_flip", "ss_id_flip") if d in decoders and S.decoder_applies(d, wk_payload)]
    if wk_fast:
        wk_decoders = wk_fast + ([d for d in decoders if d in S.RESCALE_DECODERS and S.decoder_applies(d, wk_payload)]
                                 if cfg["slow_negatives"] else [])
        wk_specs = C.make_specs(cfg["kinds"], cfg["sizes"], per_cell(cfg["n_wrongkey_images"]), seed + 2_000_000,
                                cfg["images_dir"], cfg["max_side"])
        for spec in wk_specs:
            for alpha in cfg["alphas"]:
                units.append(dict(common, type="neg_wrongkey", spec=spec, payload_id=wk_payload, scheme="ss",
                                  alpha=alpha, decoders=wk_decoders, n_wrong_keys=cfg["n_wrong_keys"], rescale=True))
    return units


# One unit writes all its records at once, when it finishes, so "a unit has records" means "a unit is done"
# (bar a process killed mid-write, which leaves at most one torn last line - dropped on resume).
_RECORD_UNIT_TYPE = {"embed": "pos", "pos": "pos", "neg_unmarked": "neg_unmarked", "neg_wrongkey": "neg_wrongkey"}


def _unit_key(u):
    unmarked = u["type"] == "neg_unmarked"
    return (u["type"], C.source_id(u["spec"]), None if unmarked else u.get("payload_id"),
            None if unmarked else u.get("scheme"), None if unmarked else u.get("alpha"))


def _record_key(r):
    return (_RECORD_UNIT_TYPE[r["kind"]], r["image_id"], r.get("payload_id"), r.get("scheme"), r.get("alpha"))


def _load_done(records_path):
    """(records already on disk, keys of the units they complete). Rewrites the file without a torn last line."""
    good, done = [], set()
    with open(records_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add(_record_key(rec))
            except (ValueError, KeyError):
                break                                   # torn final line: everything after it is untrustworthy
            good.append(line if line.endswith("\n") else line + "\n")
    with open(records_path, "w") as fh:
        fh.writelines(good)
    return len(good), done


def execute(units, jobs, records_path, progress=True, resume=False):
    """Run `units`, appending each unit's records to `records_path`. With resume=True the units whose records are
    already there are skipped (after a killed run); the file is otherwise started afresh. Returns the total number
    of records in the file."""
    try:
        from tqdm import tqdm
    except ImportError:  # tqdm is in requirements.txt, but a bench shouldn't die without it
        tqdm = None
    n, mode = 0, "w"
    if resume and os.path.exists(records_path):
        n, done = _load_done(records_path)
        units = [u for u in units if _unit_key(u) not in done]
        mode = "a"
    with open(records_path, mode) as fh:
        def sink(recs):
            nonlocal n
            for r in recs:
                fh.write(json.dumps(_clean(r), allow_nan=False) + "\n")
                n += 1
            fh.flush()

        if jobs <= 1:
            _warmup()
            it = tqdm(units, desc="bench", unit="unit") if (tqdm and progress) else units
            for u in it:
                sink(run_unit(u))
        else:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_warmup) as ex:
                it = ex.map(run_unit, units)
                if tqdm and progress:
                    it = tqdm(it, total=len(units), desc="bench", unit="unit")
                for recs in it:
                    sink(recs)
    return n


# ---------------------------------------------------------------------------
# Metadata / CLI
# ---------------------------------------------------------------------------

def code_fingerprint():
    """Hash of the wmcore sources under test, so a result can always be tied
    to the exact code that produced it."""
    h = hashlib.sha256()
    d = os.path.join(HERE, "..", "cli", "wmcore")
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            with open(os.path.join(d, f), "rb") as fh:
                h.update(f.encode() + b"\0" + fh.read())
    return h.hexdigest()[:12]


def make_meta(cfg, units, n_records, wall_s):
    import cv2
    import PIL
    return {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "suite": cfg["suite"], "seed": cfg["seed"], "jobs": cfg["jobs"], "wall_seconds": round(wall_s, 1),
        "units": len(units), "records": n_records,
        "config": {k: v for k, v in cfg.items() if k not in ("out",)},
        "wmcore_fingerprint": code_fingerprint(),
        "defaults": {"DEFAULT_SS_ALPHA": wmcore.DEFAULT_SS_ALPHA, "SS_RESCALE_RANGE": list(wmcore.SS_RESCALE_RANGE),
                     "DEFAULT_RESCALE_BUDGET_SECONDS": wmcore.DEFAULT_RESCALE_BUDGET_SECONDS},
        "env": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                "pillow": PIL.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                "machine": platform.machine()},
    }


def _parse_sizes(s):
    return [tuple(int(v) for v in p.lower().split("x")) for p in s.split(",")]


def _csv(s):
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Detection / BER / FPR / runtime benchmark for the invisible marks.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--suite", choices=sorted(SUITES), default="standard")
    p.add_argument("--images", metavar="DIR", help="use real images from DIR instead of synthetic ones (recommended for published numbers)")
    p.add_argument("--max-side", type=int, default=1600, help="downscale real images whose long side exceeds this")
    p.add_argument("--n-images", type=int, help="positive images per (kind, size) cell (or max real files)")
    p.add_argument("--n-neg", type=int, help="unmarked images for FPR")
    p.add_argument("--n-neg-rescale", type=int, help="how many of those also run the (slow) rescale search")
    p.add_argument("--n-wrongkey-images", type=int)
    p.add_argument("--n-wrong-keys", type=int, help="wrong keys tried per wrong-key image")
    p.add_argument("--rescale-images", type=int, help="positive images per cell that run the (slow) rescale decoder")
    p.add_argument("--kinds", type=_csv, help="synthetic image kinds: pink,gradnoise")
    p.add_argument("--sizes", type=_parse_sizes, help="e.g. 640x480,1280x960 (synthetic only)")
    p.add_argument("--alphas", type=lambda s: [float(v) for v in _csv(s)], help="--ss-alpha values to sweep")
    p.add_argument("--payloads", type=_csv, help=f"payload ids ({', '.join(S.PAYLOADS)})")
    p.add_argument("--attacks", type=_csv, help="attack names / families / globs (see --list-attacks)")
    p.add_argument("--decoders", type=_csv, default=["lsb", "ss", "ss_rescale"])
    p.add_argument("--rescale-budget", type=float, default=wmcore.DEFAULT_RESCALE_BUDGET_SECONDS,
                   help="max_seconds for the rescale search (0 = unbounded)")
    p.add_argument("--slow-negatives", action="store_true", help="also run the rescale decoder on wrong-key trials")
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--resume", action="store_true",
                   help="continue into --out: skip units whose records are already in records.jsonl (a killed run); "
                        "use the same arguments as the run being continued")
    p.add_argument("--jobs", type=int, default=1, help="worker processes (timings are only valid with 1)")
    p.add_argument("--out", help="output directory (default bench/results/<timestamp>)")
    p.add_argument("--baseline", metavar="SUMMARY_JSON", help="compare to a previous summary.json; exit 1 on regression")
    p.add_argument("--save-baseline", metavar="PATH", help="also write this run's summary.json to PATH")
    p.add_argument("--det-tol", type=float, default=0.05, help="max tolerated absolute detection-rate drop vs baseline")
    p.add_argument("--fpr-tol", type=float, default=0.02, help="max tolerated absolute FPR rise vs baseline")
    p.add_argument("--gate-runtime", type=float, metavar="FACTOR", help="also fail if decode p50 exceeds FACTOR x baseline")
    p.add_argument("--report-only", metavar="DIR", help="re-render summary/report from DIR/records.jsonl without running")
    p.add_argument("--list-attacks", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def resolve_config(args):
    cfg = dict(SUITES[args.suite])
    for k in ("n_images", "n_neg", "n_neg_rescale", "n_wrongkey_images", "n_wrong_keys", "rescale_images",
              "kinds", "sizes", "alphas", "payloads"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.attacks:
        names = A.select(args.attacks)
        if not names:
            sys.exit(f"--attacks {args.attacks} matched nothing; see --list-attacks")
        cfg["attacks"] = names
    for d in args.decoders:
        if d not in S.DECODERS:
            sys.exit(f"unknown decoder {d!r}; choose from {', '.join(S.DECODERS)}")
    for pid in cfg["payloads"]:
        if pid not in S.PAYLOADS:
            sys.exit(f"unknown payload id {pid!r}; choose from {', '.join(S.PAYLOADS)}")
    cfg.update(suite=args.suite, seed=args.seed, jobs=args.jobs, decoders=args.decoders, images_dir=args.images,
               max_side=args.max_side, rescale_budget=(args.rescale_budget or None),
               slow_negatives=args.slow_negatives)
    return cfg


def summarize_and_report(out_dir, meta):
    with open(os.path.join(out_dir, "records.jsonl")) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    rows = M.summarize(records)
    runtime = M.runtime_table(records)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"meta": meta, "rows": rows, "runtime": runtime}, fh, indent=1, allow_nan=False)
    R.write_csv(rows, os.path.join(out_dir, "summary.csv"))
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write(R.render_markdown(meta, rows, runtime))
    return rows, runtime


def main(argv=None):
    args = parse_args(argv)
    if args.list_attacks:
        for a in A.REGISTRY.values():
            print(f"{a.name:22s} {a.family:12s} {'gated' if a.gate else 'info '}  {a.note}")
        return 0

    if args.report_only:
        out_dir = args.report_only
        with open(os.path.join(out_dir, "meta.json")) as fh:
            meta = json.load(fh)
        rows, _ = summarize_and_report(out_dir, meta)
        print(f"re-rendered {out_dir}/report.md")
    else:
        cfg = resolve_config(args)
        out_dir = args.out or os.path.join(HERE, "results", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        units = build_plan(cfg)
        print(f"suite={cfg['suite']}  units={len(units)}  jobs={cfg['jobs']}  out={out_dir}")
        t0 = time.perf_counter()
        n = execute(units, cfg["jobs"], os.path.join(out_dir, "records.jsonl"), progress=not args.no_progress,
                    resume=args.resume)
        meta = make_meta(cfg, units, n, time.perf_counter() - t0)
        if args.resume:
            meta["resumed"] = True      # wall_seconds then covers only the resumed part
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        rows, _ = summarize_and_report(out_dir, meta)
        print(f"{n} trials in {meta['wall_seconds']}s -> {out_dir}/report.md")

    if args.save_baseline:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_baseline)), exist_ok=True)
        with open(os.path.join(out_dir, "summary.json")) as fh, open(args.save_baseline, "w") as out:
            out.write(fh.read())

    if args.baseline:
        with open(args.baseline) as fh:
            base_rows = json.load(fh)["rows"]
        problems = M.compare_summaries(base_rows, rows, det_tol=args.det_tol, fpr_tol=args.fpr_tol,
                                       runtime_factor=args.gate_runtime)
        if problems:
            print(f"\nREGRESSIONS vs {args.baseline}:")
            for line in problems:
                print("  -", line)
            return 1
        print(f"no regressions vs {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
