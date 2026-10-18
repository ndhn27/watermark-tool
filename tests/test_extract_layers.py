"""P2 (step 4/5): `extract` checks EVERY layer and reports each one on its own - found / nothing readable /
not checked and why - instead of stopping at the first layer that gives an ID and folding the rest into
nothing. The headline fields (found, text, scheme, ...) are chosen exactly as before; the old behaviour is
pinned by test_id_ledger.py / test_orient.py, this file is what is NEW: ExtractResult.layers, the --layers
text, the disagreement warning, and the one deliberate shortcut (spread layer: plain decode only when LSB
already read the ID)."""

import json

import cv2
import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import c2pa_provenance
import wmapp
import wmapp.extraction as ex
import wmcore
from wmapp import ledger as L
from wmcore import wmid

LAYERS = ["lsb", "spread", "metadata", "c2pa"]


def _image(path, w=640, h=480, seed=1):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    arr = np.clip(np.stack([xx, yy, (xx + yy) / 2], -1) + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return str(path)


@pytest.fixture
def led(tmp_path):
    return str(tmp_path / "ledger.sqlite3")


def _stamp(tmp_path, led, name="out.png", seed_img=1, **kw):
    src = _image(tmp_path / f"in_{name}", seed=seed_img)
    out = str(tmp_path / name)
    kw.setdefault("text", "@me")
    kw.setdefault("seed", 3)
    kw.setdefault("invisible", True)
    kw.setdefault("invisible_mode", "spread")
    return wmapp.watermark(src, out, stamp_id=True, ledger=led, **kw), out


def _by_layer(res):
    return {lr.layer: lr for lr in res.layers}


# --- structure ---------------------------------------------------------------------------------

def test_every_extract_reports_all_four_layers_in_a_fixed_order(tmp_path, led):
    r, out = _stamp(tmp_path, led)
    res = wmapp.extract(out, ledger=led)
    assert [lr.layer for lr in res.layers] == LAYERS
    got = _by_layer(res)
    assert got["lsb"].status == "none"
    assert got["spread"].status == "found" and got["spread"].wm_id == r.wm_id
    assert got["metadata"].status == "found" and got["metadata"].wm_id == r.wm_id
    assert got["c2pa"].status in ("none", "skipped")            # never stamped with --c2pa
    assert res.scheme == "spread" and res.wm_id == r.wm_id       # the headline is chosen as it always was


def test_a_stripped_and_recompressed_copy_shows_which_layers_survived(tmp_path, led):
    r, out = _stamp(tmp_path, led)
    reposted = str(tmp_path / "reposted.jpg")
    Image.open(out).convert("RGB").save(reposted, quality=60)    # a typical re-upload: metadata gone, pixels recompressed
    res = wmapp.extract(reposted, ledger=led)
    got = _by_layer(res)
    assert got["metadata"].status == "none"                      # the layer a platform strips...
    assert got["spread"].status == "found" and got["spread"].wm_id == r.wm_id   # ...and the one that survives
    assert res.scheme == "spread"


def test_a_free_text_payload_is_a_text_layer_not_an_id(tmp_path, led):
    out = str(tmp_path / "o.png")
    wmapp.watermark(_image(tmp_path / "in.png"), out, text="@me", seed=1, invisible=True, invisible_text="secret")
    res = wmapp.extract(out, use_ledger=False)
    lsb = _by_layer(res)["lsb"]
    assert lsb.status == "text" and lsb.text == "secret" and lsb.wm_id is None
    assert "secret" in lsb.describe() and "not a watermark ID" in lsb.describe()
    assert res.text == "secret" and res.scheme == "lsb"


def test_a_junk_lsb_decode_is_reported_as_text_and_does_not_hide_the_spread_id(tmp_path, led, monkeypatch):
    r, out = _stamp(tmp_path, led)
    monkeypatch.setattr(ex, "extract_invisible", lambda path, max_length=0: "xY")
    res = wmapp.extract(out, ledger=led)
    got = _by_layer(res)
    assert got["lsb"].status == "text" and got["spread"].status == "found"
    assert res.scheme == "spread" and res.wm_id == r.wm_id


# --- the text report ------------------------------------------------------------------------

def test_the_layer_block_is_printed_only_when_asked_and_only_adds_lines(tmp_path, led):
    r, out = _stamp(tmp_path, led)
    plain = wmapp.extract(out, ledger=led).report_lines()
    full = wmapp.extract(out, ledger=led, layers=True).report_lines()
    assert not any("layer by layer" in ln for ln in plain)       # default output unchanged
    assert full[:len(plain)] == plain                             # ...and --layers only appends
    block = full[len(plain):]
    assert "layer by layer" in block[0]
    for name in ("LSB mark", "spread-spectrum mark", "image metadata", "C2PA manifest"):
        assert any(ln.strip().startswith(name + ":") for ln in block[1:])
    assert sum(wmid.display(r.wm_id) in ln for ln in block) == 2  # spread + metadata


def test_layers_option_is_in_the_public_dict_and_message(tmp_path, led):
    r, out = _stamp(tmp_path, led)
    d = wmapp.extract(out, ledger=led, layers=True).to_dict(public=True)
    assert [lr["layer"] for lr in d["layers"]] == LAYERS
    assert d["layers_disagree"] is False
    assert "layer by layer" in d["message"]
    assert led not in json.dumps(d) and str(r.ss_key) not in json.dumps(d["layers"])   # nothing server-side leaks


# --- modes, missing keys, missing packages ---------------------------------------------------

def test_a_layer_that_was_not_checked_says_why(tmp_path, led):
    r, out = _stamp(tmp_path, led)
    only_lsb = _by_layer(wmapp.extract(out, ledger=led, mode="lsb"))
    assert only_lsb["lsb"].status == "none"
    assert all(only_lsb[k].status == "skipped" for k in ("spread", "metadata", "c2pa"))
    assert "--mode lsb" in only_lsb["spread"].note and "--mode auto" in only_lsb["metadata"].note

    only_spread = _by_layer(wmapp.extract(out, ledger=led, mode="spread"))
    assert only_spread["lsb"].status == "skipped" and only_spread["spread"].status == "found"

    no_key = wmapp.extract(out, use_ledger=False)                 # no key, no ledger: spread can't be tried
    got = _by_layer(no_key)
    assert got["spread"].status == "skipped" and "--ss-key" in got["spread"].note
    assert got["metadata"].status == "found" and no_key.scheme == "metadata"


def test_missing_c2pa_support_is_a_skipped_layer_not_a_miss(tmp_path, led, monkeypatch):
    r, out = _stamp(tmp_path, led)
    monkeypatch.setattr(c2pa_provenance, "AVAILABLE", False)
    c2pa = _by_layer(wmapp.extract(out, ledger=led))["c2pa"]
    assert c2pa.status == "skipped" and "isn't installed" in c2pa.note


def test_a_search_that_ran_out_of_time_is_inconclusive_not_none(tmp_path, led):
    plain = _image(tmp_path / "plain.png")
    res = wmapp.extract(plain, use_ledger=False, ss_key=12345, rescale=True, rescale_budget_seconds=0)
    spread = _by_layer(res)["spread"]
    assert res.budget_exhausted and spread.status == "inconclusive"
    assert "not a confirmed absence" in spread.describe()


# --- the one shortcut ---------------------------------------------------------------------------------

def test_when_lsb_already_read_the_id_the_resize_search_is_skipped(tmp_path, led):
    # A resize destroys an LSB mark, so a resize search on a copy whose LSB read fine can only burn its budget
    # (--rescale-budget-seconds, 20s by default) on a "not found". Skipped, and the layer says so.
    a, lsb_out = _stamp(tmp_path, led, name="lsb.png", invisible_mode="lsb")
    assert "lsb" in a.id_layers
    _stamp(tmp_path, led, name="spr.png", seed_img=2)             # puts a spread key in the ledger, so spread has keys to try
    res = wmapp.extract(lsb_out, ledger=led, rescale=True)
    got = _by_layer(res)
    assert res.scheme == "lsb" and res.wm_id == a.wm_id and got["lsb"].status == "found"
    assert got["spread"].status == "none" and "resize search skipped" in got["spread"].note
    assert res.rescale_tried is False and not res.budget_exhausted   # no resize search was started
    assert res.flipped is None and res.rescaled_by is None and res.confidence is None


def test_but_the_flip_search_still_runs_because_a_top_bottom_flip_does_not_break_lsb(tmp_path, led):
    # Why the shortcut is only the RESIZE search: a top-bottom flip of a lossless copy still reads through the LSB
    # layer (measured, bench/results/EXTRACT-LAYERS-NOTES.md), so "LSB read an ID" does NOT mean "not flipped".
    # Here a spread-marked copy (ID X) is flipped and then stamped with an LSB ID (Y): LSB reads Y, and the
    # spread layer must still find X - flipped back - and the two disagree.
    from PIL import ImageOps
    x, out_x = _stamp(tmp_path, led, name="x.png", seed_img=1)
    flipped = str(tmp_path / "flipped.png")
    ImageOps.flip(Image.open(out_x).convert("RGB")).save(flipped)
    y = wmapp.watermark(flipped, str(tmp_path / "y.png"), text="@me", seed=3, stamp_id=True, ledger=led,
                        invisible=True, invisible_mode="lsb")
    res = wmapp.extract(str(tmp_path / "y.png"), ledger=led, rescale=True)
    got = _by_layer(res)
    assert res.scheme == "lsb" and res.wm_id == y.wm_id
    assert got["spread"].status == "found" and got["spread"].wm_id == x.wm_id
    assert "top-bottom" in got["spread"].note and "resize search skipped" in got["spread"].note
    assert set(res.disagreement) == {x.wm_id, y.wm_id}


# --- disagreement ----------------------------------------------------------------------------------

def test_layers_that_read_different_ids_are_flagged_not_hidden(tmp_path, led):
    x, out_x = _stamp(tmp_path, led, name="x.png", seed_img=1)
    y, out_y = _stamp(tmp_path, led, name="y.png", seed_img=2)
    assert x.wm_id != y.wm_id
    info = PngInfo()
    info.add_text(wmcore.metadata.PNG_ID_KEY, Image.open(out_y).info[wmcore.metadata.PNG_ID_KEY])
    mixed = str(tmp_path / "mixed.png")                           # X's pixels, Y's metadata
    Image.open(out_x).save(mixed, pnginfo=info)

    res = wmapp.extract(mixed, ledger=led)
    assert res.scheme == "spread" and res.wm_id == x.wm_id        # headline: first match in the fixed order, as before
    assert res.disagreement == {x.wm_id: ["spread"], y.wm_id: ["metadata"]}
    lines = res.report_lines()                                    # printed WITHOUT --layers: it must not be missed
    warning = next(ln for ln in lines if ln.startswith("Warning: the layers disagree"))
    assert wmid.display(x.wm_id) in warning and wmid.display(y.wm_id) in warning
    assert res.to_dict()["layers_disagree"] is True


def test_layers_that_agree_get_no_warning(tmp_path, led):
    r, out = _stamp(tmp_path, led)
    res = wmapp.extract(out, ledger=led)
    assert not res.disagreement and not any("disagree" in ln for ln in res.report_lines())


# --- video ----------------------------------------------------------------------------------------

def _pink_pan_video(path, w=320, h=240, fps=12, n_frames=24, pad=64, seed=0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    big = np.random.default_rng(seed).integers(0, 256, (h + pad, w + pad, 3), dtype=np.uint8)
    for i in range(n_frames):
        off = i % pad
        vw.write(big[off:off + h, off:off + w])
    vw.release()
    return str(path)


def test_a_video_reports_its_two_possible_layers_and_says_why_the_others_do_not_apply(tmp_path):
    src = _pink_pan_video(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    r = wmapp.watermark(src, out, text="@niny", stamp_id=True, invisible=True, invisible_mode="spread",
                        jitter_refresh_seconds=1.0, workers=2)
    res = wmapp.extract(out, ss_key=r.ss_key, jitter_refresh_seconds=1.0, use_ledger=False)
    got = _by_layer(res)
    assert [lr.layer for lr in res.layers] == LAYERS
    assert got["lsb"].status == "skipped" and "re-encoding" in got["lsb"].note
    assert got["metadata"].status == "skipped"
    assert got["spread"].status == "found" and "frame(s) decoded this ID" in got["spread"].note
    assert got["c2pa"].status in ("none", "skipped")
    assert res.scheme == "spread" and res.video_scan is not None   # headline and frame breakdown unchanged

    unmarked = wmapp.extract(src, ss_key=r.ss_key, jitter_refresh_seconds=1.0, use_ledger=False)   # never stamped
    spread = _by_layer(unmarked)["spread"]
    assert not unmarked.found and spread.status == "none" and "0/24" in spread.note


# --- the C2PA layer in its "found" state (it had only ever been seen as "none"/"skipped" in this sandbox, whose c2pa-python
#     0.38.0 rejects the c2pa.watermarked action; with 0.37.3-0.37.10 these run - see requirements.txt) -----------------------

def _needs_working_c2pa(tmp_path, led, monkeypatch):
    monkeypatch.setattr(c2pa_provenance, "DEFAULT_CERT_DIR", tmp_path / "certs")
    try:
        r, out = _stamp(tmp_path, led, c2pa=True)
    except Exception as exc:                                    # signing refused by this c2pa-python: not what these test
        pytest.skip(f"C2PA signing is not available with this c2pa-python: {type(exc).__name__}: {str(exc)[:80]}")
    if "c2pa" not in r.id_layers:                               # 0.37.12 / 0.38.0 reject the c2pa.watermarked action and the
        pytest.skip("this c2pa-python did not sign the file (see requirements.txt)")   # run reports it without raising
    return r, out


def test_the_c2pa_layer_reports_found_when_the_metadata_is_stripped(tmp_path, led, monkeypatch):
    r, out = _needs_working_c2pa(tmp_path, led, monkeypatch)
    monkeypatch.setattr(wmcore.metadata, "read_wm_id_field", lambda path: None)         # as if a platform stripped the metadata
    res = wmapp.extract(out, ledger=led, layers=True)
    got = _by_layer(res)
    assert got["c2pa"].status == "found" and got["c2pa"].wm_id == r.wm_id
    assert got["metadata"].status == "none" and got["spread"].status == "found"
    assert not res.disagreement
    assert any(ln.strip().startswith("C2PA manifest:") and wmid.display(r.wm_id) in ln for ln in res.report_lines())


def test_a_c2pa_id_that_differs_from_the_pixel_layers_is_flagged_as_a_disagreement(tmp_path, led, monkeypatch):
    r, out = _needs_working_c2pa(tmp_path, led, monkeypatch)
    other = wmid.generate()
    monkeypatch.setattr(wmcore.metadata, "read_wm_id_field", lambda path: None)
    monkeypatch.setattr(ex, "_id_from_c2pa", lambda path: other)                         # a manifest carrying another ID
    res = wmapp.extract(out, ledger=led, layers=True)
    assert res.scheme == "spread" and res.wm_id == r.wm_id                               # the headline order is unchanged
    assert set(res.disagreement) == {r.wm_id, other} and res.disagreement[other] == ["c2pa"]
    assert any(ln.startswith("Warning: the layers disagree") for ln in res.report_lines())
