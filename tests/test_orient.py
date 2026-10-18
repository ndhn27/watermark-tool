"""Mirrored copies (left-right / top-bottom / both): wmcore.orient and its wiring into wmapp.extract.

What is pinned here, and why (measurements: bench/results/P3-NOTES.md):
  - a watermark ID is read from a mirrored copy in every orientation, at a size that is NOT a multiple of 8 and
    after a JPEG re-save (the mirror is exact, so the block grid comes back pixel for pixel);
  - a FREE-TEXT payload is read unmirrored exactly as before and is deliberately NOT read mirrored (the text
    decoder's accept rule is too loose to run on three more orientations);
  - the ID search goes over every orientation BEFORE the text decoder is consulted, so junk text from the
    unmirrored orientation cannot beat the right orientation's ID (a design bug caught before it shipped);
  - never-stamped images match nothing; orientations=("none",) is the old reader, result for result;
  - `wmapp.extract` reports the mirroring (`flipped`, a report line, to_dict) for the --ss-key path, the ledger
    path, and the --rescale path (which checks the mirrorings first, then searches for a resize as before).
"""

import io
import random

import numpy as np
import pytest
from PIL import Image

import bench_corpus as C           # bench/ on sys.path via conftest.py
import wmapp
import wmcore
from wmcore import orient, spread, wmid

KEY = 20260922
FLIP = {
    "none": lambda im: im,
    "h": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "v": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    "hv": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
}


def _photo_like(width=643, height=477, seed=0):
    """Deliberately not a multiple of 8 in either direction: a mirror moves the unused edge rows/columns to the
    other side, so this is the size that would expose a grid that did not come back exactly."""
    xx, yy = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
    rng = np.random.default_rng(seed)
    arr = np.clip(np.stack([xx, yy, (xx + yy) / 2], axis=-1) + rng.normal(0, 8, (height, width, 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")


def _valid_id(seed=1):
    return wmid.generate(random.Random(seed))


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# --- the decoder ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("how", orient.ORIENTATIONS)
def test_a_mirrored_id_is_read_in_every_orientation(how):
    wm = _valid_id(3)
    stamped = wmcore.embed_spread_spectrum(_photo_like(), wm, KEY).convert("RGB")
    got = wmcore.find_spread_spectrum_payload_or_id_oriented(FLIP[how](stamped), KEY)
    assert got == (wm, how)
    assert wmcore.find_spread_spectrum_id_oriented(FLIP[how](stamped), KEY) == (wm, how)


@pytest.mark.parametrize("how", ["h", "v", "hv"])
def test_a_mirrored_id_survives_a_jpeg_resave(how):
    wm = _valid_id(4)
    stamped = wmcore.embed_spread_spectrum(_photo_like(), wm, KEY).convert("RGB")
    assert wmcore.find_spread_spectrum_id_oriented(_jpeg(FLIP[how](stamped), 75), KEY) == (wm, how)


def test_the_old_reader_cannot_read_a_mirrored_id():
    # the "before": the bench measured 0/240 on mirrored copies; this is the same fact on one image
    wm = _valid_id(5)
    mirrored = FLIP["h"](wmcore.embed_spread_spectrum(_photo_like(), wm, KEY).convert("RGB"))
    old = wmcore.find_spread_spectrum_payload_or_id(wmcore.spread_spectrum_block_correlations(mirrored, KEY))
    assert old != wm


def test_a_text_payload_is_read_unmirrored_and_not_read_mirrored():
    stamped = wmcore.embed_spread_spectrum(_photo_like(), "@yourhandle", KEY).convert("RGB")
    assert wmcore.find_spread_spectrum_payload_or_id_oriented(stamped, KEY) == ("@yourhandle", "none")
    for how in ("h", "v", "hv"):
        # Pins the decision (bench/results/p3-flip-text-probe.txt): mirrored text is not recovered. Not `== NO_MATCH`:
        # the text decoder may, rarely, return junk on the unmirrored orientation of a mirrored image (that is the
        # old reader's behaviour, unchanged) - what must never happen is the CORRECT text coming back.
        assert wmcore.find_spread_spectrum_payload_or_id_oriented(FLIP[how](stamped), KEY).text != "@yourhandle"


def test_an_unmarked_image_matches_nothing():
    plain = _photo_like(seed=9).convert("RGB")
    assert wmcore.find_spread_spectrum_payload_or_id_oriented(plain, KEY) == orient.NO_MATCH
    assert wmcore.find_spread_spectrum_id_oriented(plain, KEY) == orient.NO_MATCH


def test_the_id_is_searched_in_every_orientation_before_any_text_is_accepted(monkeypatch):
    # The bug this pins: with a retry loop that consulted the loose text decoder on each orientation in turn, a
    # junk text hit on the unmirrored orientation would return before the mirrored one - where the real ID is -
    # was ever tried. Force the text decoder to "find" junk on everything: the mirrored ID must still win.
    wm = _valid_id(6)
    mirrored = FLIP["v"](wmcore.embed_spread_spectrum(_photo_like(), wm, KEY).convert("RGB"))
    monkeypatch.setattr(spread, "find_spread_spectrum_payload", lambda corrs, max_length=10000: "junk")
    assert wmcore.find_spread_spectrum_payload_or_id_oriented(mirrored, KEY) == (wm, "v")
    # control: with nothing better to find, the text decoder's answer is still returned, on the unmirrored image only
    plain = _photo_like(seed=9).convert("RGB")
    assert wmcore.find_spread_spectrum_payload_or_id_oriented(plain, KEY) == ("junk", "none")


@pytest.mark.parametrize("payload", ["id", "text", "none"])
def test_restricted_to_the_unmirrored_orientation_it_is_the_old_reader(payload):
    img = _photo_like(seed=2)
    if payload == "id":
        img = wmcore.embed_spread_spectrum(img, _valid_id(7), KEY)
    elif payload == "text":
        img = wmcore.embed_spread_spectrum(img, "@yourhandle", KEY)
    img = img.convert("RGB")
    old = wmcore.find_spread_spectrum_payload_or_id(wmcore.spread_spectrum_block_correlations(img, KEY))
    new = wmcore.find_spread_spectrum_payload_or_id_oriented(img, KEY, orientations=("none",))
    assert new.text == old


def test_unflip_is_its_own_inverse_and_rejects_unknown_orientations():
    a = np.arange(35).reshape(5, 7)
    for how in orient.ORIENTATIONS:
        assert np.array_equal(orient.unflip(orient.unflip(a, how), how), a)
    assert np.array_equal(orient.unflip(a, "h"), a[:, ::-1]) and np.array_equal(orient.unflip(a, "v"), a[::-1])
    with pytest.raises(ValueError):
        orient.unflip(a, "diagonal")


# --- wmapp.extract -------------------------------------------------------------------------------------------

@pytest.fixture
def led(tmp_path):
    return str(tmp_path / "ledger.sqlite3")


def _stamped(tmp_path, led, name="out.png", **kw):
    src = str(tmp_path / "in.png")
    C.load_source(dict(kind="pink", w=320, h=240, seed=8)).convert("RGB").save(src)
    out = str(tmp_path / name)
    # ss_key / ID pinned: both are random per stamp and the ID decoder is not universal (see test_id_ledger.py)
    r = wmapp.watermark(src, out, stamp_id=True, ledger=led, text="@me", seed=3, invisible=True,
                        invisible_mode="spread", ss_key=kw.pop("ss_key", 20260922), id_value=kw.pop("id_value", "V2ACCRPDPV"))
    return r, out


def _mirrored_copy(tmp_path, out, how, name):
    copy = str(tmp_path / name)
    FLIP[how](Image.open(out).convert("RGB")).save(copy)        # a fresh PIL image: the PNG's ID metadata is dropped
    return copy


@pytest.mark.parametrize("how,label", [("h", "left-right"), ("v", "top-bottom"), ("hv", "both ways")])
def test_extract_reads_a_mirrored_copy_with_the_ledger_key_and_says_so(tmp_path, led, how, label):
    r, out = _stamped(tmp_path, led)
    copy = _mirrored_copy(tmp_path, out, how, "mirrored.png")
    res = wmapp.extract(copy, ledger=led)
    assert res.found and res.wm_id == r.wm_id and res.scheme == "spread" and res.flipped == how
    assert res.record and res.record["ss_key"] == r.ss_key
    assert res.keys_tried == res.keys_available == 1                # keys, not (key, mirroring) attempts
    assert any("flipping the image back" in ln and label in ln for ln in res.report_lines())
    assert res.to_dict()["flipped"] == how and res.to_dict(public=True)["flipped"] == how


def test_extract_reads_a_mirrored_copy_with_a_given_key(tmp_path, led):
    r, out = _stamped(tmp_path, led)
    copy = _mirrored_copy(tmp_path, out, "h", "mirrored.png")
    res = wmapp.extract(copy, mode="spread", ss_key=r.ss_key)
    assert res.found and res.text == r.wm_id and res.flipped == "h"


def test_extract_finds_the_mirroring_before_it_runs_the_resize_search(tmp_path, led):
    r, out = _stamped(tmp_path, led)
    copy = _mirrored_copy(tmp_path, out, "v", "mirrored.png")
    res = wmapp.extract(copy, mode="spread", ss_key=r.ss_key, rescale=True)
    assert res.found and res.text == r.wm_id and res.flipped == "v"
    assert res.rescaled_by is None and not res.budget_exhausted


def test_an_unmirrored_copy_reports_no_mirroring(tmp_path, led):
    r, out = _stamped(tmp_path, led)
    res = wmapp.extract(out, ledger=led, use_ledger=True)
    assert res.found and res.flipped is None and res.to_dict()["flipped"] is None
    assert not any("flipping" in ln for ln in res.report_lines())
    res = wmapp.extract(out, mode="spread", ss_key=r.ss_key)
    assert res.found and res.flipped is None


def test_the_right_key_is_found_among_several_runs_on_a_mirrored_copy(tmp_path, led):
    runs = []
    for i in range(3):
        src = str(tmp_path / f"in{i}.png")
        C.load_source(dict(kind="pink", w=320, h=240, seed=20 + i)).convert("RGB").save(src)
        runs.append(wmapp.watermark(src, str(tmp_path / f"o{i}.png"), text="@me", seed=i, stamp_id=True, ledger=led,
                                    invisible=True, invisible_mode="spread", ss_key=1000 + i))
    copy = _mirrored_copy(tmp_path, str(tmp_path / "o1.png"), "hv", "m1.png")
    res = wmapp.extract(copy, ledger=led)
    assert res.found and res.wm_id == runs[1].wm_id and res.flipped == "hv"
    assert res.record["ss_key"] == 1001 and res.keys_available == 3 and res.keys_tried == 3


def test_a_mirrored_text_payload_is_still_not_found_by_extract(tmp_path):
    # the documented limit: mirrored TEXT payloads are not read (only IDs are), and nothing wrong is reported
    src = str(tmp_path / "in.png")
    C.load_source(dict(kind="pink", w=320, h=240, seed=8)).convert("RGB").save(src)
    out = str(tmp_path / "out.png")
    wmapp.watermark(src, out, text="@me", seed=3, invisible=True, invisible_mode="spread", ss_key=7,
                    invisible_text="@yourhandle")
    assert wmapp.extract(out, mode="spread", ss_key=7).text == "@yourhandle"
    res = wmapp.extract(_mirrored_copy(tmp_path, out, "h", "m.png"), mode="spread", ss_key=7)
    assert res.text != "@yourhandle" and res.flipped is None
