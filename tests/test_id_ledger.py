"""--id end to end: one short ID into every layer that can carry it, recorded in the ledger, and read back
by `extract` with no key to remember."""

import os

import cv2
import numpy as np
import pytest
from PIL import Image

import bench_corpus as C           # bench/ on sys.path via conftest.py
import c2pa_provenance
import wmapp
import wmcore
from wmapp import ledger as L
from wmapp.errors import SpecError
from wmcore import wmid


def _image(path, w=640, h=480, seed=1):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    arr = np.clip(np.stack([xx, yy, (xx + yy) / 2], -1) + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return str(path)


@pytest.fixture
def led(tmp_path):
    return str(tmp_path / "ledger.sqlite3")


def _stamp(tmp_path, led, name="out.png", src=None, **kw):
    src = src or _image(tmp_path / "in.png")
    out = str(tmp_path / name)
    kw.setdefault("text", "@me")
    kw.setdefault("seed", 3)
    r = wmapp.watermark(src, out, stamp_id=True, ledger=led, **kw)
    return r, out


# --- the pipeline -----------------------------------------------------------

def test_every_layer_that_can_carries_the_same_id_and_the_ledger_agrees(tmp_path, led):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread", grid_id=True)
    assert wmid.parse(r.wm_id) == r.wm_id and r.id_layers == ["grid", "spread", "metadata"]
    assert Image.open(out).info[wmcore.metadata.PNG_ID_KEY] == r.wm_id
    rec = L.Ledger(led, create=False).lookup(r.wm_id)
    assert rec["status"] == "done" and rec["layers"] == r.id_layers and rec["ss_key"] == r.ss_key
    assert rec["output_sha256"] == L.file_sha256(out) and rec["input_name"] == "in.png"
    assert rec["spec"]["text"] == "@me" and rec["spec"]["seed"] == 3
    note = next(n for n in r.notes if n.code == "id_stamped")
    assert wmid.display(r.wm_id) in note.message and note.data["layers"] == r.id_layers


def test_the_hidden_mark_carries_the_id_not_the_text(tmp_path, led):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="lsb", invisible_text="ignored sentence")
    assert wmapp.extract(out, mode="lsb", use_ledger=False).text == r.wm_id
    assert any(n.code == "id_text_ignored" for n in r.notes)


def test_without_stamp_id_nothing_changes_and_no_ledger_is_touched(tmp_path, led):
    src = _image(tmp_path / "in.png")
    out = str(tmp_path / "o.png")
    r = wmapp.watermark(src, out, text="@me", seed=3, invisible=True, invisible_text="hello")
    assert r.wm_id is None and r.id_layers == [] and not os.path.exists(led)
    assert wmcore.metadata.PNG_ID_KEY not in Image.open(out).info
    assert wmapp.extract(out, use_ledger=False).text == "hello"
    assert os.listdir(tmp_path).count("default-ledger.sqlite3") == 0


def test_the_id_option_pair_without_stamp_id_is_noted_and_ignored(tmp_path, led):
    r = wmapp.watermark(_image(tmp_path / "in.png"), str(tmp_path / "o.png"), text="x", grid_id=True)
    assert [n.code for n in r.notes] == ["id_options_ignored"] and r.wm_id is None


def test_the_grid_shows_the_id_only_when_asked(tmp_path, led):
    src = _image(tmp_path / "in.png")
    plain = wmapp.watermark(src, str(tmp_path / "a.png"), text="@me", seed=3, stamp_id=True, ledger=led,
                            id_value=wmid.generate())
    with_id = wmapp.watermark(src, str(tmp_path / "b.png"), text="@me", seed=3, stamp_id=True, ledger=led,
                              grid_id=True)
    bare = wmapp.watermark(src, str(tmp_path / "c.png"), text="@me", seed=3)
    px = lambda p: np.asarray(Image.open(p).convert("RGBA"))
    assert np.array_equal(px(tmp_path / "a.png"), px(tmp_path / "c.png"))      # ID off the grid: the grid is untouched
    assert not np.array_equal(px(tmp_path / "b.png"), px(tmp_path / "c.png"))
    assert "grid" in with_id.id_layers and "grid" not in plain.id_layers and bare.wm_id is None


def test_an_explicit_id_is_used_and_may_not_be_reused(tmp_path, led):
    mine = wmid.generate()
    r, _ = _stamp(tmp_path, led, id_value=wmid.display(mine).lower())         # any spelling
    assert r.wm_id == mine
    with pytest.raises(SpecError, match="already recorded"):
        _stamp(tmp_path, led, name="again.png", id_value=mine)


def test_prepare_rejects_bad_ids_batch_ids_and_unusable_ledgers(tmp_path, led):
    with pytest.raises(SpecError, match="not a valid watermark ID"):
        wmapp.prepare(text="x", stamp_id=True, ledger=led, id_value="AAAAAAAAAA")
    with pytest.raises(SpecError, match="whole --batch"):
        wmapp.prepare(text="x", stamp_id=True, ledger=led, id_value=wmid.generate(), batch=True)
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    with pytest.raises(SpecError, match="ledger"):
        wmapp.prepare(text="x", stamp_id=True, ledger=str(blocker / "l.sqlite3"))
    p = wmapp.prepare(text="x", stamp_id=True, ledger=led)
    assert p.spec.ledger == led and {"id_ledger", "id_no_hidden_mark"} <= {n.code for n in p.notes}


def test_the_id_is_reserved_before_the_file_is_stamped_and_a_failed_run_keeps_its_row(tmp_path, led, monkeypatch):
    seen = {}
    real = wmcore.save_with_metadata

    def spy(image, output_path, **kw):
        seen["status"] = L.Ledger(led, create=False).lookup(kw["wm_id"])["status"]
        raise RuntimeError("disk full")

    monkeypatch.setattr(wmcore, "save_with_metadata", spy)
    with pytest.raises(RuntimeError, match="disk full"):
        _stamp(tmp_path, led)
    assert seen["status"] == "pending"                                         # recorded BEFORE it could escape
    (rec,) = L.Ledger(led, create=False).recent()
    assert rec["status"] == "failed" and "disk full" in rec["note"]
    monkeypatch.setattr(wmcore, "save_with_metadata", real)


def test_jpeg_output_carries_the_id_in_exif(tmp_path, led):
    pytest.importorskip("piexif")
    r, out = _stamp(tmp_path, led, name="o.jpg", invisible=True, invisible_mode="spread")
    assert "metadata" in r.id_layers and wmcore.metadata.read_wm_id_field(out) == r.wm_id


def test_webp_output_gets_no_metadata_layer_and_the_record_says_so(tmp_path, led):
    r, out = _stamp(tmp_path, led, name="o.webp", invisible=True, invisible_mode="spread")
    assert "metadata" not in r.id_layers and "spread" in r.id_layers
    assert L.Ledger(led, create=False).lookup(r.wm_id)["layers"] == r.id_layers


@pytest.mark.skipif(not c2pa_provenance.AVAILABLE, reason="needs c2pa-python + cryptography")
def test_the_id_rides_in_the_c2pa_creativework_assertion(tmp_path, led, monkeypatch):
    monkeypatch.setattr(c2pa_provenance, "DEFAULT_CERT_DIR", tmp_path / "certs")
    r, out = _stamp(tmp_path, led, c2pa=True, author="Jane")
    assert r.c2pa == "signed" and "c2pa" in r.id_layers
    info = wmapp.verify(out)
    assert info["wm_id"] == r.wm_id and info["author"] == "Jane"
    # and without an ID the assertion (and read_manifest's wm_id) stay as they were
    plain = wmapp.watermark(_image(tmp_path / "p.png"), str(tmp_path / "plain.png"), text="x", c2pa=True, author="Jane")
    assert plain.c2pa == "signed" and wmapp.verify(str(tmp_path / "plain.png"))["wm_id"] is None


# --- batch -------------------------------------------------------------------

def test_a_batch_gives_every_file_its_own_id_and_shares_one_key(tmp_path, led):
    d = tmp_path / "in"
    d.mkdir()
    for i in range(3):
        _image(d / f"p{i}.png", 320, 240, seed=i)
    res = wmapp.watermark_batch(str(d), str(tmp_path / "out"),
                                wmapp.WatermarkSpec(text="@me", seed=7, stamp_id=True, ledger=led, invisible=True,
                                                    invisible_mode="spread"), workers=2)
    assert res.ok and len({r.wm_id for r in res.results}) == 3
    ledger = L.Ledger(led, create=False)
    assert ledger.count() == 3 and ledger.spread_keys() == [res.ss_key]
    for r in res.results:
        rec = ledger.lookup(r.wm_id)
        assert rec["status"] == "done" and rec["output_name"] == os.path.basename(r.output)
        assert rec["output_sha256"] == L.file_sha256(r.output)


# --- video ---------------------------------------------------------------------

def _tiny_video(path, frames=6):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (160, 120))
    if not vw.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    rng = np.random.default_rng(0)
    for f in range(frames):
        vw.write(np.clip(rng.normal(110, 25, (120, 160, 3)) + f, 0, 255).astype(np.uint8))
    vw.release()
    return str(path)


def test_a_video_id_is_recorded_and_the_layers_are_honest(tmp_path, led):
    src = _tiny_video(tmp_path / "v.mp4")
    r = wmapp.watermark(src, str(tmp_path / "o.mp4"), text="@me", seed=1, workers=1, spacing=60, font_size=14,
                        stamp_id=True, grid_id=True, ledger=led)
    assert r.kind == "video" and r.id_layers == ["grid"]
    rec = L.Ledger(led, create=False).lookup(r.wm_id)
    assert rec["kind"] == "video" and rec["status"] == "done" and rec["invisible_mode"] is None
    assert rec["output_sha256"] == L.file_sha256(str(tmp_path / "o.mp4"))


def test_a_video_id_that_is_written_nowhere_is_a_warning(tmp_path, led):
    src = _tiny_video(tmp_path / "v.mp4")
    r = wmapp.watermark(src, str(tmp_path / "o.mp4"), text="@me", seed=1, workers=1, spacing=60, font_size=14,
                        stamp_id=True, ledger=led)
    warn = [n for n in r.notes if n.code == "video_id_not_embedded"]
    assert warn and warn[0].level == "warning" and r.id_layers == []


# --- extract -------------------------------------------------------------------

def test_extract_needs_no_key_when_the_ledger_has_it(tmp_path, led):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread")
    res = wmapp.extract(out, ledger=led)
    assert res.found and res.wm_id == res.text == r.wm_id and res.scheme == "spread"
    assert res.record["id"] == r.wm_id and res.exact_match is True and res.keys_available == 1
    lines = res.report_lines()
    assert lines[0] == f"Watermark ID: {wmid.display(r.wm_id)}" and any("byte-identical" in ln for ln in lines)
    assert not any(str(r.ss_key) in ln for ln in lines)
    assert res.to_dict()["record"]["ss_key"] == r.ss_key and "ss_key" not in res.to_dict(public=True)["record"]


def test_a_heavily_recompressed_copy_is_found_by_the_id_aware_decoder(tmp_path, led):
    # JPEG q=20 is below the q>=50 the README promises. Through the full pipeline (grid on, spread mark carrying
    # the ID) the header-hunting decoder finds nothing in this file - the mark is there, no single repeat of
    # its 32-bit header is intact - while the ID decoder (fixed period, check-verified) reads it.
    # bench/results/P2-NOTES.md has the measurement; this pins the behaviour end to end.
    # ss_key and ID are pinned: both are random per stamp, and the ID decoder is not universal at q20 (9 of 10
    # keys tried when this was written), so an unpinned test would fail one run in ten. This pair: text decoder
    # reads nothing, ID decoder reads the ID.
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread", ss_key=20260921, id_value="WR5TT382BJ")
    copy = str(tmp_path / "q20.jpg")
    Image.open(out).convert("RGB").save(copy, quality=20)
    res = wmapp.extract(copy, ledger=led)
    assert res.found and res.scheme == "spread" and res.wm_id == r.wm_id
    assert res.record and res.record["ss_key"] == r.ss_key and res.exact_match is False


@pytest.mark.parametrize("scale", [0.75, 1.25])
def test_a_resized_copy_is_found_by_the_id_aware_resize_search(tmp_path, led, scale):
    # bench image pink-320x240 seed 8, stamped through the full pipeline (grid on): with --rescale, the resize
    # search that hunts for a 32-bit header finds nothing at either scale (bench/results/P2-NOTES.md, "Inside
    # --rescale"), while the ID-aware search reads the ID - both via the ledger's keys and via a given --ss-key.
    src = str(tmp_path / "in.png")
    C.load_source(dict(kind="pink", w=320, h=240, seed=8)).convert("RGB").save(src)
    out = str(tmp_path / "out.png")
    # ss_key and ID pinned (both are random per stamp; this pair is one where the text decoder's search misses at
    # both scales and the ID-aware one reads - with another key the ID search can miss too, 1 in 2 tried).
    r = wmapp.watermark(src, out, stamp_id=True, ledger=led, text="@me", seed=3, invisible=True,
                        invisible_mode="spread", ss_key=20260922, id_value="V2ACCRPDPV")
    copy = str(tmp_path / "resized.png")
    Image.open(out).convert("RGB").resize((round(320 * scale), round(240 * scale)), Image.BICUBIC).save(copy)

    via_ledger = wmapp.extract(copy, ledger=led, rescale=True)
    assert via_ledger.found and via_ledger.wm_id == r.wm_id
    assert via_ledger.rescaled_by == pytest.approx(1 / scale, rel=0.03)
    assert via_ledger.record and via_ledger.record["ss_key"] == r.ss_key

    via_key = wmapp.extract(copy, mode="spread", ss_key=r.ss_key, rescale=True)
    assert via_key.found and via_key.text == r.wm_id


def test_a_recompressed_copy_is_found_but_is_not_byte_identical(tmp_path, led):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread")
    jpg = str(tmp_path / "re.jpg")
    Image.open(out).convert("RGB").save(jpg, quality=65)
    res = wmapp.extract(jpg, ledger=led)
    assert res.found and res.wm_id == r.wm_id and res.exact_match is False
    assert any("NOT byte-identical" in ln for ln in res.report_lines())


def test_a_resized_copy_needs_rescale_and_the_ledger_key_is_used_for_it(tmp_path, led):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread")
    small = str(tmp_path / "small.png")
    Image.open(out).convert("RGB").resize((480, 360), Image.LANCZOS).save(small)
    assert not wmapp.extract(small, ledger=led).found
    res = wmapp.extract(small, ledger=led, rescale=True)
    assert res.found and res.wm_id == r.wm_id and abs(res.rescaled_by - 4 / 3) < 0.02
    assert res.record is not None


def test_the_right_key_is_found_among_several_runs(tmp_path, led):
    src = _image(tmp_path / "in.png")
    runs = [wmapp.watermark(src, str(tmp_path / f"o{i}.png"), text="@me", seed=i, stamp_id=True, ledger=led,
                            invisible=True, invisible_mode="spread", ss_key=1000 + i) for i in range(4)]
    for i in (0, 2, 3):
        res = wmapp.extract(str(tmp_path / f"o{i}.png"), ledger=led)
        assert res.wm_id == runs[i].wm_id and res.record["ss_key"] == 1000 + i and res.keys_available == 4


def test_an_explicit_key_beats_the_ledger_keys(tmp_path, led):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread", ss_key=555)
    res = wmapp.extract(out, ledger=led, ss_key=555)
    assert res.wm_id == r.wm_id and res.keys_available == 0                       # the ledger only did the lookup


def test_no_ledger_no_key_no_luck_and_the_hint_says_what_to_do(tmp_path, led):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread")
    stripped = str(tmp_path / "stripped.png")
    Image.open(out).save(stripped)                                               # same pixels, metadata gone
    assert wmcore.metadata.PNG_ID_KEY not in Image.open(stripped).info
    res = wmapp.extract(stripped, use_ledger=False)
    assert not res.found and "pass --ss-key" in res.report_lines()[0]
    res = wmapp.extract(stripped, use_ledger=False, ss_key=r.ss_key)
    assert res.wm_id == r.wm_id and res.record is None and res.ledger_path is None
    assert wmapp.extract(out, use_ledger=False).scheme == "metadata"              # ...and with metadata intact it is found there


def test_an_id_the_ledger_has_never_heard_of_is_still_reported(tmp_path, led):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread")
    other = L.Ledger(str(tmp_path / "other.sqlite3"))
    res = wmapp.extract(out, ledger=str(tmp_path / "other.sqlite3"), ss_key=r.ss_key)
    assert res.wm_id == r.wm_id and res.record is None and res.exact_match is None
    assert any("not in the ledger" in ln for ln in res.report_lines())
    assert other.count() == 0


def test_extract_uses_the_default_ledger_only_if_it_exists_and_never_creates_it(tmp_path, monkeypatch, led):
    monkeypatch.setenv(L.ENV_VAR, led)
    plain = _image(tmp_path / "plain.png")
    assert not wmapp.extract(plain).found and not os.path.exists(led)              # nothing created by reading
    r = wmapp.watermark(_image(tmp_path / "in.png"), str(tmp_path / "o.png"), text="@me", seed=1, stamp_id=True,
                        invisible=True, invisible_mode="spread")                    # default ledger = $ENV
    assert os.path.exists(led)
    assert wmapp.extract(str(tmp_path / "o.png")).record["id"] == r.wm_id


def test_an_explicit_ledger_that_does_not_exist_is_an_error_not_a_miss(tmp_path):
    with pytest.raises(SpecError, match="no such file"):
        wmapp.extract(_image(tmp_path / "p.png"), ledger=str(tmp_path / "typo.sqlite3"))


def test_spread_mode_without_any_key_source_still_asks_for_one(tmp_path):
    with pytest.raises(SpecError, match="requires --ss-key"):
        wmapp.extract(_image(tmp_path / "p.png"), mode="spread")


def test_a_junk_lsb_decode_does_not_hide_a_real_spread_id(tmp_path, led, monkeypatch):
    r, out = _stamp(tmp_path, led, invisible=True, invisible_mode="spread")
    import wmapp.extraction as ex
    monkeypatch.setattr(ex, "extract_invisible", lambda path, max_length=0: "xY")   # what an unmarked image can yield
    res = wmapp.extract(out, ledger=led)
    assert res.wm_id == r.wm_id and res.scheme == "spread"
    monkeypatch.setattr(ex, "extract_invisible", lambda path, max_length=0: "legacy text")
    res = wmapp.extract(_image(tmp_path / "p2.png"), ledger=led)                   # nothing better: legacy result stands
    assert res.text == "legacy text" and res.wm_id is None and res.scheme == "lsb"


def test_metadata_alone_yields_the_id_in_auto_mode_only(tmp_path, led):
    r, out = _stamp(tmp_path, led)                                                  # no hidden mark at all
    res = wmapp.extract(out, ledger=led)
    assert res.found and res.wm_id == r.wm_id and res.scheme == "metadata"
    assert any("weakest layer" in ln for ln in res.report_lines())
    assert not wmapp.extract(out, ledger=led, mode="lsb").found


@pytest.mark.skipif(not c2pa_provenance.AVAILABLE, reason="needs c2pa-python + cryptography")
def test_the_c2pa_manifest_is_the_last_place_extract_looks(tmp_path, led, monkeypatch):
    monkeypatch.setattr(c2pa_provenance, "DEFAULT_CERT_DIR", tmp_path / "certs")
    r, out = _stamp(tmp_path, led, c2pa=True)
    monkeypatch.setattr(wmcore.metadata, "read_wm_id_field", lambda path: None)   # as if the metadata were stripped
    res = wmapp.extract(out, ledger=led)
    assert res.wm_id == r.wm_id and res.scheme == "c2pa"


def test_text_payloads_read_back_exactly_as_before(tmp_path, led):
    src = _image(tmp_path / "in.png")
    out = str(tmp_path / "o.png")
    wmapp.watermark(src, out, text="@me", seed=1, invisible=True, invisible_text="secret")
    res = wmapp.extract(out, ledger=str(L.Ledger(led).path))
    assert res.text == "secret" and res.wm_id is None and res.record is None
    assert res.report_lines() == ["Hidden payload: secret"]


def test_an_unreadable_default_ledger_is_set_aside_with_a_note_not_a_crash(tmp_path, monkeypatch):
    bad = tmp_path / "bad.sqlite3"
    bad.write_bytes(b"not a database" * 40)
    monkeypatch.setenv(L.ENV_VAR, str(bad))
    src = _image(tmp_path / "in.png")
    out = str(tmp_path / "o.png")
    wmapp.watermark(src, out, text="@me", seed=1, invisible=True, invisible_text="secret")      # legacy stamp: no ledger involved
    res = wmapp.extract(out)
    assert res.text == "secret" and res.ledger_error and any("couldn't be read" in ln for ln in res.report_lines())
    assert str(bad) not in res.to_dict(public=True)["message"]                                    # HTTP never sees the path
    with pytest.raises(SpecError, match="usable ledger"):                                         # but naming it explicitly is an error
        wmapp.extract(out, ledger=str(bad))
