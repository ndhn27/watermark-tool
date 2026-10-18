"""The application layer (cli/wmapp): the schema everything is generated from, validation,
notes, limits, and the watermark / batch / extract use-cases - exercised through the Python
API directly, with no argparse and no HTTP in the way. The front-ends' own tests
(test_cli_shell.py, test_serve.py) and the cross-front-end proof (test_frontends_parity.py)
sit on top of this."""

import argparse
import os
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

import c2pa_provenance
import wmapp
import wmcore


def _image(path, w=320, h=240, seed=1):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    arr = np.clip(np.stack([xx, yy, (xx + yy) / 2], -1) + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return str(path)


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGBA"))


# ---------------------------------------------------------------------------
# The schema is the single source of truth
# ---------------------------------------------------------------------------

def test_option_names_and_flags_are_unique():
    names = [o.name for o in wmapp.OPTIONS]
    flags = [o.cli_flag for o in wmapp.OPTIONS]
    assert len(names) == len(set(names)) and len(flags) == len(set(flags))


def test_spec_fields_are_exactly_the_spec_options_with_schema_defaults():
    spec = wmapp.WatermarkSpec(text="x")
    assert set(spec.__dataclass_fields__) == {o.name for o in wmapp.SPEC_OPTIONS}
    for o in wmapp.SPEC_OPTIONS:
        if not o.required:
            assert getattr(spec, o.name) == o.default, o.name
    assert {o.name for o in wmapp.RUN_OPTIONS} == {"batch", "workers"}      # CLI-only, not in the spec/JSON API
    assert not {"batch", "workers"} & set(spec.__dataclass_fields__)


def test_cli_parser_defaults_match_the_schema():
    parser = argparse.ArgumentParser()
    wmapp.add_arguments(parser, wmapp.OPTIONS)
    ns = parser.parse_args(["--text", "x"])
    for o in wmapp.OPTIONS:
        if not o.required:
            assert getattr(ns, o.name) == o.default, o.name
    assert ns.use_shared_memory is True                               # --no-shared-memory stores False


def test_a_cli_namespace_and_the_python_api_build_the_same_spec():
    parser = argparse.ArgumentParser()
    wmapp.add_arguments(parser, wmapp.OPTIONS)
    ns = parser.parse_args(["--text", "@me", "--blend", "multiply", "--ink", "0.4", "--seed", "3", "--copyright", "CC"])
    assert wmapp.spec_from_namespace(wmapp.WatermarkSpec, ns) == wmapp.WatermarkSpec(
        text="@me", blend="multiply", ink=0.4, seed=3, copyright_text="CC")


def test_requires_hints_refer_to_real_options():
    names = {o.name for o in wmapp.SPEC_OPTIONS}
    for o in wmapp.SPEC_OPTIONS:
        assert all(k in names for k, _ in o.requires), o.name


def test_schema_is_json_and_can_hide_local_options():
    import json
    full = wmapp.schema()
    json.dumps(full)
    public = wmapp.schema(include_local=False)
    local = {o["name"] for o in full["watermark"] if o["local"]}
    assert local == {"font_path", "face_cascade", "adaptive_preview", "c2pa_cert", "c2pa_key", "ledger"}
    assert not local & {o["name"] for o in public["watermark"]}
    assert "batch" not in {o["name"] for o in full["watermark"]}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad, fragment", [
    ({"text": "x", "opacity": 300}, "opacity"),
    ({"text": "x", "opacity": -1}, "opacity"),
    ({"text": "x", "spacing": 0}, "spacing"),
    ({"text": "x", "ink": 1.5}, "ink"),
    ({"text": "x", "blend": "screen"}, "blend"),
    ({"text": "x", "angle": "steep"}, "angle"),
    ({"text": "x", "opacity": True}, "opacity"),
    ({"text": "x", "opacity": 90.5}, "opacity"),
    ({"text": "x", "invisible": "yes"}, "invisible"),
    ({"text": ""}, "text"),
    ({}, "text"),
    ({"text": "x", "blnd": "multiply"}, "unknown option"),
    ({"text": "x", "batch": True}, "unknown option"),
])
def test_invalid_specs_are_rejected_with_the_option_named(bad, fragment):
    with pytest.raises(wmapp.SpecError, match=fragment):
        wmapp.WatermarkSpec.from_mapping(bad)


def test_spec_error_is_a_value_error_so_old_callers_still_catch_it():
    assert issubclass(wmapp.SpecError, ValueError)


def test_local_options_are_refused_when_asked_and_allowed_otherwise():
    wmapp.WatermarkSpec.from_mapping({"text": "x", "font_path": "/some/font.ttf"})            # CLI / Python API
    with pytest.raises(wmapp.SpecError, match="font_path"):
        wmapp.WatermarkSpec.from_mapping({"text": "x", "font_path": "/some/font.ttf"}, allow_local=False)
    wmapp.WatermarkSpec.from_mapping({"text": "x", "font_path": None}, allow_local=False)      # unset is fine


def test_spec_round_trips_through_a_dict():
    spec = wmapp.WatermarkSpec(text="@me", blend="overlay", seed=9, invisible=True, invisible_mode="spread", ss_key=5)
    assert wmapp.WatermarkSpec.from_mapping(spec.to_dict()) == spec


def test_empty_optional_strings_count_as_unset():
    assert wmapp.WatermarkSpec(text="x", author="").author is None


def test_mutating_after_construction_is_revalidated_by_replace():
    spec = wmapp.WatermarkSpec(text="x")
    with pytest.raises(wmapp.SpecError):
        spec.replace(opacity=999)


# ---------------------------------------------------------------------------
# prepare(): decisions made once per run
# ---------------------------------------------------------------------------

def test_prepare_generates_one_ss_key_and_says_so():
    p = wmapp.prepare(text="x", invisible=True, invisible_mode="spread")
    assert isinstance(p.spec.ss_key, int) and p.spec.ss_key > 0
    assert [n.code for n in p.notes] == ["ss_key_generated"]
    assert str(p.spec.ss_key) in p.notes[0].message and p.notes[0].data["ss_key"] == p.spec.ss_key
    # an explicit key is respected and silent
    q = wmapp.prepare(text="x", invisible=True, invisible_mode="spread", ss_key=77)
    assert q.spec.ss_key == 77 and q.notes == ()


def test_prepare_is_idempotent_on_a_prepared_spec():
    p = wmapp.prepare(text="x", invisible=True, invisible_mode="spread")
    assert wmapp.prepare(p) is p


def test_prepare_rejects_half_a_certificate_pair():
    with pytest.raises(wmapp.SpecError, match="must be given together"):
        wmapp.prepare(text="x", c2pa=True, c2pa_cert="/x.pem")


def test_prepare_notes_ignored_c2pa_options_without_c2pa():
    p = wmapp.prepare(text="x", c2pa_title="T")
    assert [n.code for n in p.notes] == ["c2pa_options_ignored"]


def test_prepare_reports_missing_c2pa_packages(monkeypatch):
    monkeypatch.setattr(c2pa_provenance, "AVAILABLE", False)
    with pytest.raises(wmapp.UnavailableError, match="pip install c2pa-python"):
        wmapp.prepare(text="x", c2pa=True)


@pytest.mark.skipif(not c2pa_provenance.AVAILABLE, reason="needs c2pa-python + cryptography")
def test_prepare_resolves_the_dev_certificate_once(tmp_path, monkeypatch):
    monkeypatch.setattr(c2pa_provenance, "DEFAULT_CERT_DIR", tmp_path)
    first = wmapp.prepare(text="x", c2pa=True, c2pa_alg="ps256")
    assert first.spec.c2pa_cert and os.path.exists(first.spec.c2pa_cert)
    assert first.spec.c2pa_alg == c2pa_provenance.DEFAULT_ALG
    assert {n.code for n in first.notes} == {"c2pa_alg_ignored", "c2pa_dev_cert"}
    assert next(n for n in first.notes if n.code == "c2pa_dev_cert").data["created"] is True
    second = wmapp.prepare(text="x", c2pa=True)
    assert next(n for n in second.notes if n.code == "c2pa_dev_cert").data["created"] is False


def test_batch_prepare_drops_adaptive_preview_with_a_note():
    p = wmapp.prepare(text="x", adaptive=True, adaptive_preview="/tmp/mask.png", batch=True)
    assert p.spec.adaptive_preview is None and p.notes[0].code == "adaptive_preview_batch"


# ---------------------------------------------------------------------------
# watermark(): one image
# ---------------------------------------------------------------------------

def test_watermark_writes_the_file_returns_a_result_and_prints_nothing(tmp_path, capsys):
    src, out = _image(tmp_path / "in.png"), str(tmp_path / "out.png")
    r = wmapp.watermark(src, out, text="@me", seed=7)
    assert os.path.exists(out) and r.kind == "image" and r.output == out and r.c2pa == "off" and r.seconds > 0
    assert Image.open(out).size == Image.open(src).size
    assert not np.array_equal(_pixels(src), _pixels(out))
    assert capsys.readouterr().out == "" and capsys.readouterr().err == ""          # a library must not print


def test_same_seed_same_pixels_different_seed_different_pixels(tmp_path):
    src = _image(tmp_path / "in.png")
    a, b, c = (str(tmp_path / n) for n in ("a.png", "b.png", "c.png"))
    wmapp.watermark(src, a, text="@me", seed=7)
    wmapp.watermark(src, b, text="@me", seed=7)
    wmapp.watermark(src, c, text="@me", seed=8)
    assert np.array_equal(_pixels(a), _pixels(b))
    assert not np.array_equal(_pixels(a), _pixels(c))


def test_a_spec_object_a_dict_and_keywords_are_equivalent(tmp_path):
    src = _image(tmp_path / "in.png")
    outs = []
    for i, arg in enumerate((wmapp.WatermarkSpec(text="@me", seed=7, blend="multiply"),
                             {"text": "@me", "seed": 7, "blend": "multiply"}, None)):
        out = str(tmp_path / f"o{i}.png")
        if arg is None:
            wmapp.watermark(src, out, text="@me", seed=7, blend="multiply")
        else:
            wmapp.watermark(src, out, arg)
        outs.append(_pixels(out))
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2])


def test_keyword_options_override_the_spec(tmp_path):
    src = _image(tmp_path / "in.png")
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    wmapp.watermark(src, a, wmapp.WatermarkSpec(text="@me", seed=7), blend="multiply")
    wmapp.watermark(src, b, text="@me", seed=7, blend="multiply")
    assert np.array_equal(_pixels(a), _pixels(b))


def test_options_cannot_be_combined_with_a_prepared_spec(tmp_path):
    p = wmapp.prepare(text="x")
    with pytest.raises(wmapp.SpecError, match="already-prepared"):
        wmapp.watermark(_image(tmp_path / "in.png"), str(tmp_path / "o.png"), p, opacity=10)


def test_spread_spectrum_mark_round_trips_with_the_generated_key(tmp_path):
    src, out = _image(tmp_path / "in.png", 640, 480), str(tmp_path / "out.png")
    r = wmapp.watermark(src, out, text="@me", seed=7, invisible=True, invisible_mode="spread")
    assert r.ss_key and any(n.code == "ss_key_generated" for n in r.notes)
    res = wmapp.extract(out, ss_key=r.ss_key)
    assert res.found and res.text == "@me" and res.scheme == "spread"


def test_lsb_mark_round_trips_and_jpeg_output_warns(tmp_path):
    src = _image(tmp_path / "in.png")
    ok = wmapp.watermark(src, str(tmp_path / "o.png"), text="@me", invisible=True, invisible_text="hello")
    assert wmapp.extract(str(tmp_path / "o.png")).text == "hello" and ok.notes == []
    lossy = wmapp.watermark(src, str(tmp_path / "o.jpg"), text="@me", invisible=True)
    assert [n.code for n in lossy.notes] == ["lsb_lossy_format"] and lossy.notes[0].level == "warning"


def test_on_note_receives_prepare_notes_then_file_notes_in_order(tmp_path):
    seen = []
    r = wmapp.watermark(_image(tmp_path / "in.png"), str(tmp_path / "o.jpg"), text="x", c2pa_title="T",
                        invisible=True, on_note=seen.append)
    codes = ["c2pa_options_ignored", "lsb_lossy_format"]          # run-level note first, then the per-file warning
    assert [n.code for n in seen] == codes
    assert [n.code for n in r.notes] == codes                      # and the result carries the same list
    assert all(isinstance(n, wmapp.Note) for n in seen)


def test_metadata_reaches_the_png(tmp_path):
    out = str(tmp_path / "o.png")
    wmapp.watermark(_image(tmp_path / "in.png"), out, text="x", author="Jane", copyright_text="CC")
    info = Image.open(out).info
    assert "Jane" in str(info) and "CC" in str(info)


def test_adaptive_mode_writes_the_preview_mask(tmp_path):
    mask = str(tmp_path / "mask.png")
    wmapp.watermark(_image(tmp_path / "in.png"), str(tmp_path / "o.png"), text="x", adaptive=True, adaptive_preview=mask)
    assert Image.open(mask).mode == "L"


@pytest.mark.skipif(not c2pa_provenance.AVAILABLE, reason="needs c2pa-python + cryptography")
def test_c2pa_signing_reports_signed_and_a_manifest_appears(tmp_path, monkeypatch):
    monkeypatch.setattr(c2pa_provenance, "DEFAULT_CERT_DIR", tmp_path / "certs")
    out = str(tmp_path / "o.png")
    r = wmapp.watermark(_image(tmp_path / "in.png"), out, text="x", c2pa=True, author="Jane", c2pa_title="T")
    assert r.c2pa == "signed"
    assert wmapp.verify(out)["has_manifest"] is True


@pytest.mark.skipif(not c2pa_provenance.AVAILABLE, reason="needs c2pa-python + cryptography")
def test_a_c2pa_failure_degrades_to_a_note_and_keeps_the_watermarked_file(tmp_path, monkeypatch):
    monkeypatch.setattr(c2pa_provenance, "DEFAULT_CERT_DIR", tmp_path / "certs")
    monkeypatch.setattr(c2pa_provenance, "sign_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = str(tmp_path / "o.png")
    r = wmapp.watermark(_image(tmp_path / "in.png"), out, text="x", c2pa=True)
    assert os.path.exists(out) and r.c2pa == "failed"
    assert any(n.code == "c2pa_failed" and "boom" in n.message for n in r.notes)


def test_verify_without_c2pa_raises_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(c2pa_provenance, "AVAILABLE", False)
    with pytest.raises(wmapp.UnavailableError):
        wmapp.verify(_image(tmp_path / "in.png"))


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

def test_a_dense_grid_is_refused_before_any_work_is_done(tmp_path):
    src = _image(tmp_path / "in.png", 640, 480)
    with pytest.raises(wmapp.LimitError, match="watermark tiles"):
        wmapp.watermark(src, str(tmp_path / "o.png"), text="x", spacing=2, limits=wmapp.SERVER_LIMITS)
    assert not (tmp_path / "o.png").exists()
    wmapp.watermark(src, str(tmp_path / "ok.png"), text="x", spacing=250, limits=wmapp.SERVER_LIMITS)   # default is fine


def test_pixel_and_text_limits():
    lim = wmapp.Limits(max_pixels=1000, max_text_chars=5)
    spec = wmapp.WatermarkSpec(text="x")
    with pytest.raises(wmapp.LimitError, match="MP"):
        lim.check_image((100, 100), spec)
    with pytest.raises(wmapp.LimitError, match="characters"):
        lim.check_text("too long")
    wmapp.NO_LIMITS.check_image((10 ** 6, 10 ** 6), spec)                                 # the CLI/API default: unlimited


def _count_drawn_tiles(fn):
    from PIL import ImageDraw
    calls = []
    real = ImageDraw.ImageDraw.text
    ImageDraw.ImageDraw.text = lambda self, *a, **k: (calls.append(1), real(self, *a, **k))[1]
    try:
        fn()
    finally:
        ImageDraw.ImageDraw.text = real
    return len(calls)


@pytest.mark.parametrize("size, spacing", [((320, 240), 100), ((320, 240), 250), ((500, 333), 77), ((200, 200), 40)])
def test_tile_estimate_equals_the_tiles_wmcore_actually_draws(size, spacing):
    # The limit is only as good as the estimate: count real draws and demand equality, so a change to
    # the grid loops in wmcore/visible.py breaks this test instead of silently unprotecting the server.
    drawn = _count_drawn_tiles(lambda: wmcore.create_watermark_layer(size, "x", 12, 90, 30, spacing, None, seed=1))
    assert wmapp.estimate_tiles(size, spacing) == drawn


def test_tile_estimate_includes_the_adaptive_infill_grid():
    size, spacing = (320, 240), 100
    sens = np.ones((size[1], size[0]), dtype=np.float32)
    drawn = _count_drawn_tiles(lambda: wmcore.create_watermark_layer(
        size, "x", 12, 90, 30, spacing, None, seed=1, sensitivity_map=sens, adaptive_strength=0.6,
        infill_spacing_divisor=2.0))
    assert wmapp.estimate_tiles(size, spacing, adaptive=True, infill_divisor=2.0) == drawn


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _batch_dir(tmp_path, n=3):
    d = tmp_path / "in"
    d.mkdir()
    for i in range(n):
        _image(d / f"p{i}.png", seed=i)
    return str(d)


def test_batch_processes_every_image_with_a_shared_key_and_per_file_seeds(tmp_path):
    src, out = _batch_dir(tmp_path), str(tmp_path / "out")
    seen, progress = [], []
    res = wmapp.watermark_batch(src, out, wmapp.WatermarkSpec(text="@me", seed=5, invisible=True,
                                invisible_mode="spread"), workers=2, on_note=seen.append,
                                on_progress=lambda d, t: progress.append((d, t)))
    assert res.ok and len(res.results) == 3 and sorted(os.listdir(out)) == ["p0.png", "p1.png", "p2.png"]
    assert progress[0] == (0, 3) and progress[-1] == (3, 3)
    assert isinstance(res.ss_key, int) and all(wmapp.extract(os.path.join(out, f), ss_key=res.ss_key).found for f in os.listdir(out))
    assert "batch_images_start" in [n.code for n in seen] and "batch_images_done" in [n.code for n in res.notes]


def test_batch_gives_each_file_its_own_jitter_but_reproducibly(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    same = _image(tmp_path / "base.png")
    shutil.copy(same, d / "a.png")
    shutil.copy(same, d / "b.png")                                   # identical inputs...
    o1, o2 = str(tmp_path / "o1"), str(tmp_path / "o2")
    wmapp.watermark_batch(str(d), o1, wmapp.WatermarkSpec(text="@me", seed=5), workers=1)
    wmapp.watermark_batch(str(d), o2, wmapp.WatermarkSpec(text="@me", seed=5), workers=1)
    assert not np.array_equal(_pixels(o1 + "/a.png"), _pixels(o1 + "/b.png"))          # ...different pattern per file
    assert np.array_equal(_pixels(o1 + "/a.png"), _pixels(o2 + "/a.png"))              # ...same across runs


def test_batch_reports_a_bad_file_and_carries_on(tmp_path):
    src = _batch_dir(tmp_path, 2)
    with open(os.path.join(src, "broken.png"), "wb") as fh:
        fh.write(b"not a png")
    res = wmapp.watermark_batch(src, str(tmp_path / "out"), wmapp.WatermarkSpec(text="x"), workers=1)
    assert not res.ok and [os.path.basename(p) for p, _ in res.failures] == ["broken.png"]
    assert len(res.results) == 2
    assert any("FAILED" in n.message for n in res.notes)


def test_batch_with_nothing_to_do(tmp_path):
    (tmp_path / "empty").mkdir()
    res = wmapp.watermark_batch(str(tmp_path / "empty"), str(tmp_path / "out"), wmapp.WatermarkSpec(text="x"))
    assert res.results == [] and [n.code for n in res.notes] == ["no_files"]


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def _tiny_video(path, frames=8):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (160, 120))
    if not vw.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    rng = np.random.default_rng(0)
    for f in range(frames):
        vw.write(np.clip(rng.normal(110, 25, (120, 160, 3)) + f, 0, 255).astype(np.uint8))
    vw.release()
    return str(path)


def test_video_goes_through_the_same_entry_point(tmp_path, capsys):
    src, out = _tiny_video(tmp_path / "v.mp4"), str(tmp_path / "o.mp4")
    r = wmapp.watermark(src, out, text="@me", seed=1, workers=1, invisible=True, author="A", spacing=60, font_size=14)
    assert r.kind == "video" and os.path.getsize(out) > 0
    # --invisible is no longer blanket-ignored for video (see test_video_invisible.py) - with
    # invisible_mode left at its "lsb" default (video-unsupported: never survives re-encoding), it
    # produces its own dedicated warning instead, alongside --author still being image-only here
    # (no --c2pa, so nothing writes it anywhere for video).
    assert [n.code for n in r.notes] == ["video_options_ignored", "video_lsb_unsupported"]
    assert "--author is image-only" in r.notes[0].message
    assert "no hidden mark was embedded" in r.notes[1].message
    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8 and cap.read()[0]
    assert capsys.readouterr().out == ""                                                # progress off, nothing printed


def _tiny_video_with_a_cut(path, frames_per_half=6):
    # Two very differently-lit halves back to back, standing in for a hard
    # scene cut, so --scene-cut-detect has something real to catch rather
    # than only ever seeing the periodic jitter-refresh path.
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (160, 120))
    if not vw.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    rng = np.random.default_rng(0)
    for f in range(frames_per_half):
        vw.write(np.clip(rng.normal(60, 15, (120, 160, 3)) + f, 0, 255).astype(np.uint8))
    for f in range(frames_per_half):
        vw.write(np.clip(rng.normal(210, 15, (120, 160, 3)) - f, 0, 255).astype(np.uint8))
    vw.release()
    return str(path)


def test_video_with_adaptive_tracking_and_scene_cut_detection(tmp_path):
    # End-to-end smoke test for feature (5): the full apply_watermark_video
    # path with --adaptive --adaptive-track --scene-cut-detect all active
    # and a real segment boundary (jitter_refresh_seconds short enough to
    # fire at least once over 12 frames @ 6fps), not just AdaptiveTracker in
    # isolation (see test_adaptive.py for that).
    src = _tiny_video_with_a_cut(tmp_path / "cut.mp4")
    out = str(tmp_path / "o.mp4")
    r = wmapp.watermark(
        src, out, text="@me", seed=1, workers=1, spacing=40,
        adaptive=True, adaptive_track=True, adaptive_redetect_seconds=0.5,
        scene_cut_detect=True, scene_cut_threshold=0.12,
        jitter_refresh_seconds=1.0,
    )
    assert r.kind == "video" and os.path.getsize(out) > 0
    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12 and cap.read()[0]


def test_video_adaptive_track_can_be_turned_off_for_the_old_behavior(tmp_path):
    # --no-adaptive-track should still run end to end (single frame-0
    # detection reused for the whole clip, as before feature (5)).
    src = _tiny_video(tmp_path / "v2.mp4")
    out = str(tmp_path / "o2.mp4")
    r = wmapp.watermark(
        src, out, text="@me", seed=1, workers=1, spacing=40,
        adaptive=True, adaptive_track=False, jitter_refresh_seconds=1.0,
    )
    assert r.kind == "video" and os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_extract_reports_not_found_with_a_useful_hint(tmp_path):
    res = wmapp.extract(_image(tmp_path / "plain.png"))
    assert not res.found and res.text is None
    assert "pass --ss-key" in res.report_lines()[0]
    assert "try --rescale" in wmapp.extract(_image(tmp_path / "plain.png"), ss_key=5).report_lines()[0]


def test_extract_mode_and_range_errors(tmp_path):
    img = _image(tmp_path / "plain.png")
    with pytest.raises(wmapp.SpecError, match="requires --ss-key"):
        wmapp.extract(img, mode="spread")
    for bad in ((2.0, 1.0), (0.0, 2.0), (-1.0, 2.0), (1.0, 1.0)):
        with pytest.raises(wmapp.SpecError, match="MIN < MAX"):
            wmapp.extract(img, mode="spread", ss_key=1, rescale=True, rescale_range=bad)


def test_extract_rescale_recovers_a_resized_image(tmp_path):
    src, out = _image(tmp_path / "in.png", 640, 480), str(tmp_path / "o.png")
    wmapp.watermark(src, out, text="@me", seed=7, invisible=True, invisible_mode="spread", ss_key=4242)
    small = str(tmp_path / "small.png")
    Image.open(out).convert("RGB").resize((480, 360), Image.LANCZOS).save(small)
    assert not wmapp.extract(small, ss_key=4242).found                          # plain decode can't
    res = wmapp.extract(small, ss_key=4242, rescale=True)
    assert res.found and res.text == "@me" and res.scheme == "spread" and abs(res.rescaled_by - 4 / 3) < 0.02
    assert "recovered after resampling" in "\n".join(res.report_lines())


def test_extract_options_are_validated_like_watermark_options():
    with pytest.raises(wmapp.SpecError, match="mode"):
        wmapp.ExtractSpec.from_mapping({"mode": "psychic"})
    with pytest.raises(wmapp.SpecError, match="unknown option"):
        wmapp.ExtractSpec.from_mapping({"key": 5})


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_the_old_apply_watermark_entry_point_still_works_and_still_prints(tmp_path, capsys):
    import watermark
    watermark.apply_watermark(_image(tmp_path / "in.png"), str(tmp_path / "o.jpg"), "x", invisible=True)
    assert "requires a lossless format" in capsys.readouterr().out
    with pytest.raises(ValueError):                                                     # spread without a key, as before
        watermark.apply_watermark(_image(tmp_path / "in2.png"), str(tmp_path / "o2.png"), "x",
                                  invisible=True, invisible_mode="spread")


def test_extract_invisible_still_exports_its_old_names():
    import extract_invisible as e
    for name in ("extract_invisible", "extract_spread_spectrum", "extract_spread_spectrum_rescaled", "_find_payload",
                 "_decode_at", "MAX_PLAUSIBLE_LENGTH", "main"):
        assert hasattr(e, name), name
    assert e.extract_invisible is wmcore.invisible.extract_invisible
