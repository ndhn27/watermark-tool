"""The invisible mark for video (bench/results/P4-NOTES.md's step 3/5 "thêm" phase): embedding
wmcore.embed_spread_spectrum() into every frame inside cli/video.py's worker pool, with a
per-segment key rotation (video.scan_invisible() on the read-back side), and wmapp.extract()'s
video dispatch. test_wmapp.py's test_video_goes_through_the_same_entry_point covers the
invisible_mode="lsb" (unsupported, warns) case; this file is invisible_mode="spread" only."""

import cv2
import numpy as np
import pytest

import wmapp
import wmcore
import video as video_mod


def _pink_pan_video(path, w=320, h=240, fps=12, n_frames=24, pad=64, seed=0):
    """A real (if tiny) moving picture - a static noisy image panned across, not independent
    per-frame noise - so the block-DCT mark has something plausible to sit on top of, the same
    reasoning bench/experiments/video_invisible_probe.py's source used."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 256, (h + pad, w + pad, 3), dtype=np.uint8)
    for i in range(n_frames):
        off = i % pad
        vw.write(big[off:off + h, off:off + w])
    vw.release()
    return str(path)


def test_video_default_alpha_is_higher_than_the_image_one():
    # bench/results/P4-NOTES.md §5: the image default (tuned for JPEG) only decoded ~65-70% of
    # frames back after a real CRF23 H.264 encode; video needs more.
    assert wmcore.DEFAULT_SS_ALPHA_VIDEO > wmcore.DEFAULT_SS_ALPHA


def test_video_invisible_spread_round_trip(tmp_path):
    src = _pink_pan_video(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    r = wmapp.watermark(src, out, text="@niny", stamp_id=True, invisible=True, invisible_mode="spread",
                        jitter_refresh_seconds=1.0, workers=2)
    assert r.kind == "video" and r.ss_key is not None and "spread" in r.id_layers

    res = wmapp.extract(out, ss_key=r.ss_key, jitter_refresh_seconds=1.0)
    assert res.found and res.scheme == "spread" and res.wm_id == r.wm_id
    assert res.video_scan is not None
    assert res.video_scan.hits > 0
    assert res.video_scan.frames_checked == res.video_scan.frames_total == 24
    assert f"{res.video_scan.hits}/{res.video_scan.frames_checked}" in "\n".join(res.report_lines())


def test_video_invisible_not_found_still_reports_every_frame_checked(tmp_path):
    # A wrong key should come back found=False, but STILL say how much was actually scanned - the
    # "quét hết" decision (P4-NOTES.md §7 / HANDOFF v24): a video gets one decode attempt per
    # frame, so "0 hits out of N checked" is the honest report, not a bare no.
    src = _pink_pan_video(tmp_path / "in2.mp4")
    out = str(tmp_path / "out2.mp4")
    r = wmapp.watermark(src, out, text="@niny", stamp_id=True, invisible=True, invisible_mode="spread",
                        jitter_refresh_seconds=1.0, workers=2)

    # The unstamped source, not "the stamped file with a wrong key": a wrong key reads the real ID some of the time
    # BY DESIGN (the key is not a secret - spread.py), so that version of this test was a coin flip on the random key.
    res = wmapp.extract(src, ss_key=r.ss_key, jitter_refresh_seconds=1.0)
    assert not res.found
    assert res.video_scan is not None
    assert res.video_scan.hits == 0
    assert res.video_scan.frames_checked == 24
    assert "24" in "\n".join(res.report_lines())


def test_video_invisible_key_actually_rotates_per_segment(tmp_path):
    # Confirms the per-segment rotation is REAL, not a no-op that happens to work because the same
    # key is used throughout: decoding with jitter_refresh_seconds=0 (video.scan_invisible's "no
    # rotation" case - see its docstring) should only line up with whichever ONE segment's key
    # scan_invisible falls back to using throughout, so it must recover STRICTLY FEWER hits than
    # decoding with the embed-time value, on a clip that actually crosses more than one segment
    # boundary (fps=12, jitter_refresh_seconds=0.5 -> 6 frames/segment, 24 frames -> 4 segments).
    src = _pink_pan_video(tmp_path / "in3.mp4", fps=12, n_frames=24)
    out = str(tmp_path / "out3.mp4")
    r = wmapp.watermark(src, out, text="@niny", stamp_id=True, invisible=True, invisible_mode="spread",
                        jitter_refresh_seconds=0.5, workers=2)

    matched = video_mod.scan_invisible(out, r.ss_key, jitter_refresh_seconds=0.5)
    single_key = video_mod.scan_invisible(out, r.ss_key, jitter_refresh_seconds=0)
    assert matched.hits == 24                     # every frame's derived key is the one it was embedded with
    assert single_key.hits < matched.hits          # only the one segment single_key's fixed key happens to match


def test_video_invisible_mode_lsb_skips_embedding_but_grid_and_id_still_work(tmp_path):
    # Companion to test_wmapp.py's test_video_goes_through_the_same_entry_point, but with
    # --grid-id so there IS still an ID somewhere to extract from (the visible grid's OCR-free
    # text layer isn't what extract() reads - this just confirms the run completes and produces
    # a real file; wmapp.extract() has no video LSB path to try, by design - see
    # wmapp/extraction.py's _extract_video docstring).
    src = _pink_pan_video(tmp_path / "in4.mp4")
    out = str(tmp_path / "out4.mp4")
    r = wmapp.watermark(src, out, text="@niny", stamp_id=True, invisible=True, invisible_mode="lsb", workers=2)
    assert r.ss_key is None and "spread" not in r.id_layers
    assert any(n.code == "video_lsb_unsupported" for n in r.notes)


def test_extract_video_without_ss_key_or_ledger_reports_nothing_found(tmp_path):
    src = _pink_pan_video(tmp_path / "in5.mp4")
    out = str(tmp_path / "out5.mp4")
    wmapp.watermark(src, out, text="@niny", stamp_id=True, invisible=True, invisible_mode="spread", workers=2)

    res = wmapp.extract(out, use_ledger=False)
    assert not res.found and res.video_scan is None    # no key to try at all - never got to scan a single frame
