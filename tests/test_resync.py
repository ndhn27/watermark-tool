"""Crop resync (wmcore/resync.py + extract's --resync): the spread-spectrum ID read from a copy that was CROPPED -
left/right trims (the raster row length changes) and offsets that are not a multiple of 8 pixels (the block grid
moves). Measured numbers and limits: bench/results/CROP-NOTES.md."""

import io
import time

import numpy as np
import pytest
from PIL import Image

import wmapp
import wmcore
from wmcore import resync, spread, wmid

KEY = 424242


def _image(w=640, h=480, seed=3):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    tex = np.kron(rng.normal(0, 14, (-(-h // 8), -(-w // 8), 1)), np.ones((8, 8, 1)))[:h, :w]
    arr = np.clip(np.stack([xx, yy, (xx + yy) / 2], -1) + tex + rng.normal(0, 6, (h, w, 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _stamp(img=None, key=KEY, wm_id=None):
    wm_id = wm_id or wmid.generate()
    marked = wmcore.embed_spread_spectrum((img or _image()).convert("RGBA"), wm_id, key).convert("RGB")
    return marked, wm_id


def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _read(img, key=KEY, **kw):
    return resync.find_spread_spectrum_id_resync(img, key, **kw)


# --- the building blocks ------------------------------------------------------------------------------------

def test_the_filter_gives_the_block_correlations_at_every_grid_offset():
    # The whole method rests on this: one 8x8 filter == the per-block DCT correlation, at any offset.
    marked, _ = _stamp()
    cmap = resync.correlation_map(resync.luma(marked), KEY)
    for dy, dx in [(0, 0), (3, 5), (7, 2), (1, 7)]:
        old = spread.spread_spectrum_block_correlations(marked.crop((dx, dy, marked.width, marked.height)), KEY)
        new = resync.grid_at(cmap, dy, dx)
        assert new.size == old.size
        assert np.allclose(new.ravel(), old, atol=1e-2)


def test_the_stride_and_phase_search_contains_the_old_decoder_as_a_special_case():
    marked, wm = _stamp()
    grid = resync.grid_at(resync.correlation_map(resync.luma(marked), KEY), 0, 0)
    text, stride, phase, _confidence = resync.find_id_in_grid(grid)
    assert text == wm and stride == (marked.width // 8) % 112 and phase == 0


def test_a_grid_too_small_for_one_period_reads_nothing():
    assert resync.find_id_in_grid(np.ones((5, 5))) is None


# --- crops --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("box", [
    (0, 40, 640, 480),            # top trim, on the grid: always worked
    (40, 0, 640, 480),            # left trim, on the grid: row length changes
    (0, 0, 560, 480),             # right trim
    (64, 32, 536, 440),           # all four sides, on the grid
    (0, 37, 640, 480),            # top trim OFF the grid
    (37, 23, 590, 450),           # screenshot-like: every edge off the grid
])
def test_crops_that_the_plain_decoder_cannot_read_are_read(box):
    marked, wm = _stamp()
    cropped = marked.crop(box)
    r = _read(cropped)
    assert r.text == wm and r.orientation == "none" and not r.budget_exhausted
    assert r.shift == ((-box[1]) % 8, (-box[0]) % 8)


def test_the_plain_decoder_really_could_not_read_the_hard_ones():
    # Guards the claim above: if the old decoder could read these, this file would prove nothing.
    marked, wm = _stamp()
    for box in [(40, 0, 640, 480), (0, 37, 640, 480), (37, 23, 590, 450)]:
        corrs = spread.spread_spectrum_block_correlations(marked.crop(box), KEY)
        assert spread.find_spread_spectrum_id(corrs) != wm


@pytest.mark.parametrize("q", [85, 60])
def test_a_jpeg_after_the_crop_does_not_stop_it(q):
    marked, wm = _stamp()
    assert _read(_jpeg(marked.crop((37, 23, 590, 450)), q)).text == wm


@pytest.mark.parametrize("flip,op", [("h", Image.FLIP_LEFT_RIGHT), ("v", Image.FLIP_TOP_BOTTOM), ("hv", Image.ROTATE_180)])
def test_a_mirrored_and_cropped_copy_is_read_and_says_how_it_was_mirrored(flip, op):
    marked, wm = _stamp()
    r = _read(marked.transpose(op).crop((37, 23, 590, 450)))
    assert r.text == wm and r.orientation == flip


def test_the_reported_orientation_of_a_top_bottom_flip_is_the_real_one_not_an_equivalent_read():
    # To the raster search a top-bottom flip is also "rows in reverse order" (a negated stride), which it can read
    # at ~1/8 of the mark's strength WITHOUT un-flipping - sometimes, depending on the picture. That is a correct ID
    # but not what was done to the copy, so the most confident orientation must win (see the function's docstring).
    for seed in range(4):
        marked, wm = _stamp(_image(640, 480, seed=seed))
        r = _read(marked.transpose(Image.FLIP_TOP_BOTTOM).crop((37, 23, 590, 450)))
        assert r.text == wm and r.orientation == "v"
        r = _read(marked.transpose(Image.ROTATE_180).crop((37, 23, 590, 450)))
        assert r.text == wm and r.orientation == "hv"


def test_a_tiny_fragment_is_still_read_when_enough_blocks_remain():
    marked, wm = _stamp(_image(1280, 960))
    assert _read(marked.crop((301, 211, 301 + 330, 211 + 250))).text == wm        # ~5% of the area


# --- what must NOT be read ----------------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: _image(640, 480, seed=9),                                             # never stamped
    lambda: Image.fromarray(np.full((480, 640, 3), 120, np.uint8)),               # flat
    lambda: Image.fromarray(np.tile((np.arange(640) // 8 % 2 * 80 + 60).astype(np.uint8)[None, :, None], (480, 1, 3))),
    lambda: Image.fromarray(np.kron(np.random.default_rng(2).integers(0, 256, (60, 80, 1)), np.ones((8, 8, 1)))
                            .repeat(3, axis=2).astype(np.uint8)),                 # random 8x8 blocks: the nasty one
])
def test_mark_free_images_return_no_id_over_all_hypotheses(make):
    assert _read(make()).text is None


def test_too_small_an_image_is_skipped_not_an_error():
    assert _read(Image.fromarray(np.zeros((12, 12, 3), np.uint8))).text is None


def test_a_zero_budget_is_reported_as_not_finished_not_as_absent():
    marked, _ = _stamp()
    r = _read(marked, max_seconds=0)
    assert r.text is None and r.budget_exhausted


# --- the known limit, pinned so nobody "fixes" it by guessing -------------------------------------------------

def test_trimming_columns_below_gcd_of_width_and_period_loses_the_mark_for_good():
    # 448 px = 56 blocks: g = gcd(56, 112) = 56, so any left/right trim leaves fewer than g columns. A property of
    # the embedding, not of this decoder.
    marked, wm = _stamp(_image(448, 320))
    assert _read(marked.crop((56, 0, 392, 320))).text != wm
    wide, wm2 = _stamp(_image(1344, 320))            # 168 blocks: g = 56, so keeping >= 448 px is fine...
    assert _read(wide.crop((200, 0, 1100, 320))).text == wm2
    assert _read(wide.crop((200, 0, 600, 320))).text != wm2            # ...and keeping 400 px is not


# --- through extract ----------------------------------------------------------------------------------------

@pytest.fixture
def led(tmp_path):
    return str(tmp_path / "ledger.sqlite3")


def _stamped_file(tmp_path, led):
    src = tmp_path / "in.png"
    _image(1280, 960, seed=5).save(src)
    out = str(tmp_path / "out.png")
    r = wmapp.watermark(str(src), out, text="@me", seed=3, stamp_id=True, ledger=led,
                        invisible=True, invisible_mode="spread")
    return r, out


def test_extract_reads_a_screenshot_style_crop_with_no_key_via_the_ledger(tmp_path, led):
    r, out = _stamped_file(tmp_path, led)
    shot = tmp_path / "shot.jpg"
    _jpeg(Image.open(out).convert("RGB").crop((37, 23, 1230, 900)), 85).save(shot, quality=85)
    res = wmapp.extract(str(shot), ledger=led)
    assert res.found and res.wm_id == r.wm_id and res.scheme == "spread"
    assert res.regrid == ((-23) % 8, (-37) % 8) and res.resync_tried
    assert res.to_dict()["cropped"] is True
    assert any("re-aligning" in ln and "cropped" in ln for ln in res.report_lines())
    spread_layer = {lr.layer: lr for lr in res.layers}["spread"]
    assert spread_layer.status == "found" and "looks cropped" in spread_layer.note


def test_extract_with_a_given_key_reads_it_too_and_the_cli_flag_turns_it_off(tmp_path, led):
    r, out = _stamped_file(tmp_path, led)
    shot = tmp_path / "shot.png"
    Image.open(out).convert("RGB").crop((37, 23, 1230, 900)).save(shot)
    key = wmapp.extract(out, ledger=led).record["ss_key"]
    assert wmapp.extract(str(shot), ss_key=key, use_ledger=False, resync_budget_seconds=60).wm_id == r.wm_id
    off = wmapp.extract(str(shot), ss_key=key, use_ledger=False, resync=False)
    assert not off.found and not off.resync_tried
    assert any("rescale" in ln for ln in off.report_lines())


def test_an_uncropped_copy_is_not_reported_as_cropped(tmp_path, led):
    r, out = _stamped_file(tmp_path, led)
    res = wmapp.extract(out, ledger=led)
    assert res.found and res.regrid is None and not res.resync_tried       # plain decode answered; nothing extra ran
    assert res.to_dict()["cropped"] is False


def test_resync_running_out_of_time_is_inconclusive_not_a_confirmed_absence(tmp_path, led):
    r, out = _stamped_file(tmp_path, led)
    shot = tmp_path / "shot.png"
    Image.open(out).convert("RGB").crop((37, 23, 1230, 900)).save(shot)
    # mode="spread": the LSB layer reads a junk byte off some images (random pixels), and then the report says "Hidden payload"
    # instead of the not-found message this test is about - it made the test fail 1 full run in 3 before this.
    res = wmapp.extract(str(shot), ledger=led, mode="spread", resync_budget_seconds=0)
    assert res.wm_id is None and res.resync_budget_exhausted
    assert {lr.layer: lr for lr in res.layers}["spread"].status == "inconclusive"
    assert "--resync-budget-seconds" in " ".join(res.report_lines())


def test_a_mark_free_image_through_extract_still_says_not_found(tmp_path, led):
    r, _ = _stamped_file(tmp_path, led)                                    # gives the ledger a key to try
    plain = tmp_path / "plain.png"
    _image(640, 480, seed=77).save(plain)
    res = wmapp.extract(str(plain), ledger=led)
    assert res.wm_id is None and res.resync_tried and not res.resync_budget_exhausted   # (the LSB layer's junk byte is not an ID)


# --- video ---------------------------------------------------------------------------------------------------

def _pan_video(path, w=320, h=240, fps=12, n_frames=24, pad=64, seed=0):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    big = np.random.default_rng(seed).integers(0, 256, (h + pad, w + pad, 3), dtype=np.uint8)
    for i in range(n_frames):
        off = i % pad
        vw.write(big[off:off + h, off:off + w])
    vw.release()
    return str(path)


def _crop_video(src, dst, box):
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    left, top, right, bottom = box
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (right - left, bottom - top))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(np.ascontiguousarray(frame[top:bottom, left:right]))
    vw.release()
    cap.release()
    return dst


def test_a_cropped_video_is_read_frame_by_frame_and_says_it_needed_the_re_alignment(tmp_path):
    src = _pan_video(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    r = wmapp.watermark(src, out, text="@v", stamp_id=True, invisible=True, invisible_mode="spread",
                        jitter_refresh_seconds=1.0, workers=2)
    cropped = _crop_video(out, str(tmp_path / "crop.mp4"), (37, 23, 301, 229))      # off-grid, width and height change
    res = wmapp.extract(cropped, ss_key=r.ss_key, jitter_refresh_seconds=1.0, use_ledger=False)
    scan = res.video_scan
    assert res.found and res.wm_id == r.wm_id
    assert scan.hits >= scan.frames_checked // 2 and scan.regrid_hits == scan.hits
    assert any("looks cropped" in ln for ln in res.report_lines())
    assert res.to_dict()["video"]["regrid_hits"] == scan.regrid_hits

    off = wmapp.extract(cropped, ss_key=r.ss_key, jitter_refresh_seconds=1.0, use_ledger=False, resync=False)
    assert not off.found and off.video_scan.hits == 0


def test_an_uncropped_video_does_not_claim_re_alignment(tmp_path):
    src = _pan_video(tmp_path / "in.mp4")
    out = str(tmp_path / "out.mp4")
    r = wmapp.watermark(src, out, text="@v", stamp_id=True, invisible=True, invisible_mode="spread",
                        jitter_refresh_seconds=1.0, workers=2)
    res = wmapp.extract(out, ss_key=r.ss_key, jitter_refresh_seconds=1.0, use_ledger=False)
    assert res.found and res.video_scan.regrid_hits == 0
    assert not any("looks cropped" in ln for ln in res.report_lines())


# --- v28: the label bug found by the user (a crop-only copy reported as mirrored), speed-ups, crop + resize ------------

def test_the_stride_sums_by_fft_equal_the_brute_force_definition():
    rng = np.random.default_rng(0)
    g = rng.normal(0, 1, (112, 112))
    brute = np.zeros((112, 112))
    for s in range(112):
        for rho in range(112):
            brute[s] += np.roll(g[rho], rho * s)          # b[s, j] += g[rho, (j - rho * s) mod 112]
    assert np.abs(resync._stride_sums(g) - brute).max() < 1e-9


def test_the_vectorised_shift_scores_equal_the_per_offset_definition():
    rng = np.random.default_rng(1)
    cmap = rng.normal(0, 50, (203, 311)).astype(np.float32)
    fast = resync.shift_scores(cmap)
    h8, w8 = (203 - 7) // 8 * 8, (311 - 7) // 8 * 8
    for dy in range(8):
        for dx in range(8):
            ref = np.square(cmap[dy:h8:8, dx:w8:8].astype(np.float64)).mean() if False else \
                  np.square(cmap[:h8, :w8].astype(np.float64))[dy::8, dx::8].mean()
            assert fast[dy, dx] == pytest.approx(ref, rel=1e-4)


@pytest.fixture
def multi_key_ledger(tmp_path):
    """A ledger holding 6 stamps with six different keys, as a real user's ledger does after a while."""
    led = str(tmp_path / "ledger.sqlite3")
    stamps = []
    for i in range(6):
        src = tmp_path / f"in{i}.png"
        _image(640, 480, seed=60 + i).save(src)
        out = str(tmp_path / f"o{i}.png")
        r = wmapp.watermark(str(src), out, text="@me", seed=3, stamp_id=True, ledger=led,
                            invisible=True, invisible_mode="spread", ss_key=1000 + 7919 * i)
        stamps.append((out, r.wm_id))
    return led, stamps


@pytest.mark.parametrize("op,label", [(None, None), (Image.FLIP_LEFT_RIGHT, "h"), (Image.FLIP_TOP_BOTTOM, "v"),
                                      (Image.ROTATE_180, "hv")])
def test_with_a_ledger_of_several_keys_the_mirroring_label_is_the_real_one(tmp_path, multi_key_ledger, op, label):
    # The user's report: a copy that was only CROPPED came back "mirrored top-bottom". A wrong ledger key reads the real ID
    # a good part of the time (the key is not a secret), and the first key to read used to decide the orientation. The most
    # confident read over every key and orientation is the true key in the true orientation (2-3x the mass of any wrong
    # one in the measurement). 6 stamps x this orientation: every one must be labelled right, ID right.
    led, stamps = multi_key_ledger
    for i, (out, wm) in enumerate(stamps):
        img = Image.open(out).convert("RGB")
        img = img.transpose(op) if op is not None else img
        shot = tmp_path / f"shot{i}.png"
        img.crop((37, 23, 590, 450)).save(shot)
        res = wmapp.extract(str(shot), ledger=led)
        assert res.wm_id == wm and res.flipped == label, (i, res.flipped, label)
        assert res.regrid is not None                                # and it is reported as cropped


def test_a_mirrored_copy_that_is_not_cropped_is_not_reported_as_cropped(tmp_path, multi_key_ledger):
    led, stamps = multi_key_ledger
    out, wm = stamps[2]
    flipped = tmp_path / "f.png"
    Image.open(out).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT).save(flipped)
    res = wmapp.extract(str(flipped), ledger=led)
    assert res.wm_id == wm and res.flipped == "h" and res.regrid is None


def test_the_resync_search_returns_the_most_confident_read_over_all_keys_and_says_which_key():
    marked, wm = _stamp(key=111)
    cropped = marked.crop((37, 23, 590, 450))
    r = resync.find_spread_spectrum_id_resync(cropped, [222, 333, 111, 444, 555])
    assert r.text == wm and r.key == 111 and r.orientation == "none" and r.confidence > 0


# --- crop + resize ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("g", [0.9, 0.75, 1.25])
def test_a_copy_that_was_cropped_and_resized_is_read_and_the_scale_is_recovered(g):
    marked, wm = _stamp(_image(1280, 960, seed=8))
    crop = marked.crop((37, 23, 1230, 900))
    copy = _jpeg(crop.resize((round(crop.width * g), round(crop.height * g)), Image.LANCZOS if g < 1 else Image.BICUBIC), 85)
    r = resync.find_spread_spectrum_id_resync_scaled(copy, KEY)
    assert r.text == wm and r.cropped and not r.budget_exhausted
    assert r.factor == pytest.approx(1 / g, rel=0.004)


def test_a_copy_that_was_only_resized_is_read_but_not_called_cropped():
    marked, wm = _stamp(_image(1280, 960, seed=9))
    copy = _jpeg(marked.resize((960, 720), Image.LANCZOS), 85)
    r = resync.find_spread_spectrum_id_resync_scaled(copy, KEY)
    assert r.text == wm and not r.cropped and r.factor == pytest.approx(4 / 3, rel=0.004)


def test_the_scale_search_reads_nothing_on_a_mark_free_image_and_returns_quickly_enough():
    plain = _image(1280, 960, seed=10).resize((1024, 768), Image.LANCZOS)
    r = resync.find_spread_spectrum_id_resync_scaled(plain, KEY)
    assert r.text is None and not r.budget_exhausted


def test_the_scale_search_reports_running_out_of_time():
    marked, _ = _stamp(_image(1280, 960, seed=11))
    r = resync.find_spread_spectrum_id_resync_scaled(marked.resize((960, 720)), KEY, max_seconds=0)
    assert r.text is None and r.budget_exhausted


def test_extract_with_rescale_reads_a_cropped_and_zoomed_screenshot_through_a_ledger_of_several_keys(tmp_path, multi_key_ledger):
    led, stamps = multi_key_ledger
    out, wm = stamps[3]
    base = Image.open(out).convert("RGB")
    crop = base.crop((37, 23, 590, 450))
    # (a 640x480 stamp at 90% zoom: the patch sweep needs a copy of a few hundred pixels at least)
    shot = tmp_path / "shot.jpg"
    _jpeg(crop.resize((round(crop.width * 0.9), round(crop.height * 0.9)), Image.LANCZOS), 85).save(shot, quality=85)
    assert not wmapp.extract(str(shot), ledger=led).found or wmapp.extract(str(shot), ledger=led).wm_id != wm
    res = wmapp.extract(str(shot), ledger=led, rescale=True)
    assert res.wm_id == wm and res.regrid is not None and res.rescaled_by == pytest.approx(1 / 0.9, rel=0.01)
    assert any("resized too" in ln for ln in res.report_lines())
    off = wmapp.extract(str(shot), ledger=led, rescale=True, resync=False, rescale_budget_seconds=3)
    assert off.wm_id != wm                                           # the established search cannot read a crop


# --- the pre-screen: a speed-up that must not change what is read --------------------------------------------------

def test_the_prescreen_skips_the_full_search_for_every_pair_that_cannot_read(monkeypatch):
    calls = {"n": 0}
    real = resync.find_id_in_grid
    monkeypatch.setattr(resync, "find_id_in_grid", lambda g: (calls.__setitem__("n", calls["n"] + 1), real(g))[1])
    # >= 480 px on both sides: pre-screened. A smooth photo-like image, NOT _image(): its random 8x8 block texture has
    # strong grid-aligned energy of its own, which passes the gate (harmlessly: it only costs the full search).
    import cv2
    rng = np.random.default_rng(31)
    smooth = cv2.GaussianBlur(rng.normal(128, 60, (480, 640, 3)).astype(np.float32), (0, 0), 2.0)
    plain = Image.fromarray(np.clip(smooth, 0, 255).astype(np.uint8))
    assert _read(plain, key=[11, 12, 13, 14, 15, 16]).text is None
    screened = calls["n"]
    calls["n"] = 0
    assert resync.find_spread_spectrum_id_resync(plain, [11, 12, 13, 14, 15, 16], prescreen=False).text is None
    assert calls["n"] >= 6 * 4 * 3 and screened <= calls["n"] // 10     # exhaustive: 6 keys x 4 orientations x ~4 grids


def test_the_prescreen_does_not_change_what_is_read_on_cropped_copies():
    for seed, q in [(1, None), (2, 85), (3, 60), (4, 40)]:
        marked, wm = _stamp(_image(1024, 768, seed=seed), key=500 + seed)
        img = marked.crop((37, 23, 900, 700))
        img = _jpeg(img, q) if q else img
        a = resync.find_spread_spectrum_id_resync(img, [9, 500 + seed, 77], prescreen=True)
        b = resync.find_spread_spectrum_id_resync(img, [9, 500 + seed, 77], prescreen=False)
        assert a.text == b.text == wm and a.key == b.key == 500 + seed and a.orientation == b.orientation == "none"


def test_images_under_480_px_are_not_prescreened():
    marked, wm = _stamp(_image(320, 240, seed=5))
    assert resync.find_spread_spectrum_id_resync(marked.crop((13, 9, 300, 230)), KEY, prescreen=True).text == wm


# --- v28b: the scale search has no threshold; the decoder decides (board.jpg was lost to a z/r gate) -----------------

def _circuit(w=640, h=480, seed=0, amp=60, trace=55):
    """A dense, high-contrast, periodic host like a circuit-board photo: strong grid-like energy of its own, so the patch
    statistic at the TRUE scale is only ~1.3-1.7 (a clean photo: 3-5) and a threshold on it cannot work."""
    r = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    grid = amp * np.sin(2 * np.pi * xx / 6.5) * np.sin(2 * np.pi * yy / 7.3)
    traces = np.kron((r.random((h // 3 + 1, w // 3 + 1)) > 0.55).astype(np.float64), np.ones((3, 3)))[:h, :w] * trace
    g = np.clip(110 + grid + traces + r.normal(0, 5, (h, w)), 0, 255)
    return Image.fromarray(np.clip(np.stack([g * 0.45, g * 0.95, g * 0.4], -1), 0, 255).astype(np.uint8))


@pytest.mark.parametrize("key,seed", [(903, 0), (903, 1), (903, 2), (905, 0), (905, 1)])
def test_a_dense_periodic_host_is_read_after_crop_and_resize_although_its_scale_statistic_is_weak(key, seed):
    # Pairs picked because the gated version of this search (z >= 12 and r >= 1.6) read NONE of them and this one reads all
    # (bench/results/CROP-NOTES.md §8b). On this extreme host the read rate over random (key, host) pairs is ~80% (24/30):
    # whether a key reads depends on how much of the host's own periodic energy lands on that key's pattern (key 902 never
    # reads), so these are fixed, known-good pairs rather than a claim of 100%.
    marked, wm = _stamp(_circuit(640, 480, seed=seed, amp=45, trace=50), key=key)
    crop = marked.crop((35, 27, 601, 455))
    copy = _jpeg(crop.resize((round(crop.width * 0.75), round(crop.height * 0.75)), Image.LANCZOS), 85)
    r = resync.find_spread_spectrum_id_resync_scaled(copy, key)
    assert r.text == wm and r.factor == pytest.approx(4 / 3, rel=0.004)
    grid, vals = resync._sweep_all(resync.luma(copy), [resync.key_pattern(key)], resync.SCALE_RANGE,
                                   time.monotonic() + 60, 256)
    i = int(np.argmin(np.abs(grid - 4 / 3)))
    assert np.nanmax(vals[0][i - 3:i + 4]) < 1.9               # the statistic really is weak here: no threshold would do


def test_a_small_cropped_and_resized_copy_is_not_rejected_by_a_size_guard():
    # A guard tied to the sweep patch size once rejected every copy under 384 px outright (64 of 84 reads in the measurement).
    marked, wm = _stamp(_image(640, 480, seed=14), key=77)
    crop = marked.crop((13, 9, 500, 380))
    copy = _jpeg(crop.resize((round(crop.width * 0.8), round(crop.height * 0.8)), Image.LANCZOS), 85)
    assert min(copy.size) < 384
    assert resync.find_spread_spectrum_id_resync_scaled(copy, 77).text == wm


def test_the_scale_search_picks_the_right_key_from_a_ledger_at_the_scale_that_reads():
    marked, wm = _stamp(_image(1024, 768, seed=15), key=4242)
    crop = marked.crop((37, 23, 980, 730))
    copy = _jpeg(crop.resize((round(crop.width * 0.75), round(crop.height * 0.75)), Image.LANCZOS), 85)
    r = resync.find_spread_spectrum_id_resync_scaled(copy, [1, 2, 3, 4242, 5, 6])
    assert r.text == wm and r.key == 4242 and r.cropped and r.confidence > 0


def test_a_large_cropped_and_resized_copy_is_read_through_a_central_region_and_never_returns_none():
    # Above ~1.5 MP the decode runs on a central region of the restored image. A first version then re-read the WHOLE image and
    # crashed on None when only the region read (10 MP photo, x0.5 zoom): the contract is a ScaledMatch, always.
    marked, wm = _stamp(_image(2400, 1600, seed=21), key=313)
    crop = marked.crop((37, 23, 2310, 1540))
    copy = _jpeg(crop.resize((round(crop.width * 0.75), round(crop.height * 0.75)), Image.LANCZOS), 85)
    assert copy.width * copy.height * (4 / 3) ** 2 > 1_500_000                  # restored image is above the region cutoff
    r = resync.find_spread_spectrum_id_resync_scaled(copy, [11, 313, 12])
    assert isinstance(r, resync.ScaledMatch) and r.text == wm and r.key == 313 and r.cropped
    plain = _jpeg(marked.resize((1800, 1200), Image.LANCZOS), 85)               # only resized: must NOT be called cropped
    r2 = resync.find_spread_spectrum_id_resync_scaled(plain, 313)
    assert r2.text == wm and not r2.cropped


def test_a_big_copy_that_was_only_resized_is_not_called_cropped_but_a_cropped_one_is():
    # At 10 MP a 0.25% scale error shifts the grid by ~6 px at the middle of the image, so a fixed 1 px slack called a plain
    # x0.5 resize 'cropped' (measured on 3888x2592 photos). x0.75 here, not x0.5: at x0.5 + JPEG q85 the mark itself is marginal
    # (a random ID reads or not, 1 trial in 6 did not), and this test is about the label. 2400x1600 restored is 3.8 MP, above
    # the 1.5 MP region cutoff, so the region path is the one exercised.
    marked, wm = _stamp(_image(2400, 1600, seed=22), key=414)
    r = resync.find_spread_spectrum_id_resync_scaled(_jpeg(marked.resize((1800, 1200), Image.LANCZOS), 85), 414, max_seconds=60)
    assert r.text == wm and not r.cropped and r.factor == pytest.approx(4 / 3, rel=0.004)
    crop = marked.crop((37, 23, 2300, 1500))
    r2 = resync.find_spread_spectrum_id_resync_scaled(_jpeg(crop.resize((1697, 1108), Image.LANCZOS), 85), 414, max_seconds=60)
    assert r2.text == wm and r2.cropped


# --- v28b: where the patch-based tests look (a screenshot of a picture in the corner of a mostly empty screen) ----------

def _screen_with_picture(picture, scale, canvas=(1400, 900), at=(70, 50)):
    shown = picture.resize((round(picture.width * scale), round(picture.height * scale)), Image.LANCZOS)
    screen = Image.new("RGB", canvas, (0, 0, 0))
    screen.paste(shown, at)
    return screen


def test_the_patch_stays_at_the_centre_of_any_picture_that_has_content_there():
    for img in (_image(1280, 960, seed=40), _image(900, 700, seed=41)):
        y = resync.luma(img)
        assert resync.find_patch(y, 384) == ((y.shape[0] // 2, y.shape[1] // 2), "centre")


def test_the_patch_leaves_an_empty_centre_for_the_smoothest_window_that_has_content_and_says_so():
    marked, _ = _stamp(_image(960, 720, seed=42))
    y = resync.luma(_screen_with_picture(marked, 1.0, canvas=(2400, 1600), at=(30, 40)))      # picture in the corner of black
    centre, status = resync.find_patch(y, 384)
    assert status == "moved" and centre != (y.shape[0] // 2, y.shape[1] // 2)
    assert 40 <= centre[0] - 192 and centre[0] + 192 <= 40 + 720 and 30 <= centre[1] - 192 and centre[1] + 192 <= 30 + 960   # inside the picture
    assert resync.find_patch(np.zeros((1600, 2400), np.float32), 384)[1] == "none"


def test_extract_reads_a_whole_screenshot_of_a_zoomed_picture_in_the_corner_of_a_black_screen(tmp_path, multi_key_ledger):
    led, stamps = multi_key_ledger
    out, wm = stamps[1]
    screen = _screen_with_picture(Image.open(out).convert("RGB"), 0.75)
    shot = tmp_path / "screen.png"
    screen.save(shot)
    res = wmapp.extract(str(shot), ledger=led, rescale=True)
    assert res.wm_id == wm and res.regrid is not None and res.rescaled_by == pytest.approx(4 / 3, rel=0.01)


# --- v29b: a long ledger no longer has to be swept in full before the first decode ---------------------------------------

def test_a_long_ledger_is_swept_in_rounds_and_the_right_key_is_found_in_the_first_one():
    # Wrong keys read the real ID at the true scale (the key is not a secret), so the FIRST round's candidates already hit, and
    # the key is then chosen among the whole ledger at that scale. The old sweep-everything-first cost 0.43 s x N before any
    # decode (26 s at N = 60, over the whole budget).
    marked, wm = _stamp(_image(1024, 768, seed=60), key=7777)
    crop = marked.crop((37, 23, 980, 730))
    copy = _jpeg(crop.resize((round(crop.width * 0.75), round(crop.height * 0.75)), Image.LANCZOS), 85)
    rng = np.random.default_rng(1)
    keys = [int(k) for k in rng.integers(1, 2 ** 31, 59)] + [7777]                       # the true key LAST of 60
    t0 = time.monotonic()
    r = resync.find_spread_spectrum_id_resync_scaled(copy, keys, max_seconds=60)
    took = time.monotonic() - t0
    assert r.text == wm and r.key == 7777 and r.cropped
    assert r.keys_reached < len(keys) and took < 15          # not every key was swept: well under 60 x 0.4 s


def test_running_out_of_time_on_a_long_ledger_says_how_many_keys_were_swept():
    marked, _ = _stamp(_image(1024, 768, seed=61), key=1)
    copy = _jpeg(marked.resize((768, 576), Image.LANCZOS), 85)
    keys = list(range(100, 140))                                                          # none of them stamped it
    r = resync.find_spread_spectrum_id_resync_scaled(copy, keys, max_seconds=1.2)
    assert r.text is None and r.budget_exhausted and 0 <= r.keys_reached < len(keys)


def test_extract_reports_how_far_the_crop_and_resize_search_got_when_a_big_ledger_runs_out_of_budget(tmp_path):
    led = str(tmp_path / "ledger.sqlite3")
    outs = []
    for i in range(14):
        src = tmp_path / f"in{i}.png"
        _image(640, 480, seed=300 + i).save(src)
        out = str(tmp_path / f"o{i}.png")
        wmapp.watermark(str(src), out, text="@me", seed=3, stamp_id=True, ledger=led, invisible=True, invisible_mode="spread")
        outs.append(out)
    plain = tmp_path / "plain.png"
    _image(640, 480, seed=999).resize((480, 360), Image.LANCZOS).save(plain)           # never stamped
    res = wmapp.extract(str(plain), ledger=led, rescale=True, rescale_budget_seconds=3, mode="spread")
    assert res.wm_id is None and res.budget_exhausted and res.scaled_keys_reached is not None
    assert res.scaled_keys_reached < 14
    text = " ".join(res.report_lines())
    assert "the crop + resize search got through" in text and "of 14 ledger keys" in text
    assert res.to_dict()["scaled_keys_reached"] == res.scaled_keys_reached
