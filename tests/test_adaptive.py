"""Unit tests for the content-adaptive placement feature: the sensitivity
map (wmcore.compute_sensitivity_map), the ink-map math it feeds into
multiply/overlay blending (wmcore.build_ink_map), the extra masked grid
layer it feeds into create_watermark_layer, and the video-only temporal
tracker (wmcore.AdaptiveTracker) that keeps a sensitivity map current across
frames instead of freezing it at whatever frame first detected it.

These don't try to validate face-detection accuracy (that would need real
photos and a much heavier fixture) - they pin down the parts that are pure,
fast, deterministic math: shapes, ranges, and the "adaptive_strength=0 or
sensitivity_map=None behaves exactly like before" and "never crosses the
blend mode's own no-op point" guarantees the rest of the feature leans on.
"""

from unittest.mock import patch

import numpy as np
import pytest

import wmcore
from wmcore.adaptive import AdaptiveTracker


# ---------------------------------------------------------------------------
# build_ink_map
# ---------------------------------------------------------------------------

class TestBuildInkMap:
    def test_no_sensitivity_map_returns_scalar_unchanged(self):
        assert wmcore.build_ink_map(0.32, None, 0.6, "multiply") == 0.32

    def test_zero_strength_returns_scalar_unchanged(self):
        s_map = np.ones((4, 4), dtype=np.float32)
        assert wmcore.build_ink_map(0.32, s_map, 0.0, "multiply") == 0.32

    def test_alpha_mode_returns_scalar_even_with_a_map(self):
        # --ink doesn't apply to mode="alpha" at all; adaptive placement
        # there works purely through the extra grid density, not ink.
        s_map = np.ones((4, 4), dtype=np.float32)
        assert wmcore.build_ink_map(0.32, s_map, 0.6, "alpha") == 0.32

    def test_multiply_darkens_where_sensitivity_is_high(self):
        s_map = np.array([[0.0, 1.0]], dtype=np.float32)
        out = wmcore.build_ink_map(0.5, s_map, 1.0, "multiply")
        assert out.shape == (1, 2, 1)
        # sensitivity 0 -> untouched; sensitivity 1 at strength 1 -> ink*0 = 0
        np.testing.assert_allclose(out[0, 0, 0], 0.5)
        np.testing.assert_allclose(out[0, 1, 0], 0.0, atol=1e-6)

    def test_multiply_never_exceeds_base_ink(self):
        rng = np.random.RandomState(0)
        s_map = rng.rand(8, 8).astype(np.float32)
        out = wmcore.build_ink_map(0.4, s_map, 0.8, "multiply")
        assert np.all(out <= 0.4 + 1e-6)
        assert np.all(out >= 0.0)

    @pytest.mark.parametrize("ink", [0.05, 0.32, 0.49])
    def test_overlay_low_ink_never_reaches_the_noop_point(self, ink):
        s_map = np.ones((4, 4), dtype=np.float32)
        out = wmcore.build_ink_map(ink, s_map, 1.0, "overlay")
        # Even at maximum strength, ink < 0.5 must stay strictly < 0.5 -
        # touching or crossing it would flip (or erase) the effect.
        assert np.all(out < 0.5)
        assert np.all(out <= ink + 1e-6)  # moves toward 0, never past it

    @pytest.mark.parametrize("ink", [0.51, 0.68, 0.95])
    def test_overlay_high_ink_never_reaches_the_noop_point(self, ink):
        s_map = np.ones((4, 4), dtype=np.float32)
        out = wmcore.build_ink_map(ink, s_map, 1.0, "overlay")
        assert np.all(out > 0.5)
        assert np.all(out <= 1.0 + 1e-6)
        assert np.all(out >= ink - 1e-6)  # moves toward 1, never past it

    def test_unknown_mode_returns_scalar_unchanged(self):
        s_map = np.ones((4, 4), dtype=np.float32)
        assert wmcore.build_ink_map(0.32, s_map, 0.6, "not-a-real-mode") == 0.32


# ---------------------------------------------------------------------------
# compute_sensitivity_map
# ---------------------------------------------------------------------------

class TestComputeSensitivityMap:
    def test_shape_matches_input_and_values_in_range(self):
        rng = np.random.RandomState(1)
        img = rng.randint(0, 255, size=(40, 60, 3), dtype=np.uint8)
        out = wmcore.compute_sensitivity_map(img)
        assert out.shape == (40, 60)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_flat_image_has_low_sensitivity_everywhere(self):
        # A perfectly flat image has zero detail energy and (almost
        # certainly) no detected faces, so the whole map should sit near 0.
        img = np.full((80, 80, 3), 128, dtype=np.uint8)
        out = wmcore.compute_sensitivity_map(img)
        assert out.max() < 0.05

    def test_textured_region_scores_higher_than_flat_region(self):
        rng = np.random.RandomState(2)
        img = np.full((80, 160, 3), 128, dtype=np.uint8)
        # Left half stays flat; right half gets high-frequency noise, which
        # the Laplacian-based detail signal should pick up as "high detail".
        img[:, 80:] = rng.randint(0, 255, size=(80, 80, 3), dtype=np.uint8)
        out = wmcore.compute_sensitivity_map(img, detail_weight=1.0)
        assert out[:, 80:].mean() > out[:, :80].mean()

    def test_handles_tiny_images_without_crashing(self):
        img = np.zeros((3, 3, 3), dtype=np.uint8)
        out = wmcore.compute_sensitivity_map(img, downscale=4)
        assert out.shape == (3, 3)

    def test_bad_custom_cascade_path_degrades_to_detail_only(self):
        # An unreadable/invalid --face-cascade shouldn't crash a whole
        # batch run - it should just fall back to the detail-only signal.
        img = np.full((40, 40, 3), 128, dtype=np.uint8)
        out = wmcore.compute_sensitivity_map(img, face_cascade_path="/nonexistent/path.xml")
        assert out.shape == (40, 40)


# ---------------------------------------------------------------------------
# create_watermark_layer: adaptive infill grid
# ---------------------------------------------------------------------------

class TestAdaptiveGridLayer:
    def _alpha_sum(self, layer):
        return np.asarray(layer, dtype=np.float32)[..., 3].sum()

    def test_all_zero_sensitivity_matches_plain_layer_exactly(self):
        size = (200, 150)
        s_map = np.zeros((150, 200), dtype=np.float32)
        plain = wmcore.create_watermark_layer(size, "@test", spacing=250, seed=7)
        adaptive = wmcore.create_watermark_layer(
            size, "@test", spacing=250, seed=7, sensitivity_map=s_map, adaptive_strength=0.6,
        )
        np.testing.assert_array_equal(np.asarray(plain), np.asarray(adaptive))

    def test_zero_strength_matches_plain_layer_even_with_a_map(self):
        size = (200, 150)
        s_map = np.ones((150, 200), dtype=np.float32)
        plain = wmcore.create_watermark_layer(size, "@test", spacing=250, seed=7)
        adaptive = wmcore.create_watermark_layer(
            size, "@test", spacing=250, seed=7, sensitivity_map=s_map, adaptive_strength=0.0,
        )
        np.testing.assert_array_equal(np.asarray(plain), np.asarray(adaptive))

    def test_full_sensitivity_increases_total_coverage(self):
        size = (200, 150)
        s_map = np.ones((150, 200), dtype=np.float32)
        plain = wmcore.create_watermark_layer(size, "@test", spacing=250, seed=7)
        adaptive = wmcore.create_watermark_layer(
            size, "@test", spacing=250, seed=7, sensitivity_map=s_map, adaptive_strength=1.0,
        )
        assert self._alpha_sum(adaptive) > self._alpha_sum(plain)

    def test_wrong_shaped_sensitivity_map_raises(self):
        size = (200, 150)  # (width, height) -> map must be (150, 200)
        bad_map = np.ones((200, 150), dtype=np.float32)  # swapped on purpose
        with pytest.raises(ValueError):
            wmcore.create_watermark_layer(size, "@test", sensitivity_map=bad_map, adaptive_strength=0.6)


# ---------------------------------------------------------------------------
# AdaptiveTracker: video-only temporal tracking of the sensitivity map
# ---------------------------------------------------------------------------

def _panned_frames(patch_at, dx, size=(160, 220), patch_size=28, seed=5):
    """Two overlapping crops of one static scene, offset by dx along x - a
    camera pan (or the world sliding past a fixed camera) between frame A
    and frame B. The scene is a smooth, low-detail gradient everywhere
    except for one small textured "subject" patch, so:
      - compute_sensitivity_map has exactly one thing to find (the patch),
        making "did the tracked peak follow it" unambiguous to check;
      - the whole frame still undergoes genuine, coherent pixel motion (the
        gradient shifts too), which is what dense optical flow needs to
        produce a reliable, roughly-uniform flow field to warp with.
    Returns (frame_a, frame_b); the patch sits `dx` px further left in B
    than in A, matching how content recedes when a camera pans right.
    """
    h, w = size
    margin = dx + 5
    big = np.zeros((h, w + margin, 3), dtype=np.uint8)
    grad = np.linspace(60, 180, w + margin, dtype=np.uint8)
    big[:, :, :] = grad[None, :, None]
    rng = np.random.RandomState(seed)
    px, py = patch_at
    big[py:py + patch_size, px:px + patch_size] = rng.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
    return big[:, :w], big[:, dx:dx + w]


class TestAdaptiveTracker:
    def test_first_update_detects_and_returns_full_res_map(self):
        frame, _ = _panned_frames(patch_at=(20, 20), dx=15)
        tracker = AdaptiveTracker(detail_weight=1.0)
        out = tracker.update(frame)
        assert out.shape == frame.shape[:2]
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert tracker.last_cut is False  # nothing to compare against yet

    def test_tracked_peak_follows_a_panning_scene(self):
        # redetect_every_frames is set way above the number of calls we make,
        # so the second update can ONLY be explained by the flow-warp path,
        # not by a second, independent detection happening to land in the
        # right place.
        dx = 16
        frame_a, frame_b = _panned_frames(patch_at=(100, 60), dx=dx)
        tracker = AdaptiveTracker(detail_weight=1.0, redetect_every_frames=10_000, cut_threshold=0.9)
        tracker.update(frame_a)
        out_b = tracker.update(frame_b)
        assert tracker.last_cut is False

        peak_a_x = 100 + 14  # patch center's x in frame_a
        peak_b_x = peak_a_x - dx  # same physical patch, dx px further left in frame_b

        col_scores = out_b.mean(axis=0)  # collapse rows -> per-column sensitivity
        tracked_peak_x = int(np.argmax(col_scores))
        # Flow on a downscaled image is approximate - allow some slack, but
        # it must land meaningfully closer to the patch's NEW position than
        # to where it used to be (a bug that skips the warp, or warps the
        # wrong way, would instead stay at peak_a_x or overshoot to
        # peak_a_x - 2*dx - see _warp_with_flow's docstring).
        assert abs(tracked_peak_x - peak_b_x) < abs(tracked_peak_x - peak_a_x)
        assert abs(tracked_peak_x - peak_b_x) <= dx // 2

    def test_static_scene_does_not_redetect_every_frame(self):
        frame, _ = _panned_frames(patch_at=(20, 20), dx=15)
        with patch("wmcore.adaptive._sensitivity_from_small_gray", wraps=wmcore.adaptive._sensitivity_from_small_gray) as spy:
            tracker = AdaptiveTracker(detail_weight=1.0, redetect_every_frames=2, cut_threshold=0.9)
            for _ in range(5):
                tracker.update(frame)
            # calls 1 and 4 detect (frames_since_detect resets each time);
            # calls 2, 3, 5 are pure flow-tracks - see the cadence math in
            # AdaptiveTracker.update's need_detect check.
            assert spy.call_count == 2

    def test_scene_cut_forces_redetect_and_skips_flow(self):
        frame_a, _ = _panned_frames(patch_at=(20, 20), dx=15, seed=1)
        # An unrelated, much brighter scene - a stand-in for a hard cut to a
        # different shot, not a pan of the same one.
        frame_cut = np.full_like(frame_a, 250)

        with patch("wmcore.adaptive._sensitivity_from_small_gray", wraps=wmcore.adaptive._sensitivity_from_small_gray) as detect_spy, \
             patch("wmcore.adaptive._dense_flow", wraps=wmcore.adaptive._dense_flow) as flow_spy:
            tracker = AdaptiveTracker(detail_weight=1.0, redetect_every_frames=10_000, cut_threshold=0.12)
            tracker.update(frame_a)
            assert detect_spy.call_count == 1
            tracker.update(frame_cut)
            assert tracker.last_cut is True
            # The cut must trigger a fresh detection (ignoring the very high
            # redetect_every_frames) and must NOT bother computing flow
            # across content that just doesn't correspond frame-to-frame.
            assert detect_spy.call_count == 2
            assert flow_spy.call_count == 0

    def test_mismatched_frame_size_does_not_crash(self):
        # A resolution change mid-stream (unusual, but shouldn't be fatal):
        # same as a cut - detect fresh at the new size, no flow warp.
        frame_a, _ = _panned_frames(patch_at=(20, 20), dx=15, size=(100, 140))
        frame_b, _ = _panned_frames(patch_at=(20, 20), dx=15, size=(120, 200))
        tracker = AdaptiveTracker(detail_weight=1.0)
        tracker.update(frame_a)
        out = tracker.update(frame_b)
        assert out.shape == frame_b.shape[:2]
