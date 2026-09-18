"""Unit tests for the content-adaptive placement feature: the sensitivity
map (wmcore.compute_sensitivity_map), the ink-map math it feeds into
multiply/overlay blending (wmcore.build_ink_map), and the extra masked grid
layer it feeds into create_watermark_layer.

These don't try to validate face-detection accuracy (that would need real
photos and a much heavier fixture) - they pin down the parts that are pure,
fast, deterministic math: shapes, ranges, and the "adaptive_strength=0 or
sensitivity_map=None behaves exactly like before" and "never crosses the
blend mode's own no-op point" guarantees the rest of the feature leans on.
"""

import numpy as np
import pytest

import wmcore


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
