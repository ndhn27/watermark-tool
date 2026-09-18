"""Unit tests for wmcore.blend_arrays() - the numpy math shared by the
still-image compositor and the per-frame video worker.

These pin down the actual arithmetic (not just "it runs"), so a future
change to the blend formulas fails loudly here instead of only showing up
as a subtly-off-looking image that someone has to notice by eye.
"""

import numpy as np
import pytest

import wmcore


def _flat(value, shape=(2, 2, 1)):
    return np.full(shape, value, dtype=np.float32)


class TestAlphaMode:
    def test_zero_alpha_returns_base_unchanged(self):
        base = np.random.RandomState(0).rand(4, 4, 3).astype(np.float32)
        mark_rgb = np.ones((4, 4, 3), dtype=np.float32)
        out = wmcore.blend_arrays(base, _flat(0.0, (4, 4, 1)), mode="alpha", mark_rgb01=mark_rgb)
        np.testing.assert_allclose(out, base)

    def test_full_alpha_returns_mark_unchanged(self):
        base = np.zeros((4, 4, 3), dtype=np.float32)
        mark_rgb = np.random.RandomState(1).rand(4, 4, 3).astype(np.float32)
        out = wmcore.blend_arrays(base, _flat(1.0, (4, 4, 1)), mode="alpha", mark_rgb01=mark_rgb)
        np.testing.assert_allclose(out, mark_rgb)

    def test_half_alpha_is_midpoint(self):
        base = _flat(0.2, (2, 2, 3))
        mark_rgb = _flat(0.8, (2, 2, 3))
        out = wmcore.blend_arrays(base, _flat(0.5, (2, 2, 1)), mode="alpha", mark_rgb01=mark_rgb)
        np.testing.assert_allclose(out, _flat(0.5, (2, 2, 3)), atol=1e-6)

    def test_requires_mark_rgb01(self):
        base = _flat(0.5, (2, 2, 3))
        with pytest.raises(ValueError):
            wmcore.blend_arrays(base, _flat(1.0), mode="alpha")


class TestMultiplyMode:
    def test_zero_alpha_returns_base_unchanged(self):
        base = _flat(0.7, (2, 2, 3))
        out = wmcore.blend_arrays(base, _flat(0.0), mode="multiply", ink=0.2)
        np.testing.assert_allclose(out, base)

    def test_full_alpha_scales_base_by_ink(self):
        base = _flat(0.7, (2, 2, 3))
        out = wmcore.blend_arrays(base, _flat(1.0), mode="multiply", ink=0.2)
        np.testing.assert_allclose(out, base * 0.2, atol=1e-6)

    def test_output_never_exceeds_base(self):
        # multiply only darkens (ink in 0..1), so blended <= base everywhere.
        rng = np.random.RandomState(2)
        base = rng.rand(8, 8, 3).astype(np.float32)
        alpha = rng.rand(8, 8, 1).astype(np.float32)
        out = wmcore.blend_arrays(base, alpha, mode="multiply", ink=0.4)
        assert np.all(out <= base + 1e-6)


class TestOverlayMode:
    def test_zero_alpha_returns_base_unchanged(self):
        base = _flat(0.3, (2, 2, 3))
        out = wmcore.blend_arrays(base, _flat(0.0), mode="overlay", ink=0.5)
        np.testing.assert_allclose(out, base)

    def test_stays_in_valid_range(self):
        rng = np.random.RandomState(3)
        base = rng.rand(16, 16, 3).astype(np.float32)
        alpha = rng.rand(16, 16, 1).astype(np.float32)
        out = wmcore.blend_arrays(base, alpha, mode="overlay", ink=0.6)
        assert out.min() >= -1e-6
        assert out.max() <= 1 + 1e-6

    def test_ink_half_is_a_no_op_even_at_full_alpha(self):
        # ink=0.5 is the exact mathematical neutral point of this overlay
        # formula for BOTH branches (base < 0.5 and base >= 0.5): the mark
        # is fully invisible here, not just "medium strength". This is easy
        # to hit by accident since 0.5 looks like a safe middle value -
        # pin it down so a future formula change can't silently reintroduce
        # (or accidentally remove) this behavior without a test noticing.
        rng = np.random.RandomState(4)
        base = rng.rand(8, 8, 3).astype(np.float32)
        out = wmcore.blend_arrays(base, _flat(1.0, (8, 8, 1)), mode="overlay", ink=0.5)
        np.testing.assert_allclose(out, base, atol=1e-6)

    def test_dark_pixel_darkens_light_pixel_lightens(self):
        # overlay: base < 0.5 branch scales toward 0 as ink shrinks; base >
        # 0.5 branch scales toward 1 as ink grows. Sanity-check the two
        # branches move in the expected direction relative to the base.
        dark_base = _flat(0.2, (2, 2, 3))
        light_base = _flat(0.8, (2, 2, 3))
        dark_out = wmcore.blend_arrays(dark_base, _flat(1.0), mode="overlay", ink=0.1)
        light_out = wmcore.blend_arrays(light_base, _flat(1.0), mode="overlay", ink=0.9)
        assert np.all(dark_out < dark_base)
        assert np.all(light_out > light_base)


def test_unknown_mode_raises():
    base = _flat(0.5, (2, 2, 3))
    with pytest.raises(ValueError):
        wmcore.blend_arrays(base, _flat(1.0), mode="not-a-real-mode")
