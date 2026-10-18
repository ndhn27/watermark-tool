"""Blend modes: mix the watermark into local pixel values instead of a flat
alpha-over, so it reads more like ink soaking into paper than a sticker
laid on top.
"""

import numpy as np
from PIL import Image


def blend_arrays(base01, alpha01, mode="alpha", ink=0.32, mark_rgb01=None):
    """Pure-numpy blend core, shared by the still-image compositor and the
    per-frame video worker.

    base01:     HxWx3 float32 array, base pixels in 0..1.
    alpha01:    HxWx1 float32 array, watermark mask alpha in 0..1.
    mark_rgb01: HxWx3 float32 array, only needed for mode="alpha" (the
                watermark's own RGB to lay over the base). Not used for
                multiply/overlay, which derive the blended color from the
                base pixels themselves.
    Returns an HxWx3 float32 array in 0..1.
    """
    if mode == "alpha":
        if mark_rgb01 is None:
            raise ValueError('mode="alpha" requires mark_rgb01')
        return base01 * (1 - alpha01) + mark_rgb01 * alpha01

    if mode == "multiply":
        blended = base01 * ink
    elif mode == "overlay":
        blended = np.where(base01 < 0.5, 2 * base01 * ink, 1 - 2 * (1 - base01) * (1 - ink))
    else:
        raise ValueError(f"Unknown blend mode: {mode}")

    return base01 * (1 - alpha01) + blended * alpha01


def composite(base_rgba, mark_rgba, mode="alpha", ink=0.32):
    """Combine a still base image + watermark layer (both PIL RGBA images).

    mode="alpha":    standard alpha-over (original behavior).
    mode="multiply": darkens the base by `ink` wherever the mark's alpha
                      is set, scaled by that alpha.
    mode="overlay":  classic overlay blend (darkens shadows, lightens
                      highlights less aggressively than multiply), also
                      scaled by the mark's alpha.
    """
    if mode == "alpha":
        return Image.alpha_composite(base_rgba, mark_rgba)

    base01 = np.asarray(base_rgba.convert("RGB"), dtype=np.float32) / 255.0
    alpha01 = np.asarray(mark_rgba.getchannel("A"), dtype=np.float32)[..., None] / 255.0
    out01 = blend_arrays(base01, alpha01, mode=mode, ink=ink)
    out_img = Image.fromarray(np.clip(out01 * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
    return out_img.convert("RGBA")


def watermark_layer_to_arrays(mark_rgba_image):
    """Convert a PIL RGBA watermark layer into the two float32 arrays the
    video frame workers need: RGB in 0..1 and alpha in 0..1 (kept separate
    so a worker never has to re-derive them per frame).
    """
    arr = np.asarray(mark_rgba_image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3:4])
