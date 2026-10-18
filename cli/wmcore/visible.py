"""The visible, diagonal-grid watermark: a repeating tiled text layer where
every tile gets its own random position offset, opacity, and rotation
(per-tile jitter breaks the FFT-periodicity a perfectly regular grid would
have, which is what makes averaging-based removal attacks work).
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from .primitives import derive_seed, load_font


def _build_grid_layer(
    size,
    text,
    font_size,
    opacity,
    angle,
    spacing,
    font_path,
    seed,
    jitter_pos,
    jitter_opacity,
    jitter_angle,
):
    """Build a transparent RGBA layer with `text` repeated in a jittered
    grid, reproducible from `seed`. This is the plain, non-adaptive grid
    builder; create_watermark_layer() below calls it twice (base + infill)
    for adaptive placement, so both grids share this exact tiling logic.
    """
    rng = random.Random(seed)
    width, height = size
    diagonal = int(math.hypot(width, height)) + spacing * 2
    layer = Image.new("RGBA", (diagonal, diagonal), (0, 0, 0, 0))
    font = load_font(font_size, font_path)

    # Measure the text once so each tile canvas is sized correctly.
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tile_size = int(math.hypot(tw, th)) + 24

    row = 0
    for y in range(0, diagonal, spacing):
        offset_x = (spacing // 2) if row % 2 else 0
        for x in range(-spacing, diagonal + spacing, spacing):
            dx = rng.uniform(-jitter_pos, jitter_pos) * spacing
            dy = rng.uniform(-jitter_pos, jitter_pos) * spacing
            tile_angle = angle + rng.uniform(-jitter_angle, jitter_angle)
            tile_opacity = max(0, min(255, opacity + rng.randint(-jitter_opacity, jitter_opacity)))

            tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(tile)
            tdraw.text(
                ((tile_size - tw) // 2 - bbox[0], (tile_size - th) // 2 - bbox[1]),
                text,
                font=font,
                fill=(255, 255, 255, tile_opacity),
            )
            tile = tile.rotate(tile_angle, expand=True, resample=Image.BICUBIC)

            px = int(x + offset_x + dx - tile.width / 2)
            py = int(y + dy - tile.height / 2)
            layer.alpha_composite(tile, (px, py))
        row += 1

    left = (diagonal - width) // 2
    top = (diagonal - height) // 2
    return layer.crop((left, top, left + width, top + height))


def create_watermark_layer(
    size,
    text,
    font_size=36,
    opacity=90,
    angle=30,
    spacing=250,
    font_path=None,
    seed=None,
    jitter_pos=0.35,
    jitter_opacity=40,
    jitter_angle=8,
    sensitivity_map=None,
    adaptive_strength=0.6,
    infill_spacing_divisor=2.0,
):
    """Build the watermark layer for one image/frame - a plain jittered grid
    (see _build_grid_layer), optionally with a second, denser grid layered
    on top wherever `sensitivity_map` (see adaptive.py) says removal is
    likely to be hard to hide cleanly (faces, high-detail regions).

    The infill grid is masked by `sensitivity_map * adaptive_strength` and
    alpha-composited over the base grid, using its own derived seed (not
    the same seed reused - an identical pattern stamped twice would look
    like one bolder grid at the same jitter, not two independent ones).
    Outside the mask the infill layer's alpha is ~0, so the result is
    pixel-for-pixel identical to the non-adaptive path there.

    `sensitivity_map`, if given, must be an (height, width) float array in
    0..1 matching `size`. When None (the default), this is exactly the old
    single-grid behavior.
    """
    base_layer = _build_grid_layer(
        size, text, font_size, opacity, angle, spacing, font_path, seed,
        jitter_pos, jitter_opacity, jitter_angle,
    )
    if sensitivity_map is None or adaptive_strength <= 0:
        return base_layer

    width, height = size
    mask = np.clip(sensitivity_map, 0, 1).astype(np.float32)
    if mask.shape != (height, width):
        raise ValueError(
            f"sensitivity_map shape {mask.shape} does not match image size "
            f"(height={height}, width={width}) - build it from an array of "
            "this same image/frame."
        )
    mask *= float(np.clip(adaptive_strength, 0, 1))

    infill_spacing = int(max(24, spacing / max(1.0, infill_spacing_divisor)))
    infill_seed = derive_seed(seed, "adaptive-infill")
    infill_layer = _build_grid_layer(
        size, text, font_size, opacity, angle, infill_spacing, font_path, infill_seed,
        jitter_pos, jitter_opacity, jitter_angle,
    )

    infill_arr = np.array(infill_layer, dtype=np.float32)  # copy, not a view
    infill_arr[..., 3] *= mask
    infill_layer = Image.fromarray(np.clip(infill_arr, 0, 255).astype(np.uint8), "RGBA")

    return Image.alpha_composite(base_layer, infill_layer)
