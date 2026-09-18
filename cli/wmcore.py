#!/usr/bin/env python3
"""
wmcore.py - Shared watermarking primitives used by watermark.py (images),
video.py (video frames), and their multiprocessing workers.

Keeping this logic in one importable, non-"__main__" module matters for two
reasons:
  1. It's the single source of truth for the actual watermark math, so the
     image path and the video path can never silently drift apart.
  2. Functions handed to multiprocessing.Pool / ProcessPoolExecutor must be
     picklable by reference. On the "spawn" start method (the default on
     Windows and macOS), that means they must live in a plain top-level
     module that child processes can just `import` - not in the __main__
     script, and not as a closure or lambda. Everything reusable across
     processes lives here for exactly that reason.
"""

import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

try:
    import piexif
except ImportError:
    piexif = None

DEFAULT_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux/WSL2
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",      # macOS (Catalina+, stock)
    "/Library/Fonts/Arial Bold.ttf",                          # macOS (older / MS Office install)
    "C:\\Windows\\Fonts\\arialbd.ttf",                         # Windows fallback
]

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

_font_fallback_warned = False


def load_font(font_size, font_path=None):
    """Load a TrueType font, trying `font_path` (if given) then the
    platform-default candidates in DEFAULT_FONT_PATHS, in order.

    If none of those exist, we still try PIL's built-in default font at
    `font_size` (Pillow >= 10.1 supports this: `ImageFont.load_default(size=...)`
    returns a real scalable font instead of the old fixed tiny bitmap). Only
    on older Pillow, where that isn't available, do we fall back to the
    classic `load_default()` - which ignores `font_size` and is tiny/blocky
    - and warn once per process about it (this can be called once per
    output file, so a large --batch run shouldn't spam the same warning).
    """
    global _font_fallback_warned
    candidates = [font_path] if font_path else DEFAULT_FONT_PATHS
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        pass  # Pillow < 10.1: load_default() doesn't take a size argument
    if not _font_fallback_warned:
        checked = ", ".join(p for p in candidates if p)
        print(
            f"Warning: no TrueType font found (checked: {checked}) and this Pillow "
            "version's built-in fallback font is fixed-size. Text will be tiny and "
            "won't scale with --font-size. Pass --font-path /path/to/font.ttf, or "
            "upgrade Pillow to >=10.1, to fix this."
        )
        _font_fallback_warned = True
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Visible grid watermark: per-tile jitter
# ---------------------------------------------------------------------------

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
):
    """Build a transparent RGBA layer with the watermark repeated in a grid,
    where every tile gets its own random position offset, opacity, and
    rotation (all derived from `seed`, so the same seed always reproduces
    the exact same pattern).
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


# ---------------------------------------------------------------------------
# Blend modes: mix the mark into local pixel values instead of flat alpha-over
# ---------------------------------------------------------------------------

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
                      is set, scaled by that alpha - like ink soaking into
                      paper, so the result depends on the underlying pixel
                      value rather than sitting on a separable layer.
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


# ---------------------------------------------------------------------------
# Invisible backup mark: simple LSB payload (not robust to recompression)
# ---------------------------------------------------------------------------

def _text_to_bits(text):
    payload = text.encode("utf-8")
    header = len(payload).to_bytes(4, "big")
    data = header + payload
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def embed_invisible(base_rgba, text):
    """Spread `text` across the low bits of the R channel, repeated to fill
    the image (so it survives a crop that keeps a reasonably large region).
    Lossless output only (save as PNG) - JPEG recompression destroys this.
    """
    arr = np.asarray(base_rgba.convert("RGB"), dtype=np.uint8).copy()
    flat = arr[..., 0].reshape(-1)  # red channel, flattened
    bits = _text_to_bits(text)
    if not bits:
        return base_rgba
    reps = max(1, len(flat) // len(bits))
    tiled_bits = (bits * (reps + 1))[: len(flat)]
    bit_arr = np.array(tiled_bits, dtype=np.uint8)
    flat[:] = (flat & 0xFE) | bit_arr
    arr[..., 0] = flat.reshape(arr[..., 0].shape)
    out = Image.fromarray(arr, "RGB")
    return out.convert("RGBA")


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------

def save_with_metadata(image, output_path, author=None, copyright_text=None, jpeg_quality=95):
    is_jpeg = output_path.lower().endswith((".jpg", ".jpeg"))

    if is_jpeg:
        rgb = image.convert("RGB")
        exif_bytes = None
        if (author or copyright_text) and piexif is not None:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            if author:
                exif_dict["0th"][piexif.ImageIFD.Artist] = author.encode("utf-8")
            if copyright_text:
                exif_dict["0th"][piexif.ImageIFD.Copyright] = copyright_text.encode("utf-8")
            exif_bytes = piexif.dump(exif_dict)
        elif (author or copyright_text) and piexif is None:
            print("Note: install piexif (pip install piexif) to embed EXIF author/copyright in JPEGs.")
        if exif_bytes:
            rgb.save(output_path, quality=jpeg_quality, exif=exif_bytes)
        else:
            rgb.save(output_path, quality=jpeg_quality)
    else:
        pnginfo = None
        if author or copyright_text:
            pnginfo = PngImagePlugin.PngInfo()
            if author:
                pnginfo.add_text("Author", author)
            if copyright_text:
                pnginfo.add_text("Copyright", copyright_text)
        if pnginfo:
            image.save(output_path, pnginfo=pnginfo)
        else:
            image.save(output_path)
