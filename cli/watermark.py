#!/usr/bin/env python3
"""
watermark.py - Apply a tiled, diagonal grid watermark to images, with
several layers of removal-resistance stacked on top of the basic grid.

Why a diagonal grid instead of a single corner logo?
  - A corner logo is gone with a single crop, and small-area AI inpainting
    can erase it easily.
  - A diagonal grid covers the whole image, so removing it cleanly means
    an AI (or a human) essentially has to repaint the entire image. That's
    far more effort, and it tends to leave artifacts or blur, especially
    over detailed areas (faces, text in charts, etc.).

What's new in this version:
  - Per-tile jitter (position / opacity / rotation), seeded with --seed.
    A perfectly regular grid has a strong, narrow signature in the
    frequency domain (FFT), which makes it possible to estimate and
    subtract automatically, especially if an attacker has several images
    stamped with the same pattern to average over. Randomizing each tile
    independently breaks that periodicity.
  - Blend modes (--blend multiply|overlay) instead of a flat alpha
    composite. A flat-alpha white watermark sits on its own "layer" that
    can, in principle, be estimated and subtracted. Multiply/overlay mix
    the mark into the local pixel values instead, so removal requires
    reconstructing the underlying image data, not just subtracting a
    constant layer.
  - An optional invisible backup mark (--invisible), a simple LSB
    (least-significant-bit) payload spread across the image. It is NOT
    robust to recompression, resizing, or screenshots (those need
    DCT/frequency-domain or learned watermarking, which is a much bigger
    project) - it's meant only as a lightweight fallback: if the visible
    grid is cropped or painted over on a losslessly-saved copy, this can
    still confirm provenance. Use extract_invisible.py to read it back.
  - Provenance metadata: --author / --copyright get written into real
    EXIF (JPEG, via piexif) or PNG text chunks, so tools and platforms
    that read standard metadata (and, going forward, C2PA/Content
    Credentials-aware tools) see attribution even if they never look at
    the pixels at all. This is a separate, complementary signal - anyone
    can strip metadata, so don't rely on it alone.

Usage:
    # single image
    python3 watermark.py input.jpg output.jpg --text "@yourhandle"

    # whole folder (batch), reproducible jitter pattern
    python3 watermark.py in/ out/ --text "@yourhandle" --batch --seed 42

    # texture-aware blend + invisible backup + provenance metadata
    python3 watermark.py input.jpg output.jpg --text "@yourhandle" \\
        --blend multiply --invisible --author "Jane Doe" --copyright "(c) 2026 Jane Doe"

Setup:
    pip install -r ../requirements.txt
    # or: pip install pillow numpy piexif --break-system-packages
"""

import argparse
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
    "C:\\Windows\\Fonts\\arialbd.ttf",                         # Windows fallback
]


def load_font(font_size, font_path=None):
    candidates = [font_path] if font_path else DEFAULT_FONT_PATHS
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, font_size)
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

def composite(base_rgba, mark_rgba, mode="alpha", ink=0.32):
    """Combine base image + watermark layer.

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

    base = np.asarray(base_rgba.convert("RGB"), dtype=np.float32) / 255.0
    alpha = np.asarray(mark_rgba.getchannel("A"), dtype=np.float32) / 255.0
    alpha = alpha[..., None]

    if mode == "multiply":
        blended = base * ink
    elif mode == "overlay":
        blended = np.where(base < 0.5, 2 * base * ink, 1 - 2 * (1 - base) * (1 - ink))
    else:
        raise ValueError(f"Unknown blend mode: {mode}")

    out = base * (1 - alpha) + blended * alpha
    out_img = Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
    return out_img.convert("RGBA")


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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def apply_watermark(
    input_path,
    output_path,
    text,
    opacity=90,
    font_size=36,
    angle=30,
    spacing=250,
    font_path=None,
    seed=None,
    blend="alpha",
    ink=0.32,
    invisible=False,
    invisible_text=None,
    author=None,
    copyright_text=None,
):
    base = Image.open(input_path).convert("RGBA")
    watermark_layer = create_watermark_layer(
        base.size, text, font_size, opacity, angle, spacing, font_path, seed=seed
    )
    watermarked = composite(base, watermark_layer, mode=blend, ink=ink)

    if invisible:
        if output_path.lower().endswith((".jpg", ".jpeg")):
            print(
                f"Warning: --invisible payload requires a lossless format; "
                f"{output_path} is JPEG and recompression will destroy it. "
                f"Consider a .png output instead."
            )
        watermarked = embed_invisible(watermarked, invisible_text or text)

    save_with_metadata(watermarked, output_path, author=author, copyright_text=copyright_text)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply a diagonal grid watermark to an image, resistant to cropping and inpainting."
    )
    parser.add_argument("input", help="Input image file (or folder if using --batch)")
    parser.add_argument("output", help="Output image file (or folder if using --batch)")
    parser.add_argument("--text", required=True, help="Watermark text, e.g. @yourhandle or a group/brand name")
    parser.add_argument("--opacity", type=int, default=90, help="Base opacity, 0-255 (default: 90, ~35%%)")
    parser.add_argument("--font-size", type=int, default=36, help="Font size (default: 36)")
    parser.add_argument("--angle", type=int, default=30, help="Base rotation angle in degrees (default: 30)")
    parser.add_argument("--spacing", type=int, default=250, help="Spacing between watermark tiles, in px (default: 250)")
    parser.add_argument("--font-path", default=None, help="Optional path to a custom .ttf font file")
    parser.add_argument("--batch", action="store_true", help="Process a whole input folder into an output folder")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for tile jitter (reuse it to reproduce the same pattern)")
    parser.add_argument("--blend", choices=["alpha", "multiply", "overlay"], default="alpha", help="How the mark mixes with the base image (default: alpha)")
    parser.add_argument("--ink", type=float, default=0.32, help="Darkness of the mark for multiply/overlay blend, 0-1 (default: 0.32)")
    parser.add_argument("--invisible", action="store_true", help="Also embed a hidden LSB backup mark (save as PNG - JPEG destroys it)")
    parser.add_argument("--invisible-text", default=None, help="Text for the hidden mark, defaults to --text")
    parser.add_argument("--author", default=None, help="Author name to embed in EXIF/PNG metadata")
    parser.add_argument("--copyright", dest="copyright_text", default=None, help="Copyright string to embed in EXIF/PNG metadata")
    args = parser.parse_args()

    common = dict(
        text=args.text,
        opacity=args.opacity,
        font_size=args.font_size,
        angle=args.angle,
        spacing=args.spacing,
        font_path=args.font_path,
        seed=args.seed,
        blend=args.blend,
        ink=args.ink,
        invisible=args.invisible,
        invisible_text=args.invisible_text,
        author=args.author,
        copyright_text=args.copyright_text,
    )

    if args.batch:
        os.makedirs(args.output, exist_ok=True)
        exts = (".jpg", ".jpeg", ".png", ".webp")
        files = [f for f in os.listdir(args.input) if f.lower().endswith(exts)]
        if not files:
            print("No images found in the input folder.")
            return
        for fname in files:
            in_path = os.path.join(args.input, fname)
            out_path = os.path.join(args.output, fname)
            # Vary the seed per-file (if one was given) so a batch doesn't
            # stamp every image with the exact same jitter pattern, which
            # would itself become a cross-image signature.
            file_kwargs = dict(common)
            if args.seed is not None:
                file_kwargs["seed"] = args.seed + hash(fname) % 100000
            apply_watermark(in_path, out_path, **file_kwargs)
    else:
        apply_watermark(args.input, args.output, **common)


if __name__ == "__main__":
    main()
