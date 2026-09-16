#!/usr/bin/env python3
"""
watermark.py - Apply a tiled, diagonal grid watermark to images.

Why a diagonal grid instead of a single corner logo?
  - A corner logo is gone with a single crop, and small-area AI inpainting
    can erase it easily.
  - A diagonal grid covers the whole image, so removing it cleanly means
    an AI (or a human) essentially has to repaint the entire image. That's
    far more effort, and it tends to leave artifacts or blur, especially
    over detailed areas (faces, text in charts, etc.).

Usage:
    # single image
    python3 watermark.py input.jpg output.jpg --text "@yourhandle"

    # whole folder (batch)
    python3 watermark.py input_folder/ output_folder/ --text "@yourhandle" --batch

Setup:
    pip install pillow --break-system-packages
"""

import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont

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


def create_watermark_layer(size, text, font_size=36, opacity=90, angle=30, spacing=250, font_path=None):
    """Build a transparent RGBA layer containing the watermark repeated in a
    grid pattern, already rotated to the given angle."""
    width, height = size
    # The temporary layer must be at least as large as the image diagonal so
    # that rotating it doesn't leave gaps at the corners.
    diagonal = int(math.hypot(width, height)) + spacing
    layer = Image.new("RGBA", (diagonal, diagonal), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = load_font(font_size, font_path)

    text_color = (255, 255, 255, max(0, min(255, opacity)))  # white, custom alpha

    row = 0
    for y in range(0, diagonal, spacing):
        # Stagger even/odd rows so the watermark is denser and harder to dodge
        offset_x = (spacing // 2) if row % 2 else 0
        for x in range(-spacing, diagonal + spacing, spacing):
            draw.text((x + offset_x, y), text, font=font, fill=text_color)
        row += 1

    rotated = layer.rotate(angle, expand=False, resample=Image.BICUBIC)
    left = (diagonal - width) // 2
    top = (diagonal - height) // 2
    return rotated.crop((left, top, left + width, top + height))


def apply_watermark(input_path, output_path, text, opacity=90, font_size=36, angle=30, spacing=250, font_path=None):
    base = Image.open(input_path).convert("RGBA")
    watermark_layer = create_watermark_layer(base.size, text, font_size, opacity, angle, spacing, font_path)
    watermarked = Image.alpha_composite(base, watermark_layer)

    if output_path.lower().endswith((".jpg", ".jpeg")):
        watermarked = watermarked.convert("RGB")
        watermarked.save(output_path, quality=95)
    else:
        watermarked.save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply a diagonal grid watermark to an image, resistant to cropping and inpainting.")
    parser.add_argument("input", help="Input image file (or folder if using --batch)")
    parser.add_argument("output", help="Output image file (or folder if using --batch)")
    parser.add_argument("--text", required=True, help="Watermark text, e.g. @yourhandle or a group/brand name")
    parser.add_argument("--opacity", type=int, default=90, help="Opacity, 0-255 (default: 90, ~35%%)")
    parser.add_argument("--font-size", type=int, default=36, help="Font size (default: 36)")
    parser.add_argument("--angle", type=int, default=30, help="Rotation angle in degrees (default: 30)")
    parser.add_argument("--spacing", type=int, default=250, help="Spacing between watermark instances, in px (default: 250)")
    parser.add_argument("--font-path", default=None, help="Optional path to a custom .ttf font file")
    parser.add_argument("--batch", action="store_true", help="Process a whole input folder into an output folder")
    args = parser.parse_args()

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
            apply_watermark(in_path, out_path, args.text, args.opacity, args.font_size, args.angle, args.spacing, args.font_path)
    else:
        apply_watermark(args.input, args.output, args.text, args.opacity, args.font_size, args.angle, args.spacing, args.font_path)


if __name__ == "__main__":
    main()
