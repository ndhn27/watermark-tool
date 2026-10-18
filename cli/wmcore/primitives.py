"""Small, dependency-free building blocks shared across the other wmcore
submodules: font loading, path/extension constants, seed derivation, and
payload bit-packing.
"""

import os
import zlib

from PIL import ImageFont

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

    Falls back to PIL's built-in default font at `font_size` (Pillow >= 10.1
    only - older Pillow's `load_default()` ignores the size and is
    tiny/blocky; that case prints a one-time warning instead of one per file).
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


def derive_seed(base_seed, key):
    """Deterministically derive a distinct-but-reproducible seed from a base
    seed and a key (a filename, a video segment index, etc.), so several
    items sharing one base seed each get their own jitter pattern instead of
    an identical, averageable one. Returns None unchanged when base_seed is
    None.

    Uses crc32 rather than the built-in hash(): Python salts string hashing
    per-process by default, which would break run-to-run reproducibility.
    """
    if base_seed is None:
        return None
    return base_seed + (zlib.crc32(str(key).encode("utf-8")) % 100000)


def text_to_bits(text):
    """UTF-8 encode `text` and prefix a 4-byte big-endian length header,
    returned as a flat list of bits (MSB first). Shared wire format for both
    the LSB and spread-spectrum invisible marks.
    """
    payload = text.encode("utf-8")
    header = len(payload).to_bytes(4, "big")
    data = header + payload
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits
