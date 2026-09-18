#!/usr/bin/env python3
"""
extract_invisible.py - Read back the hidden LSB payload embedded by
watermark.py --invisible.

Only works on the exact bytes that were saved (a lossless PNG). Any
recompression, resize, or format conversion after embedding will destroy
the payload - this is a lightweight fallback signal, not a robust
watermark.

wmcore.embed_invisible() tiles (header + payload) bits repeatedly across
the *entire* flattened R-channel, back-to-back, specifically so a copy
survives even if part of the image is cropped or painted over. That only
helps if extraction actually looks for a surviving copy: this scans every
offset in the flattened bit-stream for a plausible header, not just
offset 0 (the top-left corner), so a repeat elsewhere in the image is
still found even when the first copy is gone. See _find_payload() below.

Usage:
    python3 extract_invisible.py watermarked.png
"""

import argparse
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

# Upper bound on a plausible decoded payload length (bytes), used only to
# cut down which 32-bit windows are worth trying as a header - not a limit
# on what embed_invisible() can write. Generous on purpose: raising it
# finds longer payloads at the cost of a few more (cheap, self-rejecting)
# candidates to check.
MAX_PLAUSIBLE_LENGTH = 10_000


def _decode_at(bits, offset, length, n):
    """Decode `length` payload bytes starting 32 bits after `offset`, or
    None if that range doesn't fit or isn't valid UTF-8."""
    start = offset + 32
    end = start + length * 8
    if end > n:
        return None
    payload_bits = np.asarray(bits[start:end], dtype=np.uint8)
    byte_vals = np.packbits(payload_bits)
    try:
        return byte_vals.tobytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def _find_payload(bits, max_length=MAX_PLAUSIBLE_LENGTH):
    """Scan the flattened LSB bit-stream for a valid header+payload at ANY
    offset, not just position 0.

    embed_invisible() repeats (32-bit length header + payload bits)
    back-to-back for as many full copies as fit in the image, so a crop or
    paint-over that destroys the copy at offset 0 (the top-left corner)
    still leaves later copies intact elsewhere - as long as something
    actually looks for them. This does, by treating every offset as a
    candidate header start rather than assuming the first one is the
    only one worth trying.

    Vectorized so it stays fast on real image sizes: every 32-bit window
    is decoded as a big-endian length in one pass, and offsets whose
    "length" isn't plausible are discarded immediately.

    A raw "does 32 bits + a plausible length decode as UTF-8" test is too
    weak on its own - across a multi-megapixel image, ordinary pixel noise
    throws up occasional short byte sequences that happen to be valid
    UTF-8 by chance (single ASCII-range bytes especially). Because the
    real mark repeats with a fixed period (32 + length*8 bits), a genuine
    copy can be corroborated by checking whether the *next* (or previous)
    period decodes to the exact same text - something random pixel data
    essentially never does by coincidence. Every candidate is checked for
    that corroboration first; only if nothing in the image corroborates
    does this fall back to the single best-looking candidate, same as
    before.
    """
    n = len(bits)
    if n < 32:
        return None

    bits_u32 = np.asarray(bits, dtype=np.uint32)
    windows = sliding_window_view(bits_u32, 32)
    weights = (1 << np.arange(31, -1, -1)).astype(np.uint32)
    lengths = windows @ weights  # decoded 32-bit length at every offset

    candidate_offsets = np.flatnonzero((lengths > 0) & (lengths <= max_length))

    fallback = None  # best single, uncorroborated candidate - last resort
    for offset in candidate_offsets:
        offset = int(offset)
        length = int(lengths[offset])
        text = _decode_at(bits, offset, length, n)
        if text is None:
            continue
        if fallback is None:
            fallback = text

        period = 32 + length * 8
        next_offset = offset + period
        if next_offset + 32 <= n and _decode_at(bits, next_offset, length, n) == text:
            return text  # a second intact copy confirms this is the real mark
        prev_offset = offset - period
        if prev_offset >= 0 and _decode_at(bits, prev_offset, length, n) == text:
            return text

    return fallback


def extract_invisible(path, max_length=MAX_PLAUSIBLE_LENGTH):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    bits = (arr[..., 0].reshape(-1) & 1)
    return _find_payload(bits, max_length=max_length)


def main():
    parser = argparse.ArgumentParser(description="Extract the hidden LSB watermark payload from an image.")
    parser.add_argument("image", help="Path to a PNG stamped with watermark.py --invisible")
    parser.add_argument(
        "--max-length", type=int, default=MAX_PLAUSIBLE_LENGTH,
        help=f"Longest payload (bytes) to consider plausible while scanning (default: {MAX_PLAUSIBLE_LENGTH})",
    )
    args = parser.parse_args()

    text = extract_invisible(args.image, max_length=args.max_length)
    if text is None:
        print("No valid hidden payload found (image may be recompressed, resized, or never stamped).")
        sys.exit(1)
    print(f"Hidden payload: {text}")


if __name__ == "__main__":
    main()
