#!/usr/bin/env python3
"""
extract_invisible.py - Read back the hidden LSB payload embedded by
watermark.py --invisible.

Only works on the exact bytes that were saved (a lossless PNG). Any
recompression, resize, or format conversion after embedding will destroy
the payload - this is a lightweight fallback signal, not a robust
watermark.

Usage:
    python3 extract_invisible.py watermarked.png
"""

import argparse
import sys

import numpy as np
from PIL import Image


def extract_invisible(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    bits = (arr[..., 0].reshape(-1) & 1).tolist()

    if len(bits) < 32:
        return None

    length_bits = bits[:32]
    length = 0
    for b in length_bits:
        length = (length << 1) | b

    total_bits_needed = 32 + length * 8
    if length <= 0 or total_bits_needed > len(bits):
        return None

    payload_bits = bits[32:total_bits_needed]
    byte_vals = bytearray()
    for i in range(0, len(payload_bits), 8):
        chunk = payload_bits[i:i + 8]
        val = 0
        for b in chunk:
            val = (val << 1) | b
        byte_vals.append(val)

    try:
        return byte_vals.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract the hidden LSB watermark payload from an image.")
    parser.add_argument("image", help="Path to a PNG stamped with watermark.py --invisible")
    args = parser.parse_args()

    text = extract_invisible(args.image)
    if text is None:
        print("No valid hidden payload found (image may be recompressed, resized, or never stamped).")
        sys.exit(1)
    print(f"Hidden payload: {text}")


if __name__ == "__main__":
    main()
