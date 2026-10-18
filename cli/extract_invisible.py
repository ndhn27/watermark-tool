#!/usr/bin/env python3
"""
extract_invisible.py - Read back an invisible mark embedded by
watermark.py --invisible, in either of its two modes:

  --invisible-mode lsb (the default): a simple least-significant-bit
    payload. Only works on the exact bytes that were saved (a lossless
    PNG) - any recompression, resize, or format conversion after embedding
    destroys it. See extract_invisible() / _find_payload() below.

  --invisible-mode spread: a block-DCT spread-spectrum payload (see the
    module comment above embed_spread_spectrum() in wmcore.py for the full
    design rationale). Survives ordinary JPEG recompression and a
    block-aligned crop, at the cost of requiring the same `--ss-key` that
    was used to embed it. See extract_spread_spectrum() below.

wmcore.embed_invisible() tiles (header + payload) bits repeatedly across
the *entire* flattened R-channel, back-to-back, specifically so a copy
survives even if part of the image is cropped or painted over. That only
helps if extraction actually looks for a surviving copy: this scans every
offset in the flattened bit-stream for a plausible header, not just
offset 0 (the top-left corner), so a repeat elsewhere in the image is
still found even when the first copy is gone. See _find_payload() below.
embed_spread_spectrum() tiles the same way across the image's 8x8 block
grid instead of across pixels; find_spread_spectrum_payload() in wmcore.py
does the analogous scan over blocks.

A file stamped with `watermark.py --id` carries a short watermark ID instead of
text. Run with no key: the ID is looked up in the ledger (~/.watermark-tool/
ledger.sqlite3, or --ledger), its recorded spread-spectrum keys are tried, and
the report says which file/run the ID belongs to and whether this file is
byte-identical to what was stamped. Metadata and a C2PA manifest are checked
for the ID too when the pixel layers yield nothing.

Usage:
    python3 extract_invisible.py suspect.jpg                 # ID stamped with --id: ID + ledger record
    python3 extract_invisible.py watermarked.png
    python3 extract_invisible.py watermarked.jpg --mode spread --ss-key 20260918
    python3 extract_invisible.py resized.jpg --mode spread --ss-key 20260918 --rescale
"""

import argparse
import sys

import wmapp
from wmcore.invisible import (  # noqa: F401  (re-exported: tests, bench/ and callers import these from here)
    MAX_PLAUSIBLE_LENGTH,
    _decode_at,
    _find_payload,
    extract_invisible,
    extract_spread_spectrum,
    extract_spread_spectrum_rescaled,
)


def main():
    parser = argparse.ArgumentParser(description="Extract an invisible watermark payload from an image.")
    parser.add_argument("image", help="Path to an image stamped with watermark.py --invisible")
    wmapp.add_arguments(parser, wmapp.EXTRACT_OPTIONS)
    args = parser.parse_args()

    try:
        result = wmapp.extract(args.image, wmapp.spec_from_namespace(wmapp.ExtractSpec, args))
    except wmapp.SpecError as exc:
        parser.error(str(exc))

    for line in result.report_lines():
        print(line)
    if not result.found:
        sys.exit(1)


if __name__ == "__main__":
    main()
