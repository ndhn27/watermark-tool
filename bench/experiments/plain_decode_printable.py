#!/usr/bin/env python3
"""Does the PLAIN (non-`--rescale`) spread-spectrum text decoder accept junk on periodic,
never-stamped images? And would filtering it lose a real payload that legitimately contains
control characters (\\t \\n \\r)?

bench/results/P2-NOTES.md §6.6 found 2/30 junk accepts on periodic images *inside the rescale
search*, and traced both to the plain decode the search does at factor 1.0 (`attempts=1`), not
to the resize search itself - the search's own two-widths-must-agree-and-be-printable rule
never saw them. This script tests that plain path directly (no search at all): the exact call
`extract_spread_spectrum()` makes - `find_spread_spectrum_payload_or_id` on `identity`, and on a
resized copy decoded as-is (which is what a plain `extract` on a resized repost actually gets:
no search, block grid already desynced) - to get a real n on the plain path itself, not an
inference from search internals.

Part A - false positives: 10 periodic patterns (from id_rescale_hypotheses.structured_images),
3 sizes, 3 states each (identity / resized x0.75 / resized x1.25, no search) = 90 plain decodes,
never stamped.

Part B - would a printable filter cost real payloads: embed text containing \\t \\n \\r (and,
since the tool ships Vietnamese examples, diacritics) for real, decode through the exact same
plain path, across a JPEG quality range, and check the payload round-trips character for
character.

    python3 bench/experiments/plain_decode_printable.py                     # both parts, before/after use --after
    python3 bench/experiments/plain_decode_printable.py --after             # run again once the filter is added
"""

import argparse
import io
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in (".", "..", os.path.join("..", "..", "cli")):
    p = os.path.abspath(os.path.join(HERE, sub))
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import bench_attacks as A  # noqa: E402
from id_rescale_hypotheses import structured_images  # noqa: E402
import wmcore  # noqa: E402

SIZES = ((640, 480), (320, 240), (1280, 960))
SEED = 20260919 + 2_000_000

CONTROL_PAYLOADS = [
    "line one\nline two\t(tabbed)\r\n",
    "\t\tindented, tab-only lead-in",
    "Bản quyền © Kiobh 2026\ncaption line two\tend",
    "\r\ntrailing/leading crlf\r\n",
]


def part_a(after):
    print(f"== Part A: plain-path false positives on periodic images ({'AFTER filter' if after else 'BEFORE filter'})")
    total = 0
    junk = 0
    examples = []
    for w, h in SIZES:
        items = structured_images(w, h, SEED)
        for name, base in items:
            key = wmcore.derive_seed(SEED, f"plain|{name}|{w}x{h}")
            for state in ("identity", "resize_0.75", "resize_1.25"):
                sig = f"{name}|{w}x{h}|{state}".encode()
                rng = np.random.default_rng([SEED, zlib.crc32(sig)])
                attacked, _ = A.REGISTRY[state].fn(base, rng)
                corrs = wmcore.spread_spectrum_block_correlations(attacked.convert("RGB"), key)
                text = wmcore.find_spread_spectrum_payload_or_id(corrs)
                total += 1
                if text is not None:
                    junk += 1
                    examples.append((w, h, name, state, text))
    print(f"   plain decodes on never-stamped periodic images: {total}")
    print(f"   returned non-None text (junk, since nothing was embedded): {junk}")
    for w, h, name, state, text in examples:
        print(f"     [{w}x{h} {name} / {state}] -> {text!r}")
    return total, junk


def part_b(after):
    print(f"\n== Part B: control-character payloads round-trip through the plain path ({'AFTER filter' if after else 'BEFORE filter'})")
    xx, yy = np.meshgrid(np.linspace(0, 255, 640), np.linspace(0, 255, 480))
    rng = np.random.default_rng(0)
    arr = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
    arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
    base = Image.fromarray(arr, "RGB").convert("RGBA")

    key = 20260921
    total = 0
    ok = 0
    fails = []
    for text in CONTROL_PAYLOADS:
        stamped = wmcore.embed_spread_spectrum(base, text, key)
        for quality in (None, 95, 75):
            total += 1
            if quality is None:
                probe = stamped.convert("RGB")
            else:
                buf = io.BytesIO()
                stamped.convert("RGB").save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                probe = Image.open(buf).convert("RGB")
            corrs = wmcore.spread_spectrum_block_correlations(probe, key)
            decoded = wmcore.find_spread_spectrum_payload_or_id(corrs)
            if decoded == text:
                ok += 1
            else:
                fails.append((text, quality, decoded))
    print(f"   control-char payload trials (4 payloads x 3 conditions): {total}")
    print(f"   round-tripped exactly: {ok}")
    for text, quality, decoded in fails:
        print(f"     FAIL payload={text!r} quality={quality} decoded={decoded!r}")
    return total, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--after", action="store_true", help="label output as post-filter (no behavior difference; the code IS what it is)")
    args = ap.parse_args()
    a_total, a_junk = part_a(args.after)
    b_total, b_ok = part_b(args.after)
    print(f"\n== Summary: periodic junk {a_junk}/{a_total}; control-char round-trip {b_ok}/{b_total}")


if __name__ == "__main__":
    main()
