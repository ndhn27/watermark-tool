#!/usr/bin/env python3
"""Should the FREE-TEXT decoder also be run on mirrored orientations? (P3 decision, bench/results/P3-NOTES.md §3)

wmcore/orient.py searches mirrored orientations for a watermark ID only. The alternative - running
`spread.find_spread_spectrum_payload` (accept rule: valid UTF-8 + printable) on all four orientations too - would
recover mirrored TEXT payloads, but every extra orientation is another chance for that loose rule to accept junk.
This script measures both sides, on the SAME images, so the decision rests on numbers:

  A. gain: marked text payloads (`short`, `medium`), mirrored (h / v / hv, and h + JPEG q75), decoded with the text
     decoder after undoing the right mirroring - how often is the payload read? And what does the text decoder
     return in the three WRONG orientations of those same images (nothing should come back)?
  B. cost: never-stamped images - natural-ish (pink / gradnoise, several sizes) and the 10 periodic patterns of
     P2-NOTES §6.6 (the ones that fooled the text decoder before the printable filter) - text decoder on all four
     orientations: how many return something? Anything returned is a false positive.

    python3 bench/experiments/flip_text_probe.py
"""

import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in (".", "..", os.path.join("..", "..", "cli")):
    p = os.path.abspath(os.path.join(HERE, sub))
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import bench_attacks as A  # noqa: E402
import bench_corpus as C  # noqa: E402
import bench_schemes as S  # noqa: E402
from id_rescale_hypotheses import structured_images  # noqa: E402
import wmcore  # noqa: E402
from wmcore import orient, spread  # noqa: E402

SEED = 20260919 + 3_000_000
SIZES = ((640, 480), (320, 240), (1280, 960))
ATTACKS = ("flip_horizontal_jpeg_75", "flip_vertical", "flip_both")
FLIP_OF = {"flip_horizontal_jpeg_75": "h", "flip_vertical": "v", "flip_both": "hv"}


def text_decode(rgb, key, orientation):
    y = orient._luma(rgb)
    corrs = spread._ss_correlations_from_luma(np.ascontiguousarray(orient.unflip(y, orientation)), key)
    return spread.find_spread_spectrum_payload(corrs)


def part_a():
    print("== A. marked TEXT payloads, mirrored: text decoder after undoing the right / a wrong mirroring")
    tot = {"right_ok": 0, "right_wrong": 0, "right_none": 0, "n": 0, "wrong_orient_returns": 0, "wrong_orient_n": 0}
    for pid in ("short", "medium"):
        for kind in ("pink", "gradnoise"):
            for (w, h) in SIZES[:2]:
                for i in range(12):
                    spec = {"kind": kind, "w": w, "h": h, "seed": SEED + i}
                    base = C.load_source(spec)
                    key = wmcore.derive_seed(SEED, f"a|{pid}|{kind}|{w}|{i}")
                    payload = S.PAYLOADS[pid]
                    stamped = S.embed("ss", base, payload, key, 12.0).convert("RGB")
                    for name in ATTACKS:
                        rng = np.random.default_rng([SEED, zlib.crc32(f"{pid}{kind}{w}{i}{name}".encode())])
                        attacked, _ = A.REGISTRY[name].fn(stamped, rng)
                        true_o = FLIP_OF[name]
                        got = text_decode(attacked, key, true_o)
                        tot["n"] += 1
                        tot["right_ok" if got == payload else "right_none" if got is None else "right_wrong"] += 1
                        for o in orient.ORIENTATIONS:
                            if o == true_o:
                                continue
                            tot["wrong_orient_n"] += 1
                            if text_decode(attacked, key, o) is not None:
                                tot["wrong_orient_returns"] += 1
    print(f"   right orientation: read {tot['right_ok']}/{tot['n']}, none {tot['right_none']}, wrong string {tot['right_wrong']}")
    print(f"   the 3 wrong orientations of the same images: returned something on "
          f"{tot['wrong_orient_returns']}/{tot['wrong_orient_n']}")
    return tot


def part_b():
    print("\n== B. never-stamped images: text decoder on each orientation (anything returned = false positive)")
    counts = {o: 0 for o in orient.ORIENTATIONS}
    n = 0
    examples = []
    # natural-ish synthetic images
    for kind in ("pink", "gradnoise"):
        for (w, h) in SIZES[:2]:
            for i in range(25):
                base = C.load_source({"kind": kind, "w": w, "h": h, "seed": SEED + 500 + i}).convert("RGB")
                key = wmcore.derive_seed(SEED, f"b|{kind}|{w}|{i}")
                n += 1
                for o in orient.ORIENTATIONS:
                    if text_decode(base, key, o) is not None:
                        counts[o] += 1
    print(f"   synthetic natural-ish (pink, gradnoise; 640x480 and 320x240; {n} images): returned per orientation: "
          + ", ".join(f"{o}={counts[o]}" for o in orient.ORIENTATIONS))
    counts_p = {o: 0 for o in orient.ORIENTATIONS}
    n_p = 0
    for (w, h) in SIZES:
        for name, base in structured_images(w, h, SEED):
            key = wmcore.derive_seed(SEED, f"bp|{name}|{w}x{h}")
            for state in ("identity", "resize_0.75", "resize_1.25"):
                rng = np.random.default_rng([SEED, zlib.crc32(f"{name}|{w}x{h}|{state}".encode())])
                attacked, _ = A.REGISTRY[state].fn(base, rng)
                n_p += 1
                for o in orient.ORIENTATIONS:
                    t = text_decode(attacked.convert("RGB"), key, o)
                    if t is not None:
                        counts_p[o] += 1
                        examples.append((w, h, name, state, o, t))
    print(f"   10 periodic patterns x 3 sizes x 3 states ({n_p} images): returned per orientation: "
          + ", ".join(f"{o}={counts_p[o]}" for o in orient.ORIENTATIONS))
    for w, h, name, state, o, t in examples:
        print(f"     [{w}x{h} {name} / {state} / orientation {o}] -> {t!r}")
    return counts, counts_p


def part_c():
    """The decoder that actually ships (ID in any orientation, else text on the image as given) against the
    old one (`find_spread_spectrum_payload_or_id`) on the periodic never-stamped images: must not return more."""
    print("\n== C. shipped mirror-aware decoder vs old decoder, periodic never-stamped images")
    n = old = new = new_only = 0
    for (w, h) in SIZES:
        for name, base in structured_images(w, h, SEED):
            key = wmcore.derive_seed(SEED, f"bp|{name}|{w}x{h}")
            for state in ("identity", "resize_0.75", "resize_1.25"):
                rng = np.random.default_rng([SEED, zlib.crc32(f"{name}|{w}x{h}|{state}".encode())])
                attacked, _ = A.REGISTRY[state].fn(base, rng)
                rgb = attacked.convert("RGB")
                o = wmcore.find_spread_spectrum_payload_or_id(wmcore.spread_spectrum_block_correlations(rgb, key))
                m = wmcore.find_spread_spectrum_payload_or_id_oriented(rgb, key)
                n += 1
                old += o is not None
                new += m.text is not None
                new_only += (m.text is not None and m.text != o)
    print(f"   {n} images: old decoder returned something on {old}, mirror-aware on {new}; "
          f"different from the old one's result on {new_only}")


def part_d():
    import time
    print("\n== D. cost of the mirror-aware decoder vs the old one (1 CPU, best of 3, block DCT included)")
    rng = np.random.default_rng(1)
    from PIL import Image
    for (w, h) in ((640, 480), (1280, 960), (4000, 3000)):
        arr = np.clip(rng.normal(128, 40, (h, w, 3)), 0, 255).astype(np.uint8)
        base = Image.fromarray(arr, "RGB")
        key = 4242
        marked = wmcore.embed_spread_spectrum(base.convert("RGBA"), "K3M9QXR2T4", key).convert("RGB") if w < 4000 else None
        def old_dec(im):
            return wmcore.find_spread_spectrum_payload_or_id(wmcore.spread_spectrum_block_correlations(im, key))
        def new_dec(im):
            return wmcore.find_spread_spectrum_payload_or_id_oriented(im, key)
        rows = [("unmarked", base)] + ([("marked, not mirrored", marked)] if marked is not None else [])
        for label, im in rows:
            t = {}
            for nm, fn in (("old", old_dec), ("new", new_dec)):
                best = 1e9
                for _ in range(3):
                    t0 = time.perf_counter()
                    fn(im)
                    best = min(best, time.perf_counter() - t0)
                t[nm] = best
            print(f"   {w}x{h} {label}: old {t['old']*1000:.0f} ms, mirror-aware {t['new']*1000:.0f} ms")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
