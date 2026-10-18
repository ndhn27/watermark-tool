#!/usr/bin/env python3
"""Where does the ID-aware spread decoder beat the header-hunting one, and how much room is left?

Compares, on the very same attacked image, three readings of a spread-spectrum mark that carries a real
(check-valid) watermark ID:

  ss        wmcore.find_spread_spectrum_payload   discovers the period by hoping one repeat's 32-bit header
                                                  survives bit-exact
  ss_id     wmcore.find_spread_spectrum_id        knows the period (112 bits); 112 phases, CRC-verified
  sum_exact "oracle" reference: the true phase, ALL 112 bits (header included) hard-thresholded after summing
            the repeats, exactly right. Not a decoder (it knows the answer's phase); it says whether the signal
            is intact. ss_id can exceed it because it ignores the header bits.

For every trial where ss_id still misses, it also reports how many of the 80 payload bits are wrong at the true
phase - the room a soft (alphabet-constrained) decoder would have to work with.

Only attacks that keep the block grid (same pixel grid, or a block-aligned top crop) are used, so the true phase
is known. Resizes and geometry are the rescale search's business (bench/run_benchmark.py).

    python3 bench/experiments/id_decoder_probe.py --sizes 640x480,320x240 --n 8
    python3 bench/experiments/id_decoder_probe.py --images DIR
    python3 bench/experiments/id_decoder_probe.py --pipeline 6      # full wmapp pipeline, grid on, JPEG q sweep
"""

import argparse
import os
import random
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", os.path.join("..", "..", "cli")):
    p = os.path.abspath(os.path.join(HERE, sub))
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import bench_attacks as A  # noqa: E402
import bench_corpus as C  # noqa: E402
import wmcore  # noqa: E402
from wmcore import wmid  # noqa: E402

PERIOD = wmcore.ID_PERIOD_BITS

ATTACKS = {
    "identity": lambda img, rng: img,
    "jpeg_50": lambda img, rng: A._jpeg_bytes(img, 50),
    "jpeg_40": lambda img, rng: A._jpeg_bytes(img, 40),
    "jpeg_30": lambda img, rng: A._jpeg_bytes(img, 30),
    "jpeg_20": lambda img, rng: A._jpeg_bytes(img, 20),
    "noise_10": lambda img, rng: A._noise(10)(img, rng)[0],
    "noise_15": lambda img, rng: A._noise(15)(img, rng)[0],
    "blur_1.0": lambda img, rng: A._blur(1.0)(img, rng)[0],
    "blur_1.5": lambda img, rng: A._blur(1.5)(img, rng)[0],
    "blur_2.0": lambda img, rng: A._blur(2.0)(img, rng)[0],
    "crop_top_64": lambda img, rng: A._crop_blocks(64, 0)(img, rng)[0],
}


def _phase(attack, width):
    """Where the 112-bit period starts in the attacked image's block raster."""
    return (-8 * (width // 8)) % PERIOD if attack == "crop_top_64" else 0


def _fold(corrs, phase):
    reps = (len(corrs) - phase) // PERIOD
    return corrs[phase:phase + reps * PERIOD].reshape(reps, PERIOD).sum(axis=0), reps


def run_images(specs, label, seed):
    rows = {}
    misses = {}      # attack -> [payload-bit errors of ss_id misses]
    for spec in specs:
        base = C.load_source(spec)
        W = base.size[0]
        uid = f"{label}|{C.source_id(spec)}"
        payload = wmid.generate(random.Random(f"{seed}|{uid}"))
        key = wmcore.derive_seed(seed, uid)
        stamped = wmcore.embed_spread_spectrum(base, payload, key).convert("RGB")
        bits = np.array(wmcore.text_to_bits(payload), dtype=np.uint8)
        for name, fn in ATTACKS.items():
            rng = np.random.default_rng([seed, zlib.crc32((uid + "|" + name).encode())])
            corrs = wmcore.spread_spectrum_block_correlations(fn(stamped, rng), key)
            comb, _ = _fold(corrs, _phase(name, W))
            got = (comb > 0).astype(np.uint8)
            ok_old = wmcore.find_spread_spectrum_payload(corrs) == payload
            ok_id = wmcore.find_spread_spectrum_id(corrs) == payload
            r = rows.setdefault(name, [0, 0, 0, 0])
            r[0] += 1
            r[1] += ok_old
            r[2] += ok_id
            r[3] += bool(np.array_equal(got, bits))
            if not ok_id:
                misses.setdefault(name, []).append(int((got[32:] != bits[32:]).sum()))
    print(f"\n== {label}  ({len(specs)} images)")
    print(f"{'attack':12s} {'n':>3s} {'ss':>4s} {'ss_id':>5s} {'sum_exact':>9s}   payload-bit errors (of 80) where ss_id missed")
    for name, (n, a, b, c) in rows.items():
        print(f"{name:12s} {n:3d} {a:4d} {b:5d} {c:9d}   {sorted(misses.get(name, []))}")


def run_pipeline(n_seeds):
    import wmapp
    print(f"\n== full wmapp pipeline: 640x480 gradnoise, grid text on, spread mark carrying the ID, JPEG re-save "
          f"({n_seeds} images)")
    qs = (50, 30, 20, 15)
    hits = {q: [0, 0] for q in qs}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(1, n_seeds + 1):
            src = os.path.join(tmp, f"in{seed}.png")
            C.load_source({"kind": "gradnoise", "w": 640, "h": 480, "seed": seed}).convert("RGB").save(src)
            out = os.path.join(tmp, f"out{seed}.png")
            r = wmapp.watermark(src, out, stamp_id=True, ledger=os.path.join(tmp, f"l{seed}.sqlite3"), text="@me",
                                seed=3, invisible=True, invisible_mode="spread")
            im = Image.open(out).convert("RGB")
            for q in qs:
                p = os.path.join(tmp, f"q{q}_{seed}.jpg")
                im.save(p, quality=q)
                corrs = wmcore.spread_spectrum_block_correlations(Image.open(p).convert("RGB"), r.ss_key)
                hits[q][0] += wmcore.find_spread_spectrum_payload(corrs) == r.wm_id
                hits[q][1] += wmcore.find_spread_spectrum_id(corrs) == r.wm_id
    print(f"{'JPEG q':>7s} {'ss':>4s} {'ss_id':>6s}   (of {n_seeds})")
    for q in qs:
        print(f"{q:7d} {hits[q][0]:4d} {hits[q][1]:6d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="640x480,320x240", help="synthetic sizes, e.g. 640x480,320x240 ('' to skip)")
    ap.add_argument("--n", type=int, default=8, help="synthetic images per kind and size")
    ap.add_argument("--images", metavar="DIR", help="real images to add")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--pipeline", type=int, default=0, metavar="N", help="also run N images through the wmapp pipeline")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    for size in [s for s in args.sizes.split(",") if s]:
        w, h = (int(v) for v in size.lower().split("x"))
        specs = [dict(kind=k, w=w, h=h, seed=s) for k in ("pink", "gradnoise") for s in range(args.n)]
        run_images(specs, f"synthetic {size}", args.seed)
    if args.images:
        specs = [dict(kind="file", path=p, max_side=args.max_side) for p in C.list_real_images(args.images)]
        run_images(specs, f"real images in {args.images}", args.seed)
    if args.pipeline:
        run_pipeline(args.pipeline)


if __name__ == "__main__":
    main()
