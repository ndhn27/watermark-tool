#!/usr/bin/env python3
"""Does the ID decoder leak more through the resize search when the key is wrong?

Plain decoding already shows it (P2-NOTES §2.1: a wrong key still reads the true payload for 16.7% of
synthetic trials with the text decoder, 29.2% with the ID decoder). What the bench's wrong-key trials never
covered is the resize search: they use unresized images (identity / JPEG q75), where the search's first step is
just the plain decode. This script wrong-keys an image that has been RESIZED - the case where the search's sweep
and its per-width decodes actually run.

For each marked image and attack it decodes with the true key (control: is the mark readable at all?) and with
K random wrong keys, using the text-decoder search (`ss_rescale`) and the ID search (`ss_id_rescale`) in turn -
separate runs, so each decoder's own search path is what is measured.

    python3 bench/experiments/id_rescale_wrongkey.py --size 320x240 --n 5 --keys 5
    python3 bench/experiments/id_rescale_wrongkey.py --size 640x480 --n 2 --keys 4 --attacks resize_0.75
"""

import argparse
import os
import random
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", os.path.join("..", "..", "cli")):
    p = os.path.abspath(os.path.join(HERE, sub))
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import bench_attacks as A  # noqa: E402
import bench_corpus as C  # noqa: E402
import bench_metrics as M  # noqa: E402
import bench_schemes as S  # noqa: E402
import wmcore  # noqa: E402
from wmcore import wmid  # noqa: E402

DECODERS = ("ss_rescale", "ss_id_rescale")


def run(size, n, n_keys, attacks, seed, budget):
    w, h = size
    tally = {d: dict(control=0, control_n=0, leak=0, wrong=0, none=0, budget=0, n=0) for d in DECODERS}
    for kind in ("pink", "gradnoise"):
        for i in range(n):
            spec = dict(kind=kind, w=w, h=h, seed=seed + i)
            base = C.load_source(spec)
            uid = f"wkr|{C.source_id(spec)}"
            payload = wmid.generate(random.Random(f"{seed}|{uid}"))
            key = wmcore.derive_seed(seed, uid)
            stamped = wmcore.embed_spread_spectrum(base, payload, key).convert("RGB")
            rng = np.random.default_rng([seed, zlib.crc32(uid.encode())])
            wrong = []
            while len(wrong) < n_keys:
                k = int(rng.integers(1, 2 ** 31))
                if k != key:
                    wrong.append(k)
            for name in attacks:
                attacked, _ = A.REGISTRY[name].fn(stamped, np.random.default_rng([seed, zlib.crc32((uid + name).encode())]))
                for dec in DECODERS:
                    t = tally[dec]
                    out = S.DECODERS[dec](attacked, key, budget)
                    t["control_n"] += 1
                    t["control"] += out.text == payload
                    for k in wrong:
                        out = S.DECODERS[dec](attacked, k, budget)
                        t["n"] += 1
                        t["budget"] += bool(out.budget_exhausted)
                        if out.text is None:
                            t["none"] += 1
                        elif out.text == payload:
                            t["leak"] += 1
                        else:
                            t["wrong"] += 1
            print(f"  ... {kind} image {i + 1}/{n} done at {size[0]}x{size[1]}", flush=True)
    print(f"\n== wrong keys on RESIZED marked images, {w}x{h}, attacks {list(attacks)}, {n_keys} wrong keys per image")
    for dec in DECODERS:
        t = tally[dec]
        lo, hi = M.wilson_interval(t["leak"], t["n"])
        print(f"   {dec:14s} true-key control {t['control']}/{t['control_n']}   wrong-key trials {t['n']}: "
              f"true payload read {t['leak']} ({t['leak'] / max(1, t['n']):.1%}, 95% CI {lo:.1%}-{hi:.1%}), "
              f"other text {t['wrong']}, nothing {t['none']}, budget hit {t['budget']}")
    return tally


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--n", type=int, default=5, help="images per kind")
    ap.add_argument("--keys", type=int, default=5, help="wrong keys per image and attack")
    ap.add_argument("--attacks", default="resize_0.75,resize_1.25")
    ap.add_argument("--budget", type=float, default=wmcore.DEFAULT_RESCALE_BUDGET_SECONDS)
    ap.add_argument("--seed", type=int, default=20260919 + 3_000_000)
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    run((w, h), args.n, args.keys, tuple(a for a in args.attacks.split(",") if a), args.seed, args.budget)


if __name__ == "__main__":
    main()
