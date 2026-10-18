#!/usr/bin/env python3
"""How many chances does the resize search give a decoder to hallucinate, and does the ID decoder take any?

The resize search decodes the whole image at several candidate widths per candidate scale, so an image with no
mark still hands the payload decoder many "hypotheses". The bench's false-positive rate counts only searches that
END in an accepted payload (which needs the same text at two widths), so with small n it can say 0/N and still
hide how close a decoder came. This script counts the hypotheses themselves.

For each never-stamped image and each of identity / resize_0.75 / resize_1.25 it runs the real resize search with
a recording decoder that, on every block-correlation vector the search decodes, asks BOTH decoders:

  text decoder (wmcore.find_spread_spectrum_payload)  - what the search actually uses, so the search path is
                                                        exactly `ss_rescale`'s
  ID decoder   (wmcore.find_spread_spectrum_id)        - only observed

and reports: decode attempts, attempts on which the text decoder returned *something* (a junk hypothesis it would
have believed, pending the two-widths-agree rule), attempts on which the ID decoder returned an ID, and searches
the text decoder ended up accepting. On an unmarked image every ID is a false one.

Caveat: the ID decoder is only observed. If it ever fired, an ID-only search would diverge from this path from
that point on; zero IDs means it never diverged.

    python3 bench/experiments/id_rescale_hypotheses.py --sizes 640x480,320x240 --n 6
    python3 bench/experiments/id_rescale_hypotheses.py --images DIR
    python3 bench/experiments/id_rescale_hypotheses.py --sizes "" --structured 640x480     # periodic-pattern stress
"""

import argparse
import os
import sys
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

ATTACKS = ("identity", "resize_0.75", "resize_1.25")


def structured_images(w, h, seed):
    """Never-stamped images built to make the resize search's coarse sweep see structure that is not a mark:
    periodic patterns on and off the 8x8 DCT grid, with a little noise so nothing is perfectly flat.
    Returns [(name, RGB image)]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    hi, lo = 190.0, 66.0

    def img(field):
        arr = np.where(field, hi, lo) + rng.normal(0, 3, (h, w))
        return Image.fromarray(np.clip(np.stack([arr] * 3, -1), 0, 255).astype(np.uint8), "RGB")

    brick_row = yy // 16
    brick_x = (xx + (brick_row % 2) * 16) % 32
    return [
        ("checker8", img(((xx // 8) + (yy // 8)) % 2 == 0)),
        ("checker16", img(((xx // 16) + (yy // 16)) % 2 == 0)),
        ("checker12_offgrid", img(((xx // 12) + (yy // 12)) % 2 == 0)),
        ("hstripes8", img((yy // 8) % 2 == 0)),
        ("vstripes8", img((xx // 8) % 2 == 0)),
        ("stripes3", img((xx // 3) % 2 == 0)),
        ("gridlines16", img((xx % 16 == 0) | (yy % 16 == 0))),
        ("bricks32x16", img((brick_x < 2) | (yy % 16 < 2))),
        ("dots8", img(((xx % 8) < 2) & ((yy % 8) < 2))),
        ("diag_stripes", img(((xx + yy) // 6) % 2 == 0)),
    ]


def run(items, label, seed, budget, id_search=False):
    tot = dict(trials=0, attempts=0, text_returned=0, id_returned=0, accepted=0, budget_hit=0, id_accepted=0)
    for name, base in items:
        uid = f"hyp|{name}"
        key = wmcore.derive_seed(seed, uid)
        for attack_name in ATTACKS:
            rng = np.random.default_rng([seed, zlib.crc32((uid + "|" + attack_name).encode())])
            attacked, _ = A.REGISTRY[attack_name].fn(base, rng)
            seen = dict(attempts=0, text_returned=0, id_returned=0)

            def recording(corrs):
                seen["attempts"] += 1
                text = wmcore.find_spread_spectrum_payload(corrs)
                if text is not None:
                    seen["text_returned"] += 1
                if wmcore.find_spread_spectrum_id(corrs) is not None:
                    seen["id_returned"] += 1
                return text                      # keep the text decoder's search path

            m = wmcore.find_spread_spectrum_payload_rescaled(attacked, key, max_seconds=budget, decoder=recording)
            tot["trials"] += 1
            tot["accepted"] += m.text is not None
            tot["budget_hit"] += bool(m.budget_exhausted)
            for k, v in seen.items():
                tot[k] += v
            id_text = None
            if id_search:                            # the ID-only search on the same image, run for real
                mi = wmcore.find_spread_spectrum_payload_rescaled(attacked, key, max_seconds=budget,
                                                                  decoder=wmcore.find_spread_spectrum_id)
                id_text = mi.text
                tot["id_accepted"] += id_text is not None
            if m.text is not None or seen["id_returned"] or m.budget_exhausted or id_text is not None:
                print(f"   [{name} / {attack_name}] attempts={seen['attempts']} text_returned={seen['text_returned']} "
                      f"accepted_text={m.text!r} ({'x%.3f' % m.factor if m.factor is not None else '-'}) ID-only search={id_text!r} "
                      f"budget_hit={bool(m.budget_exhausted)}", flush=True)
    print(f"\n== {label}: {len(items)} unmarked images x {len(ATTACKS)} attacks")
    print(f"   searches                                      {tot['trials']}")
    print(f"   decode attempts (hypotheses)                  {tot['attempts']}")
    print(f"   ... text decoder returned something           {tot['text_returned']}")
    print(f"   ... ID decoder returned an ID (all false)     {tot['id_returned']}")
    print(f"   searches the text decoder ended up accepting  {tot['accepted']}")
    print(f"   searches that hit the time budget             {tot['budget_hit']}")
    if id_search:
        print(f"   searches the ID-only search accepted          {tot['id_accepted']}")
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="640x480,320x240", help="synthetic sizes ('' to skip)")
    ap.add_argument("--n", type=int, default=6, help="synthetic unmarked images per kind and size")
    ap.add_argument("--images", metavar="DIR", help="real images to add")
    ap.add_argument("--id-search", action="store_true",
                    help="also run the ID-only search on every image (doubles the time) and report what it accepts")
    ap.add_argument("--structured", metavar="WxH", help="also run 10 periodic-pattern images of this size, e.g. 640x480")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--budget", type=float, default=wmcore.DEFAULT_RESCALE_BUDGET_SECONDS)
    ap.add_argument("--seed", type=int, default=20260919 + 1_000_000)
    args = ap.parse_args()
    for size in [s for s in args.sizes.split(",") if s]:
        w, h = (int(v) for v in size.lower().split("x"))
        specs = [dict(kind=k, w=w, h=h, seed=args.seed + s) for k in ("pink", "gradnoise") for s in range(args.n)]
        run([(C.source_id(sp), C.load_source(sp).convert("RGB")) for sp in specs], f"synthetic {size}", args.seed, args.budget)
    if args.images:
        specs = [dict(kind="file", path=p, max_side=args.max_side) for p in C.list_real_images(args.images)]
        run([(C.source_id(sp), C.load_source(sp).convert("RGB")) for sp in specs],
            f"real images in {args.images}", args.seed, args.budget)
    if args.structured:
        w, h = (int(v) for v in args.structured.lower().split("x"))
        run(structured_images(w, h, args.seed), f"structured (periodic) {args.structured}", args.seed, args.budget,
            id_search=args.id_search)


if __name__ == "__main__":
    main()
