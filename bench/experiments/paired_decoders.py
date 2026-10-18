#!/usr/bin/env python3
"""Paired comparison of decoders from a bench run's records.jsonl.

The bench report scores every decoder on its own. When two decoders were run in the SAME invocation they saw
the very same attacked image (same embed, same attack noise), so the right comparison is paired: on how many
images did one read the payload and the other not? This prints, per attack, the paired counts against a
reference decoder, wrong strings (misdecodes), and the unmarked-image / wrong-key false-positive counts.

    python3 bench/experiments/paired_decoders.py bench/results/p2-ss-id-rescale-synthetic ss_rescale ss_id_rescale ss_or_id_rescale
    python3 bench/experiments/paired_decoders.py DIR REFERENCE OTHER [OTHER...] [--all]     # --all: also rows with no difference
"""

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import bench_metrics as M  # noqa: E402


def load(d):
    with open(os.path.join(d, "records.jsonl")) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("reference")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--all", action="store_true", help="also print attacks where nothing differs")
    args = ap.parse_args()
    rows = load(args.dir)
    decs = [args.reference] + args.others

    pos = [r for r in rows if r["kind"] == "pos" and r["decoder"] in decs]
    by = collections.defaultdict(dict)                      # (image, attack) -> decoder -> record
    for r in pos:
        by[(r["image_id"], r["attack"])][r["decoder"]] = r
    attacks = sorted({a for _, a in by})
    ok = lambda rec: rec is not None and rec["outcome"] == "correct"
    wrong = lambda rec: rec is not None and rec["outcome"] == "wrong"

    print(f"== paired, positives: '{args.reference}' vs each other decoder, on the same attacked image")
    for other in args.others:
        print(f"\n-- {other} vs {args.reference}")
        print(f"{'attack':22s} {'gate':>4s} {'n':>3s} {'ref ok':>6s} {'oth ok':>6s} {'oth only':>8s} {'ref only':>8s}  wrong(ref/oth)")
        tot = collections.Counter()
        for a in attacks:
            cells = [v for (i, at), v in by.items() if at == a and args.reference in v and other in v]
            if not cells:
                continue
            n = len(cells)
            r_ok = sum(ok(v[args.reference]) for v in cells)
            o_ok = sum(ok(v[other]) for v in cells)
            o_only = sum(ok(v[other]) and not ok(v[args.reference]) for v in cells)
            r_only = sum(ok(v[args.reference]) and not ok(v[other]) for v in cells)
            w_r = sum(wrong(v[args.reference]) for v in cells)
            w_o = sum(wrong(v[other]) for v in cells)
            gate = cells[0][args.reference]["gate"]
            tot.update(n=n, r_ok=r_ok, o_ok=o_ok, o_only=o_only, r_only=r_only, w_r=w_r, w_o=w_o)
            if args.all or o_only or r_only or w_r or w_o or r_ok != o_ok:
                print(f"{a:22s} {'yes' if gate else 'no':>4s} {n:3d} {r_ok:6d} {o_ok:6d} {o_only:8d} {r_only:8d}  {w_r}/{w_o}")
        print(f"{'TOTAL':22s} {'':>4s} {tot['n']:3d} {tot['r_ok']:6d} {tot['o_ok']:6d} {tot['o_only']:8d} {tot['r_only']:8d}  {tot['w_r']}/{tot['w_o']}")

    print("\n== by size x kind (all attacks pooled): images read / trials")
    cell = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in pos:
        c = cell[(tuple(r["size"]), r["image_kind"])][r["decoder"]]
        c[0] += 1
        c[1] += r["outcome"] == "correct"
    for key, per in sorted(cell.items()):
        print(f"  {key[0][0]}x{key[0][1]} {key[1]:9s} " + "  ".join(f"{d}: {per[d][1]}/{per[d][0]}" for d in decs if d in per))

    for kind, label in (("neg_unmarked", "unmarked images (no mark exists: any returned text is false)"),
                        ("neg_wrongkey", "wrong key (exact payload = leak; other text = wrong)")):
        neg = [r for r in rows if r["kind"] == kind and r["decoder"] in decs]
        if not neg:
            continue
        print(f"\n== {label}")
        for d in decs:
            rs = [r for r in neg if r["decoder"] == d]
            if not rs:
                continue
            n = len(rs)
            leak = sum(r["outcome"] == "correct" for r in rs)
            junk = sum(r["outcome"] == "wrong" for r in rs)
            lo, hi = M.clopper_pearson(junk if kind == "neg_unmarked" else leak + junk, n)
            budget = sum(bool(r.get("budget_exhausted")) for r in rs)
            extra = f", exact payload {leak}" if kind == "neg_wrongkey" else ""
            print(f"  {d:18s} n={n:3d}  text returned {junk if kind == 'neg_unmarked' else leak + junk:3d}{extra}"
                  f"  95% upper {hi:.1%}  budget-hit {budget}")


if __name__ == "__main__":
    main()
