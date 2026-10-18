#!/usr/bin/env python3
"""P0-bis follow-up: why does 0.75x-resize + rescale-search recover `gradnoise`
images at 1280x960 far less often than at 640x480, or than `pink` at the same
size (see bench/results/P1-NOTES.md section 2)?

This isolates the visible grid layer as the cause. wmapp.watermark() always
composites the tiled grid text BEFORE embedding the spread-spectrum mark, and
_build_grid_layer() (wmcore/visible.py) uses FIXED-PIXEL spacing/font_size
(not scaled to image dimensions) and crops the tiled pattern from a canvas
centered on the image - so the exact grid phase that lands at the image's
top-left corner is size-dependent. The rescale search's cheap coarse-sweep
statistic (_ss_patch_bimodality) is anchored ONLY at that top-left corner
(for cost reasons - see rescale.py's module docstring), so a size/content
combination that puts unfavorable grid texture there defeats the search
even though the underlying mark is intact (confirmed below: the plain,
unresized decode is unaffected).

Runs three ablations, `--n` seeds each:
  1. identity (no resize): does the grid break the mark at all?
  2. resize 0.75x + rescale search, grid ON  (matches the real pipeline)
  3. resize 0.75x + rescale search, grid OFF (opacity=0 - same layer, invisible)

Usage:
    python3 bench/experiments/grid_corner_anchor.py --n 12 --out bench/results/p0bis-grid-corner-anchor
    python3 bench/experiments/grid_corner_anchor.py --n 12 --kind pink --size 1280x960 --out ...
"""
import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", "../../cli"):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, sub)))

from PIL import Image  # noqa: E402

import bench_corpus as C  # noqa: E402
import wmapp  # noqa: E402

HANDLE = "@yourhandle"


def run(kind, w, h, seeds, tmp, rescale_budget):
    gen = {"gradnoise": C._gradnoise, "pink": C._pink}[kind]
    rows = []
    for seed in seeds:
        base = gen(w, h, seed)
        src = os.path.join(tmp, "src.png")
        base.convert("RGB").save(src)
        key = 77_000 + seed

        def embed(grid_on):
            out = os.path.join(tmp, "out.png")
            kw = dict(text=HANDLE, invisible=True, invisible_mode="spread", ss_key=key,
                     invisible_text=HANDLE, seed=seed + 1)
            if not grid_on:
                kw["opacity"] = 0  # same grid layer, alpha 0: isolates its effect, changes nothing else
            wmapp.watermark(src, out, **kw)
            return Image.open(out).convert("RGB")

        row = {"seed": seed}
        # 1) identity, grid ON
        marked = embed(True)
        idp = os.path.join(tmp, "id.png")
        marked.save(idp)
        res = wmapp.extract(idp, mode="spread", ss_key=key, use_ledger=False, rescale=False)
        row["identity_grid_on"] = bool(res.found and res.text == HANDLE)

        # 2) resize 0.75x + rescale, grid ON
        resized = marked.resize((round(w * 0.75), round(h * 0.75)), Image.LANCZOS)
        rp = os.path.join(tmp, "r.png")
        resized.save(rp)
        res = wmapp.extract(rp, mode="spread", ss_key=key, use_ledger=False, rescale=True,
                            rescale_budget_seconds=rescale_budget)
        row["resize_rescale_grid_on"] = bool(res.found and res.text == HANDLE)

        # 3) resize 0.75x + rescale, grid OFF
        marked_off = embed(False)
        resized_off = marked_off.resize((round(w * 0.75), round(h * 0.75)), Image.LANCZOS)
        rp2 = os.path.join(tmp, "r2.png")
        resized_off.save(rp2)
        res = wmapp.extract(rp2, mode="spread", ss_key=key, use_ledger=False, rescale=True,
                            rescale_budget_seconds=rescale_budget)
        row["resize_rescale_grid_off"] = bool(res.found and res.text == HANDLE)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--kind", default="gradnoise", choices=["gradnoise", "pink"])
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--rescale-budget", type=float, default=10.0)
    ap.add_argument("--out", default="bench/results/p0bis-grid-corner-anchor")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(5000, 5000 + args.n))

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        rows = run(args.kind, w, h, seeds, tmp, args.rescale_budget)

    n = len(rows)

    def rate(key):
        k = sum(r[key] for r in rows)
        return k, n

    summary = {
        "kind": args.kind, "size": [w, h], "n": n, "rescale_budget": args.rescale_budget,
        "identity_grid_on": rate("identity_grid_on"),
        "resize_rescale_grid_on": rate("resize_rescale_grid_on"),
        "resize_rescale_grid_off": rate("resize_rescale_grid_off"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, f"{args.kind}_{w}x{h}.json"), "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=1)

    print(f"{args.kind} {w}x{h}  n={n}  ({summary['elapsed_s']}s)")
    print(f"  identity (no resize), grid ON        : {summary['identity_grid_on'][0]}/{n}")
    print(f"  resize 0.75x + rescale, grid ON       : {summary['resize_rescale_grid_on'][0]}/{n}")
    print(f"  resize 0.75x + rescale, grid OFF      : {summary['resize_rescale_grid_off'][0]}/{n}")


if __name__ == "__main__":
    main()
