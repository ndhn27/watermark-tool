#!/usr/bin/env python3
"""Does putting the ID in the visible grid (--grid-id) hurt the hidden spread-spectrum mark?

The spread mark is embedded AFTER the visible grid is composited, so the grid's glyph edges are part
of the host signal the mark has to survive next to. --grid-id lengthens the grid text, which means
more glyph area. This runs the real pipeline (wmapp.watermark, not bench/'s embed-only path) on four
configurations and reads each back after identity / JPEG / a 0.75x resize, with the same decoder
`extract` uses:

  A  legacy text payload,  grid text "@yourhandle"                       (before --id existed)
  B  --id (ID payload),    grid text "@yourhandle"                       (ID in the hidden mark, not the grid)
  C  --id --grid-id,       grid text "@yourhandle K3M9-QXR2-T4"-shaped   (ID in the grid too)
  D  legacy text payload,  grid text as long as C's                      (separates 'longer text' from 'ID')

    python3 bench/experiments/grid_id_interference.py --n 8 --out bench/results/p1-grid-id-640x480
    python3 bench/experiments/grid_id_interference.py --n 10 --kinds gradnoise --size 1280x960 \\
        --attacks "resize_0.75 (+rescale)" --out bench/results/p1-grid-id-1280x960-gradnoise
"""

import argparse
import io
import json
import os
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", "../../cli"):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, sub)))

from PIL import Image  # noqa: E402

import bench_corpus as C  # noqa: E402
import bench_metrics as M  # noqa: E402
import wmapp  # noqa: E402
from wmcore import wmid  # noqa: E402

HANDLE = "@yourhandle"


def _jpeg(img, q):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="images per kind")
    ap.add_argument("--kinds", default="pink,gradnoise", help="synthetic image kinds (bench_corpus): pink,gradnoise")
    ap.add_argument("--size", default="640x480", help="WxH of the synthetic images")
    ap.add_argument("--attacks", default=None, help="comma-separated subset of: identity, jpeg_q60, jpeg_q40, "
                    "'resize_0.75 (+rescale)' (default: all)")
    ap.add_argument("--rescale-budget", type=float, default=20.0)
    ap.add_argument("--out", default="bench/results/p1-grid-id")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    w, h = (int(v) for v in args.size.lower().split("x"))
    specs = C.make_specs(args.kinds.split(","), [(w, h)], args.n, seed0=4000)
    id_display = wmid.display(wmid.generate(random.Random(1)))
    long_text = f"{HANDLE} {id_display}"
    configs = {
        "A legacy payload, short grid text": dict(text=HANDLE, invisible_text=HANDLE),
        "B --id, short grid text": dict(text=HANDLE, stamp_id=True),
        "C --id --grid-id (long grid text)": dict(text=HANDLE, stamp_id=True, grid_id=True),
        "D legacy payload, long grid text": dict(text=long_text, invisible_text=HANDLE),
    }
    attacks = ["identity", "jpeg_q60", "jpeg_q40", "resize_0.75 (+rescale)"]
    if args.attacks:
        attacks = [a.strip() for a in args.attacks.split(",")]
    records = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = os.path.join(tmp, "l.sqlite3")
        for si, spec in enumerate(specs):
            src = os.path.join(tmp, f"src{si}.png")
            C.load_source(spec).convert("RGB").save(src)
            for cname, kw in configs.items():
                out = os.path.join(tmp, "o.png")
                key = 10_000 + si
                r = wmapp.watermark(src, out, seed=si + 1, invisible=True, invisible_mode="spread", ss_key=key,
                                    ledger=ledger, **kw)
                want = r.wm_id or kw["invisible_text"]
                marked = Image.open(out).convert("RGB")
                for atk in attacks:
                    if atk == "identity":
                        img, rescale = marked, False
                    elif atk.startswith("jpeg"):
                        img, rescale = _jpeg(marked, int(atk.split("_q")[1])), False
                    else:
                        img, rescale = marked.resize((round(w * 0.75), round(h * 0.75)), Image.LANCZOS), True
                    p = os.path.join(tmp, "a.png")
                    img.save(p)
                    res = wmapp.extract(p, mode="spread", ss_key=key, use_ledger=False, rescale=rescale,
                                        rescale_budget_seconds=args.rescale_budget)
                    outcome = ("correct" if res.text == want else "wrong") if res.found else \
                              ("budget" if res.budget_exhausted else "miss")
                    records.append(dict(config=cname, attack=atk, image=C.source_id(spec), outcome=outcome))
            print(f"image {si + 1}/{len(specs)}  {time.time() - t0:.0f}s", file=sys.stderr)

    rows = []
    for cname in configs:
        for atk in attacks:
            sel = [r for r in records if r["config"] == cname and r["attack"] == atk]
            k = sum(r["outcome"] == "correct" for r in sel)
            lo, hi = M.wilson_interval(k, len(sel))
            rows.append(dict(config=cname, attack=atk, n=len(sel), correct=k, rate=k / len(sel), lo=lo, hi=hi,
                             wrong=sum(r["outcome"] == "wrong" for r in sel),
                             budget=sum(r["outcome"] == "budget" for r in sel)))
    with open(os.path.join(args.out, "records.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    lines = [f"# --grid-id vs the hidden spread mark (real pipeline, {args.size} synthetic {args.kinds}, "
             f"n={len(specs)} images per cell)", "",
             "| config | attack | correct / n | rate [95% Wilson] | wrong | budget hit |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['attack']} | {r['correct']}/{r['n']} | {r['rate']:.0%} "
                     f"[{r['lo']:.0%}, {r['hi']:.0%}] | {r['wrong']} | {r['budget']} |")
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
