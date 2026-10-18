"""Large REAL photos (v28b): the six 3888x2592 (10 MP) JPEG photographs of OpenCV's stitching test set
(https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat1.jpg .. boat6.jpg; real camera shots of a
harbour), stamped through the production pipeline (`wmapp.watermark`, spread mark + ID, ledger) and attacked, then read back with
`wmapp.extract` exactly as a user would. Results are appended to --out as they come (each stamp takes ~20-40 s at 10 MP).

    python3 bench/experiments/large_real_probe.py --data /tmp/realdata --boats 1 3 5 --out /tmp/large.log
"""
import argparse, io, math, os, sys, tempfile, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cli"))
import wmapp


def jpeg(img, q):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/realdata"); ap.add_argument("--boats", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--out", default="/tmp/large.log"); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    log = open(a.out, "a", buffering=1)
    def say(s):
        print(s, flush=True); log.write(s + "\n")
    d = tempfile.mkdtemp(); led = os.path.join(d, "l.sqlite3")
    for b in a.boats:
        src = os.path.join(a.data, f"stitching_boat{b}.jpg")
        out = os.path.join(d, f"o{b}.png")
        t = time.perf_counter()
        r = wmapp.watermark(src, out, text="@real", seed=3, stamp_id=True, ledger=led, invisible=True, invisible_mode="spread")
        base = Image.open(out).convert("RGB"); W, H = base.size
        say(f"boat{b}: {W}x{H} ({W*H/1e6:.1f} MP) stamped in {time.perf_counter()-t:.0f}s; ID {r.wm_id}")
        def attack(name, img, rescale=False):
            p = os.path.join(d, "a.png"); img.save(p)
            t0 = time.perf_counter(); res = wmapp.extract(p, ledger=led, rescale=rescale, rescale_budget_seconds=40); dt = time.perf_counter() - t0
            ok = res.wm_id == r.wm_id
            say(f"  boat{b} {name:52s}: ID right {str(ok):5s} cropped={res.regrid is not None!s:5s} rescaled_by={res.rescaled_by and round(res.rescaled_by,3)}  {dt:5.1f}s  copy {img.size[0]}x{img.size[1]}")
        attack("control (no attack)", base)
        c = base.crop((37, 23, W - 91, H - 61)); attack("off-grid crop (-1.6%/side)", c)
        attack("off-grid crop + JPEG q85", jpeg(c, 85))
        attack("off-grid crop + JPEG q60", jpeg(c, 60))
        attack("aligned left/right trim 400px each side + q85", jpeg(base.crop((400, 0, W - 400, H)), 85))
        for frac in (0.25, 0.1, 0.05):
            s = math.sqrt(frac); cw, ch = int(W * s), int(H * s); cx, cy = int(rng.integers(1, W - cw)), int(rng.integers(1, H - ch)); cx += cx % 8 == 0
            attack(f"off-grid crop keeping {int(frac*100)}% area + q85", jpeg(base.crop((cx, cy, cx + cw, cy + ch)), 85))
        for g in (0.75, 0.5, 0.33):
            rs = lambda im: im.resize((int(round(im.width * g)), int(round(im.height * g))), Image.LANCZOS)
            attack(f"resize only x{g} + q85 (--rescale)", jpeg(rs(base), 85), True)
            attack(f"off-grid crop + resize x{g} + q85 (--rescale)", jpeg(rs(c), 85), True)
    say("done")


if __name__ == "__main__":
    main()
