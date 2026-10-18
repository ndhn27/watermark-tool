"""Real-photo / real-video measurements for the crop work (v28). The photos are whatever real images can be had offline:
skimage's bundled samples (astronaut, coffee, rocket, hubble_deep_field, retina), scipy's `face`, and OpenCV's sample
photos (--data DIR holding building.jpg, board.jpg, messi5.jpg, fruits.jpg, home.jpg from
https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/). The largest is 1411x1411 (2 MP, a retina micrograph);
nothing above ~2 MP was available. No real video above 768x576 either (vtest.avi from the same OpenCV folder).

    python3 bench/experiments/real_corpus_probe.py --exp crops    --data /tmp/realdata
    python3 bench/experiments/real_corpus_probe.py --exp scaled   --data /tmp/realdata
"""
import argparse, io, os, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cli"))
sys.path.insert(0, os.path.dirname(__file__))
import crop_resync_probe as X
from wmcore import resync, spread


def corpus(data):
    import skimage.data as d
    out = {}
    for n in ("astronaut", "coffee", "rocket", "hubble_deep_field", "retina"):
        out[n] = Image.fromarray(getattr(d, n)()).convert("RGB")
    try:
        import scipy.datasets as sd
        out["scipy_face"] = Image.fromarray(sd.face()).convert("RGB")
    except Exception:
        pass
    for f in ("building", "board", "messi5", "fruits", "home"):
        p = os.path.join(data or "", f + ".jpg")
        if os.path.exists(p):
            out[f] = Image.open(p).convert("RGB")
    return out


def exp_crops(args):
    rng = np.random.default_rng(args.seed)
    imgs = corpus(args.data)
    print(f"{len(imgs)} real images: " + ", ".join(f"{k} {v.width}x{v.height}" for k, v in imgs.items()))
    print(f"{'JPEG':>6s} | plain decoder (uncropped control) | any-offset crop: plain | resync  | aligned L/R crop: plain | resync")
    for qual in (None, 85, 60):
        ctl = [0, 0]; anyc = [0, 0, 0]; alg = [0, 0, 0]
        for name, base in imgs.items():
            for _ in range(args.crops):
                key = int(rng.integers(1, 2**31)); m, wm = X.stamp(base, key); w, h = m.size
                def go(img):
                    if qual: img = X.jpeg(img, qual)
                    plain = spread.find_spread_spectrum_id(spread.spread_spectrum_block_correlations(img, key)) == wm
                    new = resync.find_spread_spectrum_id_resync(img, key).text == wm
                    return plain, new
                p, _n = go(m); ctl[0] += p; ctl[1] += 1
                cl, ct = int(rng.integers(1, max(2, w // 8))), int(rng.integers(1, max(2, h // 8)))
                cl += (cl % 8 == 0); ct += (ct % 8 == 0)
                if (w // 8 - (cl // 8)) * 8 % 448 == 0: cl += 8          # skip the exact 448 px-wide corner case
                p, n = go(m.crop((cl, ct, w - int(rng.integers(0, w // 8)), h - int(rng.integers(0, h // 8)))))
                anyc[0] += p; anyc[1] += n; anyc[2] += 1
                al, at = int(rng.integers(1, max(2, w // 8 // 2))) * 8, 0
                p, n = go(m.crop((al, at, w - int(rng.integers(0, w // 8 // 2)) * 8, h)))
                alg[0] += p; alg[1] += n; alg[2] += 1
        q = "-" if qual is None else f"q{qual}"
        print(f"{q:>6s} | {ctl[0]:3d}/{ctl[1]:<3d}                             | {anyc[0]:3d}/{anyc[2]:<3d}             | {anyc[1]:3d}/{anyc[2]:<3d} | {alg[0]:3d}/{alg[2]:<3d}                | {alg[1]:3d}/{alg[2]:<3d}", flush=True)


def exp_scaled(args):
    rng = np.random.default_rng(args.seed)
    imgs = {k: v for k, v in corpus(args.data).items() if min(v.size) >= 480}
    print(f"{len(imgs)} real images >= 480 px: " + ", ".join(f"{k} {v.width}x{v.height}" for k, v in imgs.items()))
    print("crop (off-grid, ~10% each side) + resize by g + JPEG q85; production scaled search with the right key (and, in brackets, 3 other ledger keys):")
    for g in (0.9, 0.75, 1.25):
        hit = hit_ledger = n = 0; tm = []
        for name, base in imgs.items():
            for _ in range(args.crops):
                key = int(rng.integers(1, 2**31)); m, wm = X.stamp(base, key); w, h = m.size
                c = m.crop((int(w * .05) + 3, int(h * .05) + 5, w - int(w * .05) - 7, h - int(h * .05) - 2))
                cp = X.jpeg(c.resize((int(round(c.width * g)), int(round(c.height * g))), Image.BICUBIC if g > 1 else Image.LANCZOS), 85)
                t = time.perf_counter(); r = resync.find_spread_spectrum_id_resync_scaled(cp, key); tm.append(time.perf_counter() - t)
                hit += r.text == wm
                ks = [key] + [int(rng.integers(1, 2**31)) for _ in range(3)]
                r2 = resync.find_spread_spectrum_id_resync_scaled(cp, ks)
                hit_ledger += r2.text == wm; n += 1
        print(f"  g={g:5.3f}: right key alone {hit}/{n} (median {1000*np.median(tm):.0f} ms) | 4-key ledger {hit_ledger}/{n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["crops", "scaled"], required=True)
    ap.add_argument("--data", default=None); ap.add_argument("--crops", type=int, default=3); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    {"crops": exp_crops, "scaled": exp_scaled}[a.exp](a)


if __name__ == "__main__":
    main()
