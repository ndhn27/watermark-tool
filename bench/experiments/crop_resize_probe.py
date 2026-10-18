"""Crop + resize probe: can the offset-agnostic grid statistic (resync.shift_scores ratio) also find the SCALE?

A screenshot at another zoom, then cropped: the copy is the original cropped at some offset and resized by g.
Restoring f = 1/g puts the 8x8 grid back (at an unknown offset), where resync's search then works. The sweep below
asks: at which restore factor f does max(S)/median(S) of a central patch stand out, and is it 1/g?

    python3 bench/experiments/crop_resize_probe.py
"""
import io, os, sys, time
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cli"))
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import crop_resync_probe as X
from wmcore import resync


def sweep_ratio(y, key, f_grid, dst=256):
    """r(f) = max/median of the 64 grid-offset energies of the correlation map of a central patch resized by f."""
    h, w = y.shape
    out = np.zeros(len(f_grid))
    for i, f in enumerate(f_grid):
        sp = int(round(dst / f))
        if sp > min(h, w):
            out[i] = np.nan
            continue
        t, l = (h - sp) // 2, (w - sp) // 2
        patch = y[t:t + sp, l:l + sp]
        p = cv2.resize(patch, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        sc = resync.shift_scores(resync.correlation_map(p, key))
        out[i] = sc.max() / max(np.median(sc), 1e-9)
    return out


def main():
    rng = np.random.default_rng(3)
    f_grid = np.exp(np.arange(np.log(0.5), np.log(2.0), 0.002))
    base = X._pink(1280, 960, 11)
    print(f"{len(f_grid)} candidate factors in [0.5, 2.0], log step 0.2%")
    print(f"{'g (copy = orig x g)':20s} {'JPEG':>5s} | true f=1/g | peak f   | peak r | median r | z=(peak-med)/MADsigma | peak within 0.3% of truth")
    for g in (0.9, 0.8, 0.75, 0.667, 1.25, 1.5, 2.0):
        for q in (None, 85):
            key = int(rng.integers(1, 2**31))
            m, wm = X.stamp(base, key)
            crop = m.crop((37, 23, 1230, 900))
            cp = crop.resize((int(round(crop.width * g)), int(round(crop.height * g))), Image.BICUBIC if g > 1 else Image.LANCZOS)
            if q:
                cp = X.jpeg(cp, q)
            y = resync.luma(cp)
            t = time.perf_counter(); r = sweep_ratio(y, key, f_grid); dt = time.perf_counter() - t
            ok = ~np.isnan(r); med = np.nanmedian(r); mad = 1.4826 * np.nanmedian(np.abs(r[ok] - med))
            i = int(np.nanargmax(r)); z = (r[i] - med) / max(mad, 1e-9)
            truth = 1 / g
            print(f"{g:<20.3f} {('-' if q is None else 'q'+str(q)):>5s} | {truth:10.4f} | {f_grid[i]:8.4f} | {r[i]:6.2f} | {med:8.2f} | {z:8.1f} | {'YES' if abs(f_grid[i]/truth-1) < 0.003 else 'no'}   ({dt:.1f}s)", flush=True)
    print("unmarked / wrong-key controls (peak r and z over the same sweep):")
    for name, img in (("pink unmarked", X._pink(1280, 960, 99)), ("real astronaut unmarked", X.real_images()["astronaut"])):
        key = int(rng.integers(1, 2**31)); cp = img.crop((37, 23, min(img.width, 1230), min(img.height, 900)))
        cp = cp.resize((int(cp.width * 0.8), int(cp.height * 0.8)), Image.LANCZOS)
        r = sweep_ratio(resync.luma(cp), key, f_grid); ok = ~np.isnan(r); med = np.nanmedian(r); mad = 1.4826 * np.nanmedian(np.abs(r[ok] - med))
        print(f"  {name:24s}: peak r {np.nanmax(r):.2f}  median {med:.2f}  z {(np.nanmax(r)-med)/mad:.1f}")


if __name__ == "__main__":
    main()
