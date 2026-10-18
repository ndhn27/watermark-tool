"""Crop resync probe (measure-before-write for the 'screenshot + crop' hole).

The mark is tiled in RASTER order over the ORIGINAL image's 8x8 block grid, period 112 bits for an ID. A crop
that keeps the block grid and the width (top / bottom) only shifts the phase of that raster, which the ID
decoder already searches. A crop that changes the width (left / right) or starts off the 8-pixel grid does not.

Idea tested here: block (r, c) of the cropped image is block (r + R0, c + C0) of the original, so its bit
index is ((r + R0) * bw0 + c + C0) mod 112 = (r * s + c + phi) mod 112 with s = bw0 mod 112 and
phi = (R0 * bw0 + C0) mod 112. Only (s, phi) matter: 112 x 112 hypotheses, each cheap (one bincount per s, then
112 circular shifts of a 112-vector). A pixel-unaligned crop additionally needs the block grid re-cut at the
right (dy, dx) in 0..7 x 0..7 before any of that.

    python3 bench/experiments/crop_resync_probe.py --exp aligned
    python3 bench/experiments/crop_resync_probe.py --exp unaligned
"""

import argparse
import io
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cli"))
import wmcore                                              # noqa: E402
from wmcore import spread, wmid                            # noqa: E402

P = spread.ID_PERIOD_BITS          # 112
_ALPHA_BYTES = np.frombuffer(wmid.ALPHABET.encode(), dtype=np.uint8)
_OK = np.zeros(256, dtype=bool)
_OK[_ALPHA_BYTES] = True


# ---------------------------------------------------------------- prototype pieces (moved to wmcore if they pay off)

def grid_corrs(rgb, key, dy=0, dx=0):
    """2-D per-block correlations (bh, bw) for the block grid that starts at pixel (dy, dx)."""
    arr = np.asarray(rgb.convert("RGB") if hasattr(rgb, "convert") else rgb, dtype=np.float64)
    y, _, _ = spread._rgb_to_y(arr)
    y = y[dy:, dx:]
    blocks, bh, bw = spread._image_to_ss_blocks(y)
    if bh == 0 or bw == 0:
        return np.zeros((0, 0))
    dct = np.einsum("ij,abjk,kl->abil", spread._SS_C, blocks, spread._SS_CT)
    pn = spread._ss_pn_vector(key, len(spread.SS_COEFF_POSITIONS))
    return (dct[:, :, spread._SS_ROWS, spread._SS_COLS] * pn).sum(axis=-1)


def find_id_grid(corr2d):
    """Search every (s, phi). Returns (id, s, phi, confidence) of the best canonical ID, or None."""
    bh, bw = corr2d.shape
    if bh * bw < P:
        return None
    flat = corr2d.ravel()
    r = (np.arange(bh) % P)[:, None]
    c = (np.arange(bw) % P)[None, :]
    k = np.arange(32, P)                                    # payload positions after the 32-bit header
    shifts = (k[None, :] - np.arange(P)[:, None]) % P       # shifts[phi, j] = index of b read for phase phi
    best = None
    for s in range(P):
        idx = ((r * s + c) % P).ravel()
        b = np.bincount(idx, weights=flat, minlength=P)     # summed correlation per payload bit position
        pay = b[shifts]                                     # (112 phases, 80 payload bits)
        bits = (pay > 0).astype(np.uint8)
        by = np.packbits(bits, axis=1)                      # (112, 10) bytes
        ok = _OK[by].all(axis=1)
        for phi in np.flatnonzero(ok):
            text = by[phi].tobytes().decode("ascii")
            if wmid.parse(text) != text:
                continue
            conf = float(np.abs(pay[phi]).sum())
            if best is None or conf > best[3]:
                best = (text, s, int(phi), conf)
    return best


# ---------------------------------------------------------------- corpora and attacks

def synthetic(w, h, seed):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    base = np.stack([xx, yy, (xx + yy) / 2], -1)
    tex = np.kron(rng.normal(0, 14, (-(-h // 8), -(-w // 8), 1)), np.ones((8, 8, 1)))[:h, :w]
    return Image.fromarray(np.clip(base + tex + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8))


def real_images():
    import skimage.data as d
    out = {}
    for n in ("astronaut", "coffee", "rocket", "hubble_deep_field", "chelsea"):
        a = getattr(d, n)()
        out[n] = Image.fromarray(a).convert("RGB")
    return out


def jpeg(img, q):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def stamp(img, key, alpha=spread.DEFAULT_SS_ALPHA, wm=None):
    wm = wm or wmid.generate()
    return wmcore.embed_spread_spectrum(img.convert("RGBA"), wm, key, alpha=alpha).convert("RGB"), wm


# ---------------------------------------------------------------- experiments

def exp_aligned(args):
    """Block-aligned crops that change the width: current flat decoder vs (s, phi) search."""
    rng = np.random.default_rng(args.seed)
    rows = []
    corpora = {"synthetic1280x960": [synthetic(1280, 960, s) for s in range(args.n)]} if args.n else {}
    if args.real:
        corpora.update({k: [v] for k, v in real_images().items()})
    for name, imgs in corpora.items():
        for qual in (None, 85, 60):
            flat_hit = grid_hit = trials = 0
            t_flat = t_grid = 0.0
            for i, base in enumerate(imgs):
                for rep in range(args.crops):
                    key = int(rng.integers(1, 2**31))
                    marked, wm = stamp(base, key)
                    w, h = marked.size
                    l, r = (int(x) * 8 for x in rng.integers(0, max(2, w // 8 // 5), 2))
                    t, b = (int(x) * 8 for x in rng.integers(0, max(2, h // 8 // 5), 2))
                    img = marked.crop((l, t, w - r, h - b))
                    if qual:
                        img = jpeg(img, qual)
                    t0 = time.perf_counter()
                    flat = spread.find_spread_spectrum_id(spread.spread_spectrum_block_correlations(img, key))
                    t_flat += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    g = find_id_grid(grid_corrs(img, key))
                    t_grid += time.perf_counter() - t0
                    flat_hit += flat == wm
                    grid_hit += bool(g and g[0] == wm)
                    trials += 1
            q = "no JPEG" if qual is None else f"JPEG q{qual}"
            print(f"{name:20s} {q:8s} aligned crop (L/R/T/B 0-20%): flat {flat_hit:3d}/{trials}  grid {grid_hit:3d}/{trials}"
                  f"   avg {1000*t_flat/trials:6.0f} ms flat, {1000*t_grid/trials:6.0f} ms grid", flush=True)


def shift_scores(rgb, key, patch=256):
    """Cheap sub-block alignment detector: mean squared block correlation over a central patch, for each of
    the 64 (dy, dx) grid shifts. The mark adds +-alpha*|pn|^2 to every block's correlation ONLY when the block
    grid is cut where the embedder cut it, so the true shift scores highest."""
    arr = np.asarray(rgb.convert("RGB") if hasattr(rgb, "convert") else rgb, dtype=np.float64)
    y, _, _ = spread._rgb_to_y(arr)
    h, w = y.shape
    ph, pw = min(patch, (h - 8) // 8 * 8), min(patch, (w - 8) // 8 * 8)
    top, left = ((h - ph - 8) // 2) // 8 * 8, ((w - pw - 8) // 2) // 8 * 8
    sub = y[top:top + ph + 8, left:left + pw + 8]
    pn = spread._ss_pn_vector(key, len(spread.SS_COEFF_POSITIONS))
    out = np.zeros((8, 8))
    for dy in range(8):
        for dx in range(8):
            blocks, _, _ = spread._image_to_ss_blocks(sub[dy:dy + ph, dx:dx + pw])
            dct = np.einsum("ij,abjk,kl->abil", spread._SS_C, blocks, spread._SS_CT)
            c = (dct[:, :, spread._SS_ROWS, spread._SS_COLS] * pn).sum(axis=-1)
            out[dy, dx] = float(np.mean(c * c))
    return out


def exp_unaligned(args):
    rng = np.random.default_rng(args.seed)
    corpora = {"synthetic1280x960": [synthetic(1280, 960, s) for s in range(args.n)]} if args.n else {}
    if args.real:
        corpora.update({k: [v] for k, v in real_images().items() if k != "chelsea"})   # chelsea: 56 blocks wide, see notes
    print(f"{'corpus':20s} {'JPEG':8s} {'trials':>6s} | flat  oracle-shift | detector rank of true shift: top1 top2 top3 top5 | "
          f"time ms: detector, 1 full decode")
    for name, imgs in corpora.items():
        for qual in (None, 85):
            n = flat = oracle = 0
            ranks = []
            t_det = t_dec = 0.0
            for base in imgs:
                for _ in range(args.crops):
                    key = int(rng.integers(1, 2**31))
                    marked, wm = stamp(base, key)
                    w, h = marked.size
                    cx, cy = int(rng.integers(1, w // 5)), int(rng.integers(1, h // 5))
                    rx, by = int(rng.integers(0, w // 5)), int(rng.integers(0, h // 5))
                    if cx % 8 == 0 and cy % 8 == 0:
                        cx += 3
                    img = marked.crop((cx, cy, w - rx, h - by))
                    if qual:
                        img = jpeg(img, qual)
                    true = ((-cy) % 8, (-cx) % 8)
                    flat += spread.find_spread_spectrum_id(spread.spread_spectrum_block_correlations(img, key)) == wm
                    t0 = time.perf_counter()
                    g = find_id_grid(grid_corrs(img, key, *true))
                    t_dec += time.perf_counter() - t0
                    oracle += bool(g and g[0] == wm)
                    t0 = time.perf_counter()
                    S = shift_scores(img, key)
                    t_det += time.perf_counter() - t0
                    order = np.argsort(-S.ravel())
                    ranks.append(int(np.flatnonzero(order == true[0] * 8 + true[1])[0]))
                    n += 1
            r = np.array(ranks)
            top = [int(((r < k)).sum()) for k in (1, 2, 3, 5)]
            q = "no JPEG" if qual is None else f"q{qual}"
            print(f"{name:20s} {q:8s} {n:6d} | {flat:3d}   {oracle:3d}/{n} | {top[0]:4d} {top[1]:4d} {top[2]:4d} {top[3]:4d} (of {n})"
                  f" | {1000*t_det/n:6.0f}  {1000*t_dec/n:6.0f}", flush=True)


def _pink(w, h, seed):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import bench_corpus
    return bench_corpus.load_source({"kind": "pink", "w": w, "h": h, "seed": seed}).convert("RGB")


def _unmarked_zoo(rng):
    """Mark-free images, including structured ones that are the dangerous case for a hypothesis search."""
    out = {}
    out["pink640x480"] = [_pink(640, 480, 100 + i) for i in range(args_n_null)]
    out["pink1280x960"] = [_pink(1280, 960, 200 + i) for i in range(max(2, args_n_null // 3))]
    import bench_corpus
    out["gradnoise640x480"] = [bench_corpus.load_source({"kind": "gradnoise", "w": 640, "h": 480, "seed": 400 + i}).convert("RGB")
                               for i in range(args_n_null)]
    flat = np.full((480, 640, 3), 120, np.uint8)
    stripes = np.tile((np.arange(640) // 8 % 2 * 80 + 60).astype(np.uint8)[None, :, None], (480, 1, 3))
    check = np.tile((((np.arange(480)[:, None] // 8 + np.arange(640)[None, :] // 8) % 2) * 80 + 60).astype(np.uint8)[:, :, None], (1, 1, 3))
    blocky = np.kron(rng.integers(0, 256, (60, 80, 1)), np.ones((8, 8, 1))).repeat(3, axis=2).astype(np.uint8)
    out["structured (flat, stripes, checker, random-block)"] = [Image.fromarray(a) for a in (flat, stripes, check, blocky)]
    for k, v in real_images().items():
        out[f"real:{k}"] = [v]
    return out


args_n_null = 20


def exp_null(args):
    global args_n_null
    args_n_null = args.n_null
    rng = np.random.default_rng(args.seed)
    # --- part 1: how well does the cheap detector separate marked from unmarked? ---------------------------------
    print("PART 1 - detector statistic r = max(S)/median(S) over the 64 shifts (S = mean squared block correlation, 256 px patch)")
    for qual in (None, 85, 60):
        marked_r, true_rank1, n = [], 0, 0
        for i in range(args.n_null):
            base = _pink(640, 480, 300 + i)
            key = int(rng.integers(1, 2**31)); m, wm = stamp(base, key)
            cx, cy = int(rng.integers(1, 100)), int(rng.integers(1, 80))
            if cx % 8 == 0 and cy % 8 == 0:
                cx += 3
            img = m.crop((cx, cy, 640 - int(rng.integers(0, 100)), 480 - int(rng.integers(0, 80))))
            if qual:
                img = jpeg(img, qual)
            S = shift_scores(img, key)
            true = ((-cy) % 8, (-cx) % 8)
            marked_r.append(S.max() / np.median(S)); true_rank1 += int(np.argmax(S) == true[0] * 8 + true[1]); n += 1
        q = "no JPEG" if qual is None else f"JPEG q{qual}"
        a = np.array(marked_r)
        print(f"  marked, pink 640x480 unaligned crop, {q:8s}: true shift is the top-scoring one {true_rank1}/{n};  r min {a.min():.2f}  p10 {np.percentile(a,10):.2f}  median {np.median(a):.2f}")
    un = []
    for name, imgs in _unmarked_zoo(rng).items():
        for img in imgs:
            key = int(rng.integers(1, 2**31))
            S = shift_scores(img, key)
            un.append((name, S.max() / np.median(S) if np.median(S) > 0 else float("inf")))
    ratios = np.array([r for _, r in un if np.isfinite(r)])
    print(f"  UNMARKED ({len(un)} images, all kinds): r min {ratios.min():.2f}  median {np.median(ratios):.2f}  p95 {np.percentile(ratios,95):.2f}  max {ratios.max():.2f}"
          f"   (non-finite: {sum(1 for _, r in un if not np.isfinite(r))})")
    worst = sorted(un, key=lambda t: -t[1])[:3]
    print("  highest unmarked r:", ", ".join(f"{n} {r:.2f}" for n, r in worst))

    # --- part 2: false accepts of the whole search on mark-free images ------------------------------------------
    print("PART 2 - false accepts: full search (top-3 shifts x 112 strides x 112 phases) on mark-free images, random key")
    tot = bad = 0
    t_all = 0.0
    for name, imgs in _unmarked_zoo(rng).items():
        got = 0
        for img in imgs:
            key = int(rng.integers(1, 2**31))
            t0 = time.perf_counter()
            S = shift_scores(img, key)
            for flat_i in np.argsort(-S.ravel())[:3]:
                dy, dx = divmod(int(flat_i), 8)
                g = find_id_grid(grid_corrs(img, key, dy, dx))
                got += g is not None
            t_all += time.perf_counter() - t0
            tot += 1
        bad += got
        print(f"  {name:48s} {len(imgs):3d} images  IDs returned: {got}")
    print(f"  TOTAL: {tot} mark-free images x 3 shifts x 12544 hypotheses -> {bad} false IDs   (theory: 2^-40 per hypothesis = {tot*3*12544/2**40:.1e} expected)")


def exp_pipeline(args):
    """End to end through the PRODUCTION module (wmcore.resync): random crops of every kind, JPEG after, the
    detector choosing the grid offset (no oracle), unmirrored and mirrored copies; then the same search on
    mark-free images. This is the table that goes in bench/results/CROP-NOTES.md."""
    import math
    from wmcore import resync
    rng = np.random.default_rng(args.seed)
    corpora = {"pink640x480 (synthetic)": [_pink(640, 480, 700 + i) for i in range(args.n)],
               "pink1280x960 (synthetic)": [_pink(1280, 960, 800 + i) for i in range(max(2, args.n // 3))]}
    real = real_images()
    for k in ("astronaut", "coffee", "rocket", "hubble_deep_field"):
        corpora[f"real:{k}"] = [real[k]]
    kinds = [("top/bottom only, aligned", lambda w, h: (0, int(rng.integers(0, h // 6)) // 8 * 8, 0, int(rng.integers(0, h // 6)))),
             ("left/right, aligned (x8)", lambda w, h: tuple(int(v) // 8 * 8 for v in (rng.integers(1, w // 6), rng.integers(0, h // 6), rng.integers(0, w // 6), rng.integers(0, h // 6)))),
             ("any offset (screenshot-like)", lambda w, h: (int(rng.integers(1, w // 6)), int(rng.integers(1, h // 6)), int(rng.integers(0, w // 6)), int(rng.integers(0, h // 6))))]
    print(f"{'corpus':26s} {'crop kind':28s} {'JPEG':>6s} | old decoder | resync  | ms per decode (resync)")
    for name, imgs in corpora.items():
        for kname, kfn in kinds:
            for qual in (None, 85, 60):
                old = new = n = 0
                tm = 0.0
                for base in imgs:
                    for _ in range(args.crops):
                        key = int(rng.integers(1, 2**31))
                        m, wm = stamp(base, key)
                        w, h = m.size
                        cl, ct, cr, cb = kfn(w, h)
                        if (name.startswith("real") or "1280" in name) is False and (w - cl - cr) // 8 * 8 == 448:
                            cr += 8                             # avoid the 56-block-wide corner case here; it has its own table
                        img = m.crop((cl, ct, w - cr, h - cb))
                        if qual:
                            img = jpeg(img, qual)
                        old += spread.find_spread_spectrum_id(spread.spread_spectrum_block_correlations(img, key)) == wm
                        t0 = time.perf_counter()
                        r = resync.find_spread_spectrum_id_resync(img, key)
                        tm += time.perf_counter() - t0
                        new += r.text == wm
                        n += 1
                q = "-" if qual is None else f"q{qual}"
                print(f"{name:26s} {kname:28s} {q:>6s} | {old:4d}/{n:<3d}    | {new:3d}/{n:<3d} | {1000*tm/n:6.0f}", flush=True)
    # mirrored + cropped
    print("mirrored copy, then cropped at any offset (orientation search + re-alignment):")
    for name in ("pink640x480 (synthetic)", "real:astronaut"):
        for mode in ("h", "v", "hv"):
            hit = n = 0
            for base in corpora[name][:args.n]:
                for _ in range(max(2, args.crops // 2)):
                    key = int(rng.integers(1, 2**31)); m, wm = stamp(base, key); w, h = m.size
                    mir = m.transpose(Image.FLIP_LEFT_RIGHT) if mode == "h" else m.transpose(Image.FLIP_TOP_BOTTOM) if mode == "v" else m.transpose(Image.ROTATE_180)
                    img = mir.crop((int(rng.integers(1, w // 6)), int(rng.integers(1, h // 6)), w - int(rng.integers(0, w // 6)), h - int(rng.integers(0, h // 6))))
                    r = resync.find_spread_spectrum_id_resync(img, key)
                    hit += r.text == wm; n += 1
            print(f"  {name:26s} mirrored {mode:2s} + crop: {hit}/{n}")
    # mark-free
    print("MARK-FREE images through the same search (4 orientations x 4 grid offsets x 12,544 hypotheses each):")
    tot = bad = 0
    t_all = 0.0
    for name, imgs in _unmarked_zoo(rng).items():
        got = 0
        for img in imgs:
            key = int(rng.integers(1, 2**31))
            t0 = time.perf_counter()
            got += resync.find_spread_spectrum_id_resync(img, key).text is not None
            t_all += time.perf_counter() - t0
            tot += 1
        bad += got
    print(f"  {tot} mark-free images (pink, gradient+noise, flat/stripes/checker/random-block, 5 real photos): {bad} false IDs;"
          f" avg {1000*t_all/tot:.0f} ms each; theory ~ {tot*4*4*12544/2**40:.1e} expected")


def exp_limits(args):
    """Where it stops working: crop SIZE and JPEG strength, plus the width-in-blocks corner case."""
    import math
    from wmcore import resync
    rng = np.random.default_rng(args.seed)
    bases = [_pink(1280, 960, 900 + i) for i in range(args.n)]
    print("pink 1280x960, crop at a random off-grid offset keeping a given share of the AREA; reads / trials   (uncropped, same JPEG, for reference)")
    print(f"{'JPEG':>6s} | uncropped | " + " | ".join(f"area {int(f*100):3d}%" for f in (0.8, 0.5, 0.25, 0.1, 0.05)))
    for qual in (85, 60, 40, 30):
        cells = []
        n0 = 0
        for base in bases:
            key = int(rng.integers(1, 2**31)); m, wm = stamp(base, key)
            n0 += resync.find_spread_spectrum_id_resync(jpeg(m, qual), key).text == wm
        for frac in (0.8, 0.5, 0.25, 0.1, 0.05):
            hit = n = 0
            for base in bases:
                for _ in range(args.crops):
                    key = int(rng.integers(1, 2**31)); m, wm = stamp(base, key); w, h = m.size
                    sc = math.sqrt(frac); cw, ch = int(w * sc), int(h * sc)
                    cx, cy = int(rng.integers(1, max(2, w - cw))), int(rng.integers(1, max(2, h - ch)))
                    if cx % 8 == 0: cx += 3
                    img = jpeg(m.crop((cx, cy, cx + cw, cy + ch)), qual)
                    hit += resync.find_spread_spectrum_id_resync(img, key).text == wm; n += 1
            cells.append(f"{hit:2d}/{n:<2d}   ")
        print(f"  q{qual:<3d} |  {n0}/{len(bases)}      | " + " | ".join(f"{c:>9s}" for c in cells), flush=True)
    print("width-in-blocks corner case (coffee photo cut to the given width, then 7 block columns trimmed each side; 10 trials):")
    coffee = real_images()["coffee"]
    for wpx in (448, 456, 512, 560, 896):
        base = coffee.crop((0, 0, min(wpx, 600), 400)) if wpx <= 600 else None
        if base is None:
            continue
        hit = 0
        for _ in range(10):
            key = int(rng.integers(1, 2**31)); m, wm = stamp(base, key)
            hit += resync.find_spread_spectrum_id_resync(m.crop((56, 0, wpx - 56, 400)), key).text == wm
        print(f"  width {wpx} px = {wpx//8} blocks (gcd with 112 = {math.gcd(wpx//8, 112):3d}): {hit}/10")
    # timing on a big photo-sized image
    big = _pink(3000, 2000, 11); key = 777; m, wm = stamp(big, key)
    img = m.crop((101, 77, 2900, 1900))
    t0 = time.perf_counter(); r = resync.find_spread_spectrum_id_resync(img, key); t1 = time.perf_counter() - t0
    plain = time.perf_counter(); spread.find_spread_spectrum_id(spread.spread_spectrum_block_correlations(img, key)); t2 = time.perf_counter() - plain
    print(f"3000x2000 photo-size, off-grid crop: resync read it: {r.text == wm}  in {t1:.2f}s   (one plain decode of the same image: {t2:.2f}s)")
    clean = _pink(3000, 2000, 12)
    t0 = time.perf_counter(); r = resync.find_spread_spectrum_id_resync(clean, key); print(f"3000x2000 mark-free, full search (4 orientations x 4 offsets): {time.perf_counter()-t0:.2f}s, ID returned: {r.text is not None}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["aligned", "unaligned", "null", "pipeline", "limits"], required=True)
    ap.add_argument("--n", type=int, default=2, help="synthetic images")
    ap.add_argument("--crops", type=int, default=6, help="random crops per image")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--n-null", type=int, default=20, help="images per kind in the null experiment")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    {"aligned": exp_aligned, "unaligned": exp_unaligned, "null": exp_null, "pipeline": exp_pipeline, "limits": exp_limits}.get(args.exp, lambda a: sys.exit("not written yet"))(args)


if __name__ == "__main__":
    main()
