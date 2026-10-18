"""Crop resync: read the spread-spectrum watermark ID from a CROPPED copy (a screenshot trimmed to the picture, a
reframed repost, a thumbnail cut from one corner).

Why a crop used to kill the mark (spread.py): the payload is tiled in RASTER order over the ORIGINAL image's 8x8
block grid - bit i sits in block number i (row-major), period 112 bits for an ID. The decoder used to assume the
copy's own block grid IS that grid. Three things can make it not so, and they are independent:

1. The crop starts ON the 8-pixel grid but changes the image's WIDTH (left or right edge trimmed). The raster rows
   are now a different length, so consecutive rows no longer line up with the 112-bit period. Cropping only the top
   or bottom keeps the width, which is why that always worked.
2. The crop starts OFF the grid (any offset that is not a multiple of 8 - a screenshot almost always is). Every 8x8
   block the decoder cuts straddles four of the embedder's blocks and the mark averages out to nothing.
3. Both.

Both are fixed by re-deriving where the original grid was, from the copy alone:

(2) Block grid. The correlation of an 8x8 pixel window with the key's spreading pattern is a plain 2-D
cross-correlation (the per-block DCT correlation is linear in the pixels, so it equals <block, M> for one fixed 8x8
matrix M per key). One cv2.filter2D therefore gives that correlation at EVERY pixel position, i.e. at all 64 possible
grid offsets (dy, dx) at once (8 ms at 1280x960, 21 ms at 3000x2000 - against 0.1-0.8 s for the DCT of one grid). The
mark adds a large |correlation| only to blocks cut where the embedder cut them, so the offset whose blocks have the
largest mean squared correlation is where the grid is. The decoder tries the best few, and always (0, 0).

(1) Raster. Block (r, c) of the copy is block (r + R0, c + C0) of the original, whose width was bw0 blocks, so its
payload bit is ((r + R0) * bw0 + c + C0) mod 112 = (r * s + c + phi) mod 112 with s = bw0 mod 112 and
phi = (R0 * bw0 + C0) mod 112 - only the pair (s, phi) matters, 112 x 112 hypotheses. Folding the correlations
modulo 112 in both directions once (a 112 x 112 table, whatever the image size) makes every hypothesis a table
lookup, so all 12,544 are tried in ~20 ms. The old decoder is the special case s = (the copy's width) mod 112.

What counts as a hit: a canonical, check-valid watermark ID (wmid.py), same as everywhere else. 12,544 hypotheses
at 2**-40 each is ~1e-8 per attempt on an image with no mark (bench/results/CROP-NOTES.md measured 0 false IDs on
55 mark-free images, structured ones included). Free-text payloads are NOT recovered this way, for the same reason
orient.py gives for mirrored copies: their accept rule ("valid UTF-8") is far too loose to run over this many
hypotheses.

Known limit, a property of how the mark was embedded and so of ANY decoder: payload bit (r * bw0 + c) mod 112
means a given column c only ever carries bits spaced g = gcd(bw0, 112) apart, so after a left/right trim the
copy must still be at least g block-columns wide (g * 8 pixels) or some payload bits are gone for good. For almost
every width g is tiny (1280 px: g = 16 -> 128 px; 1080 or 3000 px: g = 1). It bites only on widths that are a
multiple of 56 blocks (448 px, 896 px, 1344 px, 1792 px ...): g is 56 or 112, so 448 / 896 px wide images lose
the mark under ANY left/right trim and 1344 / 1792 px ones once trimmed below 448 / 896 px
(bench/results/CROP-NOTES.md §4, rule checked on 10 cases). Top / bottom trims are never affected.

Not covered: a resize combined with a crop (the resize search in rescale.py does not re-align the grid), a crop
combined with rotation or perspective.
"""

import time
from typing import NamedTuple, Optional

import cv2
import numpy as np
from PIL import Image

from . import spread, wmid
from .orient import ORIENTATIONS, unflip

P = spread.ID_PERIOD_BITS                       # 112: period of an ID on the wire, in bits / blocks
DEFAULT_RESYNC_BUDGET_SECONDS = 10.0
DEFAULT_TOP_SHIFTS = 3                          # best-scoring (dy, dx) grid offsets tried per key, besides (0, 0)
PRESCREEN_PATCH = 320                           # px side of the central patch a (key, orientation) is pre-screened on
PRESCREEN_MIN_RATIO = 1.45                      # ...and the max/median grid-offset energy ratio it must reach there
PRESCREEN_MIN_SIDE = 480                        # smaller images are not pre-screened (they are cheap, and a small
                                                # patch's noise floor is higher)

_ALPHABET_OK = np.zeros(256, dtype=bool)
_ALPHABET_OK[np.frombuffer(wmid.ALPHABET.encode("ascii"), dtype=np.uint8)] = True
_RHO = np.arange(P)
_K = np.arange(32, P)                           # payload bit positions after the 32-bit length header


class ResyncMatch(NamedTuple):
    text: Optional[str]            # canonical watermark ID, or None
    key: Optional[int]             # the key that read it
    orientation: str               # "none" | "h" | "v" | "hv": what was done to the image (undone before decoding)
    shift: Optional[tuple]         # (dy, dx): where the 8x8 block grid starts in the (un-mirrored) copy
    stride: Optional[int]          # s above (original width in blocks, mod 112)
    phase: Optional[int]           # phi above
    budget_exhausted: bool         # ran out of time before covering every candidate (not a confirmed "nothing")
    confidence: Optional[float] = None   # total correlation mass behind the read: what picks between candidate reads
    cropped: bool = False          # the read needed a hypothesis the plain decoders do not make: a block grid that does
                                   # not start at (0, 0), or a row length other than the copy's own width


def luma(img):
    """BT.601 luma as float32 (H, W) from a PIL image or an HxWx3 array - the same weights spread.py uses."""
    arr = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img, dtype=np.float32)
    return arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


PATCH_MIN_GRADIENT = 3.0        # a patch quadrant needs at least this mean gradient (on the 4x-downsampled luma) to hold content
PATCH_MOVED_MAX_CLIPPED = 0.02    # a window chosen AFTER the centre failed must be clean of fill
PATCH_MAX_CLIPPED = 0.5         # ...and at most this share of pure black / white pixels (<= 3 or >= 252: fill, not photo; a mark
                                # is lost where it is clipped). 'Under 20 / over 235' flagged ordinary dark photos as empty.


def _window_stats(y, side, k=4):
    """For every window of ~`side` px on a k-pixel lattice: (min mean gradient over its four quadrants, max clipped share
    over them, mean gradient over them), as arrays indexed by the window's top-left on the lattice, plus (half, k)."""
    h, w = y.shape
    small = cv2.resize(y, (max(w // k, 1), max(h // k, 1)), interpolation=cv2.INTER_AREA)
    grad = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3))
    clipped = ((small <= 3) | (small >= 252)).astype(np.float32)
    half = max(2, side // 2 // k)
    qg = cv2.blur(grad, (half, half), anchor=(0, 0))
    qc = cv2.blur(clipped, (half, half), anchor=(0, 0))
    mh, mw = qg.shape[0] - half, qg.shape[1] - half
    if mh <= 0 or mw <= 0:
        return None
    quads = lambda q: (q[:mh, :mw], q[half:half + mh, :mw], q[:mh, half:half + mw], q[half:half + mh, half:half + mw])
    g, c = quads(qg), quads(qc)
    return np.minimum.reduce(g), np.maximum.reduce(c), sum(g) / 4.0, half, k


def find_patch(y, side):
    """((row, col), status): where a `side` px square patch should be taken for the patch-based tests here (the pre-screen, the
    scale sweep and its refinement, the verification region). status: "centre" - the image centre holds picture content (the
    usual case; the patch is there, as it always was); "moved" - the centre is empty (a screenshot of a picture in the corner
    of a mostly black or flat screen) and the patch is on the SMOOTHEST window that holds content everywhere in it; "none" -
    no window of that size holds content (the centre is returned and the caller should not trust a patch test).

    Measured on real X screenshots: a photo shown at 50% in the corner of a 1600x1000 screen was invisible to a patch at the
    centre of the screenshot (black), while the same photo cropped to its window was read in 0.6 s. Two first versions chose
    the MOST textured patch, and that is exactly wrong: the statistic is the mark's energy against the host's, so it falls
    with texture (correlation -0.85 with mean gradient across 20 patches of one screenshot; r = 1.4 at the most textured
    patch, 2.3 at a smooth one), and a mean-gradient score is also won by a patch straddling the picture's border."""
    h, w = y.shape
    centre = (h // 2, w // 2)
    if min(h, w) <= side * 5 // 4:
        return centre, "centre"
    st = _window_stats(y, side)
    if st is None:
        return centre, "none"
    gmin, cmax, gmean, half, k = st
    valid = (gmin > PATCH_MIN_GRADIENT) & (cmax < PATCH_MAX_CLIPPED)
    mh, mw = valid.shape
    ic, jc = min(max(h // 2 // k - half, 0), mh - 1), min(max(w // 2 // k - half, 0), mw - 1)
    if valid[ic, jc]:
        return centre, "centre"
    if not valid.any():
        return centre, "none"
    # Moving away from the centre: no black or white FILL anywhere in the window (a window a quarter black still passes the
    # lenient test above, and being the smoothest it would be chosen - it was, in the test on a black screen).
    strict = valid & (cmax < PATCH_MOVED_MAX_CLIPPED)
    pool = strict if strict.any() else valid
    i, j = np.unravel_index(int(np.argmin(np.where(pool, gmean, np.inf))), valid.shape)
    cy, cx = (i + half) * k, (j + half) * k                      # centre of the 2*half x 2*half window
    return (min(max(cy, side // 2), h - side // 2), min(max(cx, side // 2), w - side // 2)), "moved"


def texture_center(y, side):
    """find_patch() without the status."""
    return find_patch(y, side)[0]


def _patch_origin(h, w, sp, center):
    """Top-left of an `sp` px square patch centred at `center` (clamped to fit the image)."""
    cy, cx = center
    return min(max(cy - sp // 2, 0), h - sp), min(max(cx - sp // 2, 0), w - sp)


def key_pattern(key):
    """The 8x8 matrix M with <block, M> == the per-block spread-spectrum correlation of that block for `key`."""
    pn = spread._ss_pn_vector(key, len(spread.SS_COEFF_POSITIONS))
    m = np.zeros((spread.SS_BLOCK, spread.SS_BLOCK))
    for weight, (r, c) in zip(pn, spread.SS_COEFF_POSITIONS):
        m += weight * np.outer(spread._SS_C[r], spread._SS_C[c])
    return m.astype(np.float32)


def correlation_map(y, key):
    """cmap[i, j] = correlation of the 8x8 window whose top-left pixel is (i, j); only entries with
    i <= H-8, j <= W-8 are windows that fit (the rest is border filler and is never read)."""
    return cv2.filter2D(y, -1, key_pattern(key), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)


def grid_at(cmap, dy, dx):
    """The (bh, bw) correlations of the block grid that starts at pixel (dy, dx) - what
    spread_spectrum_block_correlations() computes for y[dy:, dx:], to float32 rounding."""
    h, w = cmap.shape
    bh, bw = (h - dy) // spread.SS_BLOCK, (w - dx) // spread.SS_BLOCK
    return cmap[dy:dy + bh * spread.SS_BLOCK:spread.SS_BLOCK, dx:dx + bw * spread.SS_BLOCK:spread.SS_BLOCK]


def shift_scores(cmap):
    """(8, 8) mean squared correlation of the block grid at each offset: the right one stands out when marked.
    Over the part of the map every offset can cover (a few pixels at the far edge are ignored, which changes the ranking by
    nothing measurable). Folded in two contiguous passes (rows, then columns): 0.08 ms for a 384 px patch against 0.48 ms
    for the one-shot reshape-and-sum over axes (0, 2), which is most of what a scale sweep spends outside the filter."""
    h, w = cmap.shape
    h8, w8 = (h - 7) // 8 * 8, (w - 7) // 8 * 8
    if h8 < 8 or w8 < 8:
        return np.zeros((8, 8))
    sq = np.square(cmap[:h8, :w8], dtype=np.float32)
    rows = sq.reshape(h8 // 8, 8, w8).sum(axis=0)                      # (8, w8): rows folded by their offset
    return rows.reshape(8, w8 // 8, 8).sum(axis=1, dtype=np.float64) / (h8 // 8 * (w8 // 8))


def _fold(grid):
    """G[rho, m] = sum of grid[r, c] over r = rho, c = m (mod 112): the whole image reduced to a 112 x 112 table."""
    bh, bw = grid.shape
    hp, wp = -(-bh // P) * P, -(-bw // P) * P
    pad = np.zeros((hp, wp), dtype=np.float64)
    pad[:bh, :bw] = grid
    return pad.reshape(hp // P, P, wp // P, P).sum(axis=(0, 2))


_T = (np.arange(P)[:, None] * np.arange(P)[None, :]) % P            # _T[s, k] = (s * k) mod 112
_KPHI = (_K[None, :] - np.arange(P)[:, None]) % P                       # _KPHI[phi, i] = payload position i at phase phi


def _stride_sums(g):
    """b[s, j] = sum over rho of g[rho, (j - rho * s) mod 112] for all 112 strides s at once.

    Shearing a row by rho * s is a phase ramp in the Fourier domain, so with F = fft2(g):
    fft(b[s])[k] = F[(k * s) mod 112, k] - a gather of 12,544 numbers and 112 short FFTs, instead of summing 112
    shifted copies of the table for each of the 112 strides (1.4 M gathers, 3-4x slower; same numbers to ~1e-12)."""
    f2 = np.fft.fft2(g)
    return np.fft.ifft(f2[_T, np.arange(P)[None, :]], axis=1).real


def find_id_in_grid(grid):
    """Search all 112 x 112 (stride, phase) hypotheses on one block grid. Returns (id, stride, phase, confidence) of
    the most confident canonical, check-valid ID, or None. Cost: one pass over the grid + a few ms, independent of
    image size."""
    if grid.size < P:
        return None
    b = _stride_sums(_fold(grid))                                                  # (112 s, 112 j)
    # phase phi rotates the bit positions: payload bit i at phase phi is b[s, (32 + i - phi) mod 112]
    by = np.packbits(b[:, _KPHI] > 0, axis=2)                                      # (112 s, 112 phi, 10 bytes)
    ok = _ALPHABET_OK[by].all(axis=2)
    best = None
    for s, phi in zip(*np.nonzero(ok)):
        text = by[s, phi].tobytes().decode("ascii")
        if wmid.parse(text) != text:                  # canonical form only (parse() is lenient on purpose)
            continue
        conf = float(np.abs(b[s, _KPHI[phi]]).sum())
        if best is None or conf > best[3]:
            best = (text, int(s), int(phi), conf)
    return best


def _passes_prescreen(patch, pattern, min_ratio):
    """Cheap test on a central patch: does ANY grid offset's correlation energy stand out for this key? Noise-only
    patches: median 1.12, p99 1.43, max 1.51 over 360 (image, orientation, key) draws; the right key after a crop
    reads 4.8+ uncompressed, 2.5+ at JPEG q60, and as low as ~1.5 at q20 on real photos. The gate is deliberately a
    little above the noise median: a noise pair that slips through only costs one full search, a real mark that does
    not pass is lost. Skips the full-image filter, ranking and (stride, phase) search for the (key, orientation) pairs that cannot
    be read - nearly all of them when the image has no mark or the ledger has many keys."""
    sc = shift_scores(cv2.filter2D(patch, -1, pattern, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE))
    med = float(np.median(sc))
    return med <= 0 or float(sc.max()) / med >= min_ratio


def find_spread_spectrum_id_resync(img, keys, *, orientations=ORIENTATIONS, top_shifts=DEFAULT_TOP_SHIFTS,
                                   max_seconds=DEFAULT_RESYNC_BUDGET_SECONDS, prescreen=True):
    """Read a watermark ID from a copy whose block grid / raster no longer matches the original's (see the module
    docstring). Every key is tried in every orientation (a mirrored copy is un-mirrored first, as in orient.py), each
    on the grid at offset (0, 0) and at the `top_shifts` best-scoring offsets, each with the full (stride, phase)
    search. Returns the MOST CONFIDENT read of all of them as a ResyncMatch (`confidence` = total correlation mass
    behind the ID); `budget_exhausted` is True when `max_seconds` ran out before every candidate was tried.

    Why the most confident of ALL, not the first that reads: a wrong key reads the real ID a good part of the time (the
    key is not a secret - spread.py, bench/results/CROP-NOTES.md §5), so with a ledger of several keys the first key to
    "read" is often not the one the image was stamped with, and judging the orientation with that key is noise (it
    labelled un-mirrored crops as mirrored 2 times in 8, and the labels of 96 copies over 3 ledgers were right only 56%
    of the time: CROP-NOTES.md §5b). The right key in the right orientation reads at full strength - 2-3x the mass of any
    wrong read in the measurement - so the maximum picks the true key, and the orientation with it. Likewise a
    top-bottom flip is also "rows in reverse order" to the raster search, which reads it at about 1/8 of the mark's
    strength (9 of the 16 mark coefficients have even row index, 7 odd: 2/16 survive the per-block reflection) without
    un-flipping anything - a correct ID, not what was done to the picture; the un-flipped orientation outranks it.

    `prescreen` (default on, images of 480 px or more on both sides): each (key, orientation) is first tested on a
    320 px central patch (_passes_prescreen) and only the ones that pass get the full-image search - ~1 ms instead of
    ~30 ms per pair at 1280x960 for a pair that cannot read. Pass prescreen=False for the exhaustive search.

    Cost is proportional to len(keys) x len(orientations). `keys` may be a single int or a list."""
    keys = [keys] if isinstance(keys, (int, np.integer)) else list(keys)
    deadline = time.monotonic() + max_seconds
    y0 = luma(img)
    best = None                                     # the most confident ResyncMatch so far
    exhausted = False
    for orientation in orientations:
        y = y0 if orientation == "none" else np.ascontiguousarray(unflip(y0, orientation))
        if min(y.shape) < 2 * spread.SS_BLOCK:
            continue
        patch = None
        if prescreen and min(y.shape) >= PRESCREEN_MIN_SIDE:
            c, status = find_patch(y, PRESCREEN_PATCH)
            if status != "none":                         # (no window with content: a patch test would reject a real mark)
                t, l = _patch_origin(*y.shape, PRESCREEN_PATCH, c)
                patch = np.ascontiguousarray(y[t:t + PRESCREEN_PATCH, l:l + PRESCREEN_PATCH])
        for key in keys:
            if time.monotonic() > deadline:
                exhausted = True
                break
            pattern = key_pattern(key)
            if patch is not None and not _passes_prescreen(patch, pattern, PRESCREEN_MIN_RATIO):
                continue
            cmap = cv2.filter2D(y, -1, pattern, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
            scores = shift_scores(cmap)
            ranked = [divmod(int(i), 8) for i in np.argsort(-scores, axis=None)]
            candidates = [(0, 0)] + [rc for rc in ranked if rc != (0, 0)][:top_shifts]
            for dy, dx in candidates:
                hit = find_id_in_grid(grid_at(cmap, dy, dx))
                if hit is not None and (best is None or hit[3] > best.confidence):
                    plain = (dy, dx) == (0, 0) and hit[1] == (y.shape[1] // spread.SS_BLOCK) % P
                    best = ResyncMatch(hit[0], int(key), orientation, (dy, dx), hit[1], hit[2], False, hit[3], not plain)
        if exhausted:
            break
    if best is not None:
        return best._replace(budget_exhausted=exhausted)
    return ResyncMatch(None, None, "none", None, None, None, exhausted)


# =============================================================================================================
# Crop AND resize (a screenshot taken at another zoom, then trimmed): find the scale, then resync as above.
# =============================================================================================================
#
# A resize moves the 8x8 grid onto a lattice of 8*g pixels; to read the mark the copy has to be resampled by
# f = 1/g back to the original scale, after which the crop (unknown offset, unknown width) is exactly the problem
# solved above. rescale.py already searches for f, but anchors its statistic at the image's top-left corner where the
# grid phase is known to be 0 - a crop removes that anchor. The offset-agnostic statistic used for the grid works here
# too: max/median of the 64 grid-offset energies of a central patch resampled by f. On clean photos it is 2.9-5.2 at the
# true f and 1.1-1.3 elsewhere; on a densely textured host (a circuit board photo) it is only 1.3-1.7 at the true f - but
# the true f is still the best-scoring one (ranked first in 120 of 120 sweeps with a 384 px patch, rank <= 3 with 256 px),
# while NO threshold separates it from the noise peaks of other photos (CROP-NOTES.md §8b: the first version gated on
# z >= 12 and r >= 1.6 and never read that photo). So there is no threshold on the statistic: the sweep only RANKS the
# scales, and the decoder - whose accept rule (a check-valid canonical ID) is the one thing that is reliable - decides.
#
#   1. sweep every key over the scale range, sharing the resampled patch between keys (the resample is key-independent);
#   2. verify, strongest peak first, ACROSS keys (so that a ledger of many keys spends its time on the likeliest candidates
#      if the budget is short): refine the scale on a bigger patch, decode the whole image at the refined scale and at
#      +-5 refine steps (the decode tolerates +-6 steps = 0.24% around the truth, 59 of 59 trials; the refinement lands
#      within 6 steps 97% of the time, max 8) with the grid + (stride, phase) search; first the best peak of each key,
#      then the second and third best;
#   3. once a scale reads, evaluate EVERY key at that one restored image and keep the most confident read - the same
#      rule as find_spread_spectrum_id_resync, for the same reason (a wrong key reads the real ID with an arbitrary
#      stride and offset, and a first-hit would decide `cropped` with it).

SCALE_RANGE = (0.5, 2.0)
_SWEEP_PATCH = 384              # px of the resampled patch the sweep looks at (ranks the true scale first in 120 of 120
                                # real-photo sweeps, incl. a circuit board; 256 px: rank <= 3; CROP-NOTES.md §8b)
_SWEEP_LOG_STEP = 0.0025        # 0.25% between candidate factors (the true one lands within ~0.15%)
_REFINE_PATCH = 512
_REFINE_LOG_STEP = 0.0004
_REFINE_HALF_WIDTH = 0.006      # +-0.6% around a sweep peak
_DECODE_OFFSETS = (0, 5, -5)    # in refine steps around the refined scale
_PEAKS_PER_KEY = 4
SCALE_CHUNK_KEYS = 6            # ledger keys swept per round; each round's candidates are verified before the next round is swept
STAGE3_MIN_RATIO = 1.45         # at the scale that reads, only keys whose patch statistic reaches this are read in full (as the pre-screen)
_PEAK_SEPARATION = 0.01         # peaks of one key are at least 1% apart in scale
_VERIFY_MAX_PIXELS = 1_500_000  # a candidate scale is decoded on a central region of the restored image this big at most
                                # (~23,000 blocks = 200 repeats of an ID is plenty to read one): 4x cheaper per attempt at
                                # 6 MP, the same reads. NOT confirmed on the whole image: a first version did, and on a 10 MP
                                # copy the whole image does not read at a scale that the region reads (a 0.2% scale error is
                                # 8 px of drift across 3800 px but only 2 px across the region), so it crashed on None.


class ScaledMatch(NamedTuple):
    text: Optional[str]
    key: Optional[int]
    factor: Optional[float]        # the decode-time resample that restored the original scale (1/g for a copy scaled by g)
    shift: Optional[tuple]         # (dy, dx) of the block grid in the restored image
    stride: Optional[int]
    phase: Optional[int]
    budget_exhausted: bool
    cropped: bool = False          # needed more than the ~1 px / 1 block slack a resample alone introduces
    confidence: Optional[float] = None   # total correlation mass behind the read (what picks between keys)
    keys_reached: int = 0          # how many of the keys were swept before a read, or before the budget ran out


def _resample(y, f):
    return cv2.resize(y, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def _patch_at(y, f, dst, center=None):
    """The patch of `y` around `center` (default: the middle) that is `dst` px after resampling by f, resampled (None: it
    does not fit / too small)."""
    h, w = y.shape
    sp = int(round(dst / f))
    if sp > min(h, w) or sp < 4 * spread.SS_BLOCK:  # (truncating the patch to fit made the statistic drift with f and
        return None                                 # broke small copies: 64 of 84 reads in the measurement)
    t, l = _patch_origin(h, w, sp, center or (h // 2, w // 2))
    p = _resample(np.ascontiguousarray(y[t:t + sp, l:l + sp]), f)
    return p if min(p.shape) >= 3 * spread.SS_BLOCK else None


def _ratio(p, pattern):
    sc = shift_scores(cv2.filter2D(p, -1, pattern, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE))
    med = float(np.median(sc))
    return float(sc.max() / med) if med > 0 else None


def _patch_ratio(y, pattern, f, dst, center=None):
    """max/median of the 64 grid-offset energies of the patch of `y` around `center` resampled by f (None: too big)."""
    p = _patch_at(y, f, dst, center)
    return None if p is None else _ratio(p, pattern)


def _sweep_all(y, patterns, scale_range, deadline, dst=_SWEEP_PATCH, center=None):
    """vals[k, i]: the patch statistic of key k at candidate factor grid[i] (nan where the patch does not fit). The
    resampled patch is shared by all keys. Returns (grid, vals), or (grid, None) if the deadline passed."""
    grid = np.exp(np.arange(np.log(scale_range[0]), np.log(scale_range[1]), _SWEEP_LOG_STEP))
    vals = np.full((len(patterns), len(grid)), np.nan)
    for i, f in enumerate(grid):
        if i % 32 == 0 and time.monotonic() > deadline:
            return grid, None
        p = _patch_at(y, float(f), dst, center)
        if p is None:
            continue
        for k, pattern in enumerate(patterns):
            r = _ratio(p, pattern)
            if r is not None:
                vals[k, i] = r
    return grid, vals


def _ranked_peaks(row, grid):
    """[(z, f)] best first: the `_PEAKS_PER_KEY` highest sweep values, at least `_PEAK_SEPARATION` apart in scale, with
    z against the row's own noise floor. No threshold (see the section comment): the decoder decides."""
    ok = ~np.isnan(row)
    if ok.sum() < 20:
        return []
    med = float(np.median(row[ok]))
    sigma = max(1.4826 * float(np.median(np.abs(row[ok] - med))), 0.01 * med)
    peaks = []
    for i in np.argsort(-np.where(ok, row, -np.inf)):
        if not ok[i] or len(peaks) >= _PEAKS_PER_KEY:
            break
        if all(abs(np.log(grid[i] / f)) > _PEAK_SEPARATION for _, f in peaks):
            peaks.append((float((row[i] - med) / sigma), float(grid[i])))
    return peaks


def _refine(y, pattern, f0, deadline, center=None):
    best_f, best_r = f0, -1.0
    steps = int(_REFINE_HALF_WIDTH / _REFINE_LOG_STEP)
    for k in range(-steps, steps + 1):
        if time.monotonic() > deadline:
            break
        f = f0 * float(np.exp(k * _REFINE_LOG_STEP))
        # the biggest patch that fits the copy: a small copy (a crop shrunk to 300 px) has no room for 512 px, and
        # without this fallback its scale was never refined (the first version of this change dropped it: 64 of 84)
        r = (_patch_ratio(y, pattern, f, _REFINE_PATCH, center) or _patch_ratio(y, pattern, f, _SWEEP_PATCH, center)
             or _patch_ratio(y, pattern, f, 256, center) or _patch_ratio(y, pattern, f, 192, center))
        if r is not None and r > best_r:
            best_f, best_r = f, r
    return best_f


def _central_region(restored, center=None, max_pixels=_VERIFY_MAX_PIXELS):
    """(region, (top, left)): the part of a restored image around `center` (default: the middle) if the image exceeds
    `max_pixels`, with its top-left corner on the 8-pixel grid so that the block grid reads at the same offset (dy, dx)
    as it would in the whole image."""
    h, w = restored.shape
    if h * w <= max_pixels:
        return restored, (0, 0)
    k = (max_pixels / (h * w)) ** 0.5
    rh, rw = max(int(h * k), 16), max(int(w * k), 16)
    cy, cx = center or (h // 2, w // 2)
    t = min(max(cy - rh // 2, 0), h - rh) // spread.SS_BLOCK * spread.SS_BLOCK
    l = min(max(cx - rw // 2, 0), w - rw) // spread.SS_BLOCK * spread.SS_BLOCK
    return np.ascontiguousarray(restored[t:t + rh, l:l + rw]), (t, l)


def _read_restored(restored, pattern, top_shifts):
    """The most confident (stride, phase) read of one restored image for one key: (dy, dx, hit) or None."""
    cmap = cv2.filter2D(restored, -1, pattern, anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    scores = shift_scores(cmap)
    ranked = [divmod(int(i), 8) for i in np.argsort(-scores, axis=None)]
    best = None
    for dy, dx in [(0, 0)] + [rc for rc in ranked if rc != (0, 0)][:top_shifts]:
        hit = find_id_in_grid(grid_at(cmap, dy, dx))
        if hit is not None and (best is None or hit[3] > best[2][3]):
            best = (dy, dx, hit)
    return best


_SCALE_TOLERANCE = 0.0025       # the scale search lands within ~0.25% of the true factor (refine error <= 6 steps of 0.04%)


def _scaled_match(restored_shape, region_origin, key, f, found):
    """`restored_shape`: (h, w) of the WHOLE restored image (the decode may have used a region of it); `region_origin`:
    (top, left) of that region in it (both multiples of 8, so (dy, dx) is the grid offset in the whole image too)."""
    dy, dx, hit = found
    # A resample alone moves the grid and the width by an amount that grows with the image: a scale error e shifts the grid
    # by e * (distance from the origin), 6 px at the middle of a 10 MP photo for e = 0.25%, and the width by e * width.
    # "plain" (not cropped) is judged with that much slack - a copy that was only resized read as 'cropped' in the first
    # version of this (a fixed 1 px slack), and again at 10 MP (x0.5): measured. The price: a crop smaller than the slack
    # (a few pixels at 1 MP, tens at 10 MP) is not called a crop.
    h, w = restored_shape
    t, l = region_origin
    slack_y = _SCALE_TOLERANCE * (t + h / 2) + 1
    slack_x = _SCALE_TOLERANCE * (l + w / 2) + 1
    near = lambda d, slack: min(d % 8, 8 - d % 8) <= slack
    ds = (hit[1] - (w // spread.SS_BLOCK) % P + P // 2) % P - P // 2
    plain = near(dy, slack_y) and near(dx, slack_x) and abs(ds) <= _SCALE_TOLERANCE * (w // spread.SS_BLOCK) + 1
    return ScaledMatch(hit[0], int(key), f, (dy, dx), hit[1], hit[2], False, not plain, hit[3])


def find_spread_spectrum_id_resync_scaled(img, keys, *, scale_range=SCALE_RANGE, top_shifts=DEFAULT_TOP_SHIFTS,
                                          max_seconds=DEFAULT_RESYNC_BUDGET_SECONDS, chunk_keys=SCALE_CHUNK_KEYS):
    """Read a watermark ID from a copy that was resized AND cropped (unmirrored only; see the section comment). Returns
    a ScaledMatch (the most confident read over all keys at the scale that read); `budget_exhausted` is True when
    `max_seconds` ran out before every candidate was tried (not a confirmed 'nothing'), with `keys_reached` = how many keys
    were swept by then.

    Keys are taken in the order given (the ledger lists the newest stamps first, and a repost is most often of a recent one)
    in rounds of `chunk_keys`: a round is swept (~0.4 s per key: the filter on a 384 px patch at 555 scales), its candidates
    are verified strongest first, and only then is the next round swept. The first version swept EVERY key before verifying
    anything: a marked copy then cost 0.43 s x N before its first decode (26 s at N = 60, over the whole budget) when the true
    key's candidate was ranked first and verifying it took 0.1 s. (Sharing the filter between keys through a Gram matrix of the
    16 mark coefficients is exact to 1e-7 but costs ~30 ms per scale however many keys, ~17 s for the sweep: it only wins
    above ~25 keys and is still slow, so it is not used - CROP-NOTES.md §8d.)"""
    keys = [keys] if isinstance(keys, (int, np.integer)) else list(keys)
    deadline = time.monotonic() + max_seconds
    y = luma(img)
    none = ScaledMatch(None, None, None, None, None, None, False)
    if not keys or min(y.shape) < 96:              # (this guard was tied to the sweep patch size and so silently
        return none                                # rejected every copy under 384 px when the patch grew: 64 of 84)
    patterns = [key_pattern(k) for k in keys]
    # The patch is constant in RESTORED pixels (so its statistic does not drift with the scale) and must fit the copy at
    # the true scale: a 297 px-tall copy cannot supply a 384 px patch at f = 1.25 (it needs 307 source px), so the patch
    # for a small copy is 90% of its short side (264 px there). A 384 px patch was unreadable for copies under ~385 px.
    dst = max(128, min(_SWEEP_PATCH, int(0.9 * min(y.shape)) // 8 * 8))
    center, status = find_patch(y, dst)
    if status != "centre":
        # The centre is empty, so the patch must stay inside the picture for EVERY candidate scale: the source patch is
        # dst / f px and f goes down to scale_range[0], so the window that holds content must be dst / scale_range[0] px
        # (at f = 0.5 a 192 px patch needs 384 px of picture, else it takes in the black margin and its picture-border edge
        # outscores the true scale: measured). Take the biggest window that has content and size the patch from it.
        for window in (512, 384, 256, 192):
            c, st = find_patch(y, window)
            if st == "moved":
                center, dst = c, max(96, int(window * scale_range[0]) // 8 * 8)
                break
    reached = 0
    for start in range(0, len(keys), max(1, chunk_keys)):
        if time.monotonic() > deadline:
            return none._replace(budget_exhausted=True, keys_reached=reached)
        round_idx = list(range(start, min(start + max(1, chunk_keys), len(keys))))
        grid, vals = _sweep_all(y, [patterns[i] for i in round_idx], scale_range, deadline, dst, center)
        if vals is None:
            return none._replace(budget_exhausted=True, keys_reached=reached)
        reached = round_idx[-1] + 1
        peaks = {i: _ranked_peaks(vals[j], grid) for j, i in enumerate(round_idx)}
        for rank in range(_PEAKS_PER_KEY):
            queue = sorted(((peaks[i][rank][0], i) for i in round_idx if len(peaks[i]) > rank), reverse=True)
            for _z, k in queue:
                if time.monotonic() > deadline:
                    return none._replace(budget_exhausted=True, keys_reached=reached)
                f1 = _refine(y, patterns[k], peaks[k][rank][1], deadline, center)
                for off in _DECODE_OFFSETS:
                    if time.monotonic() > deadline:
                        return none._replace(budget_exhausted=True, keys_reached=reached)
                    f = f1 * float(np.exp(off * _REFINE_LOG_STEP))
                    restored = _resample(y, f)
                    if min(restored.shape) < 2 * spread.SS_BLOCK:
                        continue
                    region, origin = _central_region(restored, (round(center[0] * f), round(center[1] * f)))
                    if _read_restored(region, patterns[k], top_shifts) is None:
                        continue
                    # The scale reads. Every key (the whole ledger, not only this round) is judged at THIS restored region
                    # and the most confident read wins; keys whose patch statistic does not stand out there are skipped
                    # without a full read (the hitting key is always read).
                    pr = _central_region(region, None, 320 * 320)[0]
                    best = None
                    for kk in range(len(keys)):
                        if kk != k and not _passes_prescreen(pr, patterns[kk], STAGE3_MIN_RATIO):
                            continue
                        found = _read_restored(region, patterns[kk], top_shifts)
                        if found is not None and (best is None or found[2][3] > best.confidence):
                            best = _scaled_match(restored.shape, origin, keys[kk], f, found)
                    if best is not None:
                        return best._replace(keys_reached=reached)
    return none._replace(keys_reached=reached)
