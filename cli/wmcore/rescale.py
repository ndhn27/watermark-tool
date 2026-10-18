"""Spread-spectrum decode after a resize ("rescale search").

The mark (see spread.py) is anchored to the ORIGINAL image's 8x8 block
grid, so once the image has been resized the decoder has to undo that
resize - resample back to (almost exactly) the original size - before the
block correlations mean anything. Two facts, measured on ~1280x960 test
images, shape how the search is done:

1. Tolerance is tiny (~1-2 px, ~0.1% of scale): the payload is tiled in
   raster order over the block grid, so the block COLUMN count has to
   match too, or every row's tiling phase is scrambled. A short list of
   "common" factors (0.5, 0.75, 1.25, 2.0...) only hits when the resize
   happened to be an exact reciprocal of one of them (4 of 13 test
   resizes). Sweeping the full 0.5x-2x range that fine with full decodes
   is ~1700 attempts - minutes per megapixel - so it isn't done that way.
2. A cheap continuous signal exists instead: where the resample scale is
   right, the per-block correlations are strongly bimodal (+-alpha-sized
   lumps); where it's wrong they look like Gaussian noise.
   mean(|c|)/std(c) is ~0.8 for noise and ~1.0 for a real mark, and it can
   be computed on a small patch anchored at the image's top-left corner
   (where the block grid's phase is known to be 0) - ~100x cheaper than a
   full decode.

So: coarse patch sweep over the whole range -> keep peaks that stand out
from the sweep's own noise floor -> refine each on bigger patches -> only
then try full-image decodes at a handful of pixel widths around each
refined estimate. The sweep runs a cheap 160px tier first, a pricier 320px
tier if that finds nothing, and a 640px tier - needing a large enough
source image to even attempt - as a last resort, since a wide resample
kernel plus JPEG can leave a small patch too little signal.

A decoded candidate isn't accepted on the first validation: the payload's
length header is mostly zero bits, so a weak/drifted decode is biased
toward returning a spurious short, "valid-looking" payload. A result must
be printable text AND decode identically at two different candidate
widths before it's accepted.

v2 adds two things on top of that search, both driven by the same
observation - a search this wide over a noisy statistic can occasionally
look successful without being trustworthy, and can also take an
unpredictable amount of time on a hostile or unusually large/noisy input:

- RESULT CONFIDENCE. A successful search already clears a hard accept/
  reject gate (corroborated at 2 widths, printable). `confidence` doesn't
  change that gate; it scores how comfortably a candidate cleared it, from
  three signals gathered "for free" along the way: how far the winning
  scale factor's coarse-sweep statistic stood out from the sweep's own
  noise floor (`z`), how many payload repeats were available to combine at
  the accepted width (`repeats` - more repeats means the "processing gain"
  described in spread.py's module docstring was doing more work), and how
  many of the tried widths corroborated the same text (`corroborating_widths`,
  always >= 2 by construction). This is a bounded [0, 1] index built from
  measured, meaningful signals - it is NOT a calibrated probability or a
  p-value, because there's no labeled corpus of real photos here to
  calibrate one against. Treat 0.3-0.5 as "accepted, but marginal - a
  human might want to double check" and 0.8+ as "about as sure as this
  tool gets", not as percent-correct figures.
- PERFORMANCE BUDGET. `max_seconds` bounds wall-clock time spent on the
  search proper (not the cheap unresized-image fast path, which runs
  regardless). It is checked between discrete units of work - before each
  sweep tier, before each candidate's refine+validate pass, before each
  width tried during validation, and periodically inside the coarse sweep
  itself - not preemptively, so a single expensive unit (a full-image
  decode attempt on a very large image) can overshoot it somewhat. Hitting
  the budget only ever makes the search stop looking sooner; it never
  relaxes the accept/reject gate, so a budget that's too tight can turn a
  real mark into a false "not found" but can never manufacture a false
  positive. `budget_exhausted=True` on the returned RescaleMatch is how a
  caller tells "genuinely not found, whole space covered" apart from
  "ran out of time, inconclusive". Pass `max_seconds=None` for the old
  unbounded behavior.

Limits, deliberately not hidden: only a uniform, non-cropping resize is
searched for (no aspect-ratio changes, no crop), and very heavy
downscaling (~0.5x, especially plus JPEG) can remove enough mid-band
energy that no search finds the mark.
"""

import math
import time
from typing import NamedTuple, Optional

import cv2
import numpy as np

from . import spread

SS_RESCALE_RANGE = (0.5, 2.0)
# (patch px, log step of the sweep, min robust z of a peak, require_full_coverage).
# Peak widths are ~1.2% at 160 px and ~0.6% at 320 px; unstamped images
# topped out around z = 3 at 160/320 px, so the margins here leave some
# headroom. Tiers 1-2 tolerate PARTIAL range coverage - on a modest-sized
# image the low end of the range needs a source window bigger than the
# image has, so those scale factors are just silently skipped within the
# sweep (see _ss_patch_bimodality's None return) while the rest of the
# range is still checked, same as the original (v1) design. The 640 px
# tier is different on purpose: it's ~4x the per-evaluation cost of the
# 320 px tier (patch area scales with the square), so paying that cost for
# only a fraction of the range - the failure mode this tier hit in testing
# on a 640x480 image, where only scale factors above ~1.3 fit - isn't worth
# it. `require_full_coverage=True` makes it an all-or-nothing tier instead:
# skipped entirely (no evaluations at all) unless the source image is big
# enough to cover the WHOLE requested scale_range, so it only ever runs -
# and only ever pays its cost - on images actually large enough to make it
# worthwhile (roughly 1300px+ on the short side, for the default range).
_SS_SWEEP_TIERS = ((160, 0.003, 3.5, False), (320, 0.0015, 4.5, False), (640, 0.001, 5.0, True))
# Refinement stages: (patch px, +-log half-span, steps), each centred on the
# previous stage's best estimate; skipped when the image is too small. The
# 960 px stage only ever runs after the 640 px sweep tier found something,
# i.e. only on images already large enough to afford it.
_SS_REFINE_STAGES = ((320, 0.006, 25), (640, 0.0025, 17), (960, 0.001, 13))

# Default wall-clock budget (seconds) for the search itself - generous
# enough for the documented "a few seconds per image, more when nothing is
# found" cost on the two original tiers, with headroom for the third tier
# added in v2. Pass max_seconds=None to disable and search exhaustively.
DEFAULT_RESCALE_BUDGET_SECONDS = 20.0

# How many coarse-sweep evaluations run between deadline checks. Each single
# evaluation is cheap (a small-patch warp + DCT), so checking every one would
# waste time on time.monotonic() calls; checking too rarely risks overshooting
# a tight budget by a noticeable amount. This is a compromise, not a precise
# tuning.
_BUDGET_CHECK_STRIDE = 40


class RescaleMatch(NamedTuple):
    """Result of find_spread_spectrum_payload_rescaled().

    `text`/`factor` are None when nothing was found (mirrors the old
    (None, None) return - see module docstring for the rest of the fields).
    """

    text: Optional[str]
    factor: Optional[float]
    confidence: float
    repeats: int
    corroborating_widths: int
    budget_exhausted: bool


def _no_match(budget_exhausted=False):
    return RescaleMatch(None, None, 0.0, 0, 0, budget_exhausted)


def _deadline_hit(deadline):
    return deadline is not None and time.monotonic() > deadline


def _repeats_at(num_correlations, payload_text):
    """How many back-to-back copies of `payload_text`'s (header + payload)
    period fit in `num_correlations` block correlations - an upper-bound
    estimate of the repeats available to combine (see spread.py's
    "processing gain" discussion), not the phase-exact count. Good enough
    as one input to a confidence estimate; not used for decoding itself.
    """
    period = 32 + len(payload_text.encode("utf-8")) * 8
    return max(0, num_correlations // period)


def _rescale_confidence(z, min_z, repeats, corroborating_widths):
    """Heuristic [0, 1] confidence index - see the module docstring's
    "RESULT CONFIDENCE" section for what this is and, just as importantly,
    what it is NOT (a calibrated probability). Each term saturates toward 1
    as its underlying signal gets stronger, so one very strong signal can't
    by itself make up for the other two being weak or absent:

    - z_term: how far the winning scale's coarse-sweep statistic stood
      above that sweep tier's own min-z accept threshold.
    - repeat_term: how many payload repeats were available to combine
      at the accepted width (saturates by ~4, matching processing gain's
      sqrt(repeats) improvement flattening out well before then).
    - corrob_term: how many independently tried widths agreed
      (always >= 2 by construction, so this term has a floor of 2/3).

    Weighted toward corroboration since that's the actual accept gate;
    z and repeats are corroborating-but-secondary evidence.
    """
    z_term = 1.0 - math.exp(-max(z, 0.0) / (2.0 * min_z))
    repeat_term = 1.0 - math.exp(-repeats / 4.0)
    corrob_term = min(1.0, corroborating_widths / 3.0)
    return 0.20 * z_term + 0.35 * repeat_term + 0.45 * corrob_term


def _plain_decode_confidence(repeats):
    """Confidence for the fast, unresized-image path: no scale search ran,
    so none of the multiple-comparison risk that motivates the corroboration
    term above applies - a single valid decode here is already the same
    (uncontested) acceptance the plain decoder has always used. Scored on
    repeats alone, with a floor above the search path's worst case since
    there's no scale-search false-positive risk to weigh against it.
    """
    return 0.6 + 0.4 * (1.0 - math.exp(-repeats / 4.0))


def _ss_patch_bimodality(y32, f, pn, patch):
    """mean(|c|)/std(c) of the block correlations of a `patch` x `patch`
    window, anchored at the top-left corner, of the luma `y32` resampled by
    decode-scale `f`. None if the image is too small for that window."""
    h, w = y32.shape
    src = int(math.ceil(patch / f)) + 4
    if src > min(h, w):
        return None
    region = y32[:src, :src]
    if f < 1.0:
        region = cv2.GaussianBlur(region, (0, 0), 0.5 / f)  # anti-alias before shrinking
    t = 0.5 * f - 0.5  # pixel-centre convention, so it lines up with PIL/cv2.resize
    m = np.array([[f, 0.0, t], [0.0, f, t]], dtype=np.float64)
    win = cv2.warpAffine(region, m, (patch, patch), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REFLECT)
    blocks, _, _ = spread._image_to_ss_blocks(win.astype(np.float64))
    dct = np.einsum("ij,abjk,kl->abil", spread._SS_C, blocks, spread._SS_CT)
    c = (dct[:, :, spread._SS_ROWS, spread._SS_COLS] * pn).sum(axis=-1).reshape(-1)
    return float(np.mean(np.abs(c)) / (np.std(c) + 1e-9))


def _ss_sweep_peaks(y32, pn, scale_range, patch, log_step, min_z, max_candidates, deadline=None,
                    require_full_coverage=False):
    """Coarse sweep of decode-scale factors; returns (peaks, hit_deadline).
    `peaks` is a list of (scale, z) for the strongest well-separated peaks
    whose robust z-score (against the sweep's own median / MAD) is at least
    `min_z`, best first. `hit_deadline` is True if the sweep was cut short
    by `deadline` (in which case the z-scoring below still runs, just over
    fewer samples - fewer samples means a noisier background estimate, so a
    deadline-truncated sweep is more conservative, not less).

    `require_full_coverage`: if True, do nothing at all (return ([], False)
    immediately, at essentially zero cost) unless the image is big enough
    for EVERY factor in `scale_range` to have a valid source window - see
    _SS_SWEEP_TIERS's comment for why the 640 px tier needs this and the
    cheaper tiers don't.
    """
    lo, hi = scale_range
    if require_full_coverage:
        h, w = y32.shape
        if math.ceil(patch / lo) + 4 > min(h, w):
            return [], False
    fs_full = np.exp(np.arange(math.log(lo), math.log(hi) + log_step, log_step))
    stats = []
    hit_deadline = False
    for i, f in enumerate(fs_full):
        if deadline is not None and i % _BUDGET_CHECK_STRIDE == 0 and _deadline_hit(deadline):
            hit_deadline = True
            break
        stats.append(_ss_patch_bimodality(y32, f, pn, patch))
    fs = fs_full[: len(stats)]
    keep = np.array([v is not None for v in stats])
    if keep.sum() < 8:
        return [], hit_deadline
    fs = fs[keep]
    st = np.array([v for v in stats if v is not None])
    bg = np.median(st)
    z = (st - bg) / max(np.median(np.abs(st - bg)) * 1.4826, 1e-9)
    peaks = []
    for i in np.argsort(-z):
        if z[i] < min_z or len(peaks) >= max_candidates:
            break
        if all(abs(int(i) - j) > 4 for j in peaks):
            peaks.append(int(i))
    return [(float(fs[i]), float(z[i])) for i in peaks], hit_deadline


def _ss_refine_scale(y32, pn, f0, deadline=None):
    h, w = y32.shape
    f = f0
    for patch, half_span, steps in _SS_REFINE_STAGES:
        if _deadline_hit(deadline):
            break
        if math.ceil(patch / f) + 4 > min(h, w):
            continue
        grid = f * np.exp(np.linspace(-half_span, half_span, steps))
        vals = [_ss_patch_bimodality(y32, g, pn, patch) for g in grid]
        vals = [-1.0 if v is None else v for v in vals]
        f = float(grid[int(np.argmax(vals))])
    return f


_printable_payload = spread._printable_payload  # single definition now lives in spread.py; see its docstring


def find_spread_spectrum_payload_rescaled(rgb_image, key, scale_range=SS_RESCALE_RANGE,
                                          max_length=10000, max_candidates=6,
                                          max_seconds=DEFAULT_RESCALE_BUDGET_SECONDS, decoder=None):
    """Like spread.spread_spectrum_block_correlations() +
    spread.find_spread_spectrum_payload(), but also searches for the resize
    that was applied after embedding (see module docstring for how, for the
    `confidence`/`budget_exhausted` fields, and for what this can't do).

    Returns a RescaleMatch. On success, `factor` is the decode-time resize
    that recovered the payload (1.0 = decoded as-is; 1.3333 = the image had
    been shrunk to 75% and was scaled back up). On failure, `text`/`factor`
    are None - check `budget_exhausted` to tell "searched everything, no
    mark" apart from "ran out of time before finishing".

    `max_seconds` (default DEFAULT_RESCALE_BUDGET_SECONDS) bounds the
    search; pass None to search exhaustively regardless of how long it
    takes. It never affects the plain unresized-image fast path below.

    `decoder(corrs) -> text | None` is what reads a payload out of the block
    correlations, both on the unresized image and at every candidate width
    the search tries (default: the text decoder, spread.find_spread_spectrum_
    payload with `max_length`). It changes only that reading - the coarse
    sweep that proposes candidate scales, the two-agreeing-widths rule and
    the deadline are the same for every decoder. Pass
    spread.find_spread_spectrum_id to read watermark IDs.
    """
    if decoder is None:
        def decoder(corrs):
            return spread.find_spread_spectrum_payload(corrs, max_length=max_length)
    deadline = None if max_seconds is None else time.monotonic() + max_seconds

    rgb = np.asarray(rgb_image.convert("RGB") if hasattr(rgb_image, "convert") else rgb_image, dtype=np.float64)
    y, _, _ = spread._rgb_to_y(rgb)
    h, w = y.shape

    # Unresized images keep the plain decoder's behavior exactly - this only
    # adds a fallback. As of the printable check inside find_spread_spectrum_
    # payload() itself (spread.py), "the plain decoder's behavior" now
    # includes that filter too; no extra *corroboration* rule is added here.
    corrs0 = spread._ss_correlations_from_luma(y, key)
    text = decoder(corrs0)
    if text is not None:
        repeats = _repeats_at(len(corrs0), text)
        return RescaleMatch(text, 1.0, _plain_decode_confidence(repeats), repeats, 1, False)

    y32 = y.astype(np.float32)
    pn = spread._ss_pn_vector(key, len(spread.SS_COEFF_POSITIONS))
    tried = []  # refined scale estimates already decoded, to skip repeats across tiers
    budget_exhausted = False

    for patch, log_step, min_z, require_full_coverage in _SS_SWEEP_TIERS:
        if _deadline_hit(deadline):
            budget_exhausted = True
            break

        peaks, hit = _ss_sweep_peaks(y32, pn, scale_range, patch, log_step, min_z, max_candidates, deadline,
                                     require_full_coverage)
        budget_exhausted = budget_exhausted or hit

        for f0, z in peaks:
            if _deadline_hit(deadline):
                budget_exhausted = True
                break
            f1 = _ss_refine_scale(y32, pn, f0, deadline)
            if any(abs(f1 / t - 1.0) < 0.003 for t in tried):
                continue
            tried.append(f1)

            # Full-image decodes at a few widths around the estimate, nearest first.
            w_est = w * f1
            reach = max(2, int(math.ceil(0.0008 * w_est)))
            widths = sorted({int(round(w_est)) + d for d in range(-reach, reach + 1)},
                            key=lambda tw: abs(tw - w_est))
            votes = {}
            for tw in widths:
                if _deadline_hit(deadline):
                    budget_exhausted = True
                    break
                th = int(round(h * tw / w))
                if tw < spread.SS_BLOCK or th < spread.SS_BLOCK:
                    continue
                interp = cv2.INTER_AREA if tw < w else cv2.INTER_LANCZOS4
                resized = cv2.resize(y32, (tw, th), interpolation=interp).astype(np.float64)
                corrs = spread._ss_correlations_from_luma(resized, key)
                cand_text = decoder(corrs)
                if cand_text is None or not _printable_payload(cand_text):
                    continue
                votes.setdefault(cand_text, []).append((tw, len(corrs)))
                if len(votes[cand_text]) >= 2:
                    hits = votes[cand_text]
                    factor = float(np.mean([tw for tw, _ in hits])) / w
                    repeats = _repeats_at(hits[-1][1], cand_text)
                    confidence = _rescale_confidence(z, min_z, repeats, len(hits))
                    return RescaleMatch(cand_text, factor, confidence, repeats, len(hits), False)

            if budget_exhausted:
                break
        if budget_exhausted:
            break

    return _no_match(budget_exhausted)
