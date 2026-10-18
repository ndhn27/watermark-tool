"""Spread-spectrum decode of a MIRRORED copy (left-right, top-bottom, or both).

The mark (spread.py) is laid out in raster order over the ORIGINAL image's 8x8 block grid. Mirror the picture
- one click in any photo app, and a common repost trick - and the block order is reversed and every block is
reflected, so the plain decoder reads noise. Nothing is lost, though: a mirror is exactly its own inverse, so
mirroring the suspect image back restores the original grid pixel for pixel (any width or height, multiple of 8
or not), and the ordinary decoder then reads the mark as if nothing had happened. Measured before this module
existed (bench/results/P3-NOTES.md): the oracle bit error rate on mirrored copies was 0.000 - the signal was
fully intact, only the decoder did not look at it the right way round - while detection was 0/240 (synthetic)
and 0/35 (7 photos).

So the search is just "undo each of the four possible mirrorings (none, left-right, top-bottom, both) and decode".
What needs care is what counts as a hit, because four attempts are four chances to be fooled:

- Mirrored orientations are searched for a watermark ID ONLY (spread.find_spread_spectrum_id): a canonical,
  check-valid 10-character ID, false-accept probability ~1e-10 per orientation on an image with no mark. The
  free-text decoder's accept rule is far looser ("valid UTF-8, printable"); running it on three more
  orientations would multiply its junk rate, so it is deliberately not run on them - a mirrored TEXT payload is
  not recovered (measured: bench/results/P3-NOTES.md).
- The ID search goes over ALL orientations before the text decoder is consulted at all. The other way round,
  the text decoder on the unmirrored orientation - which returned a wrong string for 3 of 240 mirrored
  ID-marked images before this module - could win before the correct orientation was ever tried (tests/
  test_orient.py forces exactly that and pins the order).
- A text payload on an unmirrored image reads exactly as before: same decoder, same order, same result.

The un-mirrored orientation is always tried first. A marked, unmirrored ID is answered by that first attempt (no
extra time: p50 25 ms vs 24 ms in the bench); anything else - no mark, a text payload, a mirrored copy - pays for up
to three more luma flips + block DCTs + 112-phase sums, about 3-4x the plain decode's time on an image with no mark
(38 -> 131 ms at 640x480, 1.8 s -> 5.2 s at 12 MP).
"""

from typing import NamedTuple, Optional

import numpy as np

from . import spread

# What was done to the image (which the decoder undoes): "h" = left-right, "v" = top-bottom, "hv" = both
# (a 180-degree turn). "none" first: an image that was not mirrored is answered on the first attempt.
ORIENTATIONS = ("none", "h", "v", "hv")

ORIENTATION_LABELS = {
    "none": "not mirrored",
    "h": "mirrored left-right",
    "v": "mirrored top-bottom",
    "hv": "mirrored both ways (turned 180 degrees)",
}


class OrientedMatch(NamedTuple):
    text: Optional[str]
    orientation: Optional[str]      # one of ORIENTATIONS when text is not None, else None


NO_MATCH = OrientedMatch(None, None)


def unflip(arr, orientation):
    """Undo a mirroring on an HxW or HxWxC array (a view - callers that need contiguity copy it). A flip
    is its own inverse, so this is also how a flip is applied."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"unknown orientation {orientation!r}")
    if "h" in orientation:
        arr = arr[:, ::-1]
    if "v" in orientation:
        arr = arr[::-1]
    return arr


def _luma(rgb_image):
    rgb = np.asarray(rgb_image.convert("RGB") if hasattr(rgb_image, "convert") else rgb_image, dtype=np.float64)
    y, _, _ = spread._rgb_to_y(rgb)
    return y


def find_spread_spectrum_id_oriented(rgb_image, key, orientations=ORIENTATIONS):
    """A watermark ID from `rgb_image` as it is or mirrored, or NO_MATCH. `orientations` are tried in order and
    the first that holds a check-valid ID wins (a wrong orientation never yields one, bar ~1e-10 per try)."""
    y = _luma(rgb_image)
    for orientation in orientations:
        corrs = spread._ss_correlations_from_luma(np.ascontiguousarray(unflip(y, orientation)), key)
        wm_id = spread.find_spread_spectrum_id(corrs)
        if wm_id is not None:
            return OrientedMatch(wm_id, orientation)
    return NO_MATCH


def find_spread_spectrum_payload_or_id_oriented(rgb_image, key, max_length=10000, orientations=ORIENTATIONS):
    """What a reader that doesn't know what it is looking for should call, now mirror-aware: a watermark ID in
    any orientation if there is one; otherwise whatever the plain text decoder finds on the image AS GIVEN
    (never on a mirrored orientation - see the module docstring for why). With orientations=("none",) this is
    spread.find_spread_spectrum_payload_or_id(), result for result.
    """
    y = _luma(rgb_image)
    corrs_plain = None
    for orientation in orientations:
        corrs = spread._ss_correlations_from_luma(np.ascontiguousarray(unflip(y, orientation)), key)
        if orientation == "none":
            corrs_plain = corrs
        wm_id = spread.find_spread_spectrum_id(corrs)
        if wm_id is not None:
            return OrientedMatch(wm_id, orientation)
    if "none" in orientations and corrs_plain is not None:
        text = spread.find_spread_spectrum_payload(corrs_plain, max_length=max_length)
        if text is not None:
            return OrientedMatch(text, "none")
    return NO_MATCH
