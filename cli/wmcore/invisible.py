"""Invisible backup mark: LSB payload in the red channel (embed here, and the
read-back for it - plus the file-path read-back wrappers for the spread-spectrum
mark - which used to live in cli/extract_invisible.py and moved down so the
application layer (wmapp) and that CLI can both import them without a cycle;
extract_invisible.py still re-exports every name). Fast and needs no
key, but any recompression or resize destroys it - it lives in the
least-significant bit of raw pixel values, exactly the noise a lossy
encoder is designed to throw away. See spread.py for a mark that survives
JPEG recompression.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

from .orient import ORIENTATIONS, find_spread_spectrum_payload_or_id_oriented, unflip
from .primitives import text_to_bits
from .rescale import DEFAULT_RESCALE_BUDGET_SECONDS, SS_RESCALE_RANGE, find_spread_spectrum_payload_rescaled
from .spread import (
    find_spread_spectrum_id, find_spread_spectrum_payload_or_id, spread_spectrum_block_correlations,
    spread_spectrum_block_correlations_multi,
)


def embed_invisible(base_rgba, text):
    """Spread `text` across the low bits of the R channel, repeated to fill
    the image (so it survives a crop that keeps a reasonably large region).
    Lossless output only (save as PNG) - JPEG recompression destroys this.
    """
    arr = np.asarray(base_rgba.convert("RGB"), dtype=np.uint8).copy()
    flat = arr[..., 0].reshape(-1)  # red channel, flattened
    bits = text_to_bits(text)
    if not bits:
        return base_rgba
    reps = max(1, len(flat) // len(bits))
    tiled_bits = (bits * (reps + 1))[: len(flat)]
    bit_arr = np.array(tiled_bits, dtype=np.uint8)
    flat[:] = (flat & 0xFE) | bit_arr
    arr[..., 0] = flat.reshape(arr[..., 0].shape)
    out = Image.fromarray(arr, "RGB")
    return out.convert("RGBA")


# Upper bound on a plausible decoded payload length (bytes), used only to
# cut down which 32-bit windows are worth trying as a header - not a limit
# on what embed_invisible() can write. Generous on purpose: raising it
# finds longer payloads at the cost of a few more (cheap, self-rejecting)
# candidates to check.
MAX_PLAUSIBLE_LENGTH = 10_000


def _decode_at(bits, offset, length, n):
    """Decode `length` payload bytes starting 32 bits after `offset`, or
    None if that range doesn't fit or isn't valid UTF-8."""
    start = offset + 32
    end = start + length * 8
    if end > n:
        return None
    payload_bits = np.asarray(bits[start:end], dtype=np.uint8)
    byte_vals = np.packbits(payload_bits)
    try:
        return byte_vals.tobytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def _find_payload(bits, max_length=MAX_PLAUSIBLE_LENGTH):
    """Scan the flattened LSB bit-stream for a valid header+payload at ANY
    offset, not just position 0.

    embed_invisible() repeats (32-bit length header + payload bits)
    back-to-back for as many full copies as fit in the image, so a crop or
    paint-over that destroys the copy at offset 0 (the top-left corner)
    still leaves later copies intact elsewhere - as long as something
    actually looks for them. This does, by treating every offset as a
    candidate header start rather than assuming the first one is the
    only one worth trying.

    Vectorized so it stays fast on real image sizes: every 32-bit window
    is decoded as a big-endian length in one pass, and offsets whose
    "length" isn't plausible are discarded immediately.

    A raw "does 32 bits + a plausible length decode as UTF-8" test is too
    weak on its own - across a multi-megapixel image, ordinary pixel noise
    throws up occasional short byte sequences that happen to be valid
    UTF-8 by chance (single ASCII-range bytes especially). Because the
    real mark repeats with a fixed period (32 + length*8 bits), a genuine
    copy can be corroborated by checking whether the *next* (or previous)
    period decodes to the exact same text - something random pixel data
    essentially never does by coincidence. Every candidate is checked for
    that corroboration first; only if nothing in the image corroborates
    does this fall back to the single best-looking candidate, same as
    before.
    """
    n = len(bits)
    if n < 32:
        return None

    bits_u32 = np.asarray(bits, dtype=np.uint32)
    windows = sliding_window_view(bits_u32, 32)
    weights = (1 << np.arange(31, -1, -1)).astype(np.uint32)
    lengths = windows @ weights  # decoded 32-bit length at every offset

    candidate_offsets = np.flatnonzero((lengths > 0) & (lengths <= max_length))

    fallback = None  # best single, uncorroborated candidate - last resort
    for offset in candidate_offsets:
        offset = int(offset)
        length = int(lengths[offset])
        text = _decode_at(bits, offset, length, n)
        if text is None:
            continue
        if fallback is None:
            fallback = text

        period = 32 + length * 8
        next_offset = offset + period
        if next_offset + 32 <= n and _decode_at(bits, next_offset, length, n) == text:
            return text  # a second intact copy confirms this is the real mark
        prev_offset = offset - period
        if prev_offset >= 0 and _decode_at(bits, prev_offset, length, n) == text:
            return text

    return fallback


def extract_invisible(path, max_length=MAX_PLAUSIBLE_LENGTH):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    bits = (arr[..., 0].reshape(-1) & 1)
    return _find_payload(bits, max_length=max_length)


def extract_spread_spectrum(path, key, max_length=MAX_PLAUSIBLE_LENGTH):
    """Read back a block-DCT spread-spectrum payload embedded by
    watermark.py --invisible --invisible-mode spread --ss-key KEY. `key`
    must be the exact same value used at embed time (see the module-level
    comment above wmcore.embed_spread_spectrum() for what that key does
    and doesn't guarantee). Works on a JPEG just as well as a PNG - that's
    the whole point of this mode over the default LSB one.
    """
    img = Image.open(path).convert("RGB")
    corrs = spread_spectrum_block_correlations(img, key)
    return find_spread_spectrum_payload_or_id(corrs, max_length=max_length)


def iter_spread_spectrum_payloads(path, keys, max_length=MAX_PLAUSIBLE_LENGTH):
    """extract_spread_spectrum() for a list of candidate keys, lazily: yields (key, text_or_None)
    per key in order, sharing one DCT of the image between all of them (see
    spread_spectrum_block_correlations_multi), so a caller can stop at the first hit. A key
    that isn't the right one usually decodes to None, but - see the module comment above
    embed_spread_spectrum() - not always: the caller decides what counts as a real payload
    (wmapp.extract accepts only a well-formed watermark ID).
    """
    img = Image.open(path).convert("RGB")
    for key, corrs in spread_spectrum_block_correlations_multi(img, keys):
        yield key, find_spread_spectrum_payload_or_id(corrs, max_length=max_length)


def extract_spread_spectrum_oriented(path, key, max_length=MAX_PLAUSIBLE_LENGTH, orientations=ORIENTATIONS):
    """extract_spread_spectrum() that also looks at a MIRRORED copy (left-right, top-bottom, both): returns an
    orient.OrientedMatch(text, orientation), `orientation` being what was done to the image ("none", "h",
    "v", "hv"). A watermark ID is read in any orientation; a free-text payload only unmirrored, exactly as
    extract_spread_spectrum() reads it - see wmcore/orient.py's docstring for why and for what was measured.
    `orientations` limits which are tried (e.g. only the mirrored ones: ("h", "v", "hv")).
    """
    img = Image.open(path).convert("RGB")
    return find_spread_spectrum_payload_or_id_oriented(img, key, max_length=max_length, orientations=orientations)


def iter_spread_spectrum_payloads_oriented(path, keys, max_length=MAX_PLAUSIBLE_LENGTH, orientations=ORIENTATIONS):
    """iter_spread_spectrum_payloads() with mirroring: yields (key, text_or_None, orientation), one pass per
    orientation over all `keys` (the unmirrored pass first and exactly as before, then one ID-only pass per
    mirroring). Each pass shares one DCT between its keys and the next pass's DCT is only computed if the caller
    keeps iterating, so an unmirrored image found on the first pass costs nothing extra. Like the unoriented
    iterator, what counts as a real payload is the caller's call (wmapp.extract accepts only a watermark ID).
    """
    img = Image.open(path).convert("RGB")
    keys = list(keys)
    for orientation in orientations:
        if orientation == "none":
            source = img
            def read(corrs):
                return find_spread_spectrum_payload_or_id(corrs, max_length=max_length)
        else:
            source = np.ascontiguousarray(unflip(np.asarray(img, dtype=np.uint8), orientation))
            read = find_spread_spectrum_id
        for key, corrs in spread_spectrum_block_correlations_multi(source, keys):
            yield key, read(corrs), orientation


def extract_spread_spectrum_rescaled(path, key, max_length=MAX_PLAUSIBLE_LENGTH,
                                     scale_range=SS_RESCALE_RANGE,
                                     max_seconds=DEFAULT_RESCALE_BUDGET_SECONDS, decoder=None):
    """extract_spread_spectrum() plus a search for a resize applied after
    embedding. Returns a wmcore.RescaleMatch: `factor` is the decode-time
    resize that recovered the payload (1.0 = none needed), `confidence` is a
    [0, 1] heuristic index (NOT a calibrated probability - see the "RESULT
    CONFIDENCE" section of wmcore/rescale.py's module docstring), and
    `budget_exhausted` tells a clean "not found" apart from "ran out of time
    before finishing" (see `max_seconds`). Slower than the plain decode
    (typically a few seconds per megapixel-class image when the image was
    resized; more when nothing is found, up to `max_seconds`); see the
    comment block above find_spread_spectrum_payload_rescaled() in
    wmcore/rescale.py for how it works and what it can't recover.

    `decoder(corrs) -> text | None` chooses how a payload is read at every
    width the search tries. Default: an ID if the correlations hold one, else
    the text decoder (find_spread_spectrum_payload_or_id) - the same reader as
    extract_spread_spectrum(), so text payloads behave as before and IDs are
    also read where the text decoder alone loses them (bench/results/
    P2-NOTES.md, "Inside --rescale"). Pass wmcore.find_spread_spectrum_id to
    read IDs only.
    """
    img = Image.open(path).convert("RGB")
    if decoder is None:
        def decoder(corrs):
            return find_spread_spectrum_payload_or_id(corrs, max_length=max_length)
    return find_spread_spectrum_payload_rescaled(
        img, key, scale_range=scale_range, max_length=max_length, max_seconds=max_seconds, decoder=decoder
    )
