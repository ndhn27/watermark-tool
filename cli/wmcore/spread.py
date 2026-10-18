"""Invisible backup mark v2: block-DCT spread-spectrum, in the frequency
domain rather than raw pixel LSBs (see invisible.py). One payload bit is
spread across several mid-frequency DCT coefficients of every 8x8 pixel
block - the same block grid JPEG itself uses - so it survives ordinary JPEG
recompression and a block-aligned crop far better than a pixel-LSB mark
does (Cox et al., "Secure Spread Spectrum Watermarking for Multimedia",
1997, adapted here for a multi-byte payload rather than one presence bit).

Two design choices worth flagging:

- Every block uses the SAME fixed +-1 spreading sequence (derived once from
  `key`), not one derived per block from its row/column. A per-block
  sequence needs the block's absolute position in the *original* image to
  decode, which a crop breaks (an earlier version did this and silently
  failed to decode any cropped image). A single fixed sequence has no such
  dependency, at the cost of being more amenable to averaging by a
  motivated adversary than a randomized one.
- Decoding is soft, not hard-then-matched: find_spread_spectrum_payload()
  sums the raw per-block correlation across every repeat of a candidate
  period *before* making any bit decision ("processing gain" - summing R
  noisy observations improves effective SNR by roughly sqrt(R)), which is
  what makes recompression survival realistic even though a single
  repeat's raw bit error rate can be well into double digits.

`key` plays the same role --seed plays for the visible grid's jitter: pass
the same key to decode later, and vary it per file so files don't share an
identical, averageable pattern. (Extract's crop re-alignment search, wmcore/resync.py, retries the decode on a few
shifted block grids and so lets a wrong key read the ID much more often - bench/results/CROP-NOTES.md §5.)
It is NOT a secrecy mechanism - because
every block shares one fixed sequence and decoding sums correlations
across repeats, a uniformly random wrong key has, measured on real images,
roughly a 1-in-4 chance of still decoding the payload correctly. (For an ID payload read with
find_spread_spectrum_id() below it is higher - about 3 in 10 on synthetic images, bench/results/P2-NOTES.md.)
"""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

from . import wmid
from .primitives import text_to_bits

SS_BLOCK = 8

_SS_ZIGZAG = sorted(
    ((r, c) for r in range(SS_BLOCK) for c in range(SS_BLOCK)),
    key=lambda rc: rc[0] + rc[1],
)
# Mid-band coefficients only: skip the DC term and lowest frequencies
# (perturbing those visibly shifts local brightness/color), and skip the
# highest frequencies too (exactly what JPEG quantizes hardest).
SS_COEFF_POSITIONS = _SS_ZIGZAG[10:26]
_SS_ROWS = np.array([p[0] for p in SS_COEFF_POSITIONS])
_SS_COLS = np.array([p[1] for p in SS_COEFF_POSITIONS])

DEFAULT_SS_ALPHA = 12.0

# Video gets its own, higher default. Measured (bench/results/P4-NOTES.md): embedding a frame with the
# SAME embed_spread_spectrum() at the image default (12.0), then re-encoding with real libx264 at CRF 23
# (x264's OWN default quality, not an aggressive attack) only decoded ~65-70% of frames back. Raising
# alpha to 18.0 closed that to 100% in the same test; 20.0 here for a bit of headroom. Video gives many
# decode attempts (one per frame) where a still image gives one, so this isn't trying to survive heavy
# recompression (CRF 32+) on its own - see P4-NOTES.md §4 on why per-frame % isn't the number that matters
# for video the way it is for a still image.
DEFAULT_SS_ALPHA_VIDEO = 20.0


def _dct_basis_matrix(n=SS_BLOCK):
    """Orthonormal n x n type-II DCT basis matrix C: for an n x n pixel
    block B, the 2D DCT is C @ B @ C.T and the inverse is C.T @ D @ C. Built
    as an explicit matrix so every block in an image can be transformed in
    one vectorized `einsum` call (see _image_to_ss_blocks).
    """
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    c = np.sqrt(2.0 / n) * np.cos(np.pi / n * (i + 0.5) * k)
    c[0, :] *= 1.0 / np.sqrt(2.0)
    return c


_SS_C = _dct_basis_matrix()
_SS_CT = _SS_C.T


def _fmix32(h):
    """32-bit integer hash finalizer (MurmurHash3's fmix32 mixing step) -
    good avalanche behavior from plain arithmetic, which is all a spreading-
    sequence generator needs (this is not a cryptographic primitive). All
    arithmetic is uint32 and wraps on overflow by design.
    """
    h = h.astype(np.uint32)
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x85EBCA6B)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0xC2B2AE35)
    h ^= h >> np.uint32(16)
    return h


def _ss_pn_vector(key, n):
    """The one fixed +-1 spreading sequence (length n) used in every block,
    derived from `key` alone (see the module docstring for why one shared
    sequence rather than a per-block one).
    """
    idx = np.arange(n, dtype=np.uint32)
    key_u = np.full(n, np.uint32(int(key) & 0xFFFFFFFF), dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = _fmix32(key_u ^ _fmix32(idx * np.uint32(2654435761)))
    return np.where((h & np.uint32(1)) == 0, 1.0, -1.0)


def _rgb_to_y(rgb_float):
    """ITU-R BT.601 luma from an HxWx3 float array (0..255 range). Only Y is
    used for embedding/decoding - chroma is typically subsampled by JPEG,
    so a mark living there would be even more fragile than one in Y.
    """
    r, g, b = rgb_float[..., 0], rgb_float[..., 1], rgb_float[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return y, cb, cr


def _ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.stack([r, g, b], axis=-1)


def _image_to_ss_blocks(y):
    """Split a 2D luma array into non-overlapping SS_BLOCK x SS_BLOCK tiles,
    dropping leftover rows/cols that don't fill a full block (mirroring how
    JPEG itself handles non-multiple-of-8 dimensions). Returns
    (blocks, block_rows, block_cols).
    """
    h, w = y.shape
    bh, bw = h // SS_BLOCK, w // SS_BLOCK
    trimmed = y[: bh * SS_BLOCK, : bw * SS_BLOCK]
    blocks = trimmed.reshape(bh, SS_BLOCK, bw, SS_BLOCK).transpose(0, 2, 1, 3)
    return blocks, bh, bw


def _ss_blocks_to_image(blocks):
    bh, bw = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(bh * SS_BLOCK, bw * SS_BLOCK)


def embed_spread_spectrum(base_rgba, text, key, alpha=DEFAULT_SS_ALPHA):
    """Embed `text` as a block-DCT spread-spectrum mark in the luma channel.
    `key` must be reused unchanged when decoding (see module docstring).
    Unlike embed_invisible(), this survives ordinary JPEG recompression and
    a block-aligned crop, at the cost of needing a reasonably sized image
    and being slower to embed/decode.

    Raises ValueError if the image doesn't have enough 8x8 blocks to embed
    even a single copy of the payload.
    """
    rgb = np.asarray(base_rgba.convert("RGB"), dtype=np.float64)
    y, cb, cr = _rgb_to_y(rgb)
    blocks, bh, bw = _image_to_ss_blocks(y)
    n_blocks = bh * bw

    bits = text_to_bits(text)
    if not bits:
        return base_rgba
    if n_blocks < len(bits):
        raise ValueError(
            f"Image too small for a spread-spectrum payload of this length: "
            f"only {n_blocks} usable 8x8 blocks ({bh * SS_BLOCK}x{bw * SS_BLOCK}px) "
            f"but {len(bits)} bits are needed. Use a larger image or shorter text."
        )
    reps = max(1, n_blocks // len(bits))
    tiled_bits = np.array((bits * (reps + 1))[:n_blocks], dtype=np.float64).reshape(bh, bw)
    sign = np.where(tiled_bits > 0, 1.0, -1.0)

    dct_blocks = np.einsum("ij,abjk,kl->abil", _SS_C, blocks, _SS_CT)
    pn = _ss_pn_vector(key, len(SS_COEFF_POSITIONS))
    dct_blocks[:, :, _SS_ROWS, _SS_COLS] += alpha * sign[..., None] * pn[None, None, :]
    y_mod = _ss_blocks_to_image(np.einsum("ij,abjk,kl->abil", _SS_CT, dct_blocks, _SS_C))

    y_full = y.copy()
    y_full[: bh * SS_BLOCK, : bw * SS_BLOCK] = y_mod
    out_rgb = _ycbcr_to_rgb(y_full, cb, cr)
    out_rgb = np.clip(out_rgb + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out_rgb, "RGB").convert("RGBA")


def spread_spectrum_block_correlations(rgb_image, key):
    """The raw per-block correlation against `key`'s spreading sequence, one
    float per 8x8 block in raster order - the "soft" signal
    find_spread_spectrum_payload() combines across repeats before making
    any hard bit decision. Split out so callers/tests can inspect the raw
    signal directly.
    """
    rgb = np.asarray(rgb_image.convert("RGB") if hasattr(rgb_image, "convert") else rgb_image, dtype=np.float64)
    y, _, _ = _rgb_to_y(rgb)
    return _ss_correlations_from_luma(y, key)


def _ss_correlations_from_luma(y, key):
    """spread_spectrum_block_correlations() minus the RGB->luma step, so
    rescale.py can resample luma once and reuse this directly."""
    blocks, bh, bw = _image_to_ss_blocks(y)
    dct_blocks = np.einsum("ij,abjk,kl->abil", _SS_C, blocks, _SS_CT)
    pn = _ss_pn_vector(key, len(SS_COEFF_POSITIONS))
    coeffs = dct_blocks[:, :, _SS_ROWS, _SS_COLS]
    return (coeffs * pn[None, None, :]).sum(axis=-1).reshape(-1)


def spread_spectrum_block_correlations_multi(rgb_image, keys, chunk=32):
    """spread_spectrum_block_correlations() for many keys at once, as a generator of
    (key, corrs) in the order given. The expensive part - RGB->luma and the DCT of every
    block - doesn't depend on the key, and neither does the +-1 spreading sequence's
    shape (one fixed sequence per key, applied to every block), so the per-key work is
    a single 16-element dot product per block: computed for `chunk` keys per matrix
    multiply (bounded memory on a big image) after one shared DCT. On a 12 MP photo the
    DCT is ~1.5 s and each further key costs milliseconds - the difference between
    scanning a whole ledger's keys and giving up after a handful.
    """
    rgb = np.asarray(rgb_image.convert("RGB") if hasattr(rgb_image, "convert") else rgb_image, dtype=np.float64)
    y, _, _ = _rgb_to_y(rgb)
    blocks, bh, bw = _image_to_ss_blocks(y)
    dct_blocks = np.einsum("ij,abjk,kl->abil", _SS_C, blocks, _SS_CT)
    coeffs = dct_blocks[:, :, _SS_ROWS, _SS_COLS].reshape(bh * bw, len(SS_COEFF_POSITIONS))
    keys = list(keys)
    for start in range(0, len(keys), chunk):
        group = keys[start:start + chunk]
        pn = np.stack([_ss_pn_vector(k, len(SS_COEFF_POSITIONS)) for k in group], axis=1)
        corr = coeffs @ pn
        for j, key in enumerate(group):
            yield key, np.ascontiguousarray(corr[:, j])


def _printable_payload(text):
    """True if every character is printable, or one of the three control
    characters real text payloads legitimately use for layout (tab,
    newline, CR). Also used by rescale.py's resize search; kept here since
    find_spread_spectrum_payload() below is the other, more fundamental,
    place a decoded payload needs this check.
    """
    return all(ch.isprintable() or ch in "\t\n\r" for ch in text)


def find_spread_spectrum_payload(corrs, max_length=10000):
    """Decode a payload from per-block correlation values, combining every
    repeat of a candidate period before thresholding (see module docstring
    on why this matters - a single repeat can have a double-digit percent
    raw bit error rate and still decode cleanly once enough repeats are
    summed).

    Candidate (offset, period) pairs are bootstrapped the same way
    extract_invisible.py's LSB decoder finds candidates: every 32-bit window
    of the hard-thresholded correlations is tried as a length header.
    Whichever candidate combines the most repeats, decodes as valid UTF-8,
    AND is printable text (see _printable_payload - allows tab/newline/CR)
    wins, on the theory that the genuine mark has far more repeats
    available to combine than a spurious one found by chance.

    The printable check matters even here, not just in the rescale search:
    measured on 10 hand-made periodic (never-stamped) test images, this
    plain, unresized decode alone accepted control-byte runs (NUL, repeated
    0x55, ...) as "found" on 8/90 trials before this check was added, all
    valid UTF-8 but nothing a real text payload would ever contain (see
    bench/results/P2-NOTES.md §6.6 and bench/experiments/
    plain_decode_printable.py). A real payload with legitimate control
    characters (\\t \\n \\r) still round-trips - checked in the same script
    and in tests/test_spread_spectrum.py.
    """
    corrs = np.asarray(corrs, dtype=np.float64)
    n = len(corrs)
    if n < 32:
        return None

    hard_bits = (corrs > 0).astype(np.uint32)
    windows = sliding_window_view(hard_bits, 32)
    weights = (1 << np.arange(31, -1, -1)).astype(np.uint32)
    lengths = windows @ weights
    candidate_offsets = np.flatnonzero((lengths > 0) & (lengths <= max_length))

    best = None  # (num_repeats_combined, text)
    tried_periods = set()
    for offset in candidate_offsets:
        offset = int(offset)
        length = int(lengths[offset])
        period = 32 + length * 8
        if period in tried_periods or period > n:
            continue
        tried_periods.add(period)

        phase = offset % period
        num_repeats = (n - phase) // period
        if num_repeats < 1:
            continue
        usable = phase + num_repeats * period
        combined = corrs[phase:usable].reshape(num_repeats, period).sum(axis=0)
        combined_bits = (combined > 0).astype(np.uint8)

        header_bits = combined_bits[:32]
        decoded_length = int(np.packbits(header_bits).view(">u4")[0])
        if not (0 < decoded_length <= max_length) or 32 + decoded_length * 8 > period:
            continue
        payload_bits = combined_bits[32 : 32 + decoded_length * 8]
        try:
            text = np.packbits(payload_bits).tobytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not _printable_payload(text):
            continue

        if best is None or num_repeats > best[0]:
            best = (num_repeats, text)

    return best[1] if best else None


# A --id payload has a fixed shape - 4-byte length header + WM_ID_LENGTH ASCII
# characters (wmid.py) - so its period on the wire is known in advance.
ID_PERIOD_BITS = 32 + 8 * wmid.WM_ID_LENGTH


def find_spread_spectrum_id(corrs):
    """Decode a watermark ID (wmid.py) from per-block correlations, using the
    fact that an ID's period is known: 112 bits, always.

    find_spread_spectrum_payload() has to discover the period, and it does so
    by hoping that at least one single repeat of the 32-bit length header
    survives *bit-exact* in the hard-thresholded raw correlations. That is a
    much stricter condition than the one that actually limits recovery: with a
    raw bit error rate of 20%, all 32 header bits of one repeat are right with
    probability 0.8**32 (~0.08%), yet the sum over dozens of repeats can still
    be perfectly clean. The bench calls this "signal intact, decoder loses
    sync" (bench/results/P1-NOTES.md) - combined BER 0.000 and no payload.

    Here nothing needs to be discovered. For each of the 112 possible phases
    (where in the raster the period starts - a top crop or an offset shifts
    it), sum every repeat, hard-threshold the 80 payload bits, and accept a
    phase only if those bytes are a *canonical, check-valid ID* - the same
    strict shape extract already demands. The header bits are deliberately
    not used: a bit error there would only cost a decode the payload alone
    can still make. A random 80-bit string is a valid ID with probability
    ~2**-40 (each byte must land in the 32-character alphabet: 2**-30, then
    10 check bits), so over 112 phases a false accept on a mark-free image is
    ~1e-10 - unlike find_spread_spectrum_payload(), whose accept rule is only
    "valid UTF-8".

    Returns the ID (canonical form) or None. Only usable when the payload IS
    an ID; on any other payload it simply returns None.
    """
    corrs = np.asarray(corrs, dtype=np.float64)
    n = len(corrs)
    period = ID_PERIOD_BITS
    if n < period:
        return None

    best = None  # (confidence, id)
    for phase in range(period):
        repeats = (n - phase) // period
        if repeats < 1:
            continue
        combined = corrs[phase:phase + repeats * period].reshape(repeats, period).sum(axis=0)
        payload = combined[32:]
        try:
            text = np.packbits((payload > 0).astype(np.uint8)).tobytes().decode("ascii")
        except UnicodeDecodeError:
            continue
        if wmid.parse(text) != text:          # canonical form only (parse() is lenient on purpose)
            continue
        confidence = float(np.abs(payload).sum())
        if best is None or confidence > best[0]:
            best = (confidence, text)
    return best[1] if best else None


def find_spread_spectrum_payload_or_id(corrs, max_length=10000):
    """What a reader that doesn't know what it is looking for should call: a watermark ID if the
    correlations hold one, otherwise whatever find_spread_spectrum_payload() finds.

    The ID decoder goes first because its accept rule is far stricter (a check-valid, canonical ID, versus
    "valid UTF-8"), and a payload that is not an ID always falls through to the legacy decoder unchanged - so
    text payloads read back exactly as before, and an ID that the legacy decoder would have missed, or garbled,
    is now read. Measured: bench/results/P2-NOTES.md.
    """
    wm_id = find_spread_spectrum_id(corrs)
    return wm_id if wm_id is not None else find_spread_spectrum_payload(corrs, max_length=max_length)
