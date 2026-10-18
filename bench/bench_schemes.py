"""Adapters between the benchmark and the tool's embedders/decoders.

Two SCHEMES (how a mark is written) and several DECODERS (how it is read):

  scheme "lsb"  wmcore.embed_invisible                 decoder "lsb"
  scheme "ss"   wmcore.embed_spread_spectrum           decoder "ss"          (plain, no resize search)
                                                       decoder "ss_rescale"  (plain + --rescale search)
                                                       decoder "ss_id"       (plain, for --id payloads:
                                                                              fixed 112-bit period, no header hunt)
                                                       decoders "ss_flip", "ss_id_flip": `ss` / `ss_id` that also
                                                                             undo a mirroring (wmcore/orient.py)

`ss_id` only means something on a payload that IS a watermark ID, so it is only
run against the "wmid" payload (DECODER_PAYLOAD) - a fresh, check-valid ID per
image, derived from the seed. It is compared with plain `ss` on the very same
attacked images: run `--decoders ss,ss_id --payloads wmid`.

Decoders call exactly what extract_invisible.py calls, minus file I/O (a PNG/
JPEG save-and-reload adds nothing but time; the attacks apply the lossy step
themselves). The soft observations used for BER are recomputed here with the
ORACLE geometry of the attack (see bench_attacks.Geometry) so BER reflects
how much signal is left, not whether the decoder found it.
"""

import math
import os
import random
import sys

_CLI = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cli"))
if _CLI not in sys.path:
    sys.path.insert(0, _CLI)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from typing import NamedTuple, Optional  # noqa: E402

import wmcore  # noqa: E402
from extract_invisible import _find_payload  # noqa: E402  (the LSB decoder's core)

PAYLOADS = {
    # 10 B, ID-SHAPED but NOT check-valid (its check characters are wrong; wmid.parse() rejects it) - fine as a
    # length-only stand-in, which is all P1's payload-length runs needed, useless to any decoder that verifies IDs.
    "id": "K3M9QXR2T4",
    # A real, check-valid watermark ID, generated per image from the seed (resolve_payload) rather than fixed, so
    # different images exercise different bit patterns. This is the payload `ss_id` needs.
    "wmid": None,
    "short": "@yourhandle",                                                    # 11 B, matches the unit tests
    "medium": "(c) 2026 Example Studio - all rights reserved",                 # 45 B
    "long": "(c) 2026 Example Studio - all rights reserved - contact licensing@example.com - ref 2026-09-19-0001",  # 99 B
}


# Decoders that only make sense on one payload kind (a fixed-shape decoder run on free text would just report
# "none" and look like a failure). Decoders absent from here run on every payload.
DECODER_PAYLOAD = {"ss_id": "wmid", "ss_id_rescale": "wmid", "ss_id_flip": "wmid"}


def resolve_payload(payload_id, uid, seed):
    """The payload text for a unit. Fixed payloads are looked up; "wmid" is a fresh valid ID, deterministic in
    (seed, uid) so a re-run reproduces it exactly."""
    if payload_id == "wmid":
        return wmcore.wmid.generate(random.Random(f"{seed}|{uid}|wmid"))
    return PAYLOADS[payload_id]


def decoder_applies(decoder, payload_id):
    return DECODER_PAYLOAD.get(decoder, payload_id) == payload_id


class Decoded(NamedTuple):
    text: Optional[str]
    factor: Optional[float] = None
    confidence: Optional[float] = None
    repeats: Optional[int] = None
    budget_exhausted: bool = False


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(scheme, base_rgba, payload, key, alpha):
    if scheme == "ss":
        return wmcore.embed_spread_spectrum(base_rgba, payload, key, alpha=alpha)
    if scheme == "lsb":
        return wmcore.embed_invisible(base_rgba, payload)
    raise ValueError(f"unknown scheme {scheme!r}")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_lsb(rgb, key, budget=None):
    arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    return Decoded(_find_payload(arr[..., 0].reshape(-1) & 1))


def decode_ss(rgb, key, budget=None):
    corrs = wmcore.spread_spectrum_block_correlations(rgb, key)
    return Decoded(wmcore.find_spread_spectrum_payload(corrs))


def decode_ss_rescale(rgb, key, budget=wmcore.DEFAULT_RESCALE_BUDGET_SECONDS):
    m = wmcore.find_spread_spectrum_payload_rescaled(rgb, key, max_seconds=budget)
    return Decoded(m.text, m.factor, m.confidence if m.text is not None else None,
                   m.repeats, m.budget_exhausted)


def decode_ss_id(rgb, key, budget=None):
    corrs = wmcore.spread_spectrum_block_correlations(rgb, key)
    return Decoded(wmcore.find_spread_spectrum_id(corrs))


def _rescaled(rgb, key, budget, decoder):
    m = wmcore.find_spread_spectrum_payload_rescaled(rgb, key, max_seconds=budget, decoder=decoder)
    return Decoded(m.text, m.factor, m.confidence if m.text is not None else None, m.repeats, m.budget_exhausted)


def decode_ss_id_rescale(rgb, key, budget=wmcore.DEFAULT_RESCALE_BUDGET_SECONDS):
    """The resize search, reading with the ID decoder (only IDs can come out)."""
    return _rescaled(rgb, key, budget, wmcore.find_spread_spectrum_id)


def decode_ss_or_id_rescale(rgb, key, budget=wmcore.DEFAULT_RESCALE_BUDGET_SECONDS):
    """The resize search, reading with 'an ID if there is one, else the text decoder' - what a reader that
    doesn't know the payload kind (--ss-key with free text) would use."""
    return _rescaled(rgb, key, budget, wmcore.find_spread_spectrum_payload_or_id)


def decode_ss_flip(rgb, key, budget=None):
    """`ss` + mirroring: an ID in any of the 4 orientations (none / left-right / top-bottom / both), else the text
    decoder on the image as given - what wmapp.extract uses for --ss-key (wmcore/orient.py)."""
    return Decoded(wmcore.find_spread_spectrum_payload_or_id_oriented(rgb, key).text)


def decode_ss_id_flip(rgb, key, budget=None):
    """`ss_id` + mirroring: the ID decoder on each of the 4 orientations."""
    return Decoded(wmcore.find_spread_spectrum_id_oriented(rgb, key).text)


DECODERS = {"lsb": decode_lsb, "ss": decode_ss, "ss_rescale": decode_ss_rescale, "ss_id": decode_ss_id,
            "ss_id_rescale": decode_ss_id_rescale, "ss_or_id_rescale": decode_ss_or_id_rescale,
            "ss_flip": decode_ss_flip, "ss_id_flip": decode_ss_id_flip}
DECODER_SCHEME = {"lsb": "lsb", "ss": "ss", "ss_rescale": "ss", "ss_id": "ss",
                  "ss_id_rescale": "ss", "ss_or_id_rescale": "ss", "ss_flip": "ss", "ss_id_flip": "ss"}
# The decoders that run the --rescale search: slow, so only on the rescale-eligible images / attack families.
RESCALE_DECODERS = frozenset({"ss_rescale", "ss_id_rescale", "ss_or_id_rescale"})


# ---------------------------------------------------------------------------
# Oracle soft observations (for BER)
# ---------------------------------------------------------------------------

def _crop_index(orig_wh, crop_blocks, attacked_wh):
    """Original raster block index for every block of a block-aligned crop."""
    W, H = orig_wh
    bw_o = W // wmcore.SS_BLOCK
    cy, cx = crop_blocks
    aw, ah = attacked_wh
    bh, bw = ah // wmcore.SS_BLOCK, aw // wmcore.SS_BLOCK
    rr, cc = np.meshgrid(np.arange(bh), np.arange(bw), indexing="ij")
    return ((rr + cy) * bw_o + (cc + cx)).reshape(-1)


def _unflip(rgb, flip):
    """Undo a mirror ("h" | "v" | "hv") - a flip is its own inverse."""
    from PIL import Image
    if "h" in flip:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
    if "v" in flip:
        rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
    return rgb


def oracle_soft(scheme, rgb, key, geom, orig_wh):
    """(soft, orig_idx) for BER, or None when the geometry has no oracle."""
    W, H = orig_wh
    if scheme == "ss":
        if geom.kind == "identity":
            c = wmcore.spread_spectrum_block_correlations(rgb, key)
            return c, np.arange(len(c))
        if geom.kind == "crop_blocks":
            c = wmcore.spread_spectrum_block_correlations(rgb, key)
            idx = _crop_index(orig_wh, geom.crop_blocks, rgb.size)
            return c, idx
        if geom.kind == "flip":
            c = wmcore.spread_spectrum_block_correlations(_unflip(rgb, geom.flip), key)
            return c, np.arange(len(c))
        if geom.kind == "resize":
            y, _, _ = wmcore._rgb_to_y(np.asarray(rgb.convert("RGB"), dtype=np.float64))
            interp = cv2.INTER_AREA if W < rgb.size[0] else cv2.INTER_LANCZOS4
            back = cv2.resize(y.astype(np.float32), (W, H), interpolation=interp).astype(np.float64)
            c = wmcore._ss_correlations_from_luma(back, key)
            return c, np.arange(len(c))
        return None
    if scheme == "lsb":
        arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
        lsb = (arr[..., 0] & 1).astype(np.float64) * 2.0 - 1.0
        if geom.kind == "identity":
            return lsb.reshape(-1), np.arange(lsb.size)
        if geom.kind == "crop_blocks":
            py, px = geom.crop_blocks[0] * wmcore.SS_BLOCK, geom.crop_blocks[1] * wmcore.SS_BLOCK
            rr, cc = np.meshgrid(np.arange(lsb.shape[0]), np.arange(lsb.shape[1]), indexing="ij")
            return lsb.reshape(-1), ((rr + py) * W + (cc + px)).reshape(-1)
        return None
    raise ValueError(scheme)


def factor_rel_err(geom, factor):
    """Relative error of the rescale search's recovered factor vs the true
    inverse of the applied resize (None when not applicable)."""
    if geom.kind != "resize" or factor is None:
        return None
    return abs(factor * geom.scale - 1.0)
