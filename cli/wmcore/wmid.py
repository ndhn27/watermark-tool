"""Watermark IDs: the short token every layer of one stamped output carries.

Why an ID instead of the long copyright/handle string the hidden marks used to
carry: (1) the details - who, when, which file, which key, which settings -
live in the ledger (wmapp/ledger.py), keyed by this ID, so `extract` can hand
them back and no one has to remember an --ss-key; (2) a payload with a strict
shape can be told from junk (below); (3) it is small: 112 bits on the wire
(4-byte length header + 10 bytes), which is what a future learned-watermark
layer (StegaStamp/HiDDeN-style, 30-100 bits of capacity) could carry at all.

What it does NOT buy you, measured: robustness over the short handle you'd
have used anyway. Fewer bits means more repeats per image (spread.py's module
docstring), but a 10-byte ID and an 11-byte "@handle" were detected equally
often everywhere, and on 640x480 synthetic images even a 99-byte sentence was
(16/16 under every gated attack). The size effect shows up on SMALL images: at
320x240 a 99-byte payload fails even unattacked (12/16) and under a 64 px top
crop (0/16) while the ID and the handle stay at 16/16 (bench/results/
p1-payload-length*, bench/results/P1-NOTES.md). So the ID is a robustness win
over a sentence-length --invisible-text on small or cropped images, and nothing
over a short one. Synthetic images only; not yet run on photos.

Format (10 characters, Crockford base32 - no I, L, O, U, so it survives being
read off a screen or typed by hand):

    8 data characters   40 random bits
    2 check characters  10 bits of CRC-32 over the data (salted with a version tag)

The check characters exist so a decoder can tell "a real ID with a bit error
in it" and "random junk that happened to decode as UTF-8" from an intact ID -
the LSB decoder in particular is documented to return short junk payloads from
unmarked images (bench/README.md, false positives), and a payload with a strict
shape is how that gets rejected. It is NOT a signature: anyone who can read an
ID can copy it into another file. The ledger records the stamped file's hash,
which is what lets `extract` say "this is byte-identical to what I stamped" vs
"same ID, different bytes".

Canonical form is the 10 characters with no separators (what is embedded);
display form groups them, e.g. K3M9-QXR2-T4. parse() accepts either, any case,
and Crockford's usual confusables (O->0, I/L->1).
"""

import secrets
import zlib

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
DATA_CHARS = 8
CHECK_CHARS = 2
WM_ID_LENGTH = DATA_CHARS + CHECK_CHARS
_VERSION_SALT = b"wm1:"
_DECODE = {c: i for i, c in enumerate(ALPHABET)}
_DECODE.update({"O": 0, "I": 1, "L": 1})     # Crockford's forgiving reads


def _check(data):
    crc = zlib.crc32(_VERSION_SALT + data.encode("ascii")) & 0x3FF          # 10 bits
    return ALPHABET[crc >> 5] + ALPHABET[crc & 0x1F]


def generate(rng=None):
    """A fresh random ID (canonical form). `rng` is anything with getrandbits()
    (tests pass a seeded random.Random); the default is the OS CSPRNG."""
    bits = (rng.getrandbits(5 * DATA_CHARS) if rng is not None else secrets.randbits(5 * DATA_CHARS))
    data = "".join(ALPHABET[(bits >> (5 * i)) & 0x1F] for i in range(DATA_CHARS - 1, -1, -1))
    return data + _check(data)


def parse(text):
    """The canonical ID in `text`, or None if it isn't a well-formed ID (wrong
    length, a character outside the alphabet, or the check characters don't
    match). Tolerates case, spaces and dashes."""
    if not isinstance(text, str):
        return None
    cleaned = "".join(ch for ch in text.strip().upper() if ch not in "- ")
    if len(cleaned) != WM_ID_LENGTH or any(ch not in _DECODE for ch in cleaned):
        return None
    canon = "".join(ALPHABET[_DECODE[ch]] for ch in cleaned)
    return canon if canon[DATA_CHARS:] == _check(canon[:DATA_CHARS]) else None


def is_valid(text):
    return parse(text) is not None


def display(wm_id):
    """K3M9QXR2T4 -> K3M9-QXR2-T4 (for humans; the embedded form has no dashes)."""
    canon = parse(wm_id)
    if canon is None:
        raise ValueError(f"not a watermark ID: {wm_id!r}")
    return f"{canon[:4]}-{canon[4:8]}-{canon[8:]}"

