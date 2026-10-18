"""wmcore.find_spread_spectrum_id(): the decoder that knows the payload is a watermark ID.

An ID payload has a fixed shape (112 bits on the wire), so this decoder does no period discovery: it sums the
repeats at each of the 112 possible phases and accepts a phase only if the 80 payload bits are a canonical,
check-valid ID. These tests pin down (a) that it reads what embed_spread_spectrum wrote, including after a
JPEG round trip and a block-aligned top crop (which shifts the phase), (b) the case it exists for - a channel
so noisy that no single repeat's 32-bit header survives, yet the sum over repeats is clean, and (c) what it
must refuse: unmarked images, non-ID payloads, look-alike IDs, noise.
"""

import io
import random

import numpy as np
from PIL import Image

import wmcore
from wmcore import wmid

KEY = 20260921


def _photo_like(width=640, height=480, seed=0):
    xx, yy = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
    rng = np.random.default_rng(seed)
    arr = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
    arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def _valid_id(seed=1):
    return wmid.generate(random.Random(seed))


def _decode_id(img, key=KEY):
    return wmcore.find_spread_spectrum_id(wmcore.spread_spectrum_block_correlations(img, key))


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_period_is_112_bits():
    assert wmcore.ID_PERIOD_BITS == 112 == len(wmcore.text_to_bits(_valid_id()))


def test_roundtrip():
    wm = _valid_id()
    stamped = wmcore.embed_spread_spectrum(_photo_like(), wm, KEY)
    assert _decode_id(stamped) == wm


def test_survives_jpeg():
    wm = _valid_id(2)
    stamped = wmcore.embed_spread_spectrum(_photo_like(), wm, KEY)
    assert _decode_id(_jpeg(stamped, 50)) == wm


def test_top_crop_shifts_the_phase_and_still_decodes():
    # 64 px = 8 block rows of 80 blocks = 640 blocks; 640 % 112 = 80, so the period no longer starts at block 0.
    wm = _valid_id(3)
    stamped = wmcore.embed_spread_spectrum(_photo_like(), wm, KEY).convert("RGB")
    cropped = Image.fromarray(np.ascontiguousarray(np.asarray(stamped)[64:, :]), "RGB")
    assert (8 * (640 // 8)) % wmcore.ID_PERIOD_BITS != 0
    assert _decode_id(cropped) == wm


def test_any_phase_offset_decodes_on_synthetic_correlations():
    wm = _valid_id(4)
    bits = np.array(wmcore.text_to_bits(wm), dtype=float) * 2 - 1
    rng = np.random.default_rng(0)
    for offset in (0, 1, 37, 111):
        corrs = np.tile(bits, 30)[offset:] + rng.normal(0, 0.3, 30 * 112 - offset)
        assert wmcore.find_spread_spectrum_id(corrs) == wm, offset


def test_recovers_when_no_header_survives_but_the_sum_is_clean():
    # Raw bit error rate ~30%: the chance that ALL 32 header bits of one repeat are right is 0.7**32 ~ 1e-5, so
    # find_spread_spectrum_payload() (which bootstraps from an exact header) has nothing to start from - yet 42
    # summed repeats leave a per-bit z of ~3.4, i.e. the payload is there. This is "signal intact, decoder
    # loses sync" (bench/results/P1-NOTES.md) in miniature.
    wm = _valid_id(5)
    bits = np.array(wmcore.text_to_bits(wm), dtype=float) * 2 - 1
    rng = np.random.default_rng(11)
    corrs = np.tile(bits, 42) + rng.normal(0, 1.0 / 0.5244, 42 * 112)
    raw_ber = float(np.mean((corrs > 0) != (np.tile(bits, 42) > 0)))
    assert 0.25 < raw_ber < 0.35
    assert wmcore.find_spread_spectrum_payload(corrs) != wm
    assert wmcore.find_spread_spectrum_id(corrs) == wm


def test_unmarked_image_returns_none():
    assert _decode_id(_photo_like(seed=5)) is None


def test_non_id_payload_returns_none():
    stamped = wmcore.embed_spread_spectrum(_photo_like(), "@yourhandle", KEY)
    assert _decode_id(stamped) is None


def test_check_invalid_id_is_rejected():
    # 10 well-formed characters whose check characters are wrong (the bench's old "id" payload is exactly this).
    bad = "K3M9QXR2T4"
    assert wmid.parse(bad) is None
    stamped = wmcore.embed_spread_spectrum(_photo_like(), bad, KEY)
    assert _decode_id(stamped) is None


def test_only_canonical_form_is_accepted():
    # wmid.parse() is lenient (case, confusables) for humans typing an ID; a *decoder* must not be: a lowercase
    # payload is not something this tool embeds, so reading one is evidence of a look-alike, not of our mark.
    wm = _valid_id(6)
    assert wmid.parse(wm.lower()) == wm
    stamped = wmcore.embed_spread_spectrum(_photo_like(), wm.lower(), KEY)
    assert _decode_id(stamped) is None


def test_too_few_blocks_returns_none():
    assert wmcore.find_spread_spectrum_id(np.ones(wmcore.ID_PERIOD_BITS - 1)) is None
    assert wmcore.find_spread_spectrum_id(np.array([])) is None


def test_pure_noise_is_never_accepted():
    # Not a proof of the ~1e-10 analytic false-accept rate (nothing this cheap could be) - a tripwire for the
    # accept rule being loosened by accident. The analytic argument is in find_spread_spectrum_id's docstring.
    rng = np.random.default_rng(0)
    assert all(wmcore.find_spread_spectrum_id(rng.normal(size=4800)) is None for _ in range(400))


def test_the_or_id_reader_prefers_an_id_and_leaves_text_payloads_alone():
    wm = _valid_id(7)
    stamped_id = wmcore.embed_spread_spectrum(_photo_like(), wm, KEY)
    stamped_text = wmcore.embed_spread_spectrum(_photo_like(), "@yourhandle", KEY)
    read = lambda img: wmcore.find_spread_spectrum_payload_or_id(wmcore.spread_spectrum_block_correlations(img, KEY))
    assert read(stamped_id) == wm
    assert read(stamped_text) == "@yourhandle"          # not an ID -> the legacy decoder answers, exactly as before
    assert read(_photo_like(seed=9)) is None


def test_extract_spread_spectrum_reads_an_id_from_a_file(tmp_path):
    from extract_invisible import extract_spread_spectrum      # cli/ on sys.path via conftest.py
    wm = _valid_id(8)
    path = str(tmp_path / "s.jpg")
    _jpeg(wmcore.embed_spread_spectrum(_photo_like(), wm, KEY), 60).save(path, quality=95)
    assert extract_spread_spectrum(path, KEY) == wm
