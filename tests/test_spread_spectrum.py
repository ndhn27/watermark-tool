"""Round-trip tests for the block-DCT spread-spectrum invisible watermark:
wmcore.embed_spread_spectrum() paired with wmcore.find_spread_spectrum_payload()
(exposed through extract_invisible.py's extract_spread_spectrum() for the CLI
path too). See the module-level comment above embed_spread_spectrum() in
wmcore.py for the full design rationale - these tests exist to pin down the
specific claims made there and in the README: survives JPEG recompression,
survives a block-aligned crop, needs a reasonably sized image, and a wrong
key is NOT a secrecy mechanism (unlike --seed, it can leak).

Unlike the LSB tests, a tiny (e.g. 64x64) image isn't a valid test image
here - it doesn't have enough 8x8 blocks to carry even one copy of a
realistic payload, which is itself a documented limitation (see
test_image_too_small_raises below), so these use a size that comfortably
fits many repeats instead.
"""

import io

import numpy as np
from PIL import Image

import wmcore
from extract_invisible import MAX_PLAUSIBLE_LENGTH, extract_spread_spectrum  # cli/ on sys.path via conftest.py

KEY = 20260918
TEXT = "@yourhandle"


def _solid_rgba_image(width=256, height=256, rgb=(120, 130, 140)):
    return Image.new("RGBA", (width, height), rgb + (255,))


def _photo_like_rgba_image(width=640, height=480, seed=0):
    # A flat solid color round-trips fine (see test_roundtrip_ascii_solid_image
    # below) but isn't representative of a real photo's DCT content - most of
    # the tests that matter (JPEG survival, crop survival) use this instead:
    # a smooth gradient plus per-pixel noise, closer to real sensor/photo
    # texture than either a flat fill or literal white noise.
    xx, yy = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
    rng = np.random.default_rng(seed)
    arr = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
    arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def _decode(img, key=KEY, max_length=MAX_PLAUSIBLE_LENGTH):
    corrs = wmcore.spread_spectrum_block_correlations(img, key)
    return wmcore.find_spread_spectrum_payload(corrs, max_length=max_length)


def test_roundtrip_ascii_solid_image():
    img = _solid_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, TEXT, KEY)
    assert _decode(stamped) == TEXT


def test_roundtrip_unicode():
    text = "Bản quyền © Kiobh 2026"
    img = _photo_like_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, text, KEY)
    assert _decode(stamped) == text


def test_roundtrip_tab_newline_cr():
    # find_spread_spectrum_payload() requires the decoded text be printable
    # (see its docstring and bench/results/P2-NOTES.md §6.6 / bench/experiments/
    # plain_decode_printable.py: this filter is what keeps the plain decode
    # path from accepting control-byte junk on periodic images). Tab,
    # newline and CR are the three control characters a real text payload
    # legitimately uses for layout, so the filter must not cost these.
    text = "line one\nline two\t(tabbed)\r\nlast\t\r\n"
    img = _photo_like_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, text, KEY)
    assert _decode(stamped) == text


def test_other_control_characters_do_not_roundtrip():
    # The flip side of the printable filter: a payload containing a control
    # character OTHER than tab/newline/CR is not something a real text
    # payload should hold, and the filter now refuses it - a real, if
    # narrow, behavior change from before the filter existed. Documented
    # here rather than left as a silent surprise; see handoff v21 §2 item 1.
    text = "before\x07bell\x00nul-after"
    img = _photo_like_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, text, KEY)
    assert _decode(stamped) is None


def test_roundtrip_via_cli_extract_function():
    # Exercise the same path watermark.py / extract_invisible.py actually use
    # (save to disk, read back through extract_spread_spectrum), not just the
    # in-memory wmcore functions the other tests call directly.
    img = _photo_like_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, TEXT, KEY)
    stamped.convert("RGB").save("/tmp/_wm_test_ss_ascii.png")
    assert extract_spread_spectrum("/tmp/_wm_test_ss_ascii.png", KEY) == TEXT


def test_survives_jpeg_recompression():
    # This is the headline property spread-spectrum mode has that LSB mode
    # does not - see README. Checked across a realistic quality range, not
    # just one number picked to pass.
    img = _photo_like_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, TEXT, KEY)
    for quality in (95, 85, 75, 60):
        buf = io.BytesIO()
        stamped.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        reloaded = Image.open(buf).convert("RGB")
        assert _decode(reloaded) == TEXT, f"failed to decode after JPEG quality={quality}"


def test_survives_block_aligned_crop():
    # Cropping whole rows off the top (a multiple of the 8px block size, so
    # the remaining blocks' content is byte-identical to blocks 16+ of the
    # original) should still decode - the payload is tiled across the block
    # grid for exactly this reason, mirroring how embed_invisible() tiles
    # across pixels. This is also a regression test for an earlier version
    # of this feature that keyed each block's spreading sequence to its
    # absolute (row, col) position - which silently broke decoding on ANY
    # crop, aligned or not, because the extractor only ever sees the
    # cropped image's own local block coordinates. See the module comment
    # above embed_spread_spectrum() in wmcore.py.
    img = _photo_like_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, TEXT, KEY)
    arr = np.array(stamped.convert("RGB"))
    cropped = Image.fromarray(arr[128:, :], "RGB")  # drop top 128px = 16 block-rows
    assert _decode(cropped) == TEXT


def test_wrong_key_is_not_reliable_but_can_leak():
    # Document the actual, measured behavior rather than an idealized one:
    # a far-away wrong key should not decode the exact payload most of the
    # time, but --ss-key is explicitly NOT a secrecy mechanism (see README
    # and wmcore.py's module comment) - a nearby/lucky wrong key has a real,
    # non-negligible chance of leaking through, so this only asserts that
    # the *majority* of a spread sample of wrong keys fail, not that all do.
    img = _photo_like_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, TEXT, KEY)
    correct_guesses = sum(1 for i in range(1, 31) if _decode(stamped, key=KEY + i * 7919) == TEXT)
    assert _decode(stamped, key=KEY) == TEXT  # the actual key always works
    assert correct_guesses < 15, (
        f"{correct_guesses}/30 wrong keys decoded the exact payload - "
        f"worse than the ~1-in-4 documented in the README"
    )


def test_extract_on_random_noise_does_not_false_positive():
    # Same rationale as the equivalent LSB test: a large, genuinely random
    # (never stamped) image shouldn't spuriously decode as valid UTF-8.
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    assert _decode(img) is None


def test_extract_on_unstamped_photo_returns_none():
    img = _photo_like_rgba_image()
    assert _decode(img) is None


def test_image_too_small_raises():
    # A documented limitation, not a silent failure: unlike embed_invisible()
    # (which can tile a payload across even a handful of pixels), each
    # payload bit here needs a whole 8x8 block, so a small image may not
    # have enough blocks for even a single copy.
    img = _solid_rgba_image(width=32, height=32)
    try:
        wmcore.embed_spread_spectrum(img, TEXT, KEY)
        assert False, "expected ValueError for an image too small to hold the payload"
    except ValueError as e:
        assert "too small" in str(e)


def test_roundtrip_empty_string_decodes_to_none():
    # Mirrors embed_invisible's contract for an empty payload (see
    # test_invisible.py's test_roundtrip_empty_string): the embed step
    # still writes a (length=0) header - _text_to_bits("") is not an
    # empty list, it's the 32-bit zero-length header - but
    # find_spread_spectrum_payload's `decoded_length > 0` check rejects a
    # zero-length "payload" as nothing meaningful having been embedded.
    img = _solid_rgba_image()
    stamped = wmcore.embed_spread_spectrum(img, "", KEY)
    assert _decode(stamped) is None
