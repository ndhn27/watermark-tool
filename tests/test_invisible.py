"""Round-trip tests for the LSB invisible watermark: wmcore.embed_invisible()
paired with extract_invisible.py's extract_invisible(). This is exactly the
kind of thing that's tedious to check by eye (open the image, run the
extractor, read the printed string) but trivial to pin down as an
automated assertion. embed_invisible must save losslessly (PNG) - JPEG
recompression is documented to destroy the payload, so that's not
something to assert against an encoder here, just a known property.
"""

from PIL import Image

import wmcore
from extract_invisible import extract_invisible  # cli/ is on sys.path via conftest.py


def _solid_rgba_image(width=64, height=64, rgb=(120, 130, 140)):
    return Image.new("RGBA", (width, height), rgb + (255,))


def test_roundtrip_ascii():
    img = _solid_rgba_image()
    stamped = wmcore.embed_invisible(img, "@yourhandle")
    stamped.save("/tmp/_wm_test_ascii.png")  # must be lossless (PNG) to survive
    assert extract_invisible("/tmp/_wm_test_ascii.png") == "@yourhandle"


def test_roundtrip_unicode():
    # Vietnamese text exercises the utf-8 encode/decode path, not just ascii.
    text = "Bản quyền © Kiobh 2026"
    img = _solid_rgba_image()
    stamped = wmcore.embed_invisible(img, text)
    stamped.save("/tmp/_wm_test_unicode.png")
    assert extract_invisible("/tmp/_wm_test_unicode.png") == text


def test_roundtrip_empty_string():
    img = _solid_rgba_image()
    stamped = wmcore.embed_invisible(img, "")
    stamped.save("/tmp/_wm_test_empty.png")
    # length header is 0, so extract_invisible's `length <= 0` guard returns
    # None rather than an empty string - document that behavior explicitly.
    assert extract_invisible("/tmp/_wm_test_empty.png") is None


def test_extract_on_unstamped_image_returns_none():
    # A flat image whose red-channel value has LSB 0 (100 = 0b01100100) means
    # every "length" bit read back is 0 -> decoded length is 0 -> None,
    # deterministically (as opposed to a noisy image, where a stray header
    # could rarely happen to decode as some garbage-but-valid utf-8 string).
    img = _solid_rgba_image(rgb=(100, 100, 100))
    img.save("/tmp/_wm_test_unstamped.png")
    assert extract_invisible("/tmp/_wm_test_unstamped.png") is None
