"""Tests for the decode-time rescale search:
wmcore.find_spread_spectrum_payload_rescaled() (exposed through
extract_invisible.py's extract_spread_spectrum_rescaled() and --rescale).
See the "Spread-spectrum decode after a resize" comment block in
wmcore/rescale.py for the design - including v2's confidence scoring and
performance budget - and, just as importantly, for what it deliberately
does not cover.

These use a 640x480 photo-like image so the whole file stays quick (a few
seconds per resized case; the "nothing there" cases are the slow ones because
all sweep tiers run - the 640px tier is a no-op at this image size, see
test_third_sweep_tier_is_skipped_on_small_images). The numbers behind the
design - how tight the resize tolerance is, why a fixed list of scale
factors isn't enough, how a weak decode can return a junk 1-byte payload -
come from benchmarks on ~1280x960 / 1920x1080 synthetic images, not from
these tests.
"""

import io
import random
import time

import numpy as np
import pytest
from PIL import Image

import wmcore
from wmcore import wmid
from extract_invisible import extract_spread_spectrum_rescaled  # cli/ on sys.path via conftest.py
from test_spread_spectrum import KEY, TEXT, _decode, _photo_like_rgba_image

SIZE = (640, 480)


def _stamped():
    return wmcore.embed_spread_spectrum(_photo_like_rgba_image(*SIZE), TEXT, KEY)


def _resized(stamped, scale, resample=Image.BICUBIC, jpeg_quality=None):
    w, h = SIZE
    out = stamped.convert("RGB").resize((round(w * scale), round(h * scale)), resample)
    if jpeg_quality is not None:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


def test_unresized_image_takes_the_plain_path():
    match = wmcore.find_spread_spectrum_payload_rescaled(_stamped(), KEY)
    assert match.text == TEXT
    assert match.factor == 1.0
    assert not match.budget_exhausted
    # No scale search ran, so no corroboration count in the search-path
    # sense; the plain path always reports exactly one "vote".
    assert match.corroborating_widths == 1
    assert 0.0 < match.confidence <= 1.0


@pytest.mark.parametrize(
    "scale, resample, jpeg_quality",
    [
        (0.75, Image.BICUBIC, None),   # an exact-reciprocal-of-a-round-number case ...
        (0.62, Image.LANCZOS, None),   # ... and ones a fixed "common factors" list would miss
        (1.4, Image.BICUBIC, 85),
        (1.13, Image.BILINEAR, 75),
    ],
)
def test_recovers_after_resize(scale, resample, jpeg_quality):
    stamped = _stamped()
    resized = _resized(stamped, scale, resample, jpeg_quality)
    # Pre-condition: the plain decoder really can't read this, so a pass below
    # is the search's doing, not something the old path already handled.
    assert _decode(resized) != TEXT

    match = wmcore.find_spread_spectrum_payload_rescaled(resized, KEY)
    assert match.text == TEXT
    assert match.factor == pytest.approx(1 / scale, rel=0.01)
    assert not match.budget_exhausted
    # Accepted results always corroborated at >= 2 widths and combined at
    # least one payload repeat.
    assert match.corroborating_widths >= 2
    assert match.repeats >= 1
    assert 0.0 < match.confidence <= 1.0


def test_recovers_after_resize_via_cli_function(tmp_path):
    path = tmp_path / "resized.png"
    _resized(_stamped(), 0.8).save(path)
    match = extract_spread_spectrum_rescaled(str(path), KEY)
    assert match.text == TEXT
    assert match.factor == pytest.approx(1.25, rel=0.01)


def test_unstamped_image_returns_none():
    # The property the whole "accept only what validates" design rests on: many
    # candidate scales are tried here, so one spurious decode would be enough
    # to fail this - unlike the single-decode false-positive tests in
    # test_spread_spectrum.py, which only exercise scale 1.0.
    match = wmcore.find_spread_spectrum_payload_rescaled(_photo_like_rgba_image(*SIZE), KEY)
    assert match.text is None
    assert match.factor is None
    assert match.confidence == 0.0
    assert not match.budget_exhausted  # the whole space was covered, nothing found there


def test_unstamped_resized_image_returns_none():
    plain = _photo_like_rgba_image(*SIZE).convert("RGB").resize((480, 360), Image.BICUBIC)
    match = wmcore.find_spread_spectrum_payload_rescaled(plain, KEY)
    assert match.text is None and match.factor is None


def test_scale_outside_range_is_not_found():
    # 0.35x needs a decode factor of ~2.86; the default range stops at 2.0.
    resized = _resized(_stamped(), 0.35)
    match = wmcore.find_spread_spectrum_payload_rescaled(resized, KEY)
    assert match.text is None and match.factor is None


def test_a_single_valid_looking_decode_is_not_accepted(monkeypatch):
    # Regression for the failure the benchmark found: a decode a few px off in
    # width can validate as a 1-byte payload ('h', 'i', ...) or a run of NULs.
    # Each candidate width here "decodes" to a different valid-looking string;
    # since none is corroborated at a second width, nothing may be returned.
    resized = _resized(_stamped(), 0.75)
    calls = {"n": 0}

    def flaky(corrs, max_length=10000):
        calls["n"] += 1
        return None if calls["n"] == 1 else f"junk{calls['n']}"  # 1st call = the plain, unresized decode

    monkeypatch.setattr(wmcore.spread, "find_spread_spectrum_payload", flaky)
    monkeypatch.setattr(wmcore.rescale, "_SS_SWEEP_TIERS", wmcore.rescale._SS_SWEEP_TIERS[:1])  # acceptance rule only; skip tiers 2-3
    match = wmcore.find_spread_spectrum_payload_rescaled(resized, KEY)
    assert match.text is None and match.factor is None
    assert calls["n"] > 3  # it really did try several widths rather than bailing out early


def test_the_same_decode_at_two_widths_is_accepted(monkeypatch):
    resized = _resized(_stamped(), 0.75)
    calls = {"n": 0}

    def stable(corrs, max_length=10000):
        calls["n"] += 1
        return None if calls["n"] == 1 else "@corroborated"

    monkeypatch.setattr(wmcore.spread, "find_spread_spectrum_payload", stable)
    monkeypatch.setattr(wmcore.rescale, "_SS_SWEEP_TIERS", wmcore.rescale._SS_SWEEP_TIERS[:1])  # acceptance rule only; skip tiers 2-3
    match = wmcore.find_spread_spectrum_payload_rescaled(resized, KEY)
    assert match.text == "@corroborated"
    assert match.factor == pytest.approx(1 / 0.75, rel=0.01)
    assert match.corroborating_widths == 2


def test_control_character_payloads_are_rejected(monkeypatch):
    # Corroborated but not printable (the NUL-run junk seen at off-by-a-few-px
    # widths) must still be refused.
    resized = _resized(_stamped(), 0.75)
    calls = {"n": 0}

    def nul_run(corrs, max_length=10000):
        calls["n"] += 1
        return None if calls["n"] == 1 else "\x00\x00\x00\x00"

    monkeypatch.setattr(wmcore.spread, "find_spread_spectrum_payload", nul_run)
    monkeypatch.setattr(wmcore.rescale, "_SS_SWEEP_TIERS", wmcore.rescale._SS_SWEEP_TIERS[:1])  # acceptance rule only; skip tiers 2-3
    match = wmcore.find_spread_spectrum_payload_rescaled(resized, KEY)
    assert match.text is None and match.factor is None


def test_printable_payload_helper():
    assert wmcore._printable_payload("@yourhandle")
    assert wmcore._printable_payload("Bản quyền © Kiobh 2026")
    assert wmcore._printable_payload("line one\nline two\t(tabbed)")
    assert not wmcore._printable_payload("\x00\x04\x00")
    assert not wmcore._printable_payload("ok\x07")


def test_tiny_image_does_not_crash():
    tiny = Image.fromarray(np.random.default_rng(0).integers(0, 256, (48, 48, 3), dtype=np.uint8), "RGB")
    match = wmcore.find_spread_spectrum_payload_rescaled(tiny, KEY)
    assert match.text is None and match.factor is None


def test_third_sweep_tier_is_skipped_on_small_images():
    # The 640px sweep tier requires FULL scale_range coverage (see its
    # comment in _SS_SWEEP_TIERS) and this 640x480 test image can't supply
    # a big enough source window at the range's low end, so it should
    # contribute zero peaks at essentially zero cost rather than running a
    # slow partial sweep.
    y32 = np.zeros((480, 640), dtype=np.float32)
    pn = wmcore.spread._ss_pn_vector(KEY, len(wmcore.spread.SS_COEFF_POSITIONS))
    patch, log_step, min_z, require_full_coverage = wmcore.rescale._SS_SWEEP_TIERS[2]
    assert patch == 640
    assert require_full_coverage
    start = time.monotonic()
    peaks, hit_deadline = wmcore._ss_sweep_peaks(y32, pn, wmcore.SS_RESCALE_RANGE, patch, log_step, min_z, 6,
                                                 require_full_coverage=require_full_coverage)
    assert peaks == []
    assert not hit_deadline
    assert time.monotonic() - start < 0.1  # skipped outright, not partially swept


def test_budget_exhausted_stops_the_search_without_a_false_accept(monkeypatch):
    # An already-elapsed deadline should make the search bail out immediately
    # (before even the first sweep tier) rather than ignore max_seconds.
    resized = _resized(_stamped(), 0.75)
    match = wmcore.find_spread_spectrum_payload_rescaled(resized, KEY, max_seconds=-1.0)
    assert match.text is None and match.factor is None
    assert match.budget_exhausted


def test_max_seconds_none_means_unbounded():
    # Regression: max_seconds=None must not be treated as "zero budget" - the
    # unresized fast path should still succeed exactly as it did in v1.
    match = wmcore.find_spread_spectrum_payload_rescaled(_stamped(), KEY, max_seconds=None)
    assert match.text == TEXT
    assert not match.budget_exhausted


def test_confidence_reflects_repeat_count(monkeypatch):
    # Direct unit test of the confidence heuristic itself (not the full
    # search): more corroborating widths and more repeats should never
    # produce a *lower* confidence than fewer of either, all else equal.
    low = wmcore.rescale._rescale_confidence(z=3.6, min_z=3.5, repeats=1, corroborating_widths=2)
    high = wmcore.rescale._rescale_confidence(z=3.6, min_z=3.5, repeats=8, corroborating_widths=3)
    assert 0.0 <= low < high <= 1.0


# ---- the `decoder=` hook (what reads a payload out of the correlations) --------------------------------------

def test_the_decoder_hook_reads_an_id_after_a_resize():
    wm = wmid.generate(random.Random(1))
    stamped = wmcore.embed_spread_spectrum(_photo_like_rgba_image(*SIZE), wm, KEY)
    match = wmcore.find_spread_spectrum_payload_rescaled(_resized(stamped, 0.75), KEY,
                                                         decoder=wmcore.find_spread_spectrum_id)
    assert match.text == wm
    assert match.factor == pytest.approx(1 / 0.75, rel=0.02)


def test_an_id_only_decoder_reads_nothing_from_a_text_payload():
    match = wmcore.find_spread_spectrum_payload_rescaled(_resized(_stamped(), 0.75), KEY,
                                                         decoder=wmcore.find_spread_spectrum_id)
    assert match.text is None


def test_the_decoder_hook_is_consulted_at_the_candidate_widths_not_just_once():
    calls = []

    def spy(corrs):
        calls.append(len(corrs))
        return None

    match = wmcore.find_spread_spectrum_payload_rescaled(_resized(_stamped(), 0.75), KEY, decoder=spy)
    assert match.text is None
    assert len(calls) > 1, "expected the unresized attempt plus at least one candidate width"
