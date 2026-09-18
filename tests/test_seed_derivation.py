"""Unit tests for wmcore.derive_seed() - the deterministic per-item seed
offset shared by batch mode (per-file) and video's per-segment jitter
refresh (per-segment-index). Both need the same two guarantees: reproducible
given the same (base_seed, key), and (in practice) different across keys so
several items sharing a base seed don't all get an identical, more easily
averaged-out pattern.
"""

import wmcore


def test_none_base_seed_passes_through():
    # No base seed means "fresh randomness every run" for the caller -
    # derive_seed must not turn that into a pinned value.
    assert wmcore.derive_seed(None, 0) is None
    assert wmcore.derive_seed(None, "some_file.png") is None


def test_reproducible_for_same_base_and_key():
    a = wmcore.derive_seed(42, 3)
    b = wmcore.derive_seed(42, 3)
    assert a == b


def test_different_keys_usually_differ():
    base = 42
    values = {wmcore.derive_seed(base, key) for key in range(10)}
    # crc32 collisions across 10 small ints are practically impossible.
    assert len(values) == 10


def test_different_base_seeds_differ_for_same_key():
    assert wmcore.derive_seed(1, "clip.mp4") != wmcore.derive_seed(2, "clip.mp4")


def test_accepts_non_string_keys_like_segment_indices():
    # video.py passes an int segment index directly; watermark.py passes a
    # filename string - both need to work without the caller stringifying it.
    assert wmcore.derive_seed(10, 5) == wmcore.derive_seed(10, "5")
