"""Watermark IDs (wmcore.wmid): the 10-character token every layer carries."""

import random

import pytest

from wmcore import wmid


def test_generated_ids_are_well_formed_and_distinct():
    ids = [wmid.generate() for _ in range(2000)]
    assert all(len(i) == wmid.WM_ID_LENGTH == 10 and set(i) <= set(wmid.ALPHABET) for i in ids)
    assert all(wmid.parse(i) == i for i in ids)
    assert len(set(ids)) == len(ids)                       # 40 random bits: no collision in 2000 draws


def test_a_seeded_rng_makes_ids_reproducible():
    assert wmid.generate(random.Random(7)) == wmid.generate(random.Random(7))
    assert wmid.generate(random.Random(7)) != wmid.generate(random.Random(8))


def test_parse_accepts_display_form_case_and_crockford_confusables():
    canon = wmid.generate(random.Random(1))
    assert wmid.parse(wmid.display(canon)) == canon
    assert wmid.parse(canon.lower()) == canon
    assert wmid.parse("  " + wmid.display(canon).lower() + "\n") == canon
    # O->0, I->1, L->1 are read as Crockford defines them: build an ID whose canonical form has 0s and 1s
    r = random.Random(0)
    while True:
        cand = wmid.generate(r)
        if "0" in cand[:8] and "1" in cand[:8]:
            break
    spelled = cand.replace("0", "O").replace("1", "l")
    assert wmid.parse(spelled) == cand


def test_display_form_is_grouped_and_display_refuses_junk():
    assert wmid.display("K3M9QXR2T4".replace("K3M9QXR2T4", wmid.generate(random.Random(2)))).count("-") == 2
    with pytest.raises(ValueError):
        wmid.display("not-an-id")


@pytest.mark.parametrize("seed", range(5))
def test_every_single_character_error_is_caught(seed):
    canon = wmid.generate(random.Random(seed))
    for pos in range(wmid.WM_ID_LENGTH):
        for repl in wmid.ALPHABET:
            if repl == canon[pos]:
                continue
            assert wmid.parse(canon[:pos] + repl + canon[pos + 1:]) is None, (canon, pos, repl)


def test_a_swap_of_two_adjacent_characters_is_caught_almost_always():
    rng = random.Random(3)
    caught = total = 0
    for _ in range(300):
        c = wmid.generate(rng)
        i = rng.randrange(9)
        if c[i] == c[i + 1]:
            continue
        total += 1
        caught += wmid.parse(c[:i] + c[i + 1] + c[i] + c[i + 2:]) is None
    assert caught / total > 0.99                            # 10 check bits: ~1/1024 slip through


@pytest.mark.parametrize("bad", [None, 5, "", "short", "K3M9QXR2T", "K3M9QXR2T4X", "K3M9QXR2U4", "K3M9*XR2T4", b"K3M9QXR2T4"])
def test_parse_rejects_wrong_shapes(bad):
    assert wmid.parse(bad) is None and not wmid.is_valid(bad)


def test_ordinary_text_payloads_are_not_ids():
    for text in ("@yourhandle", "(c) 2026 Example Studio", "secret", "hello", "@me"):
        assert wmid.parse(text) is None
