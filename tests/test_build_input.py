import pytest

from disto import build_input


def test_full_format():
    text = build_input("An article.", "A question?", "yes", ["a", "b", "c"])
    assert text == "[QUES] A question? [ANS] yes [DIS1] a [DIS2] b [DIS3] c [ART] An article."


def test_missing_distractors_are_padded():
    text = build_input("art", "q", "a", ["only one"])
    assert "[DIS1] only one [DIS2] [EMPT] [DIS3] [EMPT]" in text


def test_blank_distractor_is_padded():
    assert "[DIS2] [EMPT]" in build_input("art", "q", "a", ["x", "  ", "z"])


def test_whitespace_is_collapsed():
    assert build_input("a\n\n b", "q", "a", ["x"]).endswith("[ART] a b")


def test_too_many_distractors():
    with pytest.raises(ValueError):
        build_input("art", "q", "a", ["1", "2", "3", "4"])
