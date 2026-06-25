"""Tests for the corruption functions in llm.corruptions.

These functions drive the structure-destruction research results, so a silent
bug in any of them would produce wrong experimental conclusions.  Each test
verifies the corruption changes the text in exactly the intended way and
preserves what it is supposed to preserve.
"""
from __future__ import annotations

import random

from llm.corruptions import (
    PLACEHOLDERS,
    context_sizes,
    corrupt_random_letters,
    corrupt_reverse,
    corrupt_shuffle_letters,
    corrupt_shuffle_middle,
    corrupt_shuffle_words,
    make_corrupt_replace_names,
)


def setup_function() -> None:
    # Deterministic shuffles for every test.
    random.seed(0)


# ---------------------------------------------------------------------------
# shuffle_letters
# ---------------------------------------------------------------------------

class TestShuffleLetters:
    def test_preserves_length(self) -> None:
        text = b"Hello world, this is a test."
        result = corrupt_shuffle_letters(text)
        assert len(result) == len(text)

    def test_preserves_non_letters(self) -> None:
        text = b"abc, def! 123"
        result = corrupt_shuffle_letters(text).decode("utf-8")
        assert result[3:5] == ", "
        assert result[8] == "!"
        assert "123" in result

    def test_preserves_letter_multiset_within_word(self) -> None:
        text = b"Elizabeth"
        result = corrupt_shuffle_letters(text).decode("utf-8")
        assert sorted(result) == sorted("Elizabeth")

    def test_single_char_words_unchanged(self) -> None:
        text = b"a I o"
        result = corrupt_shuffle_letters(text).decode("utf-8")
        assert result == "a I o"


# ---------------------------------------------------------------------------
# shuffle_middle
# ---------------------------------------------------------------------------

class TestShuffleMiddle:
    def test_preserves_first_and_last_letter(self) -> None:
        text = b"Elizabeth"
        result = corrupt_shuffle_middle(text).decode("utf-8")
        assert result[0] == "E"
        assert result[-1] == "h"
        assert sorted(result) == sorted("Elizabeth")

    def test_short_words_unchanged(self) -> None:
        for word in (b"a", b"to", b"the", b"abc"):
            assert corrupt_shuffle_middle(word) == word

    def test_preserves_length(self) -> None:
        text = b"The quick brown fox jumps."
        assert len(corrupt_shuffle_middle(text)) == len(text)


# ---------------------------------------------------------------------------
# shuffle_words
# ---------------------------------------------------------------------------

class TestShuffleWords:
    def test_preserves_word_multiset(self) -> None:
        text = b"the cat sat on the mat."
        result = corrupt_shuffle_words(text).decode("utf-8")
        assert sorted(result.split()) == sorted("the cat sat on the mat.".split())

    def test_changes_something_on_long_input(self) -> None:
        random.seed(123)
        text = b"alpha beta gamma delta epsilon zeta eta theta iota kappa."
        result = corrupt_shuffle_words(text).decode("utf-8")
        assert result != text.decode("utf-8")


# ---------------------------------------------------------------------------
# reverse
# ---------------------------------------------------------------------------

class TestReverse:
    def test_is_byte_reversal(self) -> None:
        text = b"abcdef"
        assert corrupt_reverse(text) == b"fedcba"

    def test_is_its_own_inverse(self) -> None:
        text = b"The quick brown fox."
        assert corrupt_reverse(corrupt_reverse(text)) == text


# ---------------------------------------------------------------------------
# random_letters
# ---------------------------------------------------------------------------

class TestRandomLetters:
    def test_preserves_non_letters(self) -> None:
        text = b"abc, 123! xyz"
        result = corrupt_random_letters(text)
        for i, b in enumerate(text):
            is_letter = (65 <= b <= 90) or (97 <= b <= 122)
            if not is_letter:
                assert result[i] == b

    def test_all_letters_become_lowercase(self) -> None:
        text = b"ABCxyz"
        result = corrupt_random_letters(text)
        for b in result:
            assert 97 <= b <= 122

    def test_preserves_length(self) -> None:
        text = b"Hello, World! 42"
        assert len(corrupt_random_letters(text)) == len(text)


# ---------------------------------------------------------------------------
# replace_names  (the control case is the important one)
# ---------------------------------------------------------------------------

class TestReplaceNames:
    def test_non_austen_book_is_unchanged(self) -> None:
        corrupt = make_corrupt_replace_names("Sherlock Holmes")
        text = b"Holmes lit his pipe and considered the problem."
        assert corrupt(text) == text

    def test_unknown_book_is_unchanged(self) -> None:
        corrupt = make_corrupt_replace_names("Some Unknown Book")
        text = b"Elizabeth and Darcy walked together."
        assert corrupt(text) == text

    def test_austen_names_are_replaced(self) -> None:
        corrupt = make_corrupt_replace_names("Pride and Prejudice")
        text = b"Elizabeth smiled at Darcy."
        result = corrupt(text).decode("utf-8")
        assert "Elizabeth" not in result
        assert "Darcy" not in result
        assert any(p in result for p in PLACEHOLDERS)

    def test_word_boundaries_respected(self) -> None:
        corrupt = make_corrupt_replace_names("Pride and Prejudice")
        text = b"Janet is not Jane."
        result = corrupt(text).decode("utf-8")
        # "Janet" survives intact; standalone "Jane" is replaced.
        assert "Janet" in result
        assert "Jane." not in result

    def test_preserves_surrounding_text(self) -> None:
        corrupt = make_corrupt_replace_names("Emma")
        text = b"Emma Woodhouse, handsome, clever, and rich."
        result = corrupt(text).decode("utf-8")
        assert "handsome, clever, and rich." in result


# ---------------------------------------------------------------------------
# context_sizes helper
# ---------------------------------------------------------------------------

class TestContextSizes:
    def test_powers_of_two_through_block_size(self) -> None:
        assert context_sizes(128) == [1, 2, 4, 8, 16, 32, 64, 128]

    def test_non_power_of_two_block_size(self) -> None:
        sizes = context_sizes(100)
        assert sizes[-1] == 100
        assert sizes[:-1] == [1, 2, 4, 8, 16, 32, 64]

    def test_small_block_size(self) -> None:
        assert context_sizes(1) == [1]
