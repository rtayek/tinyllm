"""Text corruption functions for structure-destruction experiments.

These are pure byte-to-byte transformations used by
``scripts/destruction_experiments.py``.  They live in the package (rather than
the script) so they are importable and type-checked, and so they can be unit
tested.  Each corruption destroys one kind of structure while preserving
others, letting the experiment isolate what the model has learned.
"""
from __future__ import annotations

import random
import re
from typing import Callable

# Character names to replace, grouped by book.  Replacements cycle through a
# fixed list of neutral placeholders.  Books absent from this mapping are left
# unchanged by ``make_corrupt_replace_names`` (the experiment control).
AUSTEN_NAMES: dict[str, list[str]] = {
    "Pride and Prejudice": [
        "Elizabeth", "Darcy", "Bennet", "Bingley", "Jane", "Lydia",
        "Wickham", "Collins", "Charlotte", "Kitty", "Mary", "Longbourn",
    ],
    "Sense and Sensibility": [
        "Elinor", "Marianne", "Dashwood", "Willoughby", "Brandon", "Edward",
        "Ferrars", "Jennings", "Palmer", "Middleton",
    ],
    "Emma": [
        "Emma", "Knightley", "Woodhouse", "Weston", "Churchill", "Fairfax",
        "Elton", "Harriet", "Smith", "Bates",
    ],
    "Mansfield Park": [
        "Fanny", "Edmund", "Crawford", "Bertram", "Norris", "Price",
        "Rushworth", "Yates", "Grant",
    ],
    "Persuasion": [
        "Anne", "Wentworth", "Elliot", "Musgrove", "Benwick", "Harville",
        "Smith", "Russell", "Walter",
    ],
    "Northanger Abbey": [
        "Catherine", "Tilney", "Morland", "Thorpe", "Isabella", "Henry",
        "Allen", "Woodston",
    ],
}
PLACEHOLDERS: list[str] = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
]


def corrupt_shuffle_letters(text: bytes) -> bytes:
    """Shuffle all characters within each word; preserve boundaries."""
    decoded = text.decode("utf-8", errors="replace")

    def _shuffle(m: re.Match[str]) -> str:
        chars = list(m.group(0))
        random.shuffle(chars)
        return "".join(chars)

    return re.sub(r"[A-Za-z]+", _shuffle, decoded).encode("utf-8", errors="replace")


def corrupt_shuffle_middle(text: bytes) -> bytes:
    """Shuffle only the middle characters of each word; keep first and last."""
    decoded = text.decode("utf-8", errors="replace")

    def _shuffle_middle(m: re.Match[str]) -> str:
        word = m.group(0)
        if len(word) <= 3:
            return word
        middle = list(word[1:-1])
        random.shuffle(middle)
        return word[0] + "".join(middle) + word[-1]

    return re.sub(r"[A-Za-z]+", _shuffle_middle, decoded).encode("utf-8", errors="replace")


def corrupt_shuffle_words(text: bytes) -> bytes:
    """Shuffle word order within each sentence; keep words intact."""
    decoded = text.decode("utf-8", errors="replace")
    sentences = re.split(r"(?<=[.!?])\s+", decoded)
    result: list[str] = []
    for sentence in sentences:
        words = sentence.split(" ")
        random.shuffle(words)
        result.append(" ".join(words))
    return " ".join(result).encode("utf-8", errors="replace")


def corrupt_reverse(text: bytes) -> bytes:
    """Reverse the entire byte sequence."""
    return text[::-1]


def corrupt_random_letters(text: bytes) -> bytes:
    """Replace every ASCII letter with a random letter a-z; keep non-letters."""
    result = bytearray(text)
    for i, b in enumerate(result):
        if 65 <= b <= 90 or 97 <= b <= 122:
            result[i] = random.randint(97, 122)
    return bytes(result)


def make_corrupt_replace_names(book_name: str) -> Callable[[bytes], bytes]:
    """Return a corruption that replaces character names for the given book.

    Books not present in ``AUSTEN_NAMES`` are returned unchanged, which is the
    experiment's control: out-of-corpus books should show zero delta.
    """
    names = AUSTEN_NAMES.get(book_name, [])

    def _corrupt(text: bytes) -> bytes:
        if not names:
            return text
        decoded = text.decode("utf-8", errors="replace")
        for i, name in enumerate(names):
            placeholder = PLACEHOLDERS[i % len(PLACEHOLDERS)]
            decoded = re.sub(r"\b" + re.escape(name) + r"\b", placeholder, decoded)
        return decoded.encode("utf-8", errors="replace")

    return _corrupt


def context_sizes(block_size: int) -> list[int]:
    """Powers of two up to and including block_size (e.g. 1,2,4,...,128)."""
    sizes: list[int] = []
    value = 1
    while value < block_size:
        sizes.append(value)
        value *= 2
    sizes.append(block_size)
    return sizes
