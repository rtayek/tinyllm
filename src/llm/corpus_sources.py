from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable


START_MARKER = re.compile(
    r"^\*\*\* START OF .*?\*\*\*\s*$",
    re.MULTILINE | re.DOTALL,
)
END_MARKER = re.compile(
    r"^\*\*\* END OF .*?\*\*\*\s*$",
    re.MULTILINE | re.DOTALL,
)
SHERLOCK_HEADING = re.compile(
    r"^(?P<number>I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\. "
    r"(?P<title>[A-Z][A-Z â€™'\-]+)$",
    re.MULTILINE,
)
CHAPTER_HEADING = re.compile(
    r"^CHAPTER (?P<number>[IVXLCDM]+|\d+)\.?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
ILLUSTRATION_BLOCK = re.compile(r"\[Illustration.*?\]", re.DOTALL)


@dataclass(frozen=True)
class WorkSpec:
    identifier: str
    author: str
    title: str
    ebook_id: str
    source_url: str
    unit_type: str
    expected_units: int
    split_counts: tuple[int, int, int]
    cleaner: Callable[[str], str]
    splitter: Callable[[str], list[tuple[str, str]]]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lstrip("\ufeff"))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    output: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            blank_count = 0
            output.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                output.append("")
    return "\n".join(output).strip() + "\n"


def strip_gutenberg_wrapper(text: str) -> str:
    start = START_MARKER.search(text)
    end = END_MARKER.search(text, start.end() if start else 0)
    if start is None or end is None:
        raise ValueError("Expected Project Gutenberg START and END markers")
    return text[start.end() : end.start()]


def _clean_from_first_heading(text: str, heading: re.Pattern[str]) -> str:
    core = strip_gutenberg_wrapper(normalize_text(text))
    first = heading.search(core)
    if first is None:
        raise ValueError("Could not find the first logical-unit heading")
    return normalize_text(core[first.start() :])


def clean_alice(text: str) -> str:
    return _clean_from_first_heading(text, CHAPTER_HEADING)


def clean_austen(text: str) -> str:
    """Clean a standard Jane Austen Gutenberg text.

    Strips the Gutenberg wrapper and front matter up to the first chapter
    heading.  Works for Sense and Sensibility, Emma, Mansfield Park,
    Persuasion, and Northanger Abbey, all of which use plain ``CHAPTER I.``
    style headings with no volume-reset or illustration blocks.
    """
    return _clean_from_first_heading(text, CHAPTER_HEADING)


def clean_sherlock(text: str) -> str:
    return _clean_from_first_heading(text, SHERLOCK_HEADING)


def clean_pride_and_prejudice(text: str) -> str:
    core = strip_gutenberg_wrapper(normalize_text(text))
    opening = "It is a truth universally acknowledged"
    opening_index = core.find(opening)
    if opening_index < 0:
        raise ValueError("Could not find the opening of Pride and Prejudice")
    cleaned = "CHAPTER I.\n\n" + core[opening_index:]
    cleaned = ILLUSTRATION_BLOCK.sub("", cleaned)
    cleaned = CHAPTER_HEADING.sub(
        lambda match: f"CHAPTER {match.group('number').upper()}.",
        cleaned,
    )
    return normalize_text(cleaned)


def _split_units(
    text: str,
    heading_pattern: re.Pattern[str],
    expected_count: int,
) -> list[tuple[str, str]]:
    matches = list(heading_pattern.finditer(text))
    if len(matches) != expected_count:
        raise ValueError(f"Expected {expected_count} units, found {len(matches)}")
    units: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        units.append((match.group(0).strip(), normalize_text(text[match.start() : end])))
    return units


def split_sherlock_stories(text: str) -> list[tuple[str, str]]:
    return _split_units(text, SHERLOCK_HEADING, 12)


def split_alice_chapters(text: str) -> list[tuple[str, str]]:
    return _split_units(text, CHAPTER_HEADING, 12)


def split_pride_chapters(text: str) -> list[tuple[str, str]]:
    return _split_units(text, CHAPTER_HEADING, 61)


def split_sense_and_sensibility_chapters(text: str) -> list[tuple[str, str]]:
    return _split_units(text, CHAPTER_HEADING, 50)


def split_emma_chapters(text: str) -> list[tuple[str, str]]:
    return _split_units(text, CHAPTER_HEADING, 55)


def split_mansfield_park_chapters(text: str) -> list[tuple[str, str]]:
    return _split_units(text, CHAPTER_HEADING, 48)


def split_persuasion_chapters(text: str) -> list[tuple[str, str]]:
    return _split_units(text, CHAPTER_HEADING, 24)


def split_northanger_abbey_chapters(text: str) -> list[tuple[str, str]]:
    return _split_units(text, CHAPTER_HEADING, 31)


SHERLOCK = WorkSpec(
    "arthur-conan-doyle/adventures-of-sherlock-holmes",
    "Arthur Conan Doyle",
    "The Adventures of Sherlock Holmes",
    "1661",
    "https://www.gutenberg.org/ebooks/1661",
    "story",
    12,
    (8, 2, 2),
    clean_sherlock,
    split_sherlock_stories,
)
ALICE = WorkSpec(
    "lewis-carroll/alices-adventures-in-wonderland",
    "Lewis Carroll",
    "Alice's Adventures in Wonderland",
    "11",
    "https://www.gutenberg.org/ebooks/11",
    "chapter",
    12,
    (8, 2, 2),
    clean_alice,
    split_alice_chapters,
)
PRIDE = WorkSpec(
    "jane-austen/pride-and-prejudice",
    "Jane Austen",
    "Pride and Prejudice",
    "1342",
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "chapter",
    61,
    (49, 6, 6),
    clean_pride_and_prejudice,
    split_pride_chapters,
)
SENSE_AND_SENSIBILITY = WorkSpec(
    "jane-austen/sense-and-sensibility",
    "Jane Austen",
    "Sense and Sensibility",
    "161",
    "https://www.gutenberg.org/cache/epub/161/pg161.txt",
    "chapter",
    50,
    (40, 5, 5),
    clean_austen,
    split_sense_and_sensibility_chapters,
)
EMMA = WorkSpec(
    "jane-austen/emma",
    "Jane Austen",
    "Emma",
    "158",
    "https://www.gutenberg.org/cache/epub/158/pg158.txt",
    "chapter",
    55,
    (44, 5, 6),
    clean_austen,
    split_emma_chapters,
)
MANSFIELD_PARK = WorkSpec(
    "jane-austen/mansfield-park",
    "Jane Austen",
    "Mansfield Park",
    "141",
    "https://www.gutenberg.org/cache/epub/141/pg141.txt",
    "chapter",
    48,
    (38, 5, 5),
    clean_austen,
    split_mansfield_park_chapters,
)
PERSUASION = WorkSpec(
    "jane-austen/persuasion",
    "Jane Austen",
    "Persuasion",
    "105",
    "https://www.gutenberg.org/cache/epub/105/pg105.txt",
    "chapter",
    24,
    (19, 2, 3),
    clean_austen,
    split_persuasion_chapters,
)
NORTHANGER_ABBEY = WorkSpec(
    "jane-austen/northanger-abbey",
    "Jane Austen",
    "Northanger Abbey",
    "121",
    "https://www.gutenberg.org/cache/epub/121/pg121.txt",
    "chapter",
    31,
    (24, 3, 4),
    clean_austen,
    split_northanger_abbey_chapters,
)
AUSTEN_WORKS = (
    PRIDE,
    SENSE_AND_SENSIBILITY,
    EMMA,
    MANSFIELD_PARK,
    PERSUASION,
    NORTHANGER_ABBEY,
)
WORKS = (SHERLOCK, ALICE, *AUSTEN_WORKS)


