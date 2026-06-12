import hashlib
import json
from pathlib import Path

import pytest

import llm.corpus as corpus_module
from llm.corpus import (
    ALICE,
    PRIDE,
    SHERLOCK,
    clean_alice,
    clean_pride_and_prejudice,
    clean_sherlock,
    normalize_text,
    prepare_corpora,
    split_alice_chapters,
    split_pride_chapters,
    split_sherlock_stories,
)


ROMAN = (
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI",
    "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII", "XXVIII",
    "XXIX", "XXX", "XXXI", "XXXII", "XXXIII", "XXXIV", "XXXV", "XXXVI",
    "XXXVII", "XXXVIII", "XXXIX", "XL", "XLI", "XLII", "XLIII", "XLIV",
    "XLV", "XLVI", "XLVII", "XLVIII", "XLIX", "L", "LI", "LII", "LIII",
    "LIV", "LV", "LVI", "LVII", "LVIII", "LIX", "LX", "LXI",
)


def wrapped(body: str) -> str:
    return (
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        f"{body}\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    )


def chapters(count: int) -> str:
    return "\n\n".join(
        f"CHAPTER {number}.\n\nChapter body {index}"
        for index, number in enumerate(ROMAN[:count], start=1)
    )


def sherlock_stories() -> str:
    headings = (
        "I. A SCANDAL IN BOHEMIA",
        "II. THE RED-HEADED LEAGUE",
        "III. A CASE OF IDENTITY",
        "IV. THE BOSCOMBE VALLEY MYSTERY",
        "V. THE FIVE ORANGE PIPS",
        "VI. THE MAN WITH THE TWISTED LIP",
        "VII. THE ADVENTURE OF THE BLUE CARBUNCLE",
        "VIII. THE ADVENTURE OF THE SPECKLED BAND",
        "IX. THE ADVENTURE OF THE ENGINEER'S THUMB",
        "X. THE ADVENTURE OF THE NOBLE BACHELOR",
        "XI. THE ADVENTURE OF THE BERYL CORONET",
        "XII. THE ADVENTURE OF THE COPPER BEECHES",
    )
    return "\n\n".join(
        f"{heading}\n\nStory body {index}"
        for index, heading in enumerate(headings, start=1)
    )


def test_normalize_text_normalizes_newlines_and_blank_runs() -> None:
    assert normalize_text("\ufeffone \r\n\r\n\r\n\r\ntwo\t") == "one\n\n\ntwo\n"


def test_cleaners_remove_wrappers_and_front_matter() -> None:
    assert clean_alice(wrapped("Contents\n" + chapters(12))).startswith("CHAPTER I.")
    assert clean_sherlock(wrapped("Contents\n" + sherlock_stories())).startswith(
        "I. A SCANDAL IN BOHEMIA"
    )


def test_pride_cleaner_removes_illustrations_and_normalizes_headings() -> None:
    body = (
        "[Illustration: title\n\nChapter I.]\n\n"
        "It is a truth universally acknowledged, body.\n\n"
        + "\n\n".join(
            f"Chapter {number}.\n\n[Illustration]\n\nBody {index}"
            for index, number in enumerate(ROMAN[1:], start=2)
        )
    )
    cleaned = clean_pride_and_prejudice(wrapped(body))
    assert "[Illustration" not in cleaned
    assert "CHAPTER XLVI." in cleaned
    assert len(split_pride_chapters(cleaned)) == 61


def test_splitters_require_expected_unit_counts() -> None:
    assert len(split_alice_chapters(chapters(12))) == 12
    assert len(split_sherlock_stories(sherlock_stories())) == 12
    with pytest.raises(ValueError, match="Expected 12"):
        split_sherlock_stories("I. A SCANDAL IN BOHEMIA\n\nBody\n")


def test_prepare_corpora_writes_manifests_units_and_splits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpora"
    raw_sources = {
        SHERLOCK: wrapped(sherlock_stories()),
        ALICE: wrapped(chapters(12)),
        PRIDE: wrapped(
            "It is a truth universally acknowledged, body.\n\n"
            + "\n\n".join(
                f"CHAPTER {number}.\n\nBody {index}"
                for index, number in enumerate(ROMAN[1:], start=2)
            )
        ),
    }
    for spec, source in raw_sources.items():
        path = root / spec.identifier / "raw" / "gutenberg.txt"
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")

    monkeypatch.setattr(corpus_module, "WORKS", tuple(raw_sources))
    manifests = prepare_corpora(root)

    assert set(manifests) == {spec.identifier for spec in raw_sources}
    for spec in raw_sources:
        work_dir = root / spec.identifier
        manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["units"]) == spec.expected_units
        assert [
            len(manifest["splits"][name]["units"])
            for name in ("train", "validation", "test")
        ] == list(spec.split_counts)
        for record in (
            manifest["source"],
            manifest["cleaning"],
            *manifest["units"],
            manifest["splits"]["train"],
            manifest["splits"]["validation"],
            manifest["splits"]["test"],
        ):
            data = (work_dir / record["path"]).read_bytes()
            assert record["bytes"] == len(data)
            assert record["sha256"] == hashlib.sha256(data).hexdigest()
