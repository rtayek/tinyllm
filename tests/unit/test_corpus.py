from pathlib import Path

import pytest

from llm.corpus import (
    clean_alice,
    clean_sherlock,
    clean_tender_buttons,
    normalize_text,
    prepare_corpora,
    split_sherlock_stories,
)


def test_normalize_text_normalizes_newlines_and_blank_runs() -> None:
    assert normalize_text("\ufeffone \r\n\r\n\r\n\r\ntwo\t") == "one\n\n\ntwo\n"


def test_clean_alice_removes_wrapper_and_front_matter() -> None:
    raw = """*** START OF THE PROJECT GUTENBERG EBOOK 11 ***
Title
Contents
CHAPTER I.
Down the Rabbit-Hole
Body
*** END OF THE PROJECT GUTENBERG EBOOK 11 ***
License
"""

    assert clean_alice(raw) == "CHAPTER I.\nDown the Rabbit-Hole\nBody\n"


def test_clean_sherlock_and_split_stories() -> None:
    headings = [
        "I. A SCANDAL IN BOHEMIA",
        "II. THE RED-HEADED LEAGUE",
        "III. A CASE OF IDENTITY",
        "IV. THE BOSCOMBE VALLEY MYSTERY",
        "V. THE FIVE ORANGE PIPS",
        "VI. THE MAN WITH THE TWISTED LIP",
        "VII. THE ADVENTURE OF THE BLUE CARBUNCLE",
        "VIII. THE ADVENTURE OF THE SPECKLED BAND",
        "IX. THE ADVENTURE OF THE ENGINEER’S THUMB",
        "X. THE ADVENTURE OF THE NOBLE BACHELOR",
        "XI. THE ADVENTURE OF THE BERYL CORONET",
        "XII. THE ADVENTURE OF THE COPPER BEECHES",
    ]
    body = "\n\n".join(f"{heading}\n\nStory {index}" for index, heading in enumerate(headings))
    raw = (
        "Header\n*** START OF THE PROJECT GUTENBERG EBOOK SHERLOCK ***\n"
        "Title and contents\n"
        f"{body}\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SHERLOCK ***\nLicense"
    )

    cleaned = clean_sherlock(raw)
    stories = split_sherlock_stories(cleaned)

    assert cleaned.startswith(headings[0])
    assert "Header" not in cleaned
    assert "License" not in cleaned
    assert len(stories) == 12
    assert stories[-1][1].startswith(headings[-1])


def test_split_sherlock_rejects_incomplete_collection() -> None:
    with pytest.raises(ValueError, match="Expected 12"):
        split_sherlock_stories("I. A SCANDAL IN BOHEMIA\n\nBody\n")


def test_clean_tender_buttons_removes_producer_credit() -> None:
    raw = "Produced by somebody.\n\nTENDER BUTTONS\n\nOBJECTS\n\nBody\n"
    assert clean_tender_buttons(raw) == "TENDER BUTTONS\n\nOBJECTS\n\nBody\n"


def test_prepare_corpora_writes_story_level_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "clean"
    source.mkdir()
    source.joinpath("alice.txt").write_text(
        "*** START OF THE PROJECT GUTENBERG EBOOK 11 ***\n"
        "CHAPTER I.\nBody\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK 11 ***\n",
        encoding="utf-8",
    )
    headings = [
        "I. A SCANDAL IN BOHEMIA",
        "II. THE RED-HEADED LEAGUE",
        "III. A CASE OF IDENTITY",
        "IV. THE BOSCOMBE VALLEY MYSTERY",
        "V. THE FIVE ORANGE PIPS",
        "VI. THE MAN WITH THE TWISTED LIP",
        "VII. THE ADVENTURE OF THE BLUE CARBUNCLE",
        "VIII. THE ADVENTURE OF THE SPECKLED BAND",
        "IX. THE ADVENTURE OF THE ENGINEER’S THUMB",
        "X. THE ADVENTURE OF THE NOBLE BACHELOR",
        "XI. THE ADVENTURE OF THE BERYL CORONET",
        "XII. THE ADVENTURE OF THE COPPER BEECHES",
    ]
    stories = "\n\n".join(f"{heading}\n\nBody {index}" for index, heading in enumerate(headings))
    source.joinpath("sherlock.txt").write_text(
        "*** START OF THE PROJECT GUTENBERG EBOOK SHERLOCK ***\n"
        f"{stories}\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SHERLOCK ***\n",
        encoding="utf-8",
    )
    source.joinpath("tenderbuttons.txt").write_text(
        "Produced by somebody\n\nTENDER BUTTONS\n\nBody\n",
        encoding="utf-8",
    )
    source.joinpath("shakewpeare.txt").write_text("Play\r\n\r\n", encoding="utf-8")

    prepare_corpora(source, output)

    assert len(list((output / "sherlock_stories").glob("*.txt"))) == 12
    assert "IX. THE ADVENTURE" not in (
        output / "sherlock_train.txt"
    ).read_text(encoding="utf-8")
    assert (output / "sherlock_validation.txt").read_text(
        encoding="utf-8"
    ).startswith("IX. THE ADVENTURE")
    assert (output / "sherlock_test.txt").read_text(
        encoding="utf-8"
    ).startswith("XI. THE ADVENTURE")
