from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Sequence


START_MARKER = re.compile(
    r"^\*\*\* START OF .*?\*\*\*\s*$",
    re.MULTILINE | re.DOTALL,
)
END_MARKER = re.compile(
    r"^\*\*\* END OF .*?\*\*\*\s*$",
    re.MULTILINE | re.DOTALL,
)
SHERLOCK_STORY_HEADING = re.compile(
    r"^(?P<number>I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\. "
    r"(?P<title>[A-Z][A-Z ’'\-]+)$",
    re.MULTILINE,
)


def normalize_text(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip(" \t") for line in text.split("\n")]
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            blank_count = 0
            cleaned_lines.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append("")
    return "\n".join(cleaned_lines).strip() + "\n"


def strip_gutenberg_wrapper(text: str) -> str:
    start = START_MARKER.search(text)
    end = END_MARKER.search(text, start.end() if start else 0)
    if start is None or end is None:
        raise ValueError("Expected Project Gutenberg START and END markers")
    return text[start.end() : end.start()]


def clean_alice(text: str) -> str:
    core = strip_gutenberg_wrapper(normalize_text(text))
    chapter_start = re.search(r"^CHAPTER I\.\s*$", core, re.MULTILINE)
    if chapter_start is None:
        raise ValueError("Could not find Alice chapter-one heading")
    return normalize_text(core[chapter_start.start() :])


def clean_sherlock(text: str) -> str:
    core = strip_gutenberg_wrapper(normalize_text(text))
    first_story = SHERLOCK_STORY_HEADING.search(core)
    if first_story is None:
        raise ValueError("Could not find first Sherlock story heading")
    return normalize_text(core[first_story.start() :])


def clean_tender_buttons(text: str) -> str:
    normalized = normalize_text(text)
    title = re.search(r"^TENDER BUTTONS\s*$", normalized, re.MULTILINE)
    if title is None:
        raise ValueError("Could not find Tender Buttons title")
    return normalize_text(normalized[title.start() :])


def split_sherlock_stories(text: str) -> list[tuple[str, str]]:
    matches = list(SHERLOCK_STORY_HEADING.finditer(text))
    if len(matches) != 12:
        raise ValueError(f"Expected 12 Sherlock stories, found {len(matches)}")

    stories: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group("title").title()
        stories.append((title, normalize_text(text[match.start() : end])))
    return stories


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.decode().lower()).strip("-")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def join_stories(stories: Sequence[tuple[str, str]]) -> str:
    return normalize_text("\n\n".join(text for _title, text in stories))


def prepare_corpora(source_dir: Path, output_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    alice = clean_alice((source_dir / "alice.txt").read_text(encoding="utf-8"))
    outputs["alice"] = output_dir / "alice.txt"
    write_text(outputs["alice"], alice)

    sherlock = clean_sherlock(
        (source_dir / "sherlock.txt").read_text(encoding="utf-8")
    )
    outputs["sherlock"] = output_dir / "sherlock.txt"
    write_text(outputs["sherlock"], sherlock)

    stories = split_sherlock_stories(sherlock)
    story_dir = output_dir / "sherlock_stories"
    for index, (title, story) in enumerate(stories, start=1):
        write_text(story_dir / f"{index:02d}-{slugify(title)}.txt", story)

    # Story-level split: 8 train, 2 validation, 2 test.
    for split_name, selected in (
        ("train", stories[:8]),
        ("validation", stories[8:10]),
        ("test", stories[10:]),
    ):
        path = output_dir / f"sherlock_{split_name}.txt"
        write_text(path, join_stories(selected))
        outputs[f"sherlock_{split_name}"] = path

    tender_buttons = clean_tender_buttons(
        (source_dir / "tenderbuttons.txt").read_text(encoding="utf-8")
    )
    outputs["tenderbuttons"] = output_dir / "tenderbuttons.txt"
    write_text(outputs["tenderbuttons"], tender_buttons)

    shakespeare = normalize_text(
        (source_dir / "shakewpeare.txt").read_text(encoding="utf-8")
    )
    outputs["shakespeare"] = output_dir / "shakespeare.txt"
    write_text(outputs["shakespeare"], shakespeare)

    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned training corpora")
    parser.add_argument("--source-dir", type=Path, default=Path("fixtureData"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtureData/clean"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = prepare_corpora(args.source_dir, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
