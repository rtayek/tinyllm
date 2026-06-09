from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


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
    r"(?P<title>[A-Z][A-Z ’'\-]+)$",
    re.MULTILINE,
)
CHAPTER_HEADING = re.compile(
    r"^CHAPTER (?P<number>[IVXLCDM]+)\.?\s*$",
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
WORKS = (SHERLOCK, ALICE, PRIDE)


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.decode().lower()).strip("-")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _file_metadata(path: Path, relative_to: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _join_units(units: Sequence[tuple[str, str]]) -> str:
    return normalize_text("\n\n".join(text for _heading, text in units))


def prepare_work(root: Path, spec: WorkSpec) -> Path:
    work_dir = root / spec.identifier
    raw_path = work_dir / "raw" / "gutenberg.txt"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing stored source: {raw_path}")

    clean_text = spec.cleaner(raw_path.read_text(encoding="utf-8"))
    units = spec.splitter(clean_text)
    if len(units) != spec.expected_units:
        raise ValueError(
            f"{spec.identifier}: expected {spec.expected_units} units, found {len(units)}"
        )

    clean_path = work_dir / "clean" / "full.txt"
    write_text(clean_path, clean_text)
    unit_dir_name = "stories" if spec.unit_type == "story" else "chapters"
    unit_dir = work_dir / "units" / unit_dir_name
    unit_records: list[dict[str, object]] = []
    for index, (heading, unit_text) in enumerate(units, start=1):
        unit_path = unit_dir / f"{index:02d}-{slugify(heading)}.txt"
        write_text(unit_path, unit_text)
        unit_records.append(
            {"index": index, "heading": heading}
            | _file_metadata(unit_path, work_dir)
        )

    train_count, validation_count, test_count = spec.split_counts
    if train_count + validation_count + test_count != len(units):
        raise ValueError(f"Split counts do not cover all units for {spec.identifier}")
    selections = {
        "train": units[:train_count],
        "validation": units[train_count : train_count + validation_count],
        "test": units[train_count + validation_count :],
    }
    split_records: dict[str, object] = {"strategy": "ordered logical units"}
    offset = 0
    for split_name, selected in selections.items():
        split_path = work_dir / "splits" / f"{split_name}.txt"
        write_text(split_path, _join_units(selected))
        split_records[split_name] = (
            {"units": list(range(offset + 1, offset + len(selected) + 1))}
            | _file_metadata(split_path, work_dir)
        )
        offset += len(selected)

    manifest = {
        "schema_version": 1,
        "id": spec.identifier,
        "author": spec.author,
        "title": spec.title,
        "language": "en",
        "genre": "fiction",
        "encoding": "utf-8",
        "unit_type": spec.unit_type,
        "source": {
            "provider": "Project Gutenberg",
            "ebook_id": spec.ebook_id,
            "url": spec.source_url,
        }
        | _file_metadata(raw_path, work_dir),
        "cleaning": {"tool": "llm.corpus", "version": 1}
        | _file_metadata(clean_path, work_dir),
        "units": unit_records,
        "splits": split_records,
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def prepare_corpora(root: Path) -> dict[str, Path]:
    return {spec.identifier: prepare_work(root, spec) for spec in WORKS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical fiction corpora")
    parser.add_argument("--root", type=Path, default=Path("corpora"))
    parser.add_argument(
        "--download-pride",
        action="store_true",
        help="Download Pride and Prejudice if its stored raw source is missing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pride_source = args.root / PRIDE.identifier / "raw" / "gutenberg.txt"
    if args.download_pride and not pride_source.exists():
        pride_source.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PRIDE.source_url, pride_source)
    for identifier, path in prepare_corpora(args.root).items():
        print(f"{identifier}: {path}")


if __name__ == "__main__":
    main()
