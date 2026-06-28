"""Prepare Gertrude Stein's Tender Buttons in the current corpus layout.

The script downloads Project Gutenberg's UTF-8 text, strips the wrapper,
normalizes whitespace, splits the three major sections, and writes:

    corpora/gertrude-stein/tender-buttons/
      raw/gutenberg.txt
      clean/full.txt
      units/sections/
      splits/train.txt
      splits/validation.txt
      splits/test.txt
      manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Sequence

from llm.corpus_sources import normalize_text
from llm.json_utils import write_json

GUTENBERG_URL = "https://www.gutenberg.org/ebooks/15396.txt.utf-8"
IDENTIFIER = "gertrude-stein/tender-buttons"
SECTION_HEADING = re.compile(r"^(OBJECTS|FOOD|ROOMS)\.?\s*$", re.MULTILINE)
START_MARKER = re.compile(r"^\*\*\* START OF .*?\*\*\*\s*$", re.MULTILINE)
END_MARKER = re.compile(r"^\*\*\* END OF .*?\*\*\*\s*$", re.MULTILINE)


def download_raw_text(url: str = GUTENBERG_URL) -> str:
    print(f"Downloading Tender Buttons from:\n  {url}")
    with urllib.request.urlopen(url) as response:
        raw_bytes = response.read()
    text = raw_bytes.decode("utf-8", errors="replace")
    print(f"Downloaded {len(text)} characters.")
    return text


def strip_gutenberg_wrapper(text: str) -> str:
    normalized = normalize_text(text)
    start = START_MARKER.search(normalized)
    end = END_MARKER.search(normalized, start.end() if start else 0)
    if start is None or end is None:
        raise ValueError("Expected Project Gutenberg START and END markers")
    return normalized[start.end() : end.start()]


def clean_tender_buttons(text: str) -> str:
    core = strip_gutenberg_wrapper(text)
    first_section = SECTION_HEADING.search(core)
    if first_section is None:
        raise ValueError("Could not find the first Tender Buttons section heading")
    return normalize_text(core[first_section.start() :])


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_HEADING.finditer(text))
    if len(matches) != 3:
        raise ValueError(f"Expected 3 Tender Buttons sections, found {len(matches)}")
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(1).title()
        sections.append((heading, normalize_text(text[match.start() : end])))
    return sections


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def file_metadata(path: Path, relative_to: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def prepare_tender_buttons(root: Path, url: str = GUTENBERG_URL) -> Path:
    work_dir = root / IDENTIFIER
    raw_path = work_dir / "raw" / "gutenberg.txt"
    write_text(raw_path, download_raw_text(url))

    clean_text = clean_tender_buttons(raw_path.read_text(encoding="utf-8"))
    clean_path = work_dir / "clean" / "full.txt"
    write_text(clean_path, clean_text)

    sections = split_sections(clean_text)
    unit_records: list[dict[str, object]] = []
    unit_dir = work_dir / "units" / "sections"
    for index, (heading, section_text) in enumerate(sections, start=1):
        unit_path = unit_dir / f"{index:02d}-{heading.lower()}.txt"
        write_text(unit_path, section_text)
        unit_records.append(
            {"index": index, "heading": heading} | file_metadata(unit_path, work_dir)
        )

    split_records: dict[str, object] = {"strategy": "major sections"}
    for split_name, (heading, section_text) in zip(
        ("train", "validation", "test"),
        sections,
        strict=True,
    ):
        split_path = work_dir / "splits" / f"{split_name}.txt"
        write_text(split_path, section_text)
        split_records[split_name] = (
            {"section": heading} | file_metadata(split_path, work_dir)
        )

    manifest = {
        "schema_version": 1,
        "id": IDENTIFIER,
        "author": "Gertrude Stein",
        "title": "Tender Buttons",
        "language": "en",
        "genre": "poetry",
        "encoding": "utf-8",
        "unit_type": "section",
        "source": {
            "provider": "Project Gutenberg",
            "ebook_id": "15396",
            "url": url,
        }
        | file_metadata(raw_path, work_dir),
        "cleaning": {"tool": "scripts.make_tender_buttons_dataset", "version": 2}
        | file_metadata(clean_path, work_dir),
        "units": unit_records,
        "splits": split_records,
    }
    manifest_path = work_dir / "manifest.json"
    write_json(manifest_path, manifest, ensure_ascii=False)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Tender Buttons corpus")
    parser.add_argument("--root", type=Path, default=Path("corpora"))
    parser.add_argument("--url", default=GUTENBERG_URL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = prepare_tender_buttons(args.root, args.url)
    print(f"{IDENTIFIER}: {manifest_path}")


if __name__ == "__main__":
    main()
