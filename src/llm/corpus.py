from __future__ import annotations

# This module intentionally re-exports corpus source helpers for compatibility.
# pyright: reportUnusedImport=false

import argparse
import hashlib
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Sequence

from llm.corpus_sources import (
    ALICE,
    AUSTEN_WORKS,
    CHAPTER_HEADING,
    EMMA,
    END_MARKER,
    ILLUSTRATION_BLOCK,
    MANSFIELD_PARK,
    NORTHANGER_ABBEY,
    PERSUASION,
    PRIDE,
    SENSE_AND_SENSIBILITY,
    SHERLOCK,
    SHERLOCK_HEADING,
    START_MARKER,
    WORKS,
    WorkSpec,
    clean_alice,
    clean_austen,
    clean_pride_and_prejudice,
    clean_sherlock,
    normalize_text,
    split_alice_chapters,
    split_emma_chapters,
    split_mansfield_park_chapters,
    split_northanger_abbey_chapters,
    split_persuasion_chapters,
    split_pride_chapters,
    split_sense_and_sensibility_chapters,
    split_sherlock_stories,
    strip_gutenberg_wrapper,
)
from llm.json_utils import write_json


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
    write_json(manifest_path, manifest, ensure_ascii=False)
    return manifest_path


def prepare_corpora(root: Path) -> dict[str, Path]:
    return {spec.identifier: prepare_work(root, spec) for spec in WORKS}


def prepare_combined_corpus(
    root: Path,
    specs: Sequence[WorkSpec] = AUSTEN_WORKS,
    identifier: str = "jane-austen/combined",
) -> Path:
    combined_dir = root / identifier
    split_records: dict[str, object] = {"strategy": "concatenate prepared work splits"}
    for split_name in ("train", "validation", "test"):
        parts: list[str] = []
        sources: list[dict[str, object]] = []
        for spec in specs:
            split_path = root / spec.identifier / "splits" / f"{split_name}.txt"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing prepared split: {split_path}")
            parts.append(split_path.read_text(encoding="utf-8"))
            sources.append(
                {
                    "id": spec.identifier,
                    "split": split_name,
                }
                | _file_metadata(split_path, root)
            )
        output_path = combined_dir / "splits" / f"{split_name}.txt"
        write_text(output_path, _join_units([("", text) for text in parts]))
        split_records[split_name] = (
            {"sources": sources}
            | _file_metadata(output_path, combined_dir)
        )

    manifest = {
        "schema_version": 1,
        "id": identifier,
        "author": "Jane Austen",
        "title": "Combined Jane Austen corpus",
        "language": "en",
        "genre": "fiction",
        "encoding": "utf-8",
        "unit_type": "combined-split",
        "source": {
            "provider": "local prepared corpora",
            "works": [spec.identifier for spec in specs],
        },
        "splits": split_records,
    }
    manifest_path = combined_dir / "manifest.json"
    write_json(manifest_path, manifest, ensure_ascii=False)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical fiction corpora")
    parser.add_argument("--root", type=Path, default=Path("corpora"))
    parser.add_argument(
        "--download-pride",
        action="store_true",
        help="Download Pride and Prejudice if its stored raw source is missing",
    )
    parser.add_argument(
        "--download-austen",
        action="store_true",
        help="Download all five remaining Austen novels if their stored raw sources are missing",
    )
    parser.add_argument(
        "--combined-austen",
        action="store_true",
        help="Build corpora/jane-austen/combined from prepared Austen splits",
    )
    return parser.parse_args(argv)


def _download_if_missing(root: Path, spec: WorkSpec) -> None:
    source = root / spec.identifier / "raw" / "gutenberg.txt"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(spec.source_url, source)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.download_pride:
        _download_if_missing(args.root, PRIDE)
    if args.download_austen:
        for spec in (SENSE_AND_SENSIBILITY, EMMA, MANSFIELD_PARK, PERSUASION, NORTHANGER_ABBEY):
            _download_if_missing(args.root, spec)
    for identifier, path in prepare_corpora(args.root).items():
        print(f"{identifier}: {path}")
    if args.combined_austen:
        print(f"jane-austen/combined: {prepare_combined_corpus(args.root)}")


if __name__ == "__main__":
    main()
