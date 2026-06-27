from __future__ import annotations

from pathlib import Path

import pytest

from llm.split_preflight import assert_no_exact_split_reuse, split_fingerprints


def test_split_fingerprints_record_existing_hashes(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("train", encoding="utf-8")
    validation.write_text("validation", encoding="utf-8")

    fingerprints = split_fingerprints(
        {"train": str(train), "validation": str(validation), "test": None}
    )

    assert [fingerprint.split for fingerprint in fingerprints] == [
        "train",
        "validation",
    ]
    assert all(fingerprint.sha256 is not None for fingerprint in fingerprints)


def test_split_preflight_rejects_same_path(tmp_path: Path) -> None:
    split = tmp_path / "split.txt"
    split.write_text("same", encoding="utf-8")

    with pytest.raises(ValueError, match="reuses the same file"):
        assert_no_exact_split_reuse(
            {"train": str(split), "validation": str(split), "test": None}
        )


def test_split_preflight_rejects_byte_identical_files(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("same", encoding="utf-8")
    validation.write_text("same", encoding="utf-8")

    with pytest.raises(ValueError, match="byte-identical"):
        assert_no_exact_split_reuse(
            {"train": str(train), "validation": str(validation), "test": None}
        )
