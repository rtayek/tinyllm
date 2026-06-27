from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitFingerprint:
    split: str
    path: str
    resolvedPath: str
    sha256: str | None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_fingerprints(
    paths: dict[str, str | None],
) -> list[SplitFingerprint]:
    fingerprints: list[SplitFingerprint] = []
    for split, raw_path in paths.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        fingerprints.append(
            SplitFingerprint(
                split=split,
                path=raw_path,
                resolvedPath=str(path.resolve()),
                sha256=_sha256(path),
            )
        )
    return fingerprints


def assert_no_exact_split_reuse(paths: dict[str, str | None]) -> None:
    fingerprints = split_fingerprints(paths)
    seen_paths: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    for fingerprint in fingerprints:
        previous_split = seen_paths.get(fingerprint.resolvedPath)
        if previous_split is not None:
            raise ValueError(
                f"{fingerprint.split} split reuses the same file as "
                f"{previous_split}: {fingerprint.path}"
            )
        seen_paths[fingerprint.resolvedPath] = fingerprint.split

        if fingerprint.sha256 is None:
            continue
        previous_hash_split = seen_hashes.get(fingerprint.sha256)
        if previous_hash_split is not None:
            raise ValueError(
                f"{fingerprint.split} split is byte-identical to "
                f"{previous_hash_split}: {fingerprint.path}"
            )
        seen_hashes[fingerprint.sha256] = fingerprint.split
