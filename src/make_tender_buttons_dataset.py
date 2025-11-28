#!/usr/bin/env python3
"""
Download and prepare a Tender Buttons dataset for the tiny LLM.

- Downloads Project Gutenberg's plain-text UTF-8 edition of *Tender Buttons*
- Strips the Gutenberg header/footer
- Does light whitespace cleanup
- Writes to data/input.txt
"""

from __future__ import annotations

import pathlib
import textwrap

import urllib.request


GUTENBERG_URL = "https://www.gutenberg.org/ebooks/15396.txt.utf-8"

# These marker strings appear in the Project Gutenberg file and let us
# cut off the boilerplate. Keep them short to avoid brittle matching.
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK TENDER BUTTONS ***"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK TENDER BUTTONS ***"


def download_raw_text(url: str) -> str:
    print(f"Downloading Tender Buttons from:\n  {url}")
    with urllib.request.urlopen(url) as resp:
        raw_bytes = resp.read()
    text = raw_bytes.decode("utf-8", errors="replace")
    print(f"Downloaded {len(text)} characters.")
    return text


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove the standard Project Gutenberg header/footer using the
    START_MARKER and END_MARKER strings.
    """
    start_idx = text.find(START_MARKER)
    if start_idx == -1:
        raise RuntimeError("Could not find START_MARKER in text")

    # Move to the line just after the marker
    start_idx = text.find("\n", start_idx)
    if start_idx == -1:
        raise RuntimeError("Malformed text around START_MARKER")
    start_idx += 1

    end_idx = text.find(END_MARKER, start_idx)
    if end_idx == -1:
        raise RuntimeError("Could not find END_MARKER in text")

    core = text[start_idx:end_idx]
    print(f"Core text length after stripping boilerplate: {len(core)} characters.")
    return core


def normalize_whitespace(text: str) -> str:
    """
    Light cleanup:
    - Normalize Windows-style newlines
    - Collapse runs of more than 2 blank lines into exactly 2
    - Strip trailing spaces
    We DO NOT touch Stein's internal spacing or punctuation.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines: list[str] = [ln.rstrip(" \t") for ln in text.split("\n")]

    cleaned_lines: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            # Keep at most 2 consecutive blank lines
            if blank_run <= 2:
                cleaned_lines.append("")
        else:
            blank_run = 0
            cleaned_lines.append(ln)

    cleaned = "\n".join(cleaned_lines).strip("\n") + "\n"
    print(f"Cleaned text length: {len(cleaned)} characters.")
    return cleaned


def ensure_data_dir() -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def write_dataset(text: str, data_dir: pathlib.Path) -> pathlib.Path:
    out_path = data_dir / "input.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote dataset to: {out_path} (size: {out_path.stat().st_size} bytes)")
    return out_path


def main() -> None:
    print("Preparing Tender Buttons dataset for tiny LLM...\n")

    raw = download_raw_text(GUTENBERG_URL)
    core = strip_gutenberg_boilerplate(raw)
    cleaned = normalize_whitespace(core)
    data_dir = ensure_data_dir()
    out_path = write_dataset(cleaned, data_dir)

    print(
        textwrap.dedent(
            f"""
            Done.

            You can now train using this dataset by running (from the project root):

              python Main.py

            The model will read from:
              {out_path}
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
