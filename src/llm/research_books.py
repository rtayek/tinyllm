from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchBook:
    name: str
    validationPath: str

    @property
    def validation_path(self) -> Path:
        return Path(self.validationPath)


RESEARCH_BOOKS: tuple[ResearchBook, ...] = (
    ResearchBook(
        "Pride and Prejudice",
        "corpora/jane-austen/pride-and-prejudice/splits/validation.txt",
    ),
    ResearchBook(
        "Sense and Sensibility",
        "corpora/jane-austen/sense-and-sensibility/splits/validation.txt",
    ),
    ResearchBook(
        "Emma",
        "corpora/jane-austen/emma/splits/validation.txt",
    ),
    ResearchBook(
        "Mansfield Park",
        "corpora/jane-austen/mansfield-park/splits/validation.txt",
    ),
    ResearchBook(
        "Persuasion",
        "corpora/jane-austen/persuasion/splits/validation.txt",
    ),
    ResearchBook(
        "Northanger Abbey",
        "corpora/jane-austen/northanger-abbey/splits/validation.txt",
    ),
    ResearchBook(
        "Sherlock Holmes",
        "corpora/arthur-conan-doyle/"
        "adventures-of-sherlock-holmes/splits/validation.txt",
    ),
    ResearchBook(
        "Alice in Wonderland",
        "corpora/lewis-carroll/"
        "alices-adventures-in-wonderland/splits/validation.txt",
    ),
)


def research_book_pairs() -> list[tuple[str, str]]:
    return [(book.name, book.validationPath) for book in RESEARCH_BOOKS]
