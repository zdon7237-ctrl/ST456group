"""Text cleaning helpers for Project Gutenberg narrative data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


START_PATTERN = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
    re.IGNORECASE | re.MULTILINE,
)
END_PATTERN = re.compile(
    r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
    re.IGNORECASE | re.MULTILINE,
)
HEADING_TOKEN_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9\s'.,:&-]*$")
ROMAN_NUMERAL_PATTERN = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)


def strip_gutenberg_boilerplate(text: str) -> str:
    start_match = START_PATTERN.search(text)
    if start_match:
        text = text[start_match.end() :]

    end_match = END_PATTERN.search(text)
    if end_match:
        text = text[: end_match.start()]
    return text.strip()


def normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _is_heading(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return True
    if re.fullmatch(r"chapter\s+[ivxlcdm\d]+", stripped, flags=re.IGNORECASE):
        return True
    words = stripped.split()
    compact = "".join(words)
    if (
        1 < len(words) <= 6
        and len(compact) <= 40
        and HEADING_TOKEN_PATTERN.fullmatch(stripped)
        and (
            any(word.isdigit() or ROMAN_NUMERAL_PATTERN.fullmatch(word) for word in words)
            or all(word.isupper() and len(word) >= 3 for word in words)
        )
        and not stripped.endswith((".", "!", "?"))
    ):
        return True
    return False


def split_paragraphs(text: str) -> list[str]:
    text = normalise_whitespace(text)
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    paragraphs = [block for block in blocks if not _is_heading(block)]
    return paragraphs


def filter_short_paragraphs(paragraphs: list[str], min_chars: int = 40) -> list[str]:
    return [paragraph for paragraph in paragraphs if len(paragraph.strip()) >= min_chars]


def preprocess_file(input_path: Path, output_path: Path, min_chars: int = 40) -> None:
    text = input_path.read_text(encoding="utf-8")
    text = strip_gutenberg_boilerplate(text)
    paragraphs = split_paragraphs(text)
    paragraphs = filter_short_paragraphs(paragraphs, min_chars=min_chars)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(paragraphs), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a Project Gutenberg text into clean paragraphs.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--min-chars", type=int, default=40)
    args = parser.parse_args()
    preprocess_file(args.input_path, args.output_path, min_chars=args.min_chars)


if __name__ == "__main__":
    main()
