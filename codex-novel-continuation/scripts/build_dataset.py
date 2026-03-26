"""Build paragraph-level continuation datasets from raw Sherlock Holmes texts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_continuation.data_sources import SHERLOCK_SOURCES
from novel_continuation.dataset_builder import assign_split, build_examples, serialise_examples
from novel_continuation.preprocess import filter_short_paragraphs, split_paragraphs, strip_gutenberg_boilerplate


def build_dataset(raw_dir: Path, output_dir: Path, context_size: int, min_chars: int) -> dict[str, list[dict]]:
    split_examples: dict[str, list[dict]] = defaultdict(list)

    for source in SHERLOCK_SOURCES:
        raw_path = raw_dir / f"{source['book_id']}.txt"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing raw text for book_id={source['book_id']} "
                f"({source['title']}) at {raw_path}. "
                "Run scripts/download_gutenberg.py first to download the corpus."
            )
        text = raw_path.read_text(encoding="utf-8")
        text = strip_gutenberg_boilerplate(text)
        paragraphs = filter_short_paragraphs(split_paragraphs(text), min_chars=min_chars)
        examples = build_examples(
            paragraphs,
            context_size=context_size,
            book_id=source["book_id"],
            source_title=source["title"],
        )

        for index, example in enumerate(examples):
            split_name = assign_split(index, len(examples))
            example["split"] = split_name
            split_examples[split_name].append(example)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        serialise_examples(split_examples[split_name], output_dir / f"{split_name}.jsonl")

    return split_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed continuation datasets from raw texts.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=40)
    args = parser.parse_args()
    build_dataset(args.raw_dir, args.output_dir, args.context_size, args.min_chars)


if __name__ == "__main__":
    main()
