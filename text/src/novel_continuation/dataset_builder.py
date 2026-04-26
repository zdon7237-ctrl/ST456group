"""Build continuation examples from paragraph lists."""

from __future__ import annotations

import json
from pathlib import Path


def build_examples(
    paragraphs: list[str],
    context_size: int,
    book_id: str = "",
    source_title: str = "",
    chapter_id: int | None = None,
) -> list[dict]:
    if context_size < 1:
        raise ValueError("context_size must be at least 1")

    examples: list[dict] = []
    for idx in range(context_size, len(paragraphs)):
        examples.append(
            {
                "book_id": book_id,
                "source_title": source_title,
                "chapter_id": chapter_id,
                "paragraph_index": idx,
                "target_paragraph_index": idx,
                "context": paragraphs[idx - context_size : idx],
                "target": paragraphs[idx],
            }
        )
    return examples


def assign_split(example_index: int, total_examples: int) -> str:
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if total_examples == 1:
        return "train"

    train_cutoff = max(1, int(total_examples * 0.8))
    val_cutoff = max(train_cutoff + 1, int(total_examples * 0.9))

    if val_cutoff >= total_examples:
        val_cutoff = total_examples - 1
    if train_cutoff >= val_cutoff:
        train_cutoff = max(1, val_cutoff - 1)

    if example_index < train_cutoff:
        return "train"
    if example_index < val_cutoff:
        return "val"
    return "test"


def serialise_examples(examples: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
