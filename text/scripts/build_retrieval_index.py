"""Persist a simple retrieval index for processed continuation data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_continuation.retrieval import fit_tfidf_retriever


def load_candidate_records(dataset_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            target = record.get("target")
            if not target:
                raise ValueError(
                    f"Invalid retrieval dataset row in {dataset_path}: missing non-empty 'target' field"
                )
            records.append(
                {
                    "text": target,
                    "book_id": record.get("book_id"),
                    "source_title": record.get("source_title"),
                    "paragraph_index": record.get("paragraph_index"),
                    "split": record.get("split"),
                }
            )
    return records


def build_retrieval_index(dataset_path: Path, output_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Retrieval dataset not found at {dataset_path}. Build the processed dataset before indexing."
        )

    candidate_records = load_candidate_records(dataset_path)
    if not candidate_records:
        raise ValueError(
            f"No retrieval candidates found in {dataset_path}. Ensure the processed dataset contains non-empty rows."
        )

    candidates = [record["text"] for record in candidate_records]
    index = fit_tfidf_retriever(candidates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(
            {
                "vectorizer": index["vectorizer"],
                "matrix": index["matrix"],
                "candidates": candidates,
                "candidate_records": candidate_records,
                "dataset_path": str(dataset_path),
            },
            handle,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save a TF-IDF retrieval index.")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/retrieval/tfidf_index.pkl"))
    args = parser.parse_args()
    build_retrieval_index(args.dataset_path, args.output_path)


if __name__ == "__main__":
    main()
