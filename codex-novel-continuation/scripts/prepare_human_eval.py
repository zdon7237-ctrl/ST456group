"""Prepare blinded CSV files for human evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "artifacts" / "eval" / "generated_samples.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "eval" / "human_eval.csv"

REQUIRED_FIELDS = ("sample_id", "prompt", "gold_target", "generated_text")
RATING_FIELDS = (
    "contextual_coherence",
    "narrative_consistency",
    "character_consistency",
    "fluency",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Generated samples file not found: {path}. Run scripts/generate_samples.py first."
        )

    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    if not rows:
        raise ValueError(
            f"Generated samples file is empty: {path}. Expected JSONL rows with sample outputs."
        )
    return rows


def build_human_eval_rows(rows: list[dict], system_label: str) -> list[dict[str, str]]:
    prepared_rows: list[dict[str, str]] = []
    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                f"Generated sample row is missing required fields: {missing_fields}"
            )

        prepared_row = {
            "sample_id": row["sample_id"],
            "system_label": system_label,
            "prompt": row["prompt"],
            "gold_target": row["gold_target"],
            "generated_text": row["generated_text"],
        }
        for field in RATING_FIELDS:
            prepared_row[field] = ""
        prepared_rows.append(prepared_row)
    return prepared_rows


def write_human_eval_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "system_label",
        "prompt",
        "gold_target",
        "generated_text",
        *RATING_FIELDS,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a blinded CSV template for human evaluation."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--system-label",
        default="System A",
        help="Blinded system label to show reviewers.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input_path)
    prepared_rows = build_human_eval_rows(rows, system_label=args.system_label)
    write_human_eval_csv(prepared_rows, args.output_path)


if __name__ == "__main__":
    main()
