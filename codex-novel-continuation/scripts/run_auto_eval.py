"""Run lightweight automatic evaluation over generated continuations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_continuation.evaluation import evaluate_generated_rows, write_metrics_csv


def load_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Generated samples file not found: {path}. Run scripts/generate_samples.py first."
        )
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(
            f"Generated samples file is empty: {path}. Expected rows with prompt, gold_target, and generated_text."
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute automatic metrics for generated continuations.")
    parser.add_argument("--generated-path", type=Path, default=PROJECT_ROOT / "artifacts" / "eval" / "generated_samples.jsonl")
    parser.add_argument("--output-path", type=Path, default=PROJECT_ROOT / "artifacts" / "eval" / "metrics.csv")
    args = parser.parse_args()

    rows = load_jsonl(args.generated_path)
    metrics = evaluate_generated_rows(rows)
    write_metrics_csv(metrics, args.output_path)


if __name__ == "__main__":
    main()
