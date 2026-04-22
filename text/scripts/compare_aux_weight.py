"""Compare two E5 aux-weight runs using validation_main_loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_training_config_path(path: Path) -> Path:
    candidate = path / "training_config.json" if path.is_dir() else path
    if not candidate.exists():
        raise FileNotFoundError(
            f"training_config.json not found for {path}. "
            "Pass a checkpoint directory or a direct path to training_config.json."
        )
    return candidate


def load_validation_metrics(path: Path) -> tuple[Path, dict[str, float]]:
    config_path = resolve_training_config_path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    validation = payload.get("metadata", {}).get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"Missing metadata.validation in {config_path}")
    if "validation_main_loss" not in validation:
        raise ValueError(f"Missing metadata.validation.validation_main_loss in {config_path}")
    metrics = {
        "validation_main_loss": float(validation["validation_main_loss"]),
    }
    if "validation_perplexity" in validation:
        metrics["validation_perplexity"] = float(validation["validation_perplexity"])
    return config_path, metrics


def relative_gap_percent(left_loss: float, right_loss: float) -> float:
    baseline = min(left_loss, right_loss)
    if baseline == 0:
        return 0.0 if left_loss == right_loss else float("inf")
    return abs(left_loss - right_loss) / baseline * 100.0


def format_label(path: Path, resolved_path: Path) -> str:
    if path.is_dir():
        return path.name
    if resolved_path.name == "training_config.json" and resolved_path.parent.name:
        return resolved_path.parent.name
    return resolved_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two E5 aux-weight runs and recommend the final variant using validation_main_loss."
    )
    parser.add_argument("left", type=Path, help="Checkpoint directory or training_config.json for the first run.")
    parser.add_argument("right", type=Path, help="Checkpoint directory or training_config.json for the second run.")
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=1.0,
        help="If the relative validation_main_loss gap is at or below this threshold, keep both runs as a robustness check.",
    )
    args = parser.parse_args()

    left_config_path, left_metrics = load_validation_metrics(args.left)
    right_config_path, right_metrics = load_validation_metrics(args.right)
    left_label = format_label(args.left, left_config_path)
    right_label = format_label(args.right, right_config_path)

    left_loss = left_metrics["validation_main_loss"]
    right_loss = right_metrics["validation_main_loss"]
    gap_percent = relative_gap_percent(left_loss, right_loss)

    # Use validation_main_loss only so the test set never becomes a selection signal.
    if gap_percent <= args.threshold_percent:
        recommendation = (
            f"keep both {left_label} and {right_label} as a robustness check "
            f"(gap <= {args.threshold_percent:.2f}%)"
        )
    elif left_loss < right_loss:
        recommendation = f"select {left_label} (lower validation_main_loss)"
    else:
        recommendation = f"select {right_label} (lower validation_main_loss)"

    print(f"left_label: {left_label}")
    print(f"left_config: {left_config_path}")
    print(f"left_validation_main_loss: {left_loss:.6f}")
    if "validation_perplexity" in left_metrics:
        print(f"left_validation_perplexity: {left_metrics['validation_perplexity']:.6f}")
    print(f"right_label: {right_label}")
    print(f"right_config: {right_config_path}")
    print(f"right_validation_main_loss: {right_loss:.6f}")
    if "validation_perplexity" in right_metrics:
        print(f"right_validation_perplexity: {right_metrics['validation_perplexity']:.6f}")
    print(f"relative_gap_percent: {gap_percent:.6f}")
    print(f"threshold_percent: {args.threshold_percent:.2f}")
    print("selection_rule: validation_main_loss only")
    print(f"recommendation: {recommendation}")


if __name__ == "__main__":
    main()
