"""Train the plain paragraph-continuation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_continuation.training import load_training_config, train_baseline_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline_distilgpt2.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_training_config(args.config)
    # Let the CLI seed win over the config file.
    config["seed"] = args.seed
    train_baseline_model(config)


if __name__ == "__main__":
    main()
