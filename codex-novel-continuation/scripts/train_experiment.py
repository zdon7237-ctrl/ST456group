"""Train a proposal-aligned continuation experiment from a config file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_continuation.training import load_training_config, train_baseline_model, train_retrieval_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a continuation experiment defined by a config file.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_training_config(args.config)
    if config.get("use_retrieval", False):
        train_retrieval_model(config)
    else:
        train_baseline_model(config)


if __name__ == "__main__":
    main()
