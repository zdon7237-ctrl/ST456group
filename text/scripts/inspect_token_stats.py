"""Inspect token budgets for a continuation experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_continuation.training import (
    attach_retrieval,
    load_jsonl_records,
    load_model_and_tokenizer,
    load_training_config,
    resolve_max_target_tokens,
    summarise_token_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect token statistics for a proposal-aligned experiment config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    config = load_training_config(args.config)
    _model, tokenizer = load_model_and_tokenizer(config["model_name"])
    train_records = load_jsonl_records(Path(config["train_path"]))
    eval_records = load_jsonl_records(Path(config["eval_path"]))
    max_target_tokens = resolve_max_target_tokens(config, train_records, tokenizer)

    if config.get("use_retrieval", False):
        # Attach retrieval here so the token budget matches the real training prompt shape.
        train_targets = [record["target"] for record in train_records]
        eval_targets = [record["target"] for record in eval_records]
        train_records = attach_retrieval(train_records, train_targets, top_k=int(config["top_k"]))
        eval_records = attach_retrieval(
            eval_records,
            eval_targets,
            top_k=int(config["top_k"]),
            history_prefix=train_targets,
        )

    payload = {
        "config_path": str(args.config),
        "model_name": config["model_name"],
        "context_format": config["context_format"],
        "max_length": int(config["max_length"]),
        "recommended_max_target_tokens": int(max_target_tokens),
        "train": summarise_token_budget(
            train_records,
            tokenizer,
            context_format=config["context_format"],
            context_size=int(config["context_size"]),
            use_retrieval=bool(config.get("use_retrieval", False)),
            max_target_tokens=int(max_target_tokens),
            max_length=int(config["max_length"]),
        ),
        "eval": summarise_token_budget(
            eval_records,
            tokenizer,
            context_format=config["context_format"],
            context_size=int(config["context_size"]),
            use_retrieval=bool(config.get("use_retrieval", False)),
            max_target_tokens=int(max_target_tokens),
            max_length=int(config["max_length"]),
        ),
    }

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
