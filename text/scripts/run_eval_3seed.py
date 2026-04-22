"""Run 3-seed generation and evaluation for a trained continuation model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from novel_continuation.evaluation import evaluate_generated_rows, summarise_seed_metrics, write_metrics_csv
from novel_continuation.training import load_trained_model_and_tokenizer

from generate_samples import generate_rows, load_jsonl, write_jsonl

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eval"
DEFAULT_SEEDS = (13, 42, 2026)


def load_metrics_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one metrics row in {path}, found {len(rows)}")
    return {key: float(value) for key, value in rows[0].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation and automatic evaluation over multiple fixed seeds.")
    parser.add_argument("--experiment-id", required=True, help="Short experiment identifier such as e1, e3, or e5_w02.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--do-sample", action="store_true", default=True,
                        help="Enable sampling; disable with --no-do-sample for greedy decoding.")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--use-retrieval", action="store_true")
    parser.add_argument("--context-format", choices=["plain", "structured"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse per-seed outputs that already exist and only fill missing pieces.")
    args = parser.parse_args()

    rows = load_jsonl(args.input_path)
    history_rows = load_jsonl(args.history_path) if args.use_retrieval else None
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = args.experiment_id.lower()

    metrics_by_seed: list[dict[str, float]] = []
    model = None
    tokenizer = None

    def ensure_model_loaded():
        nonlocal model, tokenizer
        # Reuse one loaded checkpoint across seeds to avoid repeated startup cost.
        if model is None or tokenizer is None:
            model, tokenizer = load_trained_model_and_tokenizer(args.model_dir)
        return model, tokenizer

    for seed in args.seeds:
        generated_path = output_dir / f"generated_samples_{experiment_id}_seed{seed}.jsonl"
        metrics_path = output_dir / f"metrics_{experiment_id}_seed{seed}.csv"
        # Reuse completed artifacts so a disconnected Colab session can resume mid-run.
        if args.skip_existing and generated_path.exists() and metrics_path.exists():
            metrics_by_seed.append(load_metrics_csv(metrics_path))
            continue

        if args.skip_existing and generated_path.exists() and not metrics_path.exists():
            seed_model, seed_tokenizer = ensure_model_loaded()
            generated = load_jsonl(generated_path)
            metrics = evaluate_generated_rows(generated, model=seed_model, tokenizer=seed_tokenizer)
            write_metrics_csv(metrics, metrics_path)
            metrics_by_seed.append(metrics)
            continue

        seed_model, seed_tokenizer = ensure_model_loaded()
        generated = generate_rows(
            rows=rows,
            model_dir=args.model_dir,
            max_new_tokens=args.max_new_tokens,
            use_retrieval=args.use_retrieval,
            history_rows=history_rows,
            top_k=args.top_k,
            context_format=args.context_format,
            do_sample=args.do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
            seed=seed,
            model=seed_model,
            tokenizer=seed_tokenizer,
        )
        write_jsonl(generated, generated_path)
        metrics = evaluate_generated_rows(generated, model=seed_model, tokenizer=seed_tokenizer)
        write_metrics_csv(metrics, metrics_path)
        metrics_by_seed.append(metrics)

    # Always rebuild the summary from the full set of per-seed metrics that were collected.
    summary = summarise_seed_metrics(metrics_by_seed)
    summary_path = output_dir / f"metrics_{experiment_id}_summary.csv"
    write_metrics_csv(summary, summary_path)


if __name__ == "__main__":
    main()
