"""Generate continuation samples from a fine-tuned model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_continuation.prompting import build_prompt
from novel_continuation.training import attach_retrieval, load_trained_model_and_tokenizer, select_context_window

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "eval" / "generated_samples.jsonl"


def load_training_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "training_config.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {path}. Run scripts/build_dataset.py first to create data/processed/test.jsonl."
        )
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Input JSONL is empty: {path}. Expected prompt rows with context and target fields.")
    return rows


def validate_generation_rows(rows: list[dict], use_retrieval: bool) -> None:
    required_fields = {"context", "target"}
    for index, row in enumerate(rows):
        missing_fields = sorted(required_fields - set(row))
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(
                f"Input row {index} is missing required fields: {missing}. Expected fields include context and target."
            )


def ensure_retrieval_rows(
    rows: list[dict],
    history_rows: list[dict] | None,
    top_k: int,
) -> list[dict]:
    if not rows:
        return rows
    if all("retrieved" in row for row in rows):
        return rows

    history_rows = history_rows or []
    history_targets = [row["target"] for row in history_rows if "target" in row]
    eval_targets = [row["target"] for row in rows]
    return attach_retrieval(
        rows,
        retrieval_candidates=eval_targets,
        top_k=top_k,
        history_prefix=history_targets,
    )


def generate_rows(
    rows: list[dict],
    model_dir: Path,
    max_new_tokens: int,
    use_retrieval: bool,
    history_rows: list[dict] | None = None,
    top_k: int = 2,
    context_format: str | None = None,
    do_sample: bool = True,
    top_p: float = 0.95,
    temperature: float = 0.8,
) -> list[dict]:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Train a model first or point --model-dir to a valid checkpoint."
        )

    validate_generation_rows(rows, use_retrieval=use_retrieval)
    if use_retrieval:
        rows = ensure_retrieval_rows(rows, history_rows=history_rows, top_k=top_k)

    model, tokenizer = load_trained_model_and_tokenizer(model_dir)

    training_metadata = load_training_metadata(model_dir)
    saved_config = training_metadata.get("config", {})
    resolved_context_format = context_format or saved_config.get("context_format", "plain")
    resolved_context_size = int(saved_config.get("context_size", len(rows[0]["context"]) if rows else 1))

    generated_rows: list[dict] = []
    try:
        from tqdm import tqdm
        row_iter = tqdm(rows, desc="生成样本", unit="条")
    except ImportError:
        row_iter = rows
    for row in row_iter:
        context_window = select_context_window(row["context"], resolved_context_size)
        retrieved_items = row.get("retrieved", [])
        retrieved_texts = [item["text"] for item in retrieved_items]
        prompt = build_prompt(
            context=context_window,
            retrieved=retrieved_texts,
            include_retrieval=use_retrieval,
            context_format=resolved_context_format,
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p if do_sample else None,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()
        generated_rows.append(
            {
                "sample_id": row.get("paragraph_index"),
                "prompt": prompt,
                "gold_target": row["target"],
                "generated_text": generated_text,
                "book_id": row.get("book_id"),
                "source_title": row.get("source_title"),
            }
        )
    return generated_rows


def write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate continuation samples from a trained model.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--do-sample", action="store_true", default=True,
                        help="Enable sampling; disable with --no-do-sample for greedy decoding.")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling probability threshold (default: 0.95).")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8).")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--use-retrieval", action="store_true")
    parser.add_argument("--context-format", choices=["plain", "structured"], default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_path)
    history_rows = load_jsonl(args.history_path) if args.use_retrieval else None
    generated_rows = generate_rows(
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
    )
    write_jsonl(generated_rows, args.output_path)


if __name__ == "__main__":
    main()
