"""Training helpers for the proposal-aligned continuation experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from novel_continuation.prompting import build_prompt, build_training_text
from novel_continuation.retrieval import fit_tfidf_retriever, query_tfidf_index
from novel_continuation.trainer_runtime import ContinuationDataCollator, compute_candidate_scores


PATH_KEYS = ("train_path", "eval_path", "output_dir")
DEFAULT_TRAINING_CONFIG = {
    "context_size": 2,
    "context_format": "plain",
    "training_mode": "full",
    "aux_objective": "none",
    "aux_weight": 0.1,
    "max_target_tokens": "auto",
    "num_negative_candidates": 3,
    "selection_metric": "perplexity",
    "use_retrieval": False,
    "top_k": 0,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
}
VALID_CONTEXT_FORMATS = {"plain", "structured"}
VALID_TRAINING_MODES = {"full", "lora"}
VALID_AUX_OBJECTIVES = {"none", "ranking", "classification"}


def select_context_window(context: list[str], context_size: int) -> list[str]:
    if context_size < 1:
        raise ValueError("context_size must be at least 1")
    return context[-context_size:]


def _resolve_path(value: str, config_path: Path) -> str:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return str(raw_path)

    config_dir_candidate = (config_path.parent / raw_path).resolve()
    if config_dir_candidate.exists():
        return str(config_dir_candidate)

    project_root_candidate = (config_path.parent.parent / raw_path).resolve()
    return str(project_root_candidate)


def _normalise_config(config: dict | None) -> dict:
    merged = dict(DEFAULT_TRAINING_CONFIG)
    merged.update(config or {})
    return merged


def load_training_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config = _normalise_config(config)
    for key in PATH_KEYS:
        if key in config and isinstance(config[key], str):
            config[key] = _resolve_path(config[key], config_path)
    return config


def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def load_trained_model_and_tokenizer(model_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config_path = model_dir / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Loading a LoRA checkpoint requires the `peft` package. Install requirements.txt in Colab before inference."
            ) from exc

        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_name = adapter_config["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
    return model, tokenizer


def apply_training_mode(model, config: dict):
    training_mode = config["training_mode"]
    if training_mode == "full":
        return model
    if training_mode != "lora":
        raise ValueError(f"Unsupported training_mode: {training_mode}")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LoRA training requires the `peft` package. Install requirements.txt in Colab before running E4."
        ) from exc

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    return get_peft_model(model, lora_config)


def load_jsonl_records(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. Run scripts/build_dataset.py first."
        )

    with dataset_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    if not records:
        raise ValueError(
            f"Dataset file is empty: {dataset_path}. Run scripts/build_dataset.py to regenerate processed JSONL files."
        )
    return records


def validate_main_config(config: dict) -> None:
    config = _normalise_config(config)
    if config["context_format"] not in VALID_CONTEXT_FORMATS:
        raise ValueError(f"context_format must be one of {sorted(VALID_CONTEXT_FORMATS)}")
    if config["training_mode"] not in VALID_TRAINING_MODES:
        raise ValueError(f"training_mode must be one of {sorted(VALID_TRAINING_MODES)}")
    if config["aux_objective"] not in VALID_AUX_OBJECTIVES:
        raise ValueError(f"aux_objective must be one of {sorted(VALID_AUX_OBJECTIVES)}")


def validate_baseline_config(config: dict) -> None:
    validate_main_config(config)
    if config.get("use_retrieval", False):
        raise ValueError(
            "Baseline training does not support use_retrieval=True. Use the retrieval pipeline only as an appendix experiment."
        )


def validate_retrieval_config(config: dict) -> None:
    validate_main_config(config)
    if not config.get("use_retrieval", False):
        raise ValueError("Retrieval training requires use_retrieval=True.")
    if int(config.get("top_k", 0)) <= 0:
        raise ValueError("Retrieval training requires top_k > 0.")


def choose_max_target_tokens(
    lengths: list[int],
    percentile: float = 0.9,
    round_to: int = 16,
    cap: int = 224,
) -> int:
    if not lengths:
        raise ValueError("lengths must not be empty")

    ordered = sorted(lengths)
    percentile_index = max(0, math.ceil(len(ordered) * percentile) - 1)
    threshold = ordered[min(percentile_index, len(ordered) - 1)]
    rounded = int(math.ceil(threshold / round_to) * round_to)
    return min(rounded, cap)


def _target_token_lengths(records: list[dict], tokenizer) -> list[int]:
    return [
        len(tokenizer(record["target"], add_special_tokens=False)["input_ids"])
        for record in records
    ]


def resolve_max_target_tokens(config: dict, train_records: list[dict], tokenizer) -> int:
    configured = config.get("max_target_tokens", "auto")
    if isinstance(configured, int):
        return configured
    if isinstance(configured, str) and configured.lower() != "auto":
        return int(configured)
    return choose_max_target_tokens(_target_token_lengths(train_records, tokenizer))


def summarise_token_budget(
    records: list[dict],
    tokenizer,
    *,
    context_format: str,
    use_retrieval: bool,
    max_target_tokens: int,
    max_length: int,
    context_size: int,
) -> dict[str, float]:
    total_lengths: list[int] = []
    truncated_count = 0
    for record in records:
        context_window = select_context_window(record["context"], context_size)
        prompt_text = build_prompt(
            context=context_window,
            retrieved=[item["text"] for item in record.get("retrieved", [])],
            include_retrieval=use_retrieval,
            context_format=context_format,
        )
        prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        target_tokens = min(
            len(tokenizer(record["target"], add_special_tokens=False)["input_ids"]),
            max_target_tokens,
        )
        total_length = prompt_tokens + target_tokens
        total_lengths.append(total_length)
        if total_length > max_length:
            truncated_count += 1

    if not total_lengths:
        return {"avg_total_tokens": 0.0, "max_total_tokens": 0.0, "truncation_rate": 0.0}

    return {
        "avg_total_tokens": sum(total_lengths) / len(total_lengths),
        "max_total_tokens": float(max(total_lengths)),
        "truncation_rate": truncated_count / len(total_lengths),
    }


def attach_retrieval(
    records: list[dict],
    retrieval_candidates: list[str],
    top_k: int,
    history_prefix: list[str] | None = None,
) -> list[dict]:
    history_prefix = history_prefix or []
    all_candidates = history_prefix + retrieval_candidates
    retrieval_index = fit_tfidf_retriever(all_candidates) if all_candidates else None
    enriched: list[dict] = []
    for index, record in enumerate(records):
        query = "\n".join(record["context"])
        available_count = len(history_prefix) + min(index, len(retrieval_candidates))
        if retrieval_index is None or available_count == 0:
            retrieved = []
        else:
            retrieved = query_tfidf_index(
                index=retrieval_index,
                query=query,
                top_k=min(top_k, available_count),
                candidate_limit=available_count,
            )
        next_record = dict(record)
        next_record["retrieved"] = retrieved
        enriched.append(next_record)
    return enriched


def select_negative_candidates(records: list[dict], index: int, num_negative_candidates: int) -> list[str]:
    if num_negative_candidates <= 0 or not records:
        return []

    current = records[index]
    preferred: list[str] = []
    fallback: list[str] = []
    for candidate_index, record in enumerate(records):
        if candidate_index == index or record.get("target") == current.get("target"):
            continue

        same_book = current.get("book_id") and record.get("book_id") == current.get("book_id")
        non_adjacent = (
            current.get("paragraph_index") is not None
            and record.get("paragraph_index") is not None
            and abs(int(record["paragraph_index"]) - int(current["paragraph_index"])) > 1
        )
        if same_book:
            if non_adjacent:
                preferred.append(record["target"])
            continue

        if current.get("split") is None or record.get("split") == current.get("split"):
            fallback.append(record["target"])

    ordered: list[str] = []
    for candidate in preferred + fallback:
        if candidate not in ordered:
            ordered.append(candidate)
    if not ordered:
        return []

    negatives = ordered[:num_negative_candidates]
    pointer = 0
    while len(negatives) < num_negative_candidates:
        negatives.append(ordered[pointer % len(ordered)])
        pointer += 1
    return negatives


def prepare_training_records(
    records: list[dict],
    use_retrieval: bool = False,
    top_k: int = 0,
    retrieval_candidates: list[str] | None = None,
    history_prefix: list[str] | None = None,
    context_format: str = "plain",
    context_size: int = 2,
    aux_objective: str = "none",
    num_negative_candidates: int = 3,
) -> list[dict]:
    if use_retrieval:
        if retrieval_candidates is None:
            raise ValueError("retrieval_candidates must be provided when use_retrieval=True")
        records = attach_retrieval(
            records,
            retrieval_candidates=retrieval_candidates,
            top_k=top_k,
            history_prefix=history_prefix,
        )

    prepared = []
    for index, record in enumerate(records):
        context_window = select_context_window(record["context"], context_size)
        retrieved_texts = [item["text"] for item in record.get("retrieved", [])]
        prepared_record = {
            "context": context_window,
            "target": record["target"],
            "retrieved": retrieved_texts,
            "context_format": context_format,
            "context_size": context_size,
            "include_retrieval": use_retrieval,
            "text": build_training_text(
                context=context_window,
                retrieved=retrieved_texts,
                target=record["target"],
                include_retrieval=use_retrieval,
                context_format=context_format,
            ),
        }
        for key in ("book_id", "source_title", "paragraph_index", "split"):
            if key in record:
                prepared_record[key] = record[key]

        if aux_objective != "none":
            negatives = select_negative_candidates(records, index=index, num_negative_candidates=num_negative_candidates)
            if negatives:
                prepared_record["candidate_targets"] = [record["target"], *negatives]
                prepared_record["candidate_labels"] = [1, *([0] * len(negatives))]
        prepared.append(prepared_record)
    return prepared


def build_training_texts(
    records: list[dict],
    use_retrieval: bool = False,
    top_k: int = 0,
    retrieval_candidates: list[str] | None = None,
    history_prefix: list[str] | None = None,
    context_format: str = "plain",
    context_size: int = 2,
    aux_objective: str = "none",
    num_negative_candidates: int = 3,
) -> list[str]:
    prepared = prepare_training_records(
        records,
        use_retrieval=use_retrieval,
        top_k=top_k,
        retrieval_candidates=retrieval_candidates,
        history_prefix=history_prefix,
        context_format=context_format,
        context_size=context_size,
        aux_objective=aux_objective,
        num_negative_candidates=num_negative_candidates,
    )
    return [record["text"] for record in prepared]


def serialise_config_for_json(config: dict) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def save_training_metadata(output_dir: str, config: dict, metadata: dict[str, Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {"config": serialise_config_for_json(config), "metadata": metadata}
    (output_path / "training_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_hf_trainer(config: dict, prepared_train: list[dict], prepared_eval: list[dict], model, tokenizer):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments

    class ListDataset(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            return self.rows[index]

    class ProposalAlignedTrainer(Trainer):
        def __init__(self, *args, aux_objective: str, aux_weight: float, **kwargs):
            super().__init__(*args, **kwargs)
            self.aux_objective = aux_objective
            self.aux_weight = aux_weight

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            candidate_input_ids = inputs.pop("candidate_input_ids", None)
            candidate_attention_mask = inputs.pop("candidate_attention_mask", None)
            candidate_labels = inputs.pop("candidate_labels", None)
            candidate_class_labels = inputs.pop("candidate_class_labels", None)
            candidate_mask = inputs.pop("candidate_mask", None)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )
            total_loss = outputs.loss

            if self.aux_objective != "none" and candidate_input_ids is not None:
                batch_size, num_candidates, sequence_length = candidate_input_ids.shape
                flat_input_ids = candidate_input_ids.view(batch_size * num_candidates, sequence_length)
                flat_attention_mask = candidate_attention_mask.view(batch_size * num_candidates, sequence_length)
                flat_labels = candidate_labels.view(batch_size * num_candidates, sequence_length)
                candidate_outputs = model(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
                scores = compute_candidate_scores(candidate_outputs.logits, flat_labels).view(batch_size, num_candidates)
                if candidate_mask is not None:
                    scores = scores.masked_fill(~candidate_mask, -1e4)

                if self.aux_objective == "ranking":
                    aux_targets = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
                    aux_loss = F.cross_entropy(scores, aux_targets)
                else:
                    aux_targets = candidate_class_labels.float()
                    bce_loss = F.binary_cross_entropy_with_logits(scores, aux_targets, reduction="none")
                    aux_loss = (bce_loss * candidate_mask.float()).sum() / candidate_mask.float().sum().clamp(min=1.0)
                total_loss = total_loss + (self.aux_weight * aux_loss)

            return (total_loss, outputs) if return_outputs else total_loss

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_train_epochs=float(config["num_train_epochs"]),
        learning_rate=float(config["learning_rate"]),
        logging_steps=int(config["logging_steps"]),
        save_steps=int(config["save_steps"]),
        warmup_steps=int(config.get("warmup_steps", 0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
    )

    return ProposalAlignedTrainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(prepared_train),
        eval_dataset=ListDataset(prepared_eval),
        tokenizer=tokenizer,
        data_collator=ContinuationDataCollator(
            tokenizer,
            max_length=int(config["max_length"]),
            max_target_tokens=int(config["max_target_tokens"]),
            aux_objective=config["aux_objective"],
        ),
        aux_objective=config["aux_objective"],
        aux_weight=float(config["aux_weight"]),
    )


def build_continuation_trainer(config: dict, *, use_retrieval: bool):
    config = _normalise_config(config)
    if use_retrieval:
        validate_retrieval_config(config)
    else:
        validate_baseline_config(config)

    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    model = apply_training_mode(model, config)
    train_records = load_jsonl_records(Path(config["train_path"]))
    eval_records = load_jsonl_records(Path(config["eval_path"]))
    config["max_target_tokens"] = resolve_max_target_tokens(config, train_records, tokenizer)

    train_targets = [record["target"] for record in train_records]
    eval_targets = [record["target"] for record in eval_records]
    prepared_train = prepare_training_records(
        train_records,
        use_retrieval=use_retrieval,
        top_k=config.get("top_k", 0),
        retrieval_candidates=train_targets if use_retrieval else None,
        context_format=config["context_format"],
        context_size=int(config["context_size"]),
        aux_objective=config["aux_objective"],
        num_negative_candidates=int(config["num_negative_candidates"]),
    )
    prepared_eval = prepare_training_records(
        eval_records,
        use_retrieval=use_retrieval,
        top_k=config.get("top_k", 0),
        retrieval_candidates=eval_targets if use_retrieval else None,
        history_prefix=train_targets if use_retrieval else None,
        context_format=config["context_format"],
        context_size=int(config["context_size"]),
        aux_objective=config["aux_objective"],
        num_negative_candidates=int(config["num_negative_candidates"]),
    )

    metadata = {
        "token_budget": {
            "max_target_tokens": int(config["max_target_tokens"]),
            "train": summarise_token_budget(
                train_records,
                tokenizer,
                context_format=config["context_format"],
                use_retrieval=use_retrieval,
                max_target_tokens=int(config["max_target_tokens"]),
                max_length=int(config["max_length"]),
                context_size=int(config["context_size"]),
            ),
            "eval": summarise_token_budget(
                eval_records,
                tokenizer,
                context_format=config["context_format"],
                use_retrieval=use_retrieval,
                max_target_tokens=int(config["max_target_tokens"]),
                max_length=int(config["max_length"]),
                context_size=int(config["context_size"]),
            ),
        }
    }
    trainer = _build_hf_trainer(config, prepared_train, prepared_eval, model, tokenizer)
    return trainer, tokenizer, config, metadata


def create_trainer(config: dict):
    trainer, tokenizer, resolved_config, metadata = build_continuation_trainer(config, use_retrieval=False)
    return trainer, tokenizer, resolved_config, metadata


def create_retrieval_trainer(config: dict):
    trainer, tokenizer, resolved_config, metadata = build_continuation_trainer(config, use_retrieval=True)
    return trainer, tokenizer, resolved_config, metadata


def train_baseline_model(config: dict) -> tuple[object, object]:
    trainer, tokenizer, resolved_config, metadata = create_trainer(config)
    trainer.train()
    trainer.save_model(resolved_config["output_dir"])
    tokenizer.save_pretrained(resolved_config["output_dir"])
    save_training_metadata(resolved_config["output_dir"], resolved_config, metadata)
    return trainer, tokenizer


def train_retrieval_model(config: dict) -> tuple[object, object]:
    trainer, tokenizer, resolved_config, metadata = create_retrieval_trainer(config)
    trainer.train()
    trainer.save_model(resolved_config["output_dir"])
    tokenizer.save_pretrained(resolved_config["output_dir"])
    save_training_metadata(resolved_config["output_dir"], resolved_config, metadata)
    return trainer, tokenizer
