"""Training helpers for baseline and retrieval-augmented continuation models."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from novel_continuation.prompting import build_training_text
from novel_continuation.retrieval import fit_tfidf_retriever, query_tfidf_index


PATH_KEYS = ("train_path", "eval_path", "output_dir")


def _resolve_path(value: str, config_path: Path) -> str:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return str(raw_path)

    config_dir_candidate = (config_path.parent / raw_path).resolve()
    if config_dir_candidate.exists():
        return str(config_dir_candidate)

    project_root_candidate = (config_path.parent.parent / raw_path).resolve()
    return str(project_root_candidate)


def load_training_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

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


def validate_baseline_config(config: dict) -> None:
    if config.get("use_retrieval", False):
        raise ValueError(
            "Baseline training does not support use_retrieval=True. Use the dedicated retrieval pipeline in Task 8 instead."
        )


def validate_retrieval_config(config: dict) -> None:
    if not config.get("use_retrieval", False):
        raise ValueError("Retrieval training requires use_retrieval=True.")
    if int(config.get("top_k", 0)) <= 0:
        raise ValueError("Retrieval training requires top_k > 0.")


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


def prepare_training_records(
    records: list[dict],
    use_retrieval: bool = False,
    top_k: int = 0,
    retrieval_candidates: list[str] | None = None,
    history_prefix: list[str] | None = None,
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
    for record in records:
        retrieved_texts = [item["text"] for item in record.get("retrieved", [])]
        prepared.append(
            {
                "text": build_training_text(
                    context=record["context"],
                    retrieved=retrieved_texts,
                    target=record["target"],
                    include_retrieval=use_retrieval,
                )
            }
        )
    return prepared


def build_training_texts(
    records: list[dict],
    use_retrieval: bool = False,
    top_k: int = 0,
    retrieval_candidates: list[str] | None = None,
    history_prefix: list[str] | None = None,
) -> list[str]:
    prepared = prepare_training_records(
        records,
        use_retrieval=use_retrieval,
        top_k=top_k,
        retrieval_candidates=retrieval_candidates,
        history_prefix=history_prefix,
    )
    return [record["text"] for record in prepared]


def tokenize_records(records: list[dict], tokenizer, max_length: int):
    from datasets import Dataset

    dataset = Dataset.from_list(records)

    def _tokenize(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def create_trainer(config: dict):
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    validate_baseline_config(config)
    model, tokenizer = load_model_and_tokenizer(config["model_name"])

    train_records = load_jsonl_records(Path(config["train_path"]))
    eval_records = load_jsonl_records(Path(config["eval_path"]))

    prepared_train = prepare_training_records(
        train_records,
        use_retrieval=False,
        top_k=config.get("top_k", 0),
    )
    prepared_eval = prepare_training_records(
        eval_records,
        use_retrieval=False,
        top_k=config.get("top_k", 0),
    )

    train_dataset = tokenize_records(prepared_train, tokenizer, config["max_length"])
    eval_dataset = tokenize_records(prepared_eval, tokenizer, config["max_length"])

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=float(config["learning_rate"]),
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer, tokenizer


def create_retrieval_trainer(config: dict):
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    validate_retrieval_config(config)
    model, tokenizer = load_model_and_tokenizer(config["model_name"])

    train_records = load_jsonl_records(Path(config["train_path"]))
    eval_records = load_jsonl_records(Path(config["eval_path"]))
    train_targets = [record["target"] for record in train_records]
    eval_targets = [record["target"] for record in eval_records]

    prepared_train = prepare_training_records(
        train_records,
        use_retrieval=True,
        top_k=config["top_k"],
        retrieval_candidates=train_targets,
    )
    prepared_eval = prepare_training_records(
        eval_records,
        use_retrieval=True,
        top_k=config["top_k"],
        retrieval_candidates=eval_targets,
        history_prefix=train_targets,
    )

    train_dataset = tokenize_records(prepared_train, tokenizer, config["max_length"])
    eval_dataset = tokenize_records(prepared_eval, tokenizer, config["max_length"])

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=float(config["learning_rate"]),
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer, tokenizer


def train_baseline_model(config: dict) -> tuple[object, object]:
    trainer, tokenizer = create_trainer(config)
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    return trainer, tokenizer


def train_retrieval_model(config: dict) -> tuple[object, object]:
    trainer, tokenizer = create_retrieval_trainer(config)
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    return trainer, tokenizer
