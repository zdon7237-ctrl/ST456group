"""Automatic evaluation helpers for continuation experiments."""

from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path

from novel_continuation.trainer_runtime import (
    TARGET_SECTION_DELIMITER,
    TARGET_SECTION_PREFIX,
    encode_target_with_prefix,
)


ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
GENERATION_METRIC_KEYS = ("rouge_l", "bertscore_f1", "entity_overlap")


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", text.lower()) if token]


def _longest_common_subsequence(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0

    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i, left_token in enumerate(left, start=1):
        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_rouge_l(reference_text: str, generated_text: str) -> float:
    reference_tokens = _tokenize(reference_text)
    generated_tokens = _tokenize(generated_text)
    if not reference_tokens or not generated_tokens:
        return 0.0

    lcs = _longest_common_subsequence(reference_tokens, generated_tokens)
    precision = lcs / len(generated_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _extract_entities(text: str) -> set[str]:
    return {match.strip() for match in ENTITY_PATTERN.findall(text)}


def compute_entity_overlap(context_text: str, generated_text: str) -> float:
    context_entities = _extract_entities(context_text)
    generated_entities = _extract_entities(generated_text)
    if not context_entities or not generated_entities:
        return 0.0
    overlap = context_entities & generated_entities
    return len(overlap) / len(context_entities)


def compute_bertscore(reference_texts: list[str], generated_texts: list[str]) -> float:
    from bert_score import score

    paired = [
        (ref, gen) for ref, gen in zip(reference_texts, generated_texts)
        if ref.strip() and gen.strip()
    ]
    if not paired:
        return 0.0
    refs, gens = zip(*paired)
    _, _, f1 = score(list(gens), list(refs), lang="en", verbose=False)
    return float(f1.mean().item())


def _resolve_model_max_length(model, tokenizer) -> int:
    for attr in ("n_positions", "max_position_embeddings"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int) and 0 < model_max_length < 10**6:
        return model_max_length
    return 1024


def build_conditional_ppl_example(
    tokenizer,
    *,
    prompt: str,
    gold_target: str,
    max_length: int,
) -> tuple[list[int], list[int]]:
    delimiter_ids = tokenizer(TARGET_SECTION_DELIMITER, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prefix_only_ids = tokenizer(TARGET_SECTION_PREFIX, add_special_tokens=False)["input_ids"]
    max_target_tokens = max(0, max_length - len(delimiter_ids) - len(prefix_only_ids))
    prefix_ids, target_ids = encode_target_with_prefix(
        tokenizer,
        gold_target,
        max_target_tokens=max_target_tokens,
    )
    max_prompt_tokens = max(0, max_length - len(delimiter_ids) - len(prefix_ids) - len(target_ids))
    prompt_ids = prompt_ids[-max_prompt_tokens:] if max_prompt_tokens else []
    input_ids = prompt_ids + delimiter_ids + prefix_ids + target_ids
    labels = ([-100] * (len(prompt_ids) + len(delimiter_ids) + len(prefix_ids))) + target_ids
    return input_ids, labels


def _accumulate_shifted_nll(logits, labels) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F

    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shifted_labels.size())
    token_mask = shifted_labels.ne(-100)
    supervised_tokens = int(token_mask.sum().item())
    total_nll = float((token_losses * token_mask).sum().item())
    return total_nll, supervised_tokens


def compute_perplexity(model, tokenizer, rows: list[dict[str, str]]) -> float:
    import torch

    if not rows:
        raise ValueError("rows must not be empty")

    max_length = _resolve_model_max_length(model, tokenizer)
    total_nll = 0.0
    total_supervised_tokens = 0
    try:
        from tqdm import tqdm
        row_iter = tqdm(rows, desc="计算 perplexity", unit="条")
    except ImportError:
        row_iter = rows
    device = next(model.parameters()).device
    for row in row_iter:
        input_ids, labels = build_conditional_ppl_example(
            tokenizer,
            prompt=row["prompt"],
            gold_target=row["gold_target"],
            max_length=max_length,
        )
        if not target_token_count(labels):
            continue
        encoded = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long, device=device),
        }
        label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(**encoded)
        sample_nll, sample_supervised_tokens = _accumulate_shifted_nll(outputs.logits, label_tensor)
        total_nll += sample_nll
        total_supervised_tokens += sample_supervised_tokens
    if total_supervised_tokens == 0:
        raise ValueError("Perplexity scoring requires at least one supervised target token")
    return float(math.exp(total_nll / total_supervised_tokens))


def compute_weighted_kappa(first_rater_scores: list[int], second_rater_scores: list[int] | None) -> float | None:
    if second_rater_scores is None:
        return None
    if len(first_rater_scores) != len(second_rater_scores):
        raise ValueError("Rater score lists must have the same length")

    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(first_rater_scores, second_rater_scores, weights="quadratic"))


def evaluate_generated_rows(
    rows: list[dict[str, str]],
    *,
    model=None,
    tokenizer=None,
    bertscore_fn=compute_bertscore,
    perplexity_fn=compute_perplexity,
) -> dict[str, float]:
    if not rows:
        raise ValueError("rows must not be empty")

    required_fields = {"prompt", "gold_target", "generated_text"}
    for index, row in enumerate(rows):
        missing_fields = sorted(required_fields - set(row))
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(
                f"Generated row {index} is missing required fields: {missing}."
            )

    rouge_scores = [
        compute_rouge_l(row["gold_target"], row["generated_text"])
        for row in rows
    ]
    entity_scores = [
        compute_entity_overlap(row["prompt"], row["generated_text"])
        for row in rows
    ]
    metrics = {
        "num_samples": float(len(rows)),
        "rouge_l": sum(rouge_scores) / len(rouge_scores),
        "entity_overlap": sum(entity_scores) / len(entity_scores),
    }
    metrics["bertscore_f1"] = float(
        bertscore_fn(
            [row["gold_target"] for row in rows],
            [row["generated_text"] for row in rows],
        )
    )
    if model is not None and tokenizer is not None:
        metrics["perplexity"] = float(perplexity_fn(model, tokenizer, rows))
    return metrics


def summarise_seed_metrics(metrics_by_seed: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_by_seed:
        raise ValueError("metrics_by_seed must not be empty")

    summary: dict[str, float] = {}
    metric_keys = list(metrics_by_seed[0].keys())
    for key in metric_keys:
        values = [float(metrics[key]) for metrics in metrics_by_seed]
        summary[f"{key}_mean"] = sum(values) / len(values)
        if key in GENERATION_METRIC_KEYS:
            summary[f"{key}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return summary


def target_token_count(labels: list[int] | object) -> int:
    try:
        return int(sum(1 for label in labels if label != -100))
    except TypeError:
        return 0


def write_metrics_csv(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
