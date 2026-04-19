"""Automatic evaluation helpers for continuation experiments."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path


ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


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

    _, _, f1 = score(generated_texts, reference_texts, lang="en", verbose=False)
    return float(f1.mean().item())


def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    import torch

    if not texts:
        raise ValueError("texts must not be empty")

    losses: list[float] = []
    try:
        from tqdm import tqdm
        text_iter = tqdm(texts, desc="计算 perplexity", unit="条")
    except ImportError:
        text_iter = texts
    device = next(model.parameters()).device
    for text in text_iter:
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, labels=encoded["input_ids"])
        losses.append(float(outputs.loss.item()))
    return float(math.exp(sum(losses) / len(losses)))


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
        texts = [f"{row['prompt']}\n\n[TARGET]\n{row['gold_target']}" for row in rows]
        metrics["perplexity"] = float(perplexity_fn(model, tokenizer, texts))
    return metrics


def write_metrics_csv(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
