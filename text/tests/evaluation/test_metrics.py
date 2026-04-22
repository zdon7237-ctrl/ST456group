from pathlib import Path
import sys
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.evaluation import (
    build_conditional_ppl_example,
    compute_entity_overlap,
    compute_perplexity,
    compute_rouge_l,
    compute_weighted_kappa,
    evaluate_generated_rows,
    summarise_seed_metrics,
)


class _CharTokenizer:
    model_max_length = 64

    def __call__(self, text, add_special_tokens=False, return_tensors=None, truncation=False, max_length=None):
        input_ids = [ord(char) for char in text]
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        if return_tensors == "pt":
            return {
                "input_ids": [[token for token in input_ids]],
                "attention_mask": [[1] * len(input_ids)],
            }
        return {"input_ids": input_ids}

    def decode(self, token_ids, clean_up_tokenization_spaces=False, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


def test_compute_entity_overlap_detects_shared_names():
    score = compute_entity_overlap(
        context_text="Holmes spoke to Watson in Baker Street.",
        generated_text="Watson answered Holmes quietly.",
    )
    assert score > 0


def test_compute_entity_overlap_returns_zero_without_shared_entities():
    score = compute_entity_overlap(
        context_text="Holmes spoke to Watson in Baker Street.",
        generated_text="The rain fell over London all night.",
    )
    assert score == 0.0


def test_compute_rouge_l_rewards_longer_common_subsequence():
    high_score = compute_rouge_l(
        reference_text="Holmes entered the room and greeted Watson",
        generated_text="Holmes entered the room and greeted Watson warmly",
    )
    low_score = compute_rouge_l(
        reference_text="Holmes entered the room and greeted Watson",
        generated_text="A storm passed over the city at dusk",
    )
    assert high_score > low_score
    assert 0.0 <= high_score <= 1.0


def test_compute_rouge_l_returns_one_for_identical_text():
    score = compute_rouge_l(
        reference_text="Holmes entered the room and greeted Watson",
        generated_text="Holmes entered the room and greeted Watson",
    )
    assert score == 1.0


def test_evaluate_generated_rows_requires_expected_fields():
    rows = [
        {
            "prompt": "[CONTEXT]\nHolmes spoke.",
            "generated_text": "Watson replied.",
        }
    ]

    try:
        evaluate_generated_rows(rows)
    except ValueError as exc:
        assert "gold_target" in str(exc)
    else:
        raise AssertionError("Expected ValueError when generated rows are missing required fields")


def test_build_human_eval_rows_exposes_required_columns():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_human_eval.py"
    spec = importlib.util.spec_from_file_location("prepare_human_eval", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected prepare_human_eval.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = module.build_human_eval_rows(
        [
            {
                "sample_id": 1,
                "prompt": "[CONTEXT]\nHolmes spoke.",
                "gold_target": "Watson replied.",
                "generated_text": "Watson answered.",
            }
        ],
        system_label="System A",
    )

    row = rows[0]
    assert row["sample_id"] == 1
    assert row["system_label"] == "System A"
    assert "contextual_coherence" in row
    assert "narrative_consistency" in row
    assert "character_consistency" in row
    assert "fluency" in row


def test_prepare_human_eval_uses_project_root_defaults():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_human_eval.py"
    spec = importlib.util.spec_from_file_location("prepare_human_eval", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected prepare_human_eval.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_INPUT_PATH == module.PROJECT_ROOT / "artifacts" / "eval" / "generated_samples.jsonl"
    assert module.DEFAULT_OUTPUT_PATH == module.PROJECT_ROOT / "artifacts" / "eval" / "human_eval.csv"


def test_evaluate_generated_rows_returns_proposal_aligned_metric_keys():
    rows = [
        {
            "prompt": "[CONTEXT]\nHolmes spoke.",
            "gold_target": "Watson replied.",
            "generated_text": "Watson replied calmly.",
        }
    ]

    metrics = evaluate_generated_rows(
        rows,
        model=object(),
        tokenizer=object(),
        bertscore_fn=lambda refs, gens: 0.42,
        perplexity_fn=lambda model, tokenizer, ppl_rows: 12.5,
    )

    assert metrics["num_samples"] == 1.0
    assert metrics["perplexity"] == 12.5
    assert metrics["bertscore_f1"] == 0.42
    assert "rouge_l" in metrics
    assert "entity_overlap" in metrics


def test_compute_weighted_kappa_returns_none_for_single_rater():
    assert compute_weighted_kappa([1, 2, 3], None) is None


def test_build_conditional_ppl_example_masks_prompt_and_target_prefix():
    tokenizer = _CharTokenizer()

    input_ids, labels = build_conditional_ppl_example(
        tokenizer,
        prompt="[CONTEXT]\nHolmes spoke.",
        gold_target="Watson replied.",
        max_length=128,
    )

    supervised = [token for token, label in zip(input_ids, labels) if label != -100]
    masked = [token for token, label in zip(input_ids, labels) if label == -100]

    assert tokenizer.decode(supervised) == "Watson replied."
    assert "[TARGET]\n" in tokenizer.decode(masked)


def test_build_conditional_ppl_example_left_truncates_prompt_first():
    tokenizer = _CharTokenizer()

    input_ids, labels = build_conditional_ppl_example(
        tokenizer,
        prompt="abcdefghij",
        gold_target="XYZ",
        max_length=16,
    )

    decoded = tokenizer.decode(input_ids)
    assert decoded.endswith("\n\n[TARGET]\nXYZ")
    assert decoded.startswith("a") is False


def test_compute_perplexity_uses_token_weighted_aggregation(monkeypatch):
    import torch

    class FakeModel:
        class _Config:
            n_positions = 64

        config = _Config()

        def __init__(self):
            self._parameter = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self._parameter

        def __call__(self, **kwargs):
            return type("Outputs", (), {"logits": torch.zeros((1, 4, 8))})()

    rows = [
        {"prompt": "p1", "gold_target": "t1"},
        {"prompt": "p2", "gold_target": "t2"},
    ]
    tokenizer = _CharTokenizer()
    labels_by_row = iter(
        [
            [1, -100, 11],
            [1, -100, 21, 22, 23],
        ]
    )

    monkeypatch.setattr(
        "novel_continuation.evaluation.build_conditional_ppl_example",
        lambda tokenizer, prompt, gold_target, max_length: ([1, 2, 3], next(labels_by_row)),
    )
    contributions = iter([(2.0, 1), (12.0, 3)])
    monkeypatch.setattr(
        "novel_continuation.evaluation._accumulate_shifted_nll",
        lambda logits, labels: next(contributions),
    )

    ppl = compute_perplexity(FakeModel(), tokenizer, rows)
    assert round(ppl, 6) == round((2.718281828459045 ** 3.5), 6)


def test_summarise_seed_metrics_omits_perplexity_std():
    summary = summarise_seed_metrics(
        [
            {"num_samples": 1.0, "perplexity": 3.0, "rouge_l": 0.1, "bertscore_f1": 0.2, "entity_overlap": 0.3},
            {"num_samples": 1.0, "perplexity": 3.0, "rouge_l": 0.2, "bertscore_f1": 0.3, "entity_overlap": 0.4},
            {"num_samples": 1.0, "perplexity": 3.0, "rouge_l": 0.3, "bertscore_f1": 0.4, "entity_overlap": 0.5},
        ]
    )

    assert "perplexity_mean" in summary
    assert "perplexity_std" not in summary
    assert "rouge_l_std" in summary
