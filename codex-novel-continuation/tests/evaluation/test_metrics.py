from pathlib import Path
import sys
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.evaluation import (
    compute_entity_overlap,
    compute_rouge_l,
    evaluate_generated_rows,
)


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
