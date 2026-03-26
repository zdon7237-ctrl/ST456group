from pathlib import Path
import sys
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.training import (
    attach_retrieval,
    load_jsonl_records,
    load_training_config,
    prepare_training_records,
    validate_baseline_config,
)


def test_load_training_config_reads_model_name(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name: distilgpt2\nbatch_size: 2\n", encoding="utf-8")

    config = load_training_config(config_path)

    assert config["model_name"] == "distilgpt2"


def test_load_training_config_resolves_relative_paths_from_project_root(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "model_name: distilgpt2\n"
        "train_path: data/processed/train.jsonl\n"
        "eval_path: data/processed/val.jsonl\n"
        "output_dir: artifacts/baseline\n",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config["train_path"] == str(tmp_path / "data" / "processed" / "train.jsonl")
    assert config["eval_path"] == str(tmp_path / "data" / "processed" / "val.jsonl")
    assert config["output_dir"] == str(tmp_path / "artifacts" / "baseline")


def test_prepare_training_records_builds_target_text_without_retrieval():
    records = [
        {
            "context": ["p0", "p1"],
            "target": "p2",
        }
    ]

    prepared = prepare_training_records(records, use_retrieval=False, top_k=0)

    assert prepared[0]["text"].startswith("[CONTEXT]")
    assert "[TARGET]\np2" in prepared[0]["text"]
    assert "[RETRIEVED]" not in prepared[0]["text"]


def test_prepare_training_records_requires_explicit_retrieval_candidates():
    records = [
        {
            "context": ["p0", "p1"],
            "target": "p2",
        }
    ]

    try:
        prepare_training_records(records, use_retrieval=True, top_k=1)
    except ValueError as exc:
        assert "retrieval_candidates" in str(exc)
    else:
        raise AssertionError("Expected ValueError when retrieval candidates are missing")


def test_load_jsonl_records_raises_clear_error_for_missing_dataset(tmp_path):
    missing_path = tmp_path / "data" / "processed" / "train.jsonl"

    try:
        load_jsonl_records(missing_path)
    except FileNotFoundError as exc:
        message = str(exc)
        assert "train.jsonl" in message
        assert "build_dataset.py" in message
    else:
        raise AssertionError("Expected FileNotFoundError for missing dataset")


def test_validate_baseline_config_rejects_retrieval_mode():
    config = {
        "model_name": "distilgpt2",
        "use_retrieval": True,
    }

    try:
        validate_baseline_config(config)
    except ValueError as exc:
        message = str(exc)
        assert "Task 8" in message
        assert "use_retrieval" in message
    else:
        raise AssertionError("Expected ValueError when baseline config enables retrieval")


def test_prepare_training_records_with_retrieval_uses_only_prior_candidates():
    records = [
        {
            "context": ["first past"],
            "target": "first past",
        },
        {
            "context": ["first past"],
            "target": "second future",
        },
    ]

    prepared = prepare_training_records(
        records,
        use_retrieval=True,
        top_k=1,
        retrieval_candidates=[record["target"] for record in records],
    )

    retrieved_section = prepared[1]["text"].split("[TARGET]")[0]
    assert "[RETRIEVED]" in retrieved_section
    assert "first past" in retrieved_section
    assert "second future" not in retrieved_section


def test_attach_retrieval_for_eval_can_use_train_history_prefix():
    eval_records = [
        {
            "context": ["train clue"],
            "target": "eval continuation",
        }
    ]

    enriched = attach_retrieval(
        eval_records,
        retrieval_candidates=[record["target"] for record in eval_records],
        top_k=1,
        history_prefix=["train clue"],
    )

    assert enriched[0]["retrieved"][0]["text"] == "train clue"


def test_attach_retrieval_for_eval_uses_train_and_prior_eval_history_only():
    eval_records = [
        {
            "context": ["train clue"],
            "target": "first eval target",
        },
        {
            "context": ["second query"],
            "target": "second eval target",
        },
    ]

    enriched = attach_retrieval(
        eval_records,
        retrieval_candidates=[record["target"] for record in eval_records],
        top_k=5,
        history_prefix=["train clue"],
    )

    first_retrieved = [item["text"] for item in enriched[0]["retrieved"]]
    second_retrieved = [item["text"] for item in enriched[1]["retrieved"]]

    assert first_retrieved == ["train clue"]
    assert "train clue" in second_retrieved
    assert "first eval target" in second_retrieved
    assert "second eval target" not in second_retrieved


def test_attach_retrieval_truncates_when_top_k_exceeds_available_history():
    records = [
        {
            "context": ["first context"],
            "target": "first target",
        }
    ]

    enriched = attach_retrieval(
        records,
        retrieval_candidates=["first target"],
        top_k=3,
        history_prefix=[],
    )

    assert enriched[0]["retrieved"] == []


def test_generate_samples_can_enrich_retrieval_rows_from_history():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "generate_samples.py"
    spec = importlib.util.spec_from_file_location("generate_samples", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected generate_samples.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    eval_rows = [
        {
            "context": ["train clue"],
            "target": "first eval target",
        },
        {
            "context": ["second query"],
            "target": "second eval target",
        },
    ]

    enriched_rows = module.ensure_retrieval_rows(
        rows=eval_rows,
        history_rows=[{"target": "train clue"}],
        top_k=5,
    )

    assert enriched_rows[0]["retrieved"][0]["text"] == "train clue"
    second_retrieved = [item["text"] for item in enriched_rows[1]["retrieved"]]
    assert "train clue" in second_retrieved
    assert "first eval target" in second_retrieved
    assert "second eval target" not in second_retrieved
