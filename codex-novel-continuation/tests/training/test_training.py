from pathlib import Path
import sys
import importlib.util
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.training import (
    attach_retrieval,
    choose_max_target_tokens,
    load_trained_model_and_tokenizer,
    load_jsonl_records,
    load_training_config,
    prepare_training_records,
    select_negative_candidates,
    validate_baseline_config,
)


def test_load_training_config_reads_model_name(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name: distilgpt2\nbatch_size: 2\n", encoding="utf-8")

    config = load_training_config(config_path)

    assert config["model_name"] == "distilgpt2"
    assert config["context_format"] == "plain"
    assert config["training_mode"] == "full"
    assert config["aux_objective"] == "none"
    assert config["selection_metric"] == "perplexity"
    assert config["use_retrieval"] is False


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


def test_prepare_training_records_supports_structured_context_format():
    records = [
        {
            "context": ["p0", "p1"],
            "target": "p2",
        }
    ]

    prepared = prepare_training_records(records, use_retrieval=False, top_k=0, context_format="structured")

    assert "[PARAGRAPH 1]" in prepared[0]["text"]
    assert "[PARAGRAPH 2]" in prepared[0]["text"]


def test_prepare_training_records_respects_configured_context_size():
    records = [
        {
            "context": ["p0", "p1", "p2", "p3"],
            "target": "p4",
        }
    ]

    prepared = prepare_training_records(
        records,
        use_retrieval=False,
        top_k=0,
        context_size=2,
    )

    assert "p0" not in prepared[0]["text"]
    assert "p1" not in prepared[0]["text"]
    assert "p2" in prepared[0]["text"]
    assert "p3" in prepared[0]["text"]


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
        assert "use_retrieval" in message
        assert "appendix" in message
    else:
        raise AssertionError("Expected ValueError when baseline config enables retrieval")


def test_choose_max_target_tokens_uses_p90_rounding_and_cap():
    value = choose_max_target_tokens([10, 20, 30, 40, 155, 161, 162, 163, 164, 165])
    assert value == 176

    capped = choose_max_target_tokens([225] * 20)
    assert capped == 224


def test_select_negative_candidates_prefers_same_book_non_adjacent_rows():
    records = [
        {
            "book_id": "1661",
            "paragraph_index": 0,
            "split": "train",
            "target": "target-0",
        },
        {
            "book_id": "1661",
            "paragraph_index": 1,
            "split": "train",
            "target": "target-1",
        },
        {
            "book_id": "1661",
            "paragraph_index": 2,
            "split": "train",
            "target": "target-2",
        },
        {
            "book_id": "1661",
            "paragraph_index": 5,
            "split": "train",
            "target": "target-5",
        },
        {
            "book_id": "834",
            "paragraph_index": 9,
            "split": "train",
            "target": "memoirs-9",
        },
    ]

    negatives = select_negative_candidates(records, index=1, num_negative_candidates=2)

    assert negatives[0] == "target-5"
    assert "target-1" not in negatives
    assert "target-0" not in negatives
    assert "target-2" not in negatives


def test_prepare_training_records_uses_shared_auxiliary_candidate_shape_for_ranking_and_classification():
    records = [
        {
            "book_id": "1661",
            "paragraph_index": 0,
            "split": "train",
            "context": ["p0"],
            "target": "target-0",
        },
        {
            "book_id": "1661",
            "paragraph_index": 1,
            "split": "train",
            "context": ["p1"],
            "target": "target-1",
        },
        {
            "book_id": "1661",
            "paragraph_index": 4,
            "split": "train",
            "context": ["p4"],
            "target": "target-4",
        },
        {
            "book_id": "834",
            "paragraph_index": 8,
            "split": "train",
            "context": ["q8"],
            "target": "target-8",
        },
    ]

    ranking_prepared = prepare_training_records(
        records,
        aux_objective="ranking",
        num_negative_candidates=2,
    )
    classification_prepared = prepare_training_records(
        records,
        aux_objective="classification",
        num_negative_candidates=2,
    )

    assert ranking_prepared[1]["candidate_targets"] == classification_prepared[1]["candidate_targets"]
    assert ranking_prepared[1]["candidate_labels"] == classification_prepared[1]["candidate_labels"]
    assert ranking_prepared[1]["candidate_labels"][0] == 1
    assert ranking_prepared[1]["candidate_labels"][1:] == [0, 0]


def test_prepare_training_records_skips_auxiliary_candidates_when_disabled():
    records = [
        {
            "context": ["p0"],
            "target": "p1",
        }
    ]

    prepared = prepare_training_records(records, aux_objective="none")

    assert "candidate_targets" not in prepared[0]
    assert "candidate_labels" not in prepared[0]


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


def test_generate_rows_uses_context_size_from_training_metadata(tmp_path, monkeypatch):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "generate_samples.py"
    spec = importlib.util.spec_from_file_location("generate_samples", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected generate_samples.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "training_config.json").write_text(
        '{"config": {"context_size": 2, "context_format": "plain"}}',
        encoding="utf-8",
    )

    observed = {}

    class FakeModel:
        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 0

        def __call__(self, text, return_tensors=None, truncation=False):
            observed["prompt"] = text
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

        def decode(self, *_args, **_kwargs):
            return observed["prompt"] + "\nGenerated."

    monkeypatch.setattr(module, "load_trained_model_and_tokenizer", lambda _path: (FakeModel(), FakeTokenizer()))

    rows = [
        {
            "context": ["c0", "c1", "c2", "c3"],
            "target": "target",
            "paragraph_index": 4,
        }
    ]

    module.generate_rows(rows=rows, model_dir=model_dir, max_new_tokens=8, use_retrieval=False)

    assert "[CONTEXT]\nc2\nc3" in observed["prompt"]


def test_load_trained_model_and_tokenizer_supports_plain_checkpoint(tmp_path, monkeypatch):
    observed = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    def fake_model_from_pretrained(model_dir):
        observed["model_dir"] = str(model_dir)
        return "model"

    def fake_tokenizer_from_pretrained(model_dir):
        observed["tokenizer_dir"] = str(model_dir)
        return FakeTokenizer()

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=fake_model_from_pretrained),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_tokenizer_from_pretrained),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model, tokenizer = load_trained_model_and_tokenizer(tmp_path)

    assert model == "model"
    assert tokenizer.pad_token == tokenizer.eos_token
    assert observed["model_dir"] == str(tmp_path)


def test_load_trained_model_and_tokenizer_supports_lora_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path": "distilgpt2"}', encoding="utf-8")
    observed = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    def fake_model_from_pretrained(model_name):
        observed["base_model"] = model_name
        return "base-model"

    def fake_tokenizer_from_pretrained(model_dir):
        observed["tokenizer_dir"] = str(model_dir)
        return FakeTokenizer()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, model_dir):
            observed["adapter_base_model"] = base_model
            observed["adapter_dir"] = str(model_dir)
            return "peft-model"

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=fake_model_from_pretrained),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_tokenizer_from_pretrained),
    )
    fake_peft = types.SimpleNamespace(PeftModel=FakePeftModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    model, tokenizer = load_trained_model_and_tokenizer(tmp_path)

    assert model == "peft-model"
    assert tokenizer.pad_token == tokenizer.eos_token
    assert observed["base_model"] == "distilgpt2"
    assert observed["adapter_dir"] == str(tmp_path)
