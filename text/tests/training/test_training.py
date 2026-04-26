from pathlib import Path
import sys
import importlib.util
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from novel_continuation.prompting import build_prompt
from novel_continuation.trainer_runtime import TARGET_SECTION_PREFIX, ContinuationDataCollator, encode_target_with_prefix
from novel_continuation.training import (
    attach_retrieval,
    build_eval_rows_from_prepared_records,
    choose_max_target_tokens,
    load_trained_model_and_tokenizer,
    load_jsonl_records,
    load_training_config,
    prepare_training_records,
    select_negative_candidates,
    summarise_token_budget,
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
    assert config["use_retrieval"] is False
    assert config["seed"] == 42
    assert "selection_metric" not in config


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


def test_validate_baseline_config_rejects_removed_selection_metric():
    config = {
        "model_name": "distilgpt2",
        "selection_metric": "perplexity",
    }

    try:
        validate_baseline_config(config)
    except ValueError as exc:
        assert "selection_metric" in str(exc)
    else:
        raise AssertionError("Expected ValueError when deprecated selection_metric is provided")


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


class _CharTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    eos_token_id = 9999

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


def test_encode_target_with_prefix_splits_prefix_and_body():
    tokenizer = _CharTokenizer()

    prefix_ids, target_ids = encode_target_with_prefix(tokenizer, "Holmes", max_target_tokens=128)

    assert tokenizer.decode(prefix_ids) == TARGET_SECTION_PREFIX
    assert tokenizer.decode(target_ids) == "Holmes"


def test_encode_target_with_prefix_raises_when_boundary_is_not_preserved():
    class _BoundaryMergingTokenizer(_CharTokenizer):
        def __call__(self, text, add_special_tokens=False, return_tensors=None, truncation=False, max_length=None):
            if text == TARGET_SECTION_PREFIX:
                return {"input_ids": [1, 2]}
            if text.startswith(TARGET_SECTION_PREFIX):
                return {"input_ids": [9, 9, 9]}
            return super().__call__(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                truncation=truncation,
                max_length=max_length,
            )

    try:
        encode_target_with_prefix(_BoundaryMergingTokenizer(), "Holmes", max_target_tokens=128)
    except ValueError as exc:
        assert "prefix boundary" in str(exc)
    else:
        raise AssertionError("Expected ValueError when the tokenizer does not preserve the target prefix boundary")


def test_continuation_data_collator_masks_prompt_delimiter_and_target_prefix():
    tokenizer = _CharTokenizer()
    collator = ContinuationDataCollator(
        tokenizer,
        max_length=256,
        max_target_tokens=128,
        aux_objective="none",
    )

    batch = collator(
        [
            {
                "context": ["Holmes spoke."],
                "target": "Watson replied.",
                "retrieved": [],
                "context_format": "plain",
                "include_retrieval": False,
            }
        ]
    )

    labels = batch["labels"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()
    supervised_ids = [token for token, label in zip(input_ids, labels) if label != -100]
    masked_ids = [token for token, label in zip(input_ids, labels) if label == -100]

    assert tokenizer.decode(supervised_ids) == "Watson replied."
    masked_text = tokenizer.decode(masked_ids)
    assert "[CONTEXT]" in masked_text
    assert TARGET_SECTION_PREFIX in masked_text


def test_build_eval_rows_from_prepared_records_uses_prompt_builder_contract():
    rows = build_eval_rows_from_prepared_records(
        [
            {
                "context": ["p1", "p2"],
                "target": "p3",
                "retrieved": ["clue"],
                "context_format": "structured",
                "include_retrieval": True,
            }
        ]
    )

    assert rows[0]["gold_target"] == "p3"
    assert "[PARAGRAPH 1]" in rows[0]["prompt"]
    assert "[RETRIEVED]" in rows[0]["prompt"]
    assert rows[0]["generated_text"] == "p3"


def test_summarise_token_budget_counts_delimiter_and_target_prefix():
    tokenizer = _CharTokenizer()
    record = {"context": ["Holmes spoke."], "target": "Watson replied."}
    prompt = build_prompt(
        context=record["context"],
        retrieved=[],
        include_retrieval=False,
        context_format="plain",
    )
    expected_total = (
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        + len(tokenizer("\n\n", add_special_tokens=False)["input_ids"])
        + len(tokenizer(TARGET_SECTION_PREFIX, add_special_tokens=False)["input_ids"])
        + len(tokenizer(record["target"], add_special_tokens=False)["input_ids"])
    )

    summary = summarise_token_budget(
        [record],
        tokenizer,
        context_format="plain",
        use_retrieval=False,
        max_target_tokens=128,
        max_length=expected_total - 1,
        context_size=1,
    )

    assert summary["avg_total_tokens"] == expected_total
    assert summary["max_total_tokens"] == float(expected_total)
    assert summary["truncation_rate"] == 1.0


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

    class _FakeConfig:
        n_positions = 1024

    class FakeModel:
        config = _FakeConfig()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 0

        def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
            import torch

            observed["prompt"] = text
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

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


def test_generate_rows_passes_seed_to_generation_helper(tmp_path, monkeypatch):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "generate_samples.py"
    spec = importlib.util.spec_from_file_location("generate_samples", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected generate_samples.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    observed = {}

    class _FakeConfig:
        n_positions = 1024

    class FakeModel:
        config = _FakeConfig()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
            import torch

            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

        def decode(self, *_args, **_kwargs):
            return "Generated."

    monkeypatch.setattr(module, "load_trained_model_and_tokenizer", lambda _path: (FakeModel(), FakeTokenizer()))
    monkeypatch.setattr(module, "set_generation_seed", lambda seed: observed.setdefault("seed", seed))

    rows = [{"context": ["c0"], "target": "target"}]
    module.generate_rows(rows=rows, model_dir=model_dir, max_new_tokens=8, use_retrieval=False, seed=2026)

    assert observed["seed"] == 2026


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


def test_inspect_token_stats_passes_context_size_to_budget_summary(tmp_path, monkeypatch, capsys):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "inspect_token_stats.py"
    spec = importlib.util.spec_from_file_location("inspect_token_stats", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected inspect_token_stats.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name: distilgpt2\n", encoding="utf-8")
    output_path = tmp_path / "stats.json"

    observed = {"calls": []}

    monkeypatch.setattr(
        module,
        "load_training_config",
        lambda _path: {
            "model_name": "distilgpt2",
            "train_path": "train.jsonl",
            "eval_path": "eval.jsonl",
            "context_format": "plain",
            "context_size": 2,
            "max_length": 512,
            "use_retrieval": False,
        },
    )
    monkeypatch.setattr(module, "load_model_and_tokenizer", lambda _name: (object(), object()))
    monkeypatch.setattr(
        module,
        "load_jsonl_records",
        lambda _path: [{"context": ["p0", "p1"], "target": "p2"}],
    )
    monkeypatch.setattr(module, "resolve_max_target_tokens", lambda _config, _records, _tokenizer: 128)

    def fake_summarise(records, tokenizer, **kwargs):
        observed["calls"].append(kwargs)
        return {
            "avg_total_tokens": 12.0,
            "max_total_tokens": 16.0,
            "truncation_rate": 0.0,
        }

    monkeypatch.setattr(module, "summarise_token_budget", fake_summarise)
    monkeypatch.setattr(sys, "argv", ["inspect_token_stats.py", "--config", str(config_path), "--output-path", str(output_path)])

    module.main()

    assert output_path.exists()
    assert len(observed["calls"]) == 2
    assert all(call["context_size"] == 2 for call in observed["calls"])


def test_run_eval_3seed_writes_per_seed_outputs_and_summary(tmp_path, monkeypatch):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_eval_3seed.py"
    spec = importlib.util.spec_from_file_location("run_eval_3seed", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected run_eval_3seed.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "eval"
    written_metrics = []

    monkeypatch.setattr(module, "load_jsonl", lambda _path: [{"context": ["c1"], "target": "t1"}])
    monkeypatch.setattr(module, "load_trained_model_and_tokenizer", lambda _path: ("model", "tokenizer"))
    monkeypatch.setattr(
        module,
        "generate_rows",
        lambda **kwargs: [{"prompt": "[CONTEXT]\nc1", "gold_target": "t1", "generated_text": f"g-{kwargs['seed']}"}],
    )
    monkeypatch.setattr(module, "write_jsonl", lambda rows, path: None)

    def fake_evaluate(rows, model=None, tokenizer=None):
        generated_text = rows[0]["generated_text"]
        if generated_text.endswith("13"):
            return {"num_samples": 1.0, "perplexity": 3.0, "rouge_l": 0.1, "bertscore_f1": 0.2, "entity_overlap": 0.3}
        if generated_text.endswith("42"):
            return {"num_samples": 1.0, "perplexity": 3.0, "rouge_l": 0.2, "bertscore_f1": 0.3, "entity_overlap": 0.4}
        return {"num_samples": 1.0, "perplexity": 3.0, "rouge_l": 0.3, "bertscore_f1": 0.4, "entity_overlap": 0.5}

    monkeypatch.setattr(module, "evaluate_generated_rows", fake_evaluate)
    monkeypatch.setattr(module, "write_metrics_csv", lambda metrics, path: written_metrics.append((metrics, path.name)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval_3seed.py",
            "--experiment-id",
            "e3",
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    summary_metrics = dict(written_metrics[-1][0])
    assert written_metrics[-1][1] == "metrics_e3_summary.csv"
    assert "perplexity_mean" in summary_metrics
    assert "perplexity_std" not in summary_metrics
    assert "rouge_l_std" in summary_metrics


def test_run_eval_3seed_skip_existing_reuses_metrics_and_backfills_missing_metrics(tmp_path, monkeypatch):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_eval_3seed.py"
    spec = importlib.util.spec_from_file_location("run_eval_3seed", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected run_eval_3seed.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    input_path = tmp_path / "test.jsonl"
    input_path.write_text("[]", encoding="utf-8")

    generated_13 = output_dir / "generated_samples_e3_seed13.jsonl"
    generated_13.write_text(
        '{"prompt": "[CONTEXT]\\nc1", "gold_target": "t1", "generated_text": "g-13"}\n',
        encoding="utf-8",
    )
    metrics_13 = output_dir / "metrics_e3_seed13.csv"
    metrics_13.write_text(
        "num_samples,perplexity,rouge_l,bertscore_f1,entity_overlap\n"
        "1.0,3.0,0.1,0.2,0.3\n",
        encoding="utf-8",
    )
    generated_42 = output_dir / "generated_samples_e3_seed42.jsonl"
    generated_42.write_text(
        '{"prompt": "[CONTEXT]\\nc1", "gold_target": "t1", "generated_text": "g-42"}\n',
        encoding="utf-8",
    )

    observed = {"model_loads": 0, "metrics_writes": []}

    def fake_load_jsonl(path):
        if Path(path) == input_path:
            return [{"context": ["c1"], "target": "t1"}]
        if Path(path) == generated_42:
            return [{"prompt": "[CONTEXT]\nc1", "gold_target": "t1", "generated_text": "g-42"}]
        raise AssertionError(f"Unexpected load_jsonl path: {path}")

    monkeypatch.setattr(module, "load_jsonl", fake_load_jsonl)

    def fake_load_model(_path):
        observed["model_loads"] += 1
        return "model", "tokenizer"

    monkeypatch.setattr(module, "load_trained_model_and_tokenizer", fake_load_model)
    monkeypatch.setattr(module, "generate_rows", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("generate_rows should not run")))
    monkeypatch.setattr(module, "write_jsonl", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("write_jsonl should not run")))
    monkeypatch.setattr(
        module,
        "evaluate_generated_rows",
        lambda rows, model=None, tokenizer=None: {
            "num_samples": 1.0,
            "perplexity": 3.0,
            "rouge_l": 0.2,
            "bertscore_f1": 0.3,
            "entity_overlap": 0.4,
        },
    )
    monkeypatch.setattr(module, "write_metrics_csv", lambda metrics, path: observed["metrics_writes"].append((metrics, path.name)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval_3seed.py",
            "--experiment-id",
            "e3",
            "--model-dir",
            str(model_dir),
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "13",
            "42",
            "--skip-existing",
        ],
    )

    module.main()

    assert observed["model_loads"] == 1
    assert observed["metrics_writes"][0][1] == "metrics_e3_seed42.csv"
    summary_metrics = dict(observed["metrics_writes"][-1][0])
    assert observed["metrics_writes"][-1][1] == "metrics_e3_summary.csv"
    assert round(summary_metrics["rouge_l_mean"], 6) == 0.15
    assert "perplexity_std" not in summary_metrics


def test_compare_aux_weight_reports_validation_loss_and_recommendation(tmp_path, monkeypatch, capsys):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compare_aux_weight.py"
    spec = importlib.util.spec_from_file_location("compare_aux_weight", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("Expected compare_aux_weight.py to be importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    left_dir = tmp_path / "e5_aux_ranking"
    left_dir.mkdir()
    (left_dir / "training_config.json").write_text(
        '{"metadata": {"validation": {"validation_main_loss": 1.000000, "validation_perplexity": 12.0}}}',
        encoding="utf-8",
    )
    right_config = tmp_path / "training_config_right.json"
    right_config.write_text(
        '{"metadata": {"validation": {"validation_main_loss": 1.025000, "validation_perplexity": 12.5}}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_aux_weight.py",
            str(left_dir),
            str(right_config),
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert "left_validation_main_loss: 1.000000" in output
    assert "right_validation_main_loss: 1.025000" in output
    assert "selection_rule: validation_main_loss only" in output
    assert "recommendation: select e5_aux_ranking" in output
