# Novel Continuation Project Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and evaluate a retrieval-augmented English novel continuation system that predicts the next paragraph from the same narrative world and tests whether retrieval improves contextual coherence.

**Architecture:** The project has four layers: data collection and preprocessing, retrieval, paragraph-level continuation generation, and evaluation. We will start with a plain `distilgpt2` baseline, then add TF-IDF or BM25 retrieval over earlier paragraphs from Sherlock Holmes texts, and compare the systems using automatic metrics plus human evaluation.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Hugging Face Datasets, scikit-learn, rank-bm25, pytest, Jupyter/Colab, pandas

---

## Proposed Repository Structure

- `requirements.txt`
- `README.md`
- `data/README.md`
- `configs/baseline_distilgpt2.yaml`
- `configs/retrieval_distilgpt2.yaml`
- `scripts/download_gutenberg.py`
- `scripts/build_dataset.py`
- `scripts/build_retrieval_index.py`
- `scripts/train_baseline.py`
- `scripts/train_retrieval_model.py`
- `scripts/generate_samples.py`
- `scripts/run_auto_eval.py`
- `scripts/prepare_human_eval.py`
- `src/novel_continuation/__init__.py`
- `src/novel_continuation/data_sources.py`
- `src/novel_continuation/preprocess.py`
- `src/novel_continuation/dataset_builder.py`
- `src/novel_continuation/retrieval.py`
- `src/novel_continuation/prompting.py`
- `src/novel_continuation/training.py`
- `src/novel_continuation/evaluation.py`
- `tests/test_imports.py`
- `tests/data/test_preprocess.py`
- `tests/data/test_dataset_builder.py`
- `tests/retrieval/test_retrieval.py`
- `tests/training/test_prompting.py`
- `tests/evaluation/test_metrics.py`
- `docs/human-eval-rubric.md`
- `docs/experiments/experiment-log.md`

## Milestones

1. Set up a clean project skeleton and dependency list.
2. Download and clean the Sherlock Holmes corpus.
3. Build paragraph-level continuation datasets with leak-free splits.
4. Implement the retrieval baseline and inspect retrieved passages.
5. Train the plain continuation baseline.
6. Train the retrieval-augmented continuation model.
7. Run automatic evaluation and prepare human evaluation.
8. Write up results, ablations, and failure analysis.

### Task 1: Create the Project Skeleton

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\requirements.txt`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\__init__.py`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\tests\test_imports.py`
- Modify: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\README.md`

**Step 1: Write the failing test**

Add `tests/test_imports.py` with checks that the package imports and exposes a version string.

```python
from novel_continuation import __version__


def test_package_exposes_version():
    assert isinstance(__version__, str)
    assert __version__
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_imports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'novel_continuation'`

**Step 3: Write minimal implementation**

Create `src/novel_continuation/__init__.py`.

```python
__version__ = "0.1.0"
```

Create `requirements.txt` with the initial stack:

```text
torch
transformers
datasets
scikit-learn
rank-bm25
pandas
nltk
evaluate
bert-score
rouge-score
pytest
```

Update `README.md` with a short project overview and a setup section.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_imports.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt README.md src/novel_continuation/__init__.py tests/test_imports.py
git commit -m "chore: scaffold novel continuation project"
```

### Task 2: Add Data Source Metadata and Downloader

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\data\README.md`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\data_sources.py`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\scripts\download_gutenberg.py`
- Test: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\tests\data\test_sources.py`

**Step 1: Write the failing test**

```python
from novel_continuation.data_sources import SHERLOCK_SOURCES


def test_sherlock_sources_have_urls():
    assert SHERLOCK_SOURCES
    for item in SHERLOCK_SOURCES:
        assert item["title"]
        assert item["url"].startswith("https://")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_sources.py -v`
Expected: FAIL because `data_sources.py` does not exist yet

**Step 3: Write minimal implementation**

- Define `SHERLOCK_SOURCES` in `src/novel_continuation/data_sources.py`.
- Include at least `The Adventures of Sherlock Holmes`, `The Memoirs of Sherlock Holmes`, `The Hound of the Baskervilles`, and one backup source.
- Write `scripts/download_gutenberg.py` to download raw `.txt` files into `data/raw/`.
- Document the source list and folder layout in `data/README.md`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_sources.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add data/README.md src/novel_continuation/data_sources.py scripts/download_gutenberg.py tests/data/test_sources.py
git commit -m "feat: add sherlock data source manifest"
```

### Task 3: Clean Raw Text and Split into Paragraphs

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\preprocess.py`
- Test: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\tests\data\test_preprocess.py`

**Step 1: Write the failing test**

```python
from novel_continuation.preprocess import split_paragraphs


def test_split_paragraphs_removes_empty_blocks():
    raw_text = "CHAPTER I\n\nFirst paragraph.\n\n\nSecond paragraph."
    result = split_paragraphs(raw_text)
    assert result == ["First paragraph.", "Second paragraph."]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: FAIL because `split_paragraphs` is not implemented

**Step 3: Write minimal implementation**

Implement in `src/novel_continuation/preprocess.py`:

- `strip_gutenberg_boilerplate(text: str) -> str`
- `normalise_whitespace(text: str) -> str`
- `split_paragraphs(text: str) -> list[str]`
- `filter_short_paragraphs(paragraphs: list[str], min_chars: int) -> list[str]`

Also add a small command entry in the module so it can be used from later scripts.

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/novel_continuation/preprocess.py tests/data/test_preprocess.py
git commit -m "feat: add text cleaning and paragraph splitting"
```

### Task 4: Build the Paragraph Continuation Dataset

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\dataset_builder.py`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\scripts\build_dataset.py`
- Test: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\tests\data\test_dataset_builder.py`

**Step 1: Write the failing test**

```python
from novel_continuation.dataset_builder import build_examples


def test_build_examples_uses_only_previous_paragraphs():
    paragraphs = ["p0", "p1", "p2", "p3"]
    examples = build_examples(paragraphs, context_size=2)
    assert examples[0]["context"] == ["p0", "p1"]
    assert examples[0]["target"] == "p2"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_dataset_builder.py -v`
Expected: FAIL because `dataset_builder.py` does not exist yet

**Step 3: Write minimal implementation**

Implement in `src/novel_continuation/dataset_builder.py`:

- `build_examples(paragraphs, context_size)`
- `assign_split(example_index, total_examples)` using contiguous blocks
- `serialise_examples(examples, output_path)`

Create `scripts/build_dataset.py` to:

- read cleaned paragraphs from `data/raw/`
- build examples with metadata such as `book_id`, `chapter_id`, `paragraph_id`
- save `train.jsonl`, `val.jsonl`, `test.jsonl` into `data/processed/`

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_dataset_builder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/novel_continuation/dataset_builder.py scripts/build_dataset.py tests/data/test_dataset_builder.py
git commit -m "feat: add continuation dataset builder"
```

### Task 5: Implement the Retrieval Baseline

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\retrieval.py`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\scripts\build_retrieval_index.py`
- Test: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\tests\retrieval\test_retrieval.py`

**Step 1: Write the failing test**

```python
from novel_continuation.retrieval import retrieve_top_k


def test_retrieve_top_k_returns_earlier_passages_only():
    corpus = ["holmes lit the pipe", "watson entered the room", "the fog was dense"]
    query = "watson in the room"
    result = retrieve_top_k(query=query, candidates=corpus[:2], top_k=1)
    assert result == ["watson entered the room"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_retrieval.py -v`
Expected: FAIL because `retrieve_top_k` is not implemented

**Step 3: Write minimal implementation**

Implement in `src/novel_continuation/retrieval.py`:

- `fit_tfidf_retriever(candidates)`
- `retrieve_top_k(query, candidates, top_k)`
- optional `BM25Retriever` wrapper

Create `scripts/build_retrieval_index.py` to persist the paragraph index for later reuse.

**Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_retrieval.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/novel_continuation/retrieval.py scripts/build_retrieval_index.py tests/retrieval/test_retrieval.py
git commit -m "feat: add paragraph retrieval baseline"
```

### Task 6: Implement Prompt Formatting for Baseline and Retrieval Models

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\prompting.py`
- Test: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\tests\training\test_prompting.py`

**Step 1: Write the failing test**

```python
from novel_continuation.prompting import build_prompt


def test_build_prompt_includes_context_and_retrieved_sections():
    prompt = build_prompt(
        context=["p1", "p2"],
        retrieved=["old clue"],
        include_retrieval=True,
    )
    assert "[CONTEXT]" in prompt
    assert "[RETRIEVED]" in prompt
    assert "old clue" in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_prompting.py -v`
Expected: FAIL because `prompting.py` does not exist yet

**Step 3: Write minimal implementation**

Implement in `src/novel_continuation/prompting.py`:

- `build_prompt(context, retrieved, include_retrieval)`
- `build_training_text(context, retrieved, target)`

Prompt format:

```text
[CONTEXT]
...

[RETRIEVED]
...

[TARGET]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_prompting.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/novel_continuation/prompting.py tests/training/test_prompting.py
git commit -m "feat: add prompt builders for continuation training"
```

### Task 7: Train the Plain Generator Baseline

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\configs\baseline_distilgpt2.yaml`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\training.py`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\scripts\train_baseline.py`
- Modify: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\README.md`

**Step 1: Write the failing test**

```python
from novel_continuation.training import load_training_config


def test_load_training_config_reads_model_name(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name: distilgpt2\n")
    cfg = load_training_config(config_path)
    assert cfg["model_name"] == "distilgpt2"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_imports.py tests/training/test_prompting.py -v`
Expected: FAIL because `training.py` does not exist yet

**Step 3: Write minimal implementation**

Implement in `src/novel_continuation/training.py`:

- config loading
- tokenizer and model loading
- dataset tokenisation
- trainer setup for causal language modelling

Create `configs/baseline_distilgpt2.yaml` with:

- `model_name: distilgpt2`
- `max_length: 512`
- `batch_size: 2`
- `gradient_accumulation_steps: 4`
- `num_train_epochs: 1` for smoke tests

Create `scripts/train_baseline.py` to train on `data/processed/train.jsonl` and save checkpoints to `artifacts/baseline/`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_imports.py tests/training/test_prompting.py tests -v`
Expected: PASS for unit tests

**Step 5: Commit**

```bash
git add configs/baseline_distilgpt2.yaml src/novel_continuation/training.py scripts/train_baseline.py README.md
git commit -m "feat: add distilgpt2 baseline training pipeline"
```

### Task 8: Train the Retrieval-Augmented Generator

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\configs\retrieval_distilgpt2.yaml`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\scripts\train_retrieval_model.py`
- Modify: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\training.py`
- Modify: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\retrieval.py`

**Step 1: Write the failing test**

Add a test that checks retrieved paragraphs are injected into the training prompt before the target text.

```python
from novel_continuation.prompting import build_training_text


def test_build_training_text_places_retrieval_before_target():
    text = build_training_text(["p1"], ["clue"], "target")
    assert text.index("[RETRIEVED]") < text.index("[TARGET]")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_prompting.py -v`
Expected: FAIL if retrieval-aware training text is missing

**Step 3: Write minimal implementation**

- Extend `training.py` so the data loader attaches retrieved paragraphs to each example.
- Create `scripts/train_retrieval_model.py` to:
  - load the processed dataset
  - build or load the retrieval index
  - create retrieval-augmented prompts
  - fine-tune the same generator backbone
- Save checkpoints to `artifacts/retrieval/`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_prompting.py tests/retrieval/test_retrieval.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add configs/retrieval_distilgpt2.yaml scripts/train_retrieval_model.py src/novel_continuation/training.py src/novel_continuation/retrieval.py tests/training/test_prompting.py
git commit -m "feat: add retrieval-augmented training pipeline"
```

### Task 9: Add Automatic Evaluation and Sample Generation

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\src\novel_continuation\evaluation.py`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\scripts\generate_samples.py`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\scripts\run_auto_eval.py`
- Test: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\tests\evaluation\test_metrics.py`

**Step 1: Write the failing test**

```python
from novel_continuation.evaluation import compute_entity_overlap


def test_compute_entity_overlap_detects_shared_names():
    score = compute_entity_overlap(
        context_text="Holmes spoke to Watson in Baker Street.",
        generated_text="Watson answered Holmes quietly.",
    )
    assert score > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL because `evaluation.py` does not exist yet

**Step 3: Write minimal implementation**

Implement in `src/novel_continuation/evaluation.py`:

- `compute_perplexity(...)`
- `compute_bertscore(...)`
- `compute_rouge_l(...)`
- `compute_entity_overlap(context_text, generated_text)`

Create `scripts/generate_samples.py` to save model outputs for the same held-out prompts.
Create `scripts/run_auto_eval.py` to evaluate baseline and retrieval models on the test set and save a CSV table to `artifacts/eval/metrics.csv`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/novel_continuation/evaluation.py scripts/generate_samples.py scripts/run_auto_eval.py tests/evaluation/test_metrics.py
git commit -m "feat: add automatic evaluation pipeline"
```

### Task 10: Prepare Human Evaluation and Experiment Tracking

**Files:**
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\scripts\prepare_human_eval.py`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\docs\human-eval-rubric.md`
- Create: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\docs\experiments\experiment-log.md`
- Modify: `D:\111_temu_商品\gaogou456\st456-project-wt2026-gaogou456\README.md`

**Step 1: Write the failing test**

Write a test that asserts the human-eval export contains blinded system names and rubric columns.

```python
import csv


def test_human_eval_csv_has_required_columns(tmp_path):
    path = tmp_path / "human_eval.csv"
    path.write_text("sample_id,system_label,contextual_coherence,character_consistency\n")
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert "system_label" in row
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL after the new test is added until export helpers exist

**Step 3: Write minimal implementation**

- Create `scripts/prepare_human_eval.py` to export a blinded CSV with:
  - sample id
  - source context
  - gold next paragraph
  - baseline output
  - retrieval-model output
  - shuffled system labels
- Write `docs/human-eval-rubric.md` with the 1-5 scoring rubric for:
  - contextual coherence
  - narrative consistency
  - character consistency
  - fluency
- Write `docs/experiments/experiment-log.md` to track:
  - config name
  - dataset version
  - checkpoint path
  - metrics
  - qualitative notes

**Step 4: Run test to verify it passes**

Run: `pytest tests -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/prepare_human_eval.py docs/human-eval-rubric.md docs/experiments/experiment-log.md README.md
git commit -m "feat: add human evaluation materials and experiment log"
```

## Experiment Execution Order

Run the experiments in this order:

1. Smoke-test the pipeline on a tiny subset of one Holmes text.
2. Train the plain short-context baseline.
3. Train the plain longer-context baseline.
4. Train the retrieval-augmented model with `top_k=2`.
5. Repeat the retrieval model with one small ablation:
   - `top_k=1` or `top_k=3`
   - TF-IDF versus BM25
6. Generate held-out samples for all systems.
7. Run automatic evaluation.
8. Prepare the blinded human-eval sheet and collect scores.
9. Summarise findings and write the report figures and tables.

## Team Split Suggestion

- Member 1: data download, cleaning, dataset building
- Member 2: retrieval and prompt construction
- Member 3: baseline and retrieval model training in Colab
- Member 4: evaluation, human-eval preparation, experiment logging

Everyone should still review the final report, error analysis, and presentation.

## Definition of Done

The project is ready for report writing when all of the following are true:

- `data/processed/train.jsonl`, `val.jsonl`, and `test.jsonl` exist
- one plain baseline checkpoint exists
- one retrieval-augmented checkpoint exists
- `artifacts/eval/metrics.csv` exists
- a blinded human-eval sheet exists
- `docs/experiments/experiment-log.md` is filled in with final runs
- the README documents how to reproduce the pipeline
