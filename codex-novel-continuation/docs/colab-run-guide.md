# Colab Run Guide

This guide is the main Colab execution path for the updated proposal-aligned project.

## Goal

Run the main E1-E5 experiment line in Google Colab, use the shared 3-seed evaluation CLI, and treat human evaluation as a deferred optional appendix.

## Main Sequence

1. install dependencies
2. download raw Sherlock Holmes texts
3. build processed continuation datasets
4. inspect token budgets
5. run E1-E5 plus the E5 `aux_weight=0.2` companion run
6. run 3-seed held-out generation and automatic metrics
7. compare E5 variants using validation main loss
8. optionally export a deferred human-eval CSV appendix
9. optionally run retrieval as appendix

## Setup

```python
!git clone <YOUR_REPO_URL>
%cd st456-project-wt2026-gaogou456
!pip install -r requirements.txt
```

If you want persistent storage:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Build the Dataset

```python
!python scripts/download_gutenberg.py
!python scripts/build_dataset.py --context-size 4 --min-chars 40
```

Expected outputs:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

## Inspect Token Budgets

Run this before the main experiments so `max_target_tokens` and truncation rates are visible:

```python
!python scripts/inspect_token_stats.py --config configs/e1_distilgpt2_plain_full.yaml
!python scripts/inspect_token_stats.py --config configs/e3_distilgpt2_structured_long_context.yaml
!python scripts/inspect_token_stats.py --config configs/e5_distilgpt2_structured_aux_ranking.yaml
```

If `k=4` experiments show heavy truncation under `max_length=512`, update those configs to `768` and rerun the inspection step.

## Main Experiment Configs

| Experiment | Config |
|---|---|
| E1 | `configs/e1_distilgpt2_plain_full.yaml` |
| E2 | `configs/e2_distilgpt2_structured_full.yaml` |
| E3 | `configs/e3_distilgpt2_structured_long_context.yaml` |
| E4 | `configs/e4_distilgpt2_structured_lora.yaml` |
| E5 | `configs/e5_distilgpt2_structured_aux_ranking.yaml` |
| E5-wide | `configs/e5_distilgpt2_structured_aux_ranking_wide.yaml` |

Interpretation rule:

- E1 vs E2: input structure
- E2 vs E3: context length
- E3 vs E4: full vs LoRA
- E3 vs E5: no auxiliary objective vs coherence-oriented auxiliary objective

## Train the Main Experiments

Run the configs one by one:

```python
!python scripts/train_experiment.py --config configs/e1_distilgpt2_plain_full.yaml --seed 42
!python scripts/train_experiment.py --config configs/e2_distilgpt2_structured_full.yaml --seed 42
!python scripts/train_experiment.py --config configs/e3_distilgpt2_structured_long_context.yaml --seed 42
!python scripts/train_experiment.py --config configs/e4_distilgpt2_structured_lora.yaml --seed 42
!python scripts/train_experiment.py --config configs/e5_distilgpt2_structured_aux_ranking.yaml --seed 42
!python scripts/train_experiment.py --config configs/e5_distilgpt2_structured_aux_ranking_wide.yaml --seed 42
```

Start with E1 as a smoke test before launching the rest.

## Run 3-Seed Automatic Evaluation

Example for E3:

```python
!python scripts/run_eval_3seed.py \
  --experiment-id e3 \
  --model-dir artifacts/e3_long_context \
  --output-dir artifacts/eval
```

Repeat for E1-E5 and the E5-wide run. Each evaluation writes:

- `generated_samples_<exp>_seed13.jsonl`
- `generated_samples_<exp>_seed42.jsonl`
- `generated_samples_<exp>_seed2026.jsonl`
- `metrics_<exp>_seed13.csv`
- `metrics_<exp>_seed42.csv`
- `metrics_<exp>_seed2026.csv`
- `metrics_<exp>_summary.csv`

Summary CSV columns should include:

- `num_samples_mean`
- `perplexity_mean`
- `bertscore_f1_mean`
- `bertscore_f1_std`
- `rouge_l_mean`
- `rouge_l_std`
- `entity_overlap_mean`
- `entity_overlap_std`

`perplexity` is deterministic for a fixed checkpoint, so the summary keeps its mean only; sampling variance is reported only for generation-based metrics.

If a Colab session drops midway through evaluation, rerun with `--skip-existing` to reuse completed seeds:

```python
!python scripts/run_eval_3seed.py \
  --experiment-id e3 \
  --model-dir artifacts/e3_long_context \
  --output-dir artifacts/eval \
  --skip-existing
```

## Select the Final E5 Variant

Use `metadata.validation.validation_main_loss` as the selection rule:

```python
!python scripts/compare_aux_weight.py \
  artifacts/e5_aux_ranking \
  artifacts/e5_aux_ranking_wide
```

- If one run is clearly lower, keep that run as the final E5.
- If the gap is smaller than about 1%, keep both in the results table as a robustness check.

## Deferred / Optional Human Evaluation

Example for E3:

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_e3_seed13.jsonl \
  --output-path artifacts/eval/human_eval_e3.csv \
  --system-label "System E3"
```

Use the rubric in `docs/human-eval-rubric.md` only if you later decide to add a human-evaluation appendix. It is not part of the default mainline pipeline.

## Appendix Experiment

Retrieval is no longer part of the main experiment line, but you can still run it as an appendix:

```python
!python scripts/train_retrieval_model.py --config configs/retrieval_distilgpt2.yaml
!python scripts/generate_samples.py --model-dir artifacts/retrieval --use-retrieval --history-path data/processed/train.jsonl --output-path artifacts/eval/generated_samples_retrieval.jsonl
!python scripts/run_auto_eval.py --model-dir artifacts/retrieval --generated-path artifacts/eval/generated_samples_retrieval.jsonl --output-path artifacts/eval/metrics_retrieval.csv
```

## Final Outputs to Keep

- `artifacts/e1_plain_full/`
- `artifacts/e2_structured_full/`
- `artifacts/e3_long_context/`
- `artifacts/e4_lora/`
- `artifacts/e5_aux_ranking/`
- `artifacts/e5_aux_ranking_wide/`
- `artifacts/eval/generated_samples_*_seed*.jsonl`
- `artifacts/eval/metrics_*_seed*.csv`
- `artifacts/eval/metrics_*_summary.csv`
- `artifacts/eval/human_eval_*.csv` if you choose to run the appendix human-eval step
