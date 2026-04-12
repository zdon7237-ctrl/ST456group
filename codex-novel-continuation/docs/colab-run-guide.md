# Colab Run Guide

This guide is the main Colab execution path for the updated proposal-aligned project.

## Goal

Run the main E1-E5 experiment line in Google Colab, then export automatic and human-evaluation outputs.

## Main Sequence

1. install dependencies
2. download raw Sherlock Holmes texts
3. build processed continuation datasets
4. inspect token budgets
5. run E1-E5
6. generate held-out samples
7. compute automatic metrics
8. export human-eval CSV
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

Interpretation rule:

- E1 vs E2: input structure
- E2 vs E3: context length
- E3 vs E4: full vs LoRA
- E3 vs E5: no auxiliary objective vs coherence-oriented auxiliary objective

## Train the Main Experiments

Run the configs one by one:

```python
!python scripts/train_experiment.py --config configs/e1_distilgpt2_plain_full.yaml
!python scripts/train_experiment.py --config configs/e2_distilgpt2_structured_full.yaml
!python scripts/train_experiment.py --config configs/e3_distilgpt2_structured_long_context.yaml
!python scripts/train_experiment.py --config configs/e4_distilgpt2_structured_lora.yaml
!python scripts/train_experiment.py --config configs/e5_distilgpt2_structured_aux_ranking.yaml
```

Start with E1 as a smoke test before launching the rest.

## Generate Samples

Example for E3:

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/e3_long_context \
  --output-path artifacts/eval/generated_samples_e3.jsonl
```

Repeat for E1-E5 so each experiment has its own sample file.

## Run Automatic Evaluation

Example for E3:

```python
!python scripts/run_auto_eval.py \
  --model-dir artifacts/e3_long_context \
  --generated-path artifacts/eval/generated_samples_e3.jsonl \
  --output-path artifacts/eval/metrics_e3.csv
```

Each output CSV should contain:

- `num_samples`
- `perplexity`
- `bertscore_f1`
- `rouge_l`
- `entity_overlap`

## Export Human Evaluation

Example for E3:

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_e3.jsonl \
  --output-path artifacts/eval/human_eval_e3.csv \
  --system-label "System E3"
```

Use the rubric in `docs/human-eval-rubric.md`.

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
- `artifacts/eval/generated_samples_*.jsonl`
- `artifacts/eval/metrics_*.csv`
- `artifacts/eval/human_eval_*.csv`
