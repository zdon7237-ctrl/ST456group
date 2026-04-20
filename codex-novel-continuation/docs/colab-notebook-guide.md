# Colab Notebook Guide

Copy each block into a separate Colab cell.

## Cell 1: Clone and install

```python
!git clone <YOUR_REPO_URL>
%cd st456-project-wt2026-gaogou456
!pip install -r requirements.txt
```

## Cell 2: Download and build the dataset

```python
!python scripts/download_gutenberg.py
!python scripts/build_dataset.py --context-size 4 --min-chars 40
```

## Cell 3: Inspect processed files

```python
!ls data/processed
!head -n 2 data/processed/train.jsonl
```

## Cell 4: Inspect token budgets

```python
!python scripts/inspect_token_stats.py --config configs/e1_distilgpt2_plain_full.yaml
!python scripts/inspect_token_stats.py --config configs/e3_distilgpt2_structured_long_context.yaml
!python scripts/inspect_token_stats.py --config configs/e5_distilgpt2_structured_aux_ranking.yaml
```

## Cell 5: Main experiment matrix

```python
!cat docs/experiments/main-experiment-matrix.md
```

## Cell 6: Train E1

```python
!python scripts/train_experiment.py --config configs/e1_distilgpt2_plain_full.yaml --seed 42
```

## Cell 7: Train E2

```python
!python scripts/train_experiment.py --config configs/e2_distilgpt2_structured_full.yaml --seed 42
```

## Cell 8: Train E3

```python
!python scripts/train_experiment.py --config configs/e3_distilgpt2_structured_long_context.yaml --seed 42
```

## Cell 9: Train E4

```python
!python scripts/train_experiment.py --config configs/e4_distilgpt2_structured_lora.yaml --seed 42
```

## Cell 10: Train E5

```python
!python scripts/train_experiment.py --config configs/e5_distilgpt2_structured_aux_ranking.yaml --seed 42
```

## Cell 11: Train E5-wide (`aux_weight=0.2`)

```python
!python scripts/train_experiment.py --config configs/e5_distilgpt2_structured_aux_ranking_wide.yaml --seed 42
```

## Cell 12: Run 3-seed evaluation for one experiment

```python
!python scripts/run_eval_3seed.py \
  --experiment-id e3 \
  --model-dir artifacts/e3_long_context \
  --output-dir artifacts/eval
```

`metrics_*_summary.csv` reports means for all metrics, but standard deviation only for generation-based metrics. `perplexity` is deterministic for a fixed checkpoint, so it keeps a mean/single-value interpretation only.

If a Colab session disconnects, rerun the same command with `--skip-existing` to reuse completed seeds and rebuild the summary.

## Cell 13: Compare E5 variants with validation main loss

```python
!python scripts/compare_aux_weight.py \
  artifacts/e5_aux_ranking \
  artifacts/e5_aux_ranking_wide
```

Use `metadata.validation.validation_main_loss` to choose the final E5 variant.

## Cell 14: Deferred / optional human-eval CSV export

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_e3_seed13.jsonl \
  --output-path artifacts/eval/human_eval_e3.csv \
  --system-label "System E3"
```

This step is not part of the default main evaluation pipeline.

## Cell 15: Open the deferred rubric

```python
!cat docs/human-eval-rubric.md
```

## Cell 16: Appendix retrieval experiment

```python
!python scripts/train_retrieval_model.py --config configs/retrieval_distilgpt2.yaml
```

## Cell 17: Update the experiment log

```python
!cat docs/experiments/experiment-log.md
```
