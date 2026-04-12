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
!python scripts/train_experiment.py --config configs/e1_distilgpt2_plain_full.yaml
```

## Cell 7: Train E2

```python
!python scripts/train_experiment.py --config configs/e2_distilgpt2_structured_full.yaml
```

## Cell 8: Train E3

```python
!python scripts/train_experiment.py --config configs/e3_distilgpt2_structured_long_context.yaml
```

## Cell 9: Train E4

```python
!python scripts/train_experiment.py --config configs/e4_distilgpt2_structured_lora.yaml
```

## Cell 10: Train E5

```python
!python scripts/train_experiment.py --config configs/e5_distilgpt2_structured_aux_ranking.yaml
```

## Cell 11: Generate samples for one experiment

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/e3_long_context \
  --output-path artifacts/eval/generated_samples_e3.jsonl
```

## Cell 12: Run automatic evaluation

```python
!python scripts/run_auto_eval.py \
  --model-dir artifacts/e3_long_context \
  --generated-path artifacts/eval/generated_samples_e3.jsonl \
  --output-path artifacts/eval/metrics_e3.csv
```

## Cell 13: Export human-eval CSV

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_e3.jsonl \
  --output-path artifacts/eval/human_eval_e3.csv \
  --system-label "System E3"
```

## Cell 14: Open the rubric

```python
!cat docs/human-eval-rubric.md
```

## Cell 15: Appendix retrieval experiment

```python
!python scripts/train_retrieval_model.py --config configs/retrieval_distilgpt2.yaml
```

## Cell 16: Update the experiment log

```python
!cat docs/experiments/experiment-log.md
```
