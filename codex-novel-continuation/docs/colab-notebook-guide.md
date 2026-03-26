# Colab Notebook Guide

This guide is the notebook-style version of the Colab instructions.
Copy each block into a separate Colab cell and run them from top to bottom.

Replace `<YOUR_REPO_URL>` with your repository URL before running.

## Cell 1: Optional Drive Mount

Use this cell if you want checkpoints and outputs to persist.
If you only want a quick temporary run, skip this cell.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2: Clone the Repository

If you are using Drive:

```python
%cd /content/drive/MyDrive
!git clone <YOUR_REPO_URL>
%cd st456-project-wt2026-gaogou456
```

If you are not using Drive:

```python
%cd /content
!git clone <YOUR_REPO_URL>
%cd st456-project-wt2026-gaogou456
```

If the repo is already cloned:

```python
!git pull
```

## Cell 3: Install Dependencies

```python
!pip install -r requirements.txt
```

## Cell 4: Check the Repository Layout

This is optional, but useful for sanity-checking that you are in the right folder.

```python
!ls
!ls scripts
!ls configs
```

## Cell 5: Download the Sherlock Holmes Texts

```python
!python scripts/download_gutenberg.py
```

Expected result:

- raw Gutenberg `.txt` files appear in `data/raw/`

## Cell 6: Build the Processed Dataset

```python
!python scripts/build_dataset.py
```

Expected result:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

Optional custom version:

```python
!python scripts/build_dataset.py --context-size 3 --min-chars 40
```

## Cell 7: Inspect Processed Files

```python
!ls data/processed
!head -n 2 data/processed/train.jsonl
!head -n 2 data/processed/val.jsonl
!head -n 2 data/processed/test.jsonl
```

## Cell 8: Inspect the Baseline Config

```python
!cat configs/baseline_distilgpt2.yaml
```

## Cell 9: Train the Baseline Model

```python
!python scripts/train_baseline.py
```

Expected result:

- checkpoint files under `artifacts/baseline/`

## Cell 10: Inspect the Retrieval Config

```python
!cat configs/retrieval_distilgpt2.yaml
```

## Cell 11: Train the Retrieval Model

```python
!python scripts/train_retrieval_model.py
```

Expected result:

- checkpoint files under `artifacts/retrieval/`

## Cell 12: Generate Baseline Samples

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/baseline \
  --output-path artifacts/eval/generated_samples_baseline.jsonl
```

## Cell 13: Generate Retrieval Samples

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/retrieval \
  --use-retrieval \
  --history-path data/processed/train.jsonl \
  --output-path artifacts/eval/generated_samples_retrieval.jsonl
```

This retrieval version uses:

- `train.jsonl` as the history source
- only prior text for retrieval history
- no current or future target leakage

## Cell 14: Run Automatic Evaluation for Baseline

```python
!python scripts/run_auto_eval.py \
  --generated-path artifacts/eval/generated_samples_baseline.jsonl \
  --output-path artifacts/eval/metrics_baseline.csv
```

## Cell 15: Run Automatic Evaluation for Retrieval

```python
!python scripts/run_auto_eval.py \
  --generated-path artifacts/eval/generated_samples_retrieval.jsonl \
  --output-path artifacts/eval/metrics_retrieval.csv
```

## Cell 16: Export Human Evaluation CSV for Baseline

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_baseline.jsonl \
  --output-path artifacts/eval/human_eval_baseline.csv \
  --system-label "System A"
```

## Cell 17: Export Human Evaluation CSV for Retrieval

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_retrieval.jsonl \
  --output-path artifacts/eval/human_eval_retrieval.csv \
  --system-label "System B"
```

## Cell 18: Inspect Final Outputs

```python
!ls artifacts
!ls artifacts/baseline
!ls artifacts/retrieval
!ls artifacts/eval
```

## Cell 19: View the Metric Files

```python
!cat artifacts/eval/metrics_baseline.csv
!cat artifacts/eval/metrics_retrieval.csv
```

## Cell 20: Open the Human Evaluation Rubric

```python
!cat docs/human-eval-rubric.md
```

## Cell 21: Update the Experiment Log

```python
!cat docs/experiments/experiment-log.md
```

After each meaningful run, copy the template in that file and record:

- config path
- dataset path
- checkpoint path
- metrics
- notes

## Fast Smoke Test Version

If this is your first Colab run, use this smaller sequence first:

```python
!pip install -r requirements.txt
!python scripts/download_gutenberg.py
!python scripts/build_dataset.py
!head -n 1 data/processed/train.jsonl
!head -n 1 data/processed/test.jsonl
!cat configs/baseline_distilgpt2.yaml
```

Then, if the outputs look correct, continue with training.

## Common Errors

### `ModuleNotFoundError`

Run:

```python
!pip install -r requirements.txt
```

### Dataset files missing

Run:

```python
!python scripts/download_gutenberg.py
!python scripts/build_dataset.py
```

### Retrieval generation complains about missing history

Use:

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/retrieval \
  --use-retrieval \
  --history-path data/processed/train.jsonl \
  --output-path artifacts/eval/generated_samples_retrieval.jsonl
```

### Colab session reset and files disappeared

Use the Google Drive version of the setup and keep the repository in Drive.
