# Colab Run Guide

If you want a cell-by-cell version designed to be copied directly into Colab,
see `docs/colab-notebook-guide.md`.

## Goal

This guide shows how to run the project end to end in Google Colab:

1. install dependencies
2. download the Sherlock Holmes corpus
3. build processed datasets
4. train the baseline model
5. train the retrieval-augmented model
6. generate samples
7. run automatic evaluation
8. export human evaluation material

## Recommended Colab Setup

- Use a GPU runtime in Colab.
- Open `Runtime` -> `Change runtime type` -> select `T4 GPU` if available.
- If you want outputs to persist, mount Google Drive.

## Option A: Run in Temporary Colab Storage

Use this if you only want to test the pipeline quickly.

```python
!git clone <YOUR_REPO_URL>
%cd st456-project-wt2026-gaogou456
!pip install -r requirements.txt
```

## Option B: Run with Google Drive

Use this if you want checkpoints and outputs to persist across sessions.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive
!git clone <YOUR_REPO_URL>
%cd st456-project-wt2026-gaogou456
!pip install -r requirements.txt
```

If the repository is already cloned:

```python
%cd /content/drive/MyDrive/st456-project-wt2026-gaogou456
!git pull
!pip install -r requirements.txt
```

## Step 1: Download the Raw Text Data

```python
!python scripts/download_gutenberg.py
```

Expected output:

- raw `.txt` files downloaded into `data/raw/`

## Step 2: Build the Processed Dataset

```python
!python scripts/build_dataset.py
```

Expected output:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

Optional custom settings:

```python
!python scripts/build_dataset.py --context-size 3 --min-chars 40
```

## Step 3: Train the Baseline Model

```python
!python scripts/train_baseline.py
```

Expected output:

- model checkpoint under `artifacts/baseline/`

If you want to inspect the baseline config first:

```python
!cat configs/baseline_distilgpt2.yaml
```

## Step 4: Train the Retrieval-Augmented Model

```python
!python scripts/train_retrieval_model.py
```

Expected output:

- model checkpoint under `artifacts/retrieval/`

If you want to inspect the retrieval config first:

```python
!cat configs/retrieval_distilgpt2.yaml
```

## Step 5: Generate Baseline Samples

```python
!python scripts/generate_samples.py --model-dir artifacts/baseline
```

Expected output:

- `artifacts/eval/generated_samples.jsonl`

If you want separate output files:

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/baseline \
  --output-path artifacts/eval/generated_samples_baseline.jsonl
```

## Step 6: Generate Retrieval Model Samples

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/retrieval \
  --use-retrieval \
  --history-path data/processed/train.jsonl \
  --output-path artifacts/eval/generated_samples_retrieval.jsonl
```

Expected behavior:

- retrieval context is built automatically for evaluation
- only prior text is used as retrieval history
- no current or future target leakage should occur

## Step 7: Run Automatic Evaluation

For baseline:

```python
!python scripts/run_auto_eval.py \
  --generated-path artifacts/eval/generated_samples_baseline.jsonl \
  --output-path artifacts/eval/metrics_baseline.csv
```

For retrieval:

```python
!python scripts/run_auto_eval.py \
  --generated-path artifacts/eval/generated_samples_retrieval.jsonl \
  --output-path artifacts/eval/metrics_retrieval.csv
```

Expected output:

- CSV file with lightweight automatic metrics such as `rouge_l` and `entity_overlap`

## Step 8: Export Human Evaluation CSV

For baseline:

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_baseline.jsonl \
  --output-path artifacts/eval/human_eval_baseline.csv \
  --system-label "System A"
```

For retrieval:

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_retrieval.jsonl \
  --output-path artifacts/eval/human_eval_retrieval.csv \
  --system-label "System B"
```

Use the scoring guide in `docs/human-eval-rubric.md`.

## Recommended Execution Order

Run these commands in order:

```python
!pip install -r requirements.txt
!python scripts/download_gutenberg.py
!python scripts/build_dataset.py
!python scripts/train_baseline.py
!python scripts/train_retrieval_model.py
!python scripts/generate_samples.py --model-dir artifacts/baseline --output-path artifacts/eval/generated_samples_baseline.jsonl
!python scripts/generate_samples.py --model-dir artifacts/retrieval --use-retrieval --history-path data/processed/train.jsonl --output-path artifacts/eval/generated_samples_retrieval.jsonl
!python scripts/run_auto_eval.py --generated-path artifacts/eval/generated_samples_baseline.jsonl --output-path artifacts/eval/metrics_baseline.csv
!python scripts/run_auto_eval.py --generated-path artifacts/eval/generated_samples_retrieval.jsonl --output-path artifacts/eval/metrics_retrieval.csv
!python scripts/prepare_human_eval.py --input-path artifacts/eval/generated_samples_baseline.jsonl --output-path artifacts/eval/human_eval_baseline.csv --system-label "System A"
!python scripts/prepare_human_eval.py --input-path artifacts/eval/generated_samples_retrieval.jsonl --output-path artifacts/eval/human_eval_retrieval.csv --system-label "System B"
```

## Useful Output Files

- `data/raw/`: downloaded Gutenberg text files
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `artifacts/baseline/`: baseline checkpoint
- `artifacts/retrieval/`: retrieval checkpoint
- `artifacts/eval/generated_samples_baseline.jsonl`
- `artifacts/eval/generated_samples_retrieval.jsonl`
- `artifacts/eval/metrics_baseline.csv`
- `artifacts/eval/metrics_retrieval.csv`
- `artifacts/eval/human_eval_baseline.csv`
- `artifacts/eval/human_eval_retrieval.csv`

## Common Problems

### Problem: `ModuleNotFoundError`

Run:

```python
!pip install -r requirements.txt
```

Then rerun the command.

### Problem: dataset file not found

You probably skipped dataset building.

Run:

```python
!python scripts/download_gutenberg.py
!python scripts/build_dataset.py
```

### Problem: generated samples file not found

You probably skipped sample generation.

Run one of:

```python
!python scripts/generate_samples.py --model-dir artifacts/baseline
```

or

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/retrieval \
  --use-retrieval \
  --history-path data/processed/train.jsonl
```

### Problem: retrieval generation complains about missing retrieval history

Use:

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/retrieval \
  --use-retrieval \
  --history-path data/processed/train.jsonl
```

Do not omit `--history-path` for retrieval evaluation.

### Problem: Colab disconnected and files are gone

Use Google Drive and rerun from the project directory stored there.

## Notes for the Report

- Record every meaningful run in `docs/experiments/experiment-log.md`
- Use `docs/human-eval-rubric.md` for human scoring
- Report baseline and retrieval results separately
- Keep the generated outputs for qualitative examples in the final report


这是一份为你翻译好的 **Colab 运行指南**。我保持了原始文档的结构和专业术语的准确性，确保你在中文语境下也能丝滑操作。

---

# Colab 运行指南

## 目标

本指南将展示如何在 Google Colab 中端到端地运行该项目：

1. 安装依赖项
2. 下载福尔摩斯语料库 (Sherlock Holmes corpus)
3. 构建处理后的数据集
4. 训练基准模型 (Baseline model)
5. 训练检索增强模型 (Retrieval-augmented model)
6. 生成样本
7. 运行自动评估
8. 导出人工评估材料

## 推荐的 Colab 设置

- 在 Colab 中使用 **GPU 运行时**。
- 打开 `修改` -> `笔记本设置` -> 选择 `T4 GPU`（如果可用）。
- 如果希望输出文件持久化保存，请挂载 **Google Drive**。

## 选项 A：在 Colab 临时存储中运行

如果你只想快速测试流水线，请使用此选项。

```python
!git clone <YOUR_REPO_URL>
%cd st456-project-wt2026-gaogou456
!pip install -r requirements.txt
```

## 选项 B：使用 Google Drive 运行

如果你希望模型检查点（Checkpoints）和输出结果在不同会话之间持久保存，请使用此选项。

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive
!git clone <YOUR_REPO_URL>
%cd st456-project-wt2026-gaogou456
!pip install -r requirements.txt
```

如果仓库已经克隆过了：

```python
%cd /content/drive/MyDrive/st456-project-wt2026-gaogou456
!git pull
!pip install -r requirements.txt
```

---

## 第 1 步：下载原始文本数据

```python
!python scripts/download_gutenberg.py
```

**预期输出：**
- 原始 `.txt` 文件下载至 `data/raw/` 目录。

## 第 2 步：构建处理后的数据集

```python
!python scripts/build_dataset.py
```

**预期输出：**
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

**可选自定义设置：**

```python
!python scripts/build_dataset.py --context-size 3 --min-chars 40
```

## 第 3 步：训练基准模型 (Baseline)

```python
!python scripts/train_baseline.py
```

**预期输出：**
- 模型检查点保存在 `artifacts/baseline/` 路径下。

如果你想先检查基准配置：

```python
!cat configs/baseline_distilgpt2.yaml
```

## 第 4 步：训练检索增强模型 (Retrieval Model)

```python
!python scripts/train_retrieval_model.py
```

**预期输出：**
- 模型检查点保存在 `artifacts/retrieval/` 路径下。

如果你想先检查检索配置：

```python
!cat configs/retrieval_distilgpt2.yaml
```

## 第 5 步：生成基准模型样本

```python
!python scripts/generate_samples.py --model-dir artifacts/baseline
```

**预期输出：**
- `artifacts/eval/generated_samples.jsonl`

如果你想指定单独的输出文件：

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/baseline \
  --output-path artifacts/eval/generated_samples_baseline.jsonl
```

## 第 6 步：生成检索模型样本

```python
!python scripts/generate_samples.py \
  --model-dir artifacts/retrieval \
  --use-retrieval \
  --history-path data/processed/train.jsonl \
  --output-path artifacts/eval/generated_samples_retrieval.jsonl
```

**预期行为：**
- 评估时会自动构建检索上下文。
- 仅使用先前的文本作为检索历史。
- **不会发生**当前或未来目标的标签泄漏（Target leakage）。

## 第 7 步：运行自动评估

**针对基准模型：**

```python
!python scripts/run_auto_eval.py \
  --generated-path artifacts/eval/generated_samples_baseline.jsonl \
  --output-path artifacts/eval/metrics_baseline.csv
```

**针对检索模型：**

```python
!python scripts/run_auto_eval.py \
  --generated-path artifacts/eval/generated_samples_retrieval.jsonl \
  --output-path artifacts/eval/metrics_retrieval.csv
```

**预期输出：**
- 包含轻量级自动评估指标（如 `rouge_l` 和 `entity_overlap`）的 CSV 文件。

## 第 8 步：导出人工评估 CSV

**针对基准模型：**

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_baseline.jsonl \
  --output-path artifacts/eval/human_eval_baseline.csv \
  --system-label "System A"
```

**针对检索模型：**

```python
!python scripts/prepare_human_eval.py \
  --input-path artifacts/eval/generated_samples_retrieval.jsonl \
  --output-path artifacts/eval/human_eval_retrieval.csv \
  --system-label "System B"
```

请参考 `docs/human-eval-rubric.md` 中的评分指南。

---

## 推荐执行顺序

请按顺序运行以下命令：

```python
!pip install -r requirements.txt
!python scripts/download_gutenberg.py
!python scripts/build_dataset.py
!python scripts/train_baseline.py
!python scripts/train_retrieval_model.py
!python scripts/generate_samples.py --model-dir artifacts/baseline --output-path artifacts/eval/generated_samples_baseline.jsonl
!python scripts/generate_samples.py --model-dir artifacts/retrieval --use-retrieval --history-path data/processed/train.jsonl --output-path artifacts/eval/generated_samples_retrieval.jsonl
!python scripts/run_auto_eval.py --generated-path artifacts/eval/generated_samples_baseline.jsonl --output-path artifacts/eval/metrics_baseline.csv
!python scripts/run_auto_eval.py --generated-path artifacts/eval/generated_samples_retrieval.jsonl --output-path artifacts/eval/metrics_retrieval.csv
!python scripts/prepare_human_eval.py --input-path artifacts/eval/generated_samples_baseline.jsonl --output-path artifacts/eval/human_eval_baseline.csv --system-label "System A"
!python scripts/prepare_human_eval.py --input-path artifacts/eval/generated_samples_retrieval.jsonl --output-path artifacts/eval/human_eval_retrieval.csv --system-label "System B"
```

---

## 常用输出文件

* `data/raw/`: 下载的 Gutenberg 文本原文件
* `data/processed/train.jsonl`
* `data/processed/val.jsonl`
* `data/processed/test.jsonl`
* `artifacts/baseline/`: 基准模型检查点
* `artifacts/retrieval/`: 检索模型检查点
* `artifacts/eval/generated_samples_baseline.jsonl`
* `artifacts/eval/generated_samples_retrieval.jsonl`
* `artifacts/eval/metrics_baseline.csv`
* `artifacts/eval/metrics_retrieval.csv`
* `artifacts/eval/human_eval_baseline.csv`
* `artifacts/eval/human_eval_retrieval.csv`

---

## 常见问题

### 问题：出现 `ModuleNotFoundError`
**解决方法：** 运行以下命令，然后重新执行报错的脚本。
```python
!pip install -r requirements.txt
```

### 问题：找不到数据集文件
**原因：** 你可能跳过了数据集构建步骤。
**解决方法：** 运行：
```python
!python scripts/download_gutenberg.py
!python scripts/build_dataset.py
```

### 问题：找不到生成的样本文件
**原因：** 你可能跳过了样本生成步骤。
**解决方法：** 运行以下命令之一：
```python
!python scripts/generate_samples.py --model-dir artifacts/baseline
```
或者
```python
!python scripts/generate_samples.py \
  --model-dir artifacts/retrieval \
  --use-retrieval \
  --history-path data/processed/train.jsonl
```

### 问题：检索生成报错，提示缺少检索历史
**解决方法：** 在进行检索模型评估时，**请勿省略** `--history-path` 参数：
```python
!python scripts/generate_samples.py \
  --model-dir artifacts/retrieval \
  --use-retrieval \
  --history-path data/processed/train.jsonl
```

### 问题：Colab 断开连接导致文件丢失
**解决方法：** 请使用 Google Drive 挂载方案，并从存储在 Drive 中的项目目录重新运行。

---

## 报告注意事项

* 在 `docs/experiments/experiment-log.md` 中记录每一次有意义的运行。
* 使用 `docs/human-eval-rubric.md` 进行人工评分。
* 分别报告基准模型和检索模型的结果。
* 保留生成的输出，以便在最终报告中提供定性示例（Qualitative examples）。

---
这份指南翻译完了！如果你在运行过程中遇到具体的报错，或者需要我帮你写一段自动上传 CSV 到 Drive 的脚本，随时告诉我。想试试看吗？
