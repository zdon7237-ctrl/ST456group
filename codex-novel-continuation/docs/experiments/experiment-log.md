# Experiment Log

Use this template to record each meaningful run.

## Quick Validation Mode

Notebook 支持 `QUICK_VALIDATION = True` 快速验证模式：

- 训练：所有实验 epochs 降为 1
- 评估：仅使用前 20 条测试样本
- 用途：验证全流程是否跑通，不用于最终结果

正式实验请确保 `QUICK_VALIDATION = False`。

## Entry Template

- Date:
- Experiment ID:
- Experiment name:
- Config path:
- Dataset path or version:
- Model checkpoint output:
- Main comparison axis:
- Context format / context size:
- Fine-tuning mode:
- Auxiliary objective:
- Retrieval setting:
- Key metrics:
- Notes:

## Example Entry

- Date: 2026-03-25
- Experiment ID: E1
- Experiment name: plain-full smoke
- Config path: `configs/e1_distilgpt2_plain_full.yaml`
- Dataset path or version: `data/processed/train.jsonl`
- Model checkpoint output: `artifacts/e1_plain_full`
- Main comparison axis: baseline
- Context format / context size: `plain`, `k=2`
- Fine-tuning mode: `full`
- Auxiliary objective: `none`
- Retrieval setting: disabled
- Key metrics: pending
- Notes: smoke test in Colab
