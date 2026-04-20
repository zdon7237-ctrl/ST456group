[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hZWYFfQo)
# ST456 Course Project

## Project Overview

This repository studies contextual coherence in Sherlock Holmes paragraph continuation with a Colab-first workflow. The main project narrative now follows the updated proposal:

- context design
- fine-tuning strategy
- coherence-oriented auxiliary objectives

Retrieval is still available in code, but it is treated as an appendix / optional ablation rather than the main contribution.

## Main Experiment Matrix

| ID | Goal | Core setup |
|---|---|---|
| E1 | Baseline | `distilgpt2` + `k=2` + `plain` + `full` + `aux=none` |
| E2 | Input structure | `distilgpt2` + `k=2` + `structured` + `full` + `aux=none` |
| E3 | Longer context | `distilgpt2` + `k=4` + `structured` + `full` + `aux=none` |
| E4 | Fine-tuning strategy | `distilgpt2` + `k=4` + `structured` + `lora` + `aux=none` |
| E5 | Training objective | `distilgpt2` + `k=4` + `structured` + `full` + `aux=ranking` |

Interpret the results only through these pairwise comparisons:

- E1 vs E2: input structure
- E2 vs E3: context length
- E3 vs E4: full fine-tuning vs LoRA
- E3 vs E5: no auxiliary objective vs coherence-oriented auxiliary objective

## Evaluation

The headline automatic metrics are:

- `perplexity`
- `bertscore_f1`
- `rouge_l`

The main evaluation pipeline uses the shared 3-seed generation workflow and reports mean values across seeds.

The project also keeps:

- `entity_overlap` as a diagnostic metric
- Human evaluation is deferred; `prepare_human_eval.py` is retained for future use but is not part of the main evaluation pipeline.

## Setup

### Local setup

```bash
pip install -r requirements.txt
```

### Colab setup

```python
!git clone <your-private-repo-url>
%cd st456-project-wt2026-gaogou456
!pip install -r requirements.txt
```

## Main Workflow

1. Download the Sherlock Holmes corpus.
2. Build paragraph-level continuation datasets.
3. Inspect token budgets before training.
4. Train E1-E5 plus the E5 `aux_weight=0.2` companion run.
5. Run 3-seed held-out generation and automatic evaluation.
6. Compare E5 variants with validation main loss, then report test metrics.
7. Human evaluation is deferred; optionally export sheets later if you decide to add an appendix.
8. Optionally run retrieval as an appendix experiment.

## Important Scripts

- `scripts/download_gutenberg.py`
- `scripts/build_dataset.py`
- `scripts/inspect_token_stats.py`
- `scripts/train_experiment.py`
- `scripts/generate_samples.py`
- `scripts/run_auto_eval.py`
- `scripts/run_eval_3seed.py`
- `scripts/compare_aux_weight.py`
- `scripts/prepare_human_eval.py`

## Important Configs

- `configs/e1_distilgpt2_plain_full.yaml`
- `configs/e2_distilgpt2_structured_full.yaml`
- `configs/e3_distilgpt2_structured_long_context.yaml`
- `configs/e4_distilgpt2_structured_lora.yaml`
- `configs/e5_distilgpt2_structured_aux_ranking.yaml`
- `configs/e5_distilgpt2_structured_aux_ranking_wide.yaml`
- `configs/retrieval_distilgpt2.yaml` for the appendix experiment

## Colab Guides

- `docs/colab-run-guide.md`
- `docs/colab-notebook-guide.md`
- `docs/experiments/main-experiment-matrix.md`

## Course Information

Deadline: Thursday **30/04/2026, 23:59**

The project should demonstrate neural network architecture design, implementation, training, evaluation, and interpretation. The final report should follow the ICML style and stay within the course page limits.
