# Main Experiment Matrix

Use this table when writing the report so the comparison logic stays explicit.

| ID | Goal | Setup |
|---|---|---|
| E1 | Baseline | `distilgpt2` + `k=2` + `plain` + `full` + `aux=none` |
| E2 | Input structure | `distilgpt2` + `k=2` + `structured` + `full` + `aux=none` |
| E3 | Longer context | `distilgpt2` + `k=4` + `structured` + `full` + `aux=none` |
| E4 | Fine-tuning strategy | `distilgpt2` + `k=4` + `structured` + `lora` + `aux=none` |
| E5 | Training objective | `distilgpt2` + `k=4` + `structured` + `full` + `aux=ranking` |

## Allowed Interpretations

- E1 vs E2: input structure
- E2 vs E3: context length
- E3 vs E4: full fine-tuning vs LoRA
- E3 vs E5: no auxiliary objective vs coherence-oriented auxiliary objective

## Appendix

- A1: retrieval-augmented continuation
- S1: optional `gpt2` confirmation run if time and Colab budget allow
