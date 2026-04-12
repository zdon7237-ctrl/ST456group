# Findings and Decisions

Updated: 2026-04-10 (implementation sync)

---

## Requirements

- The project should run primarily in Google Colab
- The updated root proposal is now the main source of truth
- The main research contribution should focus on deep learning methodology
- Retrieval is no longer the main headline contribution
- The final evaluation plan should include:
  - Perplexity
  - BERTScore
  - ROUGE-L
  - human evaluation if time permits

---

## Current Codebase Findings

- The data source list matches the updated proposal:
  - `The Adventures of Sherlock Holmes`
  - `The Memoirs of Sherlock Holmes`
  - `The Hound of the Baskervilles`
  - `The Return of Sherlock Holmes`
- The preprocessing pipeline already handles:
  - Gutenberg header/footer stripping
  - paragraph splitting
  - short-paragraph filtering
- The dataset builder already creates supervised continuation samples from prior paragraphs to the next paragraph
- The repository already has a Colab-first structure with runnable scripts for:
  - download
  - preprocessing
  - training
  - generation
  - evaluation

---

## Proposal Alignment Findings

- The repository README and Colab guides now describe the project through E1-E5 rather than baseline-vs-retrieval
- The training code now supports:
  - `plain` and `structured` context formatting
  - `full` and `lora` training modes
  - `none`, `ranking`, and `classification` auxiliary objective modes
  - token-budget driven `max_target_tokens`
- Automatic evaluation now exposes:
  - `perplexity`
  - `bertscore_f1`
  - `rouge_l`
  - `entity_overlap`
- Retrieval remains available as appendix code and config, but not as the main project storyline
- A clear `gpt2` comparison path is still not implemented
- A root-level Chinese Colab guide and a run-all notebook now exist for easier handoff and execution

---

## Verification Findings

- Colab execution looks more important than local execution for this project
- Local verification still matters because it catches obvious regressions before Colab runs
- The fresh-checkout dataset-builder import issue has been fixed in tests
- Local retrieval code now has a fallback path when `rank_bm25` is unavailable
- The full local test suite currently verifies as `60 passed`

---

## Decisions

| Decision | Rationale |
|----------|-----------|
| Keep `docs/PLAN.md`, `docs/PROGRESS.md`, and `docs/findings.md` as living files | Makes ongoing work easy to inspect and continue |
| Treat Google Colab as the primary runtime | Matches the actual intended experiment environment |
| Use the updated root proposal as the reference point for future changes | Prevents the repo from drifting back to the old retrieval-first framing |
| Prioritize proposal alignment and evaluation alignment before larger new experiments | This reduces submission risk fastest |

---

## Open Questions

- Should retrieval remain as an optional ablation or be removed from the main report storyline entirely?
- Is the minimum acceptable model comparison:
  - `distilgpt2` baseline + LoRA variant
  - or `distilgpt2` vs `gpt2`
  - or both, if time allows?
- What is the lightest auxiliary coherence objective that is still convincing and feasible in Colab?

---

## Issues Encountered

| Issue | Current status |
|-------|----------------|
| Older repository framing still emphasizes retrieval | resolved in README and Colab guides |
| LoRA path missing | resolved |
| Auxiliary coherence objective missing | resolved |
| Auto-eval output does not yet match the proposal | resolved |
| Local pytest collection issue in dataset-builder tests | resolved |
| Planning catchup script could not run locally because Python access was denied in this environment | noted |
| Colab smoke run has not been executed yet | open |

---

## Useful Reference Files

- `proposal.md`
- `proposal_CN.md`
- `docs/plans/2026-03-25-novel-continuation-design.md`
- `docs/plans/2026-03-25-novel-continuation-implementation.md`
- `docs/COLAB使用指南.md`
- `docs/ST456_colab_一键运行.ipynb`
- `codex-novel-continuation/README.md`
- `codex-novel-continuation/docs/colab-run-guide.md`
- `codex-novel-continuation/docs/colab-notebook-guide.md`

---

## Working Rule

Add new findings here whenever we discover:

- a requirement change
- a technical blocker
- an experiment decision
- a verification failure
- a meaningful result from Colab runs
