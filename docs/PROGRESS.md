# ST456 Novel Continuation Project Progress

Date: 2026-04-10
Current phase: Phase 3/4 - implementation and experiment prep
Status: in progress
Primary runtime: Google Colab

---

## Current Status

The project now has a proposal-aligned main line for E1-E5. The biggest remaining gap is no longer repository structure; it is real Colab execution, experiment outputs, and final report-facing analysis.

Tracking files were added on 2026-04-10 to make follow-up work easier to inspect and continue.

---

## What Is Already Done

- [x] Root proposal updated
- [x] Sherlock Holmes Gutenberg sources defined
- [x] Paragraph preprocessing and dataset builder implemented
- [x] `distilgpt2` baseline training path implemented
- [x] Retrieval path retained as appendix experiment
- [x] Generation, automatic evaluation, and human-eval export scripts exist
- [x] Colab guides rewritten for the E1-E5 main line
- [x] Root-level Chinese Colab guide added under `docs/`
- [x] Root-level run-all Colab notebook added under `docs/`
- [x] Root-level Colab entry switched from GitHub clone mode to direct ZIP upload mode
- [x] Root-level Colab notebook instructions translated to Chinese
- [x] Tracking files created under `docs/`
- [x] Proposal-aligned experiment configs for E1-E5 added
- [x] `LoRA` support added
- [x] Auxiliary objective switching added for `none` / `ranking` / `classification`
- [x] Auto-eval updated to output `perplexity`, `bertscore_f1`, `rouge_l`, and `entity_overlap`
- [x] Local pytest suite is green
- [x] All E1-E5 configs updated: `num_train_epochs` 1→3, `warmup_steps` 0→20
- [x] `generate_samples.py` updated: nucleus sampling enabled by default (`do_sample=True, top_p=0.95, temperature=0.8`)
- [x] `inspect_token_stats.py` fixed to pass `context_size` through the token-budget path

---

## What Needs Attention Next

- [ ] Run the first Colab smoke test on E1
- [ ] Run token-budget inspection on E1/E3/E5 in Colab
- [ ] Decide whether E3-E5 should stay at `max_length=512` or move to `768`
- [ ] Run E1-E5 and collect real metrics
- [ ] Decide whether appendix retrieval is worth running before the deadline

---

## Current Findings That Affect Progress

- Colab compatibility still looks good because scripts use project-relative paths
- The repository main line is now centered on context design, training mode, and auxiliary objective comparisons
- `max_target_tokens` is now computed from token-length statistics instead of being hard-coded
- Retrieval remains available, but only as appendix machinery
- The next meaningful risk is runtime and memory behavior in Colab, not repository structure

---

## Immediate Next Actions

1. Run `scripts/inspect_token_stats.py` for E1, E3, and E5 in Colab.
2. Run an E1 smoke training job in Colab.
3. If E1 succeeds, launch E2-E5 in order and keep the experiment log updated.

---

## Session Log

- 2026-04-10: Reviewed the updated `proposal.md` and compared it with the current repository.
- 2026-04-10: Checked training, evaluation, dataset, and Colab guide files.
- 2026-04-10: Confirmed that Colab is the intended runtime and should remain the primary execution path.
- 2026-04-10: Identified several proposal-code mismatches, especially around retrieval scope, LoRA, auxiliary coherence objectives, and evaluation metrics.
- 2026-04-10: Added `docs/PLAN.md`, `docs/PROGRESS.md`, and `docs/findings.md` as living tracking files.
- 2026-04-10: Fixed the fresh-checkout dataset-builder test import issue and made local retrieval tests resilient when `rank_bm25` is unavailable.
- 2026-04-10: Added proposal-aligned training config fields, token-budget logic, LoRA support, and auxiliary objective switching.
- 2026-04-10: Added E1-E5 configs, a generic `train_experiment.py` entrypoint, and a token inspection script.
- 2026-04-10: Rewrote README and Colab guides around the E1-E5 main line and moved retrieval to appendix status.
- 2026-04-10: Added `docs/COLAB使用指南.md` and `docs/ST456_colab_一键运行.ipynb` for direct Colab use from the repo root.
- 2026-04-12: Reworked the root-level Colab guide and notebook so they no longer depend on GitHub and instead use direct ZIP upload in Colab.
- 2026-04-12: Translated the root-level Colab notebook instructions, comments, and runtime messages into Chinese.
- 2026-04-12: Fixed the Colab token-stats regression in `scripts/inspect_token_stats.py` by passing `context_size` into the budget summary path.
- 2026-04-12: Verified the local pytest suite with `65 passed`.
- 2026-04-12: Proposal alignment audit: updated all E1-E5 configs to `num_train_epochs: 3` and `warmup_steps: 20` to match proposal claims.
- 2026-04-12: Added nucleus sampling support to `generate_samples.py` (`do_sample=True, top_p=0.95, temperature=0.8`) to align with Holtzman et al. reference in proposal.

---

## Update Rule

After each meaningful change, update:

- `Current phase`
- `What needs attention next`
- `Session log`
- any blocker or resolved issue that changes priorities
