# ST456 Novel Continuation Execution Plan

Updated: 2026-04-10 (implementation sync)
Primary runtime: Google Colab
Project focus: Sherlock Holmes paragraph continuation with the updated proposal as the source of truth

---

## Project Goal

Bring the repository, experiments, and documentation into line with the updated proposal:

- study contextual coherence in English novel paragraph continuation
- use pre-trained GPT-2 family models as the main backbone
- compare context modeling choices
- compare full fine-tuning vs LoRA
- compare standard generation loss vs an auxiliary coherence objective
- evaluate with Perplexity, BERTScore, ROUGE-L, and small-scale human evaluation if time permits

---

## Current Snapshot

- Root proposal has been updated
- Existing codebase now supports:
  - Gutenberg data download
  - paragraph preprocessing
  - `(first k paragraphs, next paragraph)` dataset building
  - proposal-aligned configs for E1-E5
  - `plain` vs `structured` context formatting
  - `full` vs `LoRA` training mode selection
  - `none` / `ranking` / `classification` auxiliary-objective paths
  - generation and proposal-aligned auto-eval export
- Colab is the intended execution environment for main experiments

---

## Phase Plan

### Phase 1: Tracking and audit setup
Status: done

- [x] Create persistent tracking files under `docs/`
- [x] Review updated proposal against current codebase
- [x] Record initial alignment gaps and Colab-related notes

### Phase 2: Proposal alignment
Status: done

- [x] Update repository-level messaging so retrieval is no longer framed as the core contribution
- [x] Update Colab guides to reflect the new experiment priorities
- [x] Define the minimum experiment matrix that still satisfies the updated proposal
- [x] Position retrieval as an optional appendix experiment

### Phase 3: Core experiment support
Status: in progress

- [x] Keep the full fine-tuning baseline path working
- [ ] Add a `gpt2` comparison path if compute budget allows
- [x] Add LoRA training support
- [x] Add configurable context-window / structured-context comparisons

### Phase 4: Coherence-objective support
Status: in progress

- [x] Choose a lightweight auxiliary coherence objective
- [x] Add config support for auxiliary loss weight
- [ ] Run a smoke test for the auxiliary objective path

### Phase 5: Evaluation alignment
Status: done

- [x] Wire Perplexity into the automatic evaluation pipeline
- [x] Wire BERTScore into the automatic evaluation pipeline
- [x] Keep ROUGE-L in the final metrics output
- [x] Keep human evaluation export and rubric usable in Colab

### Phase 6: Final experiment execution and reporting
Status: pending

- [ ] Run the final Colab experiment set
- [ ] Fill the experiment log with real runs
- [ ] Summarize key findings and failure cases
- [ ] Prepare report tables, figures, and contribution notes

---

## Minimum Acceptable Deliverable

The project is in a safe state for submission when all of the following are true:

- the Colab pipeline can run end to end
- the repository clearly matches the updated proposal
- at least one baseline and one improved method can be compared fairly
- the final evaluation output includes Perplexity, BERTScore, and ROUGE-L
- human-evaluation materials can be exported

---

## Risks and Watchpoints

- `gpt2` still remains optional and has not been added to the main run line
- The auxiliary-objective path is implemented, but the Colab smoke run has not been executed yet
- Retrieval still exists in code and appendix docs, so report wording should remain disciplined

---

## Working Rule

Update this file whenever:

- the project scope changes
- the experiment plan changes
- the minimum deliverable changes
- a major blocker appears or is resolved
