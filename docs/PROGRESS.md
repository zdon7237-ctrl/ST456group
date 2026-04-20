# ST456 Novel Continuation Project Progress

Date: 2026-04-20
Current phase: Phase 6 - 正式实验执行与报告准备
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
- [x] E1-E5 快速验证（QUICK_VALIDATION）在 Colab 上跑通，20 条样本
- [x] 快速验证结果已下载到本地 `colab_results/`
- [x] 代码 bug 修复：`generate_samples.py` 生成文本提取改用 token 切片（避免评测污染）
- [x] 代码 bug 修复：`retrieval.py` 变量遮蔽问题（`index` → `idx`）
- [x] Notebook 参数已改为正式实验模式（`QUICK_VALIDATION = False`）

---

## What Needs Attention Next

- [ ] 在 Colab 上跑正式实验（QUICK_VALIDATION=False，完整测试集 + 3 epochs）
- [ ] 下载正式实验结果，替换 `colab_results/` 中的快速验证数据
- [ ] 确认正式结果与快速验证趋势一致
- [ ] 组员完成人工评估（human_eval CSV 打分）
- [ ] 制作报告图表（指标柱状图、生成样本对比表）
- [ ] 撰写最终报告（NeurIPS 风格，最多 12 页）

---

## 快速验证结果摘要（20 条样本，仅供参考）

| 实验 | Perplexity ↓ | ROUGE-L ↑ | BERTScore ↑ | Entity Overlap ↑ |
|------|-------------|-----------|-------------|-------------------|
| E1 plain+full | 32.71 | 0.1256 | 0.8275 | 0.1200 |
| E2 structured+full | 28.37 | 0.1267 | 0.8263 | 0.1295 |
| E3 structured+long | 27.62 | 0.1121 | 0.8229 | 0.0568 |
| E4 structured+LoRA | 31.81 | 0.1079 | 0.8219 | 0.1102 |
| E5 structured+aux | 28.43 | 0.1097 | 0.8213 | 0.0725 |

---

## Current Findings That Affect Progress

- Colab compatibility still looks good because scripts use project-relative paths
- The repository main line is now centered on context design, training mode, and auxiliary objective comparisons
- `max_target_tokens` is now computed from token-length statistics instead of being hard-coded
- Retrieval remains available, but only as appendix machinery
- The next meaningful risk is runtime and memory behavior in Colab, not repository structure

---

## Immediate Next Actions

1. 压缩项目为 ZIP，上传到 Colab，Run All 跑正式实验（预计 30-60 分钟）
2. 下载正式结果 ZIP，解压替换本地 `colab_results/`
3. 对比正式结果与快速验证趋势是否一致
4. 分配人工评估任务给组员（每人评 5 条 × 5 实验 = 25 条）
5. 根据正式指标制作报告图表
6. 撰写报告

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
- 2026-04-20: E1-E5 快速验证结果已完成并下载到本地 `colab_results/`
- 2026-04-20: 修复 `generate_samples.py` 生成文本提取 bug（字符串匹配→token 切片）
- 2026-04-20: 修复 `retrieval.py` 变量遮蔽 bug（`index`→`idx`）
- 2026-04-20: Notebook 切换为正式实验模式（`QUICK_VALIDATION = False`）
- 2026-04-20: 更新追踪文件，记录快速验证指标和下一步计划

---

## Update Rule

After each meaningful change, update:

- `Current phase`
- `What needs attention next`
- `Session log`
- any blocker or resolved issue that changes priorities
