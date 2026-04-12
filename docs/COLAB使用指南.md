# ST456 Google Colab 使用指南

这份指南对应当前 ST456 项目的新版主线：

- E1-E5 是主实验
- retrieval 只作为 appendix / 附加实验
- 主要运行目录是仓库里的 `codex-novel-continuation/`

推荐直接使用同目录下的 notebook：

- `docs/ST456_colab_一键运行.ipynb`

这个 notebook 设计成适合在 Colab 里直接点击 `Run all`。你只需要先改参数区里的仓库地址和开关。

---

## 1. 为什么建议用 Colab

- `distilgpt2`、`LoRA`、自动评估这些步骤更适合在 Colab GPU 上跑
- 项目现在已经按 Colab-first 方式整理过
- 主线实验 E1-E5、自动评估和人评导出都已经有脚本入口
- 本地环境不需要先完全配置好，也能推进主要实验

---

## 2. 这次要跑什么

主实验矩阵固定为：

| 实验 | 目的 | 配置 |
|---|---|---|
| E1 | 基线 | `distilgpt2` + `k=2` + `plain` + `full` + `aux=none` |
| E2 | 输入结构 | `distilgpt2` + `k=2` + `structured` + `full` + `aux=none` |
| E3 | 上下文长度 | `distilgpt2` + `k=4` + `structured` + `full` + `aux=none` |
| E4 | 微调策略 | `distilgpt2` + `k=4` + `structured` + `lora` + `aux=none` |
| E5 | 训练目标 | `distilgpt2` + `k=4` + `structured` + `full` + `aux=ranking` |

报告里只按下面四组关系解释：

- E1 vs E2：输入结构
- E2 vs E3：上下文长度
- E3 vs E4：full vs LoRA
- E3 vs E5：是否加入 coherence-oriented auxiliary objective

附加实验：

- retrieval appendix

---

## 3. 推荐的使用方式

### 方式 A：直接上传 notebook 到 Colab

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 选择“上传”
3. 上传：
   - `docs/ST456_colab_一键运行.ipynb`
4. 打开 notebook 后，把第一段参数里的：
   - `REPO_URL`
   - `REPO_DIR_NAME`
   - 是否跑完整主线
   - 是否跑 appendix retrieval
   改好
5. 选择：
   - `Runtime -> Change runtime type -> GPU`
6. 点击：
   - `Runtime -> Run all`

### 方式 B：先把仓库推到 GitHub，再从 Colab 打开

如果你们已经把这份 notebook 提交进仓库，也可以直接在 Colab 里从 GitHub 打开它，然后同样执行 `Run all`。

---

## 4. Notebook 会自动做什么

按默认流程，notebook 会：

1. 可选挂载 Google Drive
2. clone / 更新仓库
3. 进入 `codex-novel-continuation/`
4. 安装 `requirements.txt`
5. 下载 Sherlock Holmes 数据
6. 构建 `train/val/test` 数据集
7. 跑 token budget 检查
8. 训练主实验
9. 生成样本
10. 计算自动评估指标
11. 导出人工评测 CSV
12. 可选跑 retrieval appendix
13. 可选打包并下载结果

---

## 5. 参数建议

### 最稳妥的第一次运行

第一次建议这样设置：

- `RUN_FULL_MAINLINE = False`
- `RUN_APPENDIX_RETRIEVAL = False`
- `DOWNLOAD_RESULTS_ZIP = True`

这样 notebook 会先跑：

- 数据准备
- token stats
- E1 smoke
- E1 样本生成
- E1 自动评估
- E1 人评导出

确认流程完全打通之后，再把：

- `RUN_FULL_MAINLINE = True`

继续跑 E2-E5。

### 是否挂载 Google Drive

如果你担心 Colab 断线后结果丢失，建议：

- `USE_GOOGLE_DRIVE = True`

否则可以先用临时目录快速试跑。

---

## 6. 关键输出文件

主实验输出会在 `codex-novel-continuation/` 下生成：

- `artifacts/e1_plain_full/`
- `artifacts/e2_structured_full/`
- `artifacts/e3_long_context/`
- `artifacts/e4_lora/`
- `artifacts/e5_aux_ranking/`
- `artifacts/eval/generated_samples_*.jsonl`
- `artifacts/eval/metrics_*.csv`
- `artifacts/eval/human_eval_*.csv`

如果开启下载打包，notebook 还会生成：

- `artifacts/colab_results.zip`

---

## 7. 常见问题

### 1. Colab 提示没有 GPU

解决方法：

- `Runtime -> Change runtime type -> GPU`

### 2. `REPO_URL` 没填导致 clone 失败

解决方法：

- 把参数区里的 `REPO_URL` 改成你们真实的 GitHub 仓库地址

### 3. E3-E5 太慢或显存紧张

先看 notebook 自动输出的 token stats：

- 如果 `max_length=512` 截断率太高，可以把对应 config 改成 `768`
- 如果 LoRA 或 aux objective 太慢，先只跑 E1 smoke 验证整条链路

### 4. 结果丢失

解决方法：

- 打开 `USE_GOOGLE_DRIVE = True`
- 或者启用 `DOWNLOAD_RESULTS_ZIP = True`

### 5. retrieval 为什么没有放进主线

因为现在新版 proposal 的主线是：

- context design
- fine-tuning strategy
- auxiliary coherence objective

retrieval 只保留为 appendix / optional ablation。

---

## 8. 推荐执行顺序

建议按下面顺序推进：

1. 先跑 notebook 默认 smoke
2. 检查 E1 是否完整产出
3. 再把 `RUN_FULL_MAINLINE` 打开
4. 跑 E2-E5
5. 最后再决定要不要跑 retrieval appendix

---

## 9. 相关文件

- 主 notebook：`docs/ST456_colab_一键运行.ipynb`
- 主 Colab 说明：`codex-novel-continuation/docs/colab-run-guide.md`
- notebook 版说明：`codex-novel-continuation/docs/colab-notebook-guide.md`
- 主实验矩阵：`codex-novel-continuation/docs/experiments/main-experiment-matrix.md`
- 人评 rubric：`codex-novel-continuation/docs/human-eval-rubric.md`
