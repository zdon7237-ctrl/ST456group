# ST456 Google Colab 使用指南

这份指南对应当前 ST456 项目的新版主线：

- E1-E5 是主实验
- retrieval 只作为 appendix / 附加实验
- 主要运行目录是压缩包里的 `codex-novel-continuation/`

推荐直接使用同目录下的 notebook：

- `docs/ST456_colab_一键运行.ipynb`

现在这套方案已经改成：

- 不用 GitHub
- 不用 `git clone`
- 直接把本地项目压成 zip 上传到 Colab

---

## 1. 你需要准备什么

你只需要准备两个东西：

1. 一个项目压缩包
2. 这个 notebook 文件

### 项目压缩包建议怎么做

建议把整个项目根目录压成一个 zip，也就是把这个文件夹打包：

- `D:\111_temu_商品\gaogou456\ST456group`

推荐压缩包命名：

- `ST456group.zip`

压缩包里至少要包含：

- `codex-novel-continuation/`
- `docs/`
- `proposal.md`

最关键的是 `codex-novel-continuation/` 必须在压缩包里。

### Notebook 文件

上传这个文件到 Colab：

- `docs/ST456_colab_一键运行.ipynb`

---

## 2. 为什么建议用 Colab

- `distilgpt2`、LoRA、自动评估这些步骤更适合在 Colab GPU 上跑
- 项目现在已经按 Colab-first 方式整理过
- 主线实验 E1-E5 和自动评估都已经有脚本入口；人评导出保留为 deferred optional 流程
- 本地不需要先把环境完全配好，也能推进主要实验

---

## 3. 这次要跑什么

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

## 4. 推荐的使用方式

### 第一步：打开 Colab

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 选择“上传”
3. 上传：
   - `docs/ST456_colab_一键运行.ipynb`

### 第二步：打开 GPU

在 Colab 菜单里选择：

- `Runtime -> Change runtime type -> GPU`

### 第三步：运行 notebook

1. 先看第一格参数区
2. 按需要修改：
   - `QUICK_VALIDATION`（快速验证模式，默认开启）
   - `USE_GOOGLE_DRIVE`
   - `RUN_FULL_MAINLINE`
   - `RUN_APPENDIX_RETRIEVAL`
   - `DOWNLOAD_RESULTS_ZIP`
3. 点击：
   - `Runtime -> Run all`

### 第四步：上传 zip

运行到上传步骤时，Colab 会弹出上传控件。

你只需要上传：

- `ST456group.zip`

notebook 会自动：

- 解压压缩包
- 找到 `codex-novel-continuation/`
- 进入该目录
- 安装依赖并继续运行

---

## 5. Notebook 会自动做什么

按默认流程，notebook 会：

1. 可选挂载 Google Drive
2. 让你上传 `ST456group.zip`
3. 自动解压 zip
4. 自动找到 `codex-novel-continuation/`
5. 安装 `requirements.txt`
6. 下载 Sherlock Holmes 数据
7. 构建 `train/val/test` 数据集
8. 跑 token budget 检查
9. 训练主实验
10. 运行 3-seed 样本生成与自动评估
11. 比较 E5 不同 `aux_weight` 的 validation main loss
12. 可选导出人工评测 CSV
13. 可选跑 retrieval appendix
14. 可选打包并下载结果

---

## 6. 参数建议

### 快速验证模式（推荐第一次使用）

第一次建议这样设置：

- `QUICK_VALIDATION = True`
- `RUN_FULL_MAINLINE = True`
- `RUN_APPENDIX_RETRIEVAL = False`
- `DOWNLOAD_RESULTS_ZIP = True`

快速验证模式会：

- 把所有实验的训练轮数从 3 降到 1
- 只用前 20 条测试样本做生成和评估

这样可以在几分钟内跑通 E1-E5 全流程，确认代码和环境没有问题。

### 正式实验设置

确认快速验证通过后，改为：

- `QUICK_VALIDATION = False`
- `RUN_FULL_MAINLINE = True`

这样会用完整的 3 epochs 训练和全量测试集评估。

正式评估默认会对每个实验用 3 个固定 seed 做生成和自动评测，最终汇总成 `metrics_*_summary.csv`。

其中：

- `perplexity` 对固定 checkpoint 是确定性的，因此 summary 里只保留均值/单值解释
- `ROUGE-L`、`BERTScore`、`entity_overlap` 会额外报告 seed 间波动

### 保守的分步设置

如果不确定环境是否正常，可以先：

- `QUICK_VALIDATION = True`
- `RUN_FULL_MAINLINE = False`

这样只跑 E1 一个实验的快速验证，确认没问题后再逐步打开。

### 是否挂载 Google Drive

如果你担心 Colab 断线后结果丢失，建议：

- `USE_GOOGLE_DRIVE = True`

否则可以先用 Colab 临时目录快速试跑。

---

## 7. 关键输出文件

主实验输出会在 `codex-novel-continuation/` 下生成：

- `artifacts/e1_plain_full/`
- `artifacts/e2_structured_full/`
- `artifacts/e3_long_context/`
- `artifacts/e4_lora/`
- `artifacts/e5_aux_ranking/`
- `artifacts/e5_aux_ranking_wide/`
- `artifacts/eval/generated_samples_*_seed*.jsonl`
- `artifacts/eval/metrics_*_seed*.csv`
- `artifacts/eval/metrics_*_summary.csv`
- `artifacts/eval/human_eval_*.csv`（如果你手动开启可选的人评导出）

如果你同时跑了 E5 的两个 `aux_weight` 版本，建议直接运行：

```bash
python codex-novel-continuation/scripts/compare_aux_weight.py \
  codex-novel-continuation/artifacts/e5_aux_ranking \
  codex-novel-continuation/artifacts/e5_aux_ranking_wide
```

再按输出里的 `validation_main_loss` 结论决定最终把哪一个 E5 放进主结果表。

如果开启下载打包，notebook 还会生成：

- `artifacts/colab_results.zip`

---

## 8. 常见问题

### 1. Colab 提示没有 GPU

解决方法：

- `Runtime -> Change runtime type -> GPU`

### 2. 上传 zip 后找不到 `codex-novel-continuation/`

解决方法：

- 检查压缩包里是否真的包含 `codex-novel-continuation` 文件夹
- 最稳妥的方式是直接把整个 `ST456group` 文件夹压缩，而不是只挑部分文件

### 3. E3-E5 太慢或显存紧张

先看 notebook 自动输出的 token stats：

- 如果 `max_length=512` 截断率太高，可以把对应 config 改成 `768`
- 如果 LoRA 或 auxiliary objective 太慢，先只跑 E1 smoke 验证整条链路

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

## 9. 推荐执行顺序

建议按下面顺序推进：

1. 先跑 notebook 默认 smoke
2. 检查 E1 是否完整产出
3. 再把 `RUN_FULL_MAINLINE` 打开
4. 跑 E2-E5
5. 最后再决定要不要跑 retrieval appendix

---

## 10. 相关文件

- 主 notebook：`docs/ST456_colab_一键运行.ipynb`
- 主 Colab 说明：`codex-novel-continuation/docs/colab-run-guide.md`
- notebook 版说明：`codex-novel-continuation/docs/colab-notebook-guide.md`
- 主实验矩阵：`codex-novel-continuation/docs/experiments/main-experiment-matrix.md`
- 人评 rubric：`codex-novel-continuation/docs/human-eval-rubric.md`
