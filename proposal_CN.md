# 项目提案表

请根据以下表格提供所需信息，并尽量做到简洁且信息充分。

**1. 你的项目标题是什么？**

基于预训练 GPT-2 微调的福尔摩斯小说段落续写中的上下文连贯性研究

**2. 你想解决什么问题？**

我们希望研究：在英语小说段落续写任务中，如何通过更合适的深度学习方法提升生成文本的上下文连贯性与人物一致性。具体而言，给定福尔摩斯系列小说中的前 `k` 个段落，模型需要生成下一个段落，同时保持与已有情节、场景和人物行为的一致性。

我们的核心研究问题是：不同的深度学习设计选择，是否能够显著提升小说续写中的长距离连贯性。

具体来说，我们将关注以下几个问题：

- 更长或更有结构的上下文输入，是否能提升叙事连贯性；
- 不同微调策略，如全参数微调与 LoRA，是否会影响生成质量；
- 在标准语言模型训练目标之外，引入辅助连贯性目标，是否能进一步改善人物一致性和情节一致性。

本项目重点关注深度学习方法本身，如输入表示、训练方案和损失函数设计，而不是把主要贡献放在检索模块或提示工程上。

**3. 你计划在项目中使用哪些深度学习方法？**

我们将把该任务建模为基于预训练自回归 Transformer 的条件文本生成任务。我们不会从头训练模型，而是对已有预训练语言模型进行微调。计划使用的基线模型包括：

- `distilgpt2`，约 8200 万参数；
- `gpt2` small，约 1.24 亿参数。

我们计划比较以下几类方法：

- 标准因果语言模型微调：
  - 输入为前 `k` 个段落；
  - 输出为目标下一段；
  - 使用 token-level cross-entropy 作为基本损失函数。

- 不同的上下文建模策略：
  - 短上下文窗口与长上下文窗口的比较；
  - 简单拼接输入与显式段落分隔标记的比较；
  - 研究输入结构是否会影响连贯性表现。

- 不同微调策略：
  - 全参数微调；
  - LoRA 参数高效微调。

- 不同训练目标设计：
  - 仅使用标准生成损失；
  - 在生成损失基础上加入辅助连贯性目标，例如 next-paragraph ranking 或 coherence classification。

训练方面，我们计划使用 GPT-2 tokenizer、AdamW 优化器、学习率 `5e-5`、warmup、mini-batch 训练、梯度裁剪，以及基于验证集表现的模型选择。我们也会调节上下文长度、batch size、epoch 数和辅助损失权重等超参数。

评测方面，我们计划使用：

- perplexity
- BERTScore
- ROUGE-L
- 如时间允许，再加入小规模人工评测，重点考察上下文连贯性和人物一致性。

如果时间允许，我们可能会把一个简单的检索增强版本作为附加对照实验，但不会把它作为项目的核心贡献。

**4. 你将使用什么数据集？请提供数据集信息以及链接，并简要说明其适用性。**

我们计划使用 Project Gutenberg 上公开领域的福尔摩斯系列文本作为主要数据集，包括：

- *The Adventures of Sherlock Holmes*（eBook 1661）
- *The Memoirs of Sherlock Holmes*（eBook 834）
- *The Hound of the Baskervilles*（eBook 2852）
- *The Return of Sherlock Holmes*（eBook 108）

这些文本共享稳定的叙事世界和重复出现的核心角色，如 Sherlock Holmes 和 Dr Watson，因此非常适合研究跨段落和跨篇章的一致性问题。

数据预处理流程计划包括：

- 去除 Gutenberg 页眉页脚等无关文本；
- 按段落切分全文；
- 过滤过短、噪声较大的段落；
- 构建 `(前 k 段, 下一段)` 的监督学习样本；
- 划分训练集、验证集和测试集，用于评估模型泛化能力。

数据集链接：

- [Project Gutenberg](https://www.gutenberg.org/)
- [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661)
- [The Memoirs of Sherlock Holmes](https://www.gutenberg.org/ebooks/834)
- [The Hound of the Baskervilles](https://www.gutenberg.org/ebooks/2852)
- [The Return of Sherlock Holmes](https://www.gutenberg.org/ebooks/108)

**5. 你的项目将基于哪些关键参考文献？**

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019.
- Fan, Lewis, and Dauphin, *Hierarchical Neural Story Generation*, ACL 2018.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020.

我们与已有工作的区别在于：我们并不试图提出一个大规模、复杂的新型故事生成系统，而是聚焦于一个更清晰、可控的实验问题，即在一个规模适中的文学语料上，系统比较不同上下文建模方式、微调策略和训练目标对小说段落续写质量的影响。

**请说明你的项目提案是否已准备好接受审批（Yes/No）：**

No
