# 项目提案表

请根据以下表格提供所需信息，并尽量做到简洁且信息充分。

**1. 你的项目标题是什么？**

基于检索增强语言模型的英语小说上下文连贯续写研究

**2. 你想解决什么问题？**

我们希望研究检索增强方法是否能够提升英语小说续写任务中的上下文连贯性。给定同一叙事世界中的若干前文段落，任务是生成下一个段落，同时保持与先前故事事件、场景设定以及人物行为的一致性。我们的核心目标是考察：在长文本叙事中，引入对早期相关段落的检索，是否能够帮助模型更好地保持长距离情节连贯性，以及这种改进是否也会带来更稳定的人物一致性。

**3. 你计划在项目中使用哪些深度学习方法？**

我们计划将该问题建模为基于 Transformer 的条件文本生成任务，并采用 PyTorch 与 Hugging Face Transformers 实现。基线系统将使用 `distilgpt2` 进行段落级续写训练：输入为前文上下文，输出为目标下一段。改进模型将在此基础上加入检索增强机制，先根据当前上下文，从同一叙事世界中已经出现过的早期段落里检索最相关内容，再把这些检索结果与局部上下文一起组织成结构化提示词输入生成模型。我们计划首先使用 TF-IDF 作为轻量级检索方法，并将 BM25 作为可比较方案。训练输入将采用带有 `[CONTEXT]`、`[RETRIEVED]` 和 `[TARGET]` 标记的格式，以便比较无检索基线与检索增强模型。评测方面，我们计划结合自动指标与人工评价来分析模型表现，自动指标将包括 perplexity、BERTScore、ROUGE-L，以及用于衡量人物与叙事元素一致性的实体重叠率。

**4. 你将使用什么数据集？请提供数据集信息以及链接（如果有），并简要说明其适用性。**

我们计划使用 Project Gutenberg 上公开领域的福尔摩斯系列文本作为主要数据集。一个可行的数据集范围包括四部作品：*The Adventures of Sherlock Holmes*（[eBook 1661](https://www.gutenberg.org/ebooks/1661)）、*The Memoirs of Sherlock Holmes*（[eBook 834](https://www.gutenberg.org/ebooks/834)）、*The Hound of the Baskervilles*（[eBook 2852](https://www.gutenberg.org/ebooks/2852)）以及 *The Return of Sherlock Holmes*（[eBook 108](https://www.gutenberg.org/ebooks/108)）。这些文本共享稳定的叙事世界和重复出现的核心角色，例如 Sherlock Holmes 和 Dr Watson，因此适合研究跨段落与跨篇章的一致性问题。我们计划对原始文本进行清洗，去除 Project Gutenberg 的前后说明文字与章节标题，按段落切分文本，过滤过短段落，并构建段落级上下文-目标样本，再采用连续区块划分训练集、验证集和测试集，以尽量避免信息泄漏。该数据集合法可获取、语言统一、规模适中，并且适合在 Colab 环境中完成训练与评测。

**5. 你的项目将基于哪些关键参考文献（例如研究论文）？**

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI, 2019.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
- Fan, Lewis, and Dauphin, *Hierarchical Neural Story Generation*, ACL 2018.
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020.

**请说明你的项目提案是否已准备好接受审阅（Yes/No）：**

No
