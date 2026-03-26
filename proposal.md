## Project proposal form

Please provide the information requested in the following form. Try provide concise and informative answers.

**1. What is your project title?**

Retrieval-Augmented Language Models for Context-Coherent English Novel Continuation

**2. What is the problem that you want to solve?**

We want to study whether retrieval augmentation can improve contextual coherence in English novel continuation. Given several preceding paragraphs from the same narrative world, the task is to generate the next paragraph while remaining consistent with earlier story events, settings, and character behaviour. Our main goal is to improve long-range contextual coherence, while also examining whether better coherence leads to better character consistency.

**3. What deep learning methodologies do you plan to use in your project?**

We plan to formulate the problem as a conditional text generation task using transformer-based language models. Our baseline models will be autoregressive generators such as DistilGPT-2 or GPT-2 fine-tuned on paragraph-level continuation data. Our proposed method is a retrieval-augmented continuation model: given the current context, we retrieve relevant earlier paragraphs from the same narrative world using a lightweight retriever such as TF-IDF or BM25, and provide these retrieved passages together with the local context to the generator. We will compare short-context, longer-context, and retrieval-augmented models using both automatic metrics and human evaluation. Our main automatic evaluation metrics will include perplexity, BERTScore, and ROUGE-L, with optional entity-consistency checks to measure whether key characters and narrative entities remain aligned with the context.

**4. What dataset will you use? Provide information about the dataset, and a URL for the dataset if available. Briefly discuss suitability of the dataset for your problem.**

We plan to use public-domain Sherlock Holmes texts from Project Gutenberg as our primary dataset. This corpus contains multiple stories and longer works set in the same narrative world, with recurring characters such as Sherlock Holmes and Dr Watson, which makes it suitable for studying contextual coherence and character consistency. A practical starting point is *The Adventures of Sherlock Holmes* ([Project Gutenberg eBook 1661](https://www.gutenberg.org/ebooks/1661)), and we plan to extend this to other Sherlock Holmes texts available on Project Gutenberg. The dataset is suitable because it is legally accessible, written in English, large enough for Colab-scale fine-tuning, and rich in recurring narrative elements that support retrieval-based continuation.

**5. List key references (e.g. research papers) that your project will be based on?**

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Radford et al., *Language Models are Unsupervised Multitask Learners*, OpenAI, 2019.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
- Fan, Lewis, and Dauphin, *Hierarchical Neural Story Generation*, ACL 2018.
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020.

**Please indicate whether your project proposal is ready for review (Yes/No):**

Yes

## Feedback (to be provided by the course lecturer)
