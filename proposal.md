## Project proposal form



**1. What is your project title?**
A Study on Contextual Coherence in Sherlock Holmes Novel Paragraph Continuation via Fine-Tuning Pre-trained GPT-2


**2. What is the problem that you want to solve?**
We aim to investigate: in the task of English novel paragraph continuation, how more suitable deep learning methods can improve the contextual coherence and character consistency of generated text. Specifically, given the first `k` paragraphs from the Sherlock Holmes series, the model is required to generate the next paragraph while maintaining consistency with existing plot, setting, and character behavior.

Our core research question is: whether different deep learning design choices can significantly improve long-range coherence in novel continuation.

Concretely, we will focus on the following questions:

- Whether longer or more structured context inputs can improve narrative coherence;
- Whether different fine-tuning strategies, such as full-parameter fine-tuning versus LoRA, affect generation quality;
- Whether introducing auxiliary coherence objectives beyond the standard language model training objective can further improve character consistency and plot consistency.

This project focuses on deep learning methods themselves — such as input representation, training schemes, and loss function design — rather than placing the primary contribution on retrieval modules or prompt engineering.

**3. What deep learning methodologies do you plan to use in your project?**

We will model this task as a conditional text generation task based on pre-trained autoregressive Transformers. Rather than training from scratch, we will fine-tune existing pre-trained language models. The baseline models we plan to use include:

- `distilgpt2`
- `gpt2`

We plan to compare the following categories of methods:

- Standard causal language model fine-tuning:
  - Input consists of the first `k` paragraphs;
  - Output is the target next paragraph;
  - Token-level cross-entropy is used as the basic loss function.

- Different context modeling strategies:
  - Comparison of short versus long context windows;
  - Comparison of simple concatenation versus explicit paragraph separator tokens;
  - Investigation of whether input structure affects coherence performance.

- Different fine-tuning strategies:
  - Full-parameter fine-tuning;
  - LoRA parameter-efficient fine-tuning.

- Different training objective designs:
  - Using only the standard generation loss;
  - Adding an auxiliary coherence objective on top of the generation loss, such as next-paragraph ranking or coherence classification.

For training, we plan to use the GPT-2 tokenizer, AdamW optimizer, a learning rate of `5e-5`, warmup, mini-batch training, gradient clipping, and model selection based on validation set performance. We will also tune hyperparameters such as context length, batch size, number of epochs, and auxiliary loss weight.

For evaluation, we plan to use:

- Perplexity
- BERTScore
- ROUGE-L
- A small-scale human evaluation focusing on contextual coherence and character consistency, if time permits.


**4. What dataset will you use? Provide information about the dataset, and a URL for the dataset if available. Briefly discuss suitability of the dataset for your problem.**
We plan to use public-domain Sherlock Holmes texts from Project Gutenberg as our primary dataset, including:

- *The Adventures of Sherlock Holmes* (eBook 1661)
- *The Memoirs of Sherlock Holmes* (eBook 834)
- *The Hound of the Baskervilles* (eBook 2852)
- *The Return of Sherlock Holmes* (eBook 108)

These texts share a stable narrative world and recurring core characters, such as Sherlock Holmes and Dr. Watson, making them well-suited for studying cross-paragraph and cross-chapter consistency.

The planned data preprocessing pipeline includes:

- Removing irrelevant text such as Gutenberg headers and footers;
- Splitting the full text into paragraphs;
- Filtering out paragraphs that are too short or overly noisy;
- Constructing supervised learning samples in the form of `(first k paragraphs, next paragraph)`;
- Splitting into training, validation, and test sets to evaluate model generalization.

Dataset links:

- [Project Gutenberg](https://www.gutenberg.org/)
- [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661)
- [The Memoirs of Sherlock Holmes](https://www.gutenberg.org/ebooks/834)
- [The Hound of the Baskervilles](https://www.gutenberg.org/ebooks/2852)
- [The Return of Sherlock Holmes](https://www.gutenberg.org/ebooks/108)

**5. List key references (e.g. research papers) that your project will be based on?**


- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019.
- Fan, Lewis, and Dauphin, *Hierarchical Neural Story Generation*, ACL 2018.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020.


**Please indicate whether your project proposal is ready for review (Yes/No):**

## Feedback (to be provided by the course lecturer)
