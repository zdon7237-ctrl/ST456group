[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hZWYFfQo)
# ST456 course project

Deadline: Thursday **30/04/2026, 23:59**

---

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

---

## Project task

You are free to propose a task for your project, subject to approval by the course lecturer.
Your proposal may be immediately approved or you may be asked to revise and resubmit it.

Your project should demonstrate a strong understanding of the topics covered in the course,
including neural network architecture design, implementation, training,, evaluation and interpretation.

Your implementation must be in TensorFlow or PyTorch, using a dataset suitable for your problem.

## Group work

Your project is a group project. Each group **MUST** consist of **FOUR** students, except in cases where the total number of students is not a multiple of four.  

You are free to form groups as you wish, and we will provide support to facilitate group formation.  

Each group member is expected to contribute fairly and meaningfully to the technical aspects of the project.  

## GitHub

Each group will be assigned a private repository within the ST456 organisation.

Your entire **source code and report** must be included in this repository. Depending on the **size of your dataset(s)**, you can provide only the links to the external data sources.

## Report

### Format

Your report should follows the [ICML latex style files](https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip).

The report must not exceed **eight** pages, excluding references, with unlimited space for references.
Your may include an appendix for additional details.

### Structure

Your report should follow the standard structure of a research paper. Below is a recommended layout:

* **Abstract:** A concise summary of your project, covering your chosen task, proposed solution, and key results.

* **Introduction:** A clear description of your problem, its importance, your solution, and a summary of your main findings.  

* **Related work:** A review of relevant research and how your work compares to existing studies.

* **The architecture:** A detailed explanation of the neural network architectures and methodologies used.

* **Training methods:** A description of the training process, including optimisers, hyperparameter settings, and techniques to mitigate overfitting.  

* **Numerical results:**
   * Clearly define the goals of your numerical evaluation.
   * Describe the datasets used.
   * Specify evaluation metrics (e.g., loss functions, accuracy).
   * Summarise any baseline methods used for comparison.
   * Present and discuss numerical results.
* **Conclusion:** A summary of your findings, along with potential future research directions.

* **Bibliography** A list of all references cited in your report. Proper citations are required for any concepts or results taken from existing research.

* **Statement of individual contributions:** A summary of the technical contributions of each group member.


### Writing tips

Here are some useful resources on writing research papers:

* Jean-Yves Le Boudec, [How to Write a Paper](https://leboudec.github.io/leboudec/resources/paper.html)
* John N. Tsitsiklis, [A Few Tips on Writing Papers with Mathematical Content](http://web.mit.edu/jnt/www/Papers/R-20-write-v5.pdf), last update 2020
* Jon Turner, [How to Write a Great Research Paper](https://www.arl.wustl.edu/~pcrowley/cse/591/writingResearchPapers.pdf)

### Marking

Marking criteria can be found here [here](https://github.com/lse-st456/lectures2026/blob/main/assessment/project/ST456-project-marking.pdf).

### Important dates

* Week 8
   * A Google form for for group formation released.
* Week 9   
   * Q\&A session at the end of Lecture 8.
* Week 10
   * Groups finalised.
   * Each group submits a project proposal in the README.md of their project GitHub repository by Friday end of the day.
* Week 11
   * All project proposals discussed and approved.  
