# Retrieval-Augmented Novel Continuation Design

Date: 2026-03-25
Status: Draft

## Project Title

Retrieval-Augmented Language Models for Context-Coherent English Novel Continuation

## Design 1: Task Definition and Scope

### Task

Given a passage of prior text from the same narrative world, generate the next paragraph of the story.

### Input

- The current continuation context, such as the previous 2-4 paragraphs or a fixed token window.
- Optionally, retrieved passages from earlier text in the same narrative world.

### Output

- One next paragraph of story continuation.

### Main Research Question

Does retrieval augmentation improve contextual coherence in English novel continuation?

### Secondary Research Question

If contextual coherence improves, does character consistency also improve?

### Scope

- Focus on continuation within the same narrative world rather than open-domain story generation.
- Use English literary text.
- Build a paragraph-level continuation task rather than chapter-level generation.

## Design 2: Dataset Choice and Retrieval-Augmented Model Structure

### Dataset Strategy

The dataset should come from the same narrative world so that recurring settings, entities, and character behavior can be learned and reused during continuation.

Recommended dataset strategy:

- Primary choice: a public-domain book series or story collection with recurring characters and stable world-building.
- Best practical option: Sherlock Holmes texts from Project Gutenberg.
- Backup option: a public-domain fantasy series such as Oz if the group prefers stronger world continuity across books.

Why Sherlock Holmes is a strong starting point:

- It is easy to collect legally.
- It contains multiple long and short works with recurring characters such as Holmes and Watson.
- It provides enough text volume for Colab-scale fine-tuning.
- It supports evaluation of both contextual coherence and character consistency.

### Dataset Construction

Convert the corpus into paragraph-level supervised samples:

- Context: previous 2-4 paragraphs or a fixed-length token window.
- Target: the next paragraph.
- Retrieval pool: earlier paragraphs from the same book, or from earlier texts in the same narrative world.

Data split rules:

- Train, validation, and test should be split by contiguous text blocks to avoid leakage.
- Retrieval during validation and test must only access text that occurs before the target paragraph.
- No future paragraphs should ever be available to the retriever.

### Model Architecture

Recommended base generator:

- `distilgpt2` for a lighter Colab baseline.
- `gpt2` if GPU memory and runtime allow.

Recommended retriever:

- First version: TF-IDF or BM25 over paragraph chunks.
- Optional extension: semantic retrieval with sentence embeddings if time permits.

Proposed retrieval-augmented pipeline:

1. Take the current context window.
2. Use it as a query to retrieve the top-k most relevant earlier paragraphs.
3. Concatenate current context and retrieved evidence into one structured prompt.
4. Fine-tune the generator to predict the next paragraph.

Example input format:

```text
[CONTEXT]
paragraph t-3
paragraph t-2
paragraph t-1

[RETRIEVED]
relevant earlier paragraph 1
relevant earlier paragraph 2

[TASK]
Generate the next paragraph.
```

### Experimental Variants

At minimum, compare:

- Baseline A: short-context generator without retrieval.
- Baseline B: longer-context generator without retrieval.
- Proposed model: context plus retrieved passages.

Useful ablations:

- Different numbers of retrieved passages.
- Lexical retrieval versus semantic retrieval.
- Retrieval from same book only versus same narrative world corpus.

### Why This Structure Fits the Project

- It directly targets long-range contextual coherence.
- It is feasible to run in Colab.
- It gives a clean baseline-versus-improved-model comparison.
- It naturally supports both quantitative and qualitative evaluation.

## Pending Next Design Sections

- Evaluation metrics and human evaluation protocol.
- Training pipeline and Colab workflow.
- Risks, limitations, and fallback plan.

## Design 3: Evaluation Metrics and Human Evaluation Protocol

### Evaluation Goals

The evaluation should test whether retrieval augmentation improves:

- Contextual coherence with the immediately preceding text.
- Consistency with earlier story events and facts.
- Character consistency as a secondary outcome.
- General text quality, such as fluency and readability.

### Automatic Metrics

Recommended automatic metrics:

- Perplexity: measures how well the model fits the continuation distribution.
- BERTScore: the preferred semantic similarity metric for comparing the generated paragraph with the gold next paragraph.
- ROUGE-L: a lightweight lexical-structural overlap metric that is more informative than unigram overlap alone.
- BLEU: optional auxiliary lexical overlap baseline.
- Entity consistency score: a task-specific diagnostic based on overlap or agreement of important character names, locations, and key entities between context and generation.

These metrics should not be treated as sufficient on their own because creative writing can have many valid continuations.

Recommended primary automatic metrics:

- Perplexity
- BERTScore
- ROUGE-L

Recommended optional diagnostic metric:

- Entity consistency or character/entity overlap

### Retrieval-Specific Evaluation

To support the main research question, include diagnostics that reflect retrieval usefulness:

- Retrieval relevance: similarity between the query context and retrieved passages.
- Citation-style overlap: whether generated entities, locations, or events are grounded in retrieved material.
- Ablation by top-k: test whether coherence changes as more retrieved passages are used.

These diagnostics help explain why the proposed model performs better or worse than the baselines.

### Human Evaluation

Human evaluation is essential because story continuation quality is not fully captured by automatic metrics.

Recommended rubric:

- Contextual coherence: does the generated paragraph follow naturally from the given preceding text?
- Narrative consistency: does it avoid contradicting earlier events or facts?
- Character consistency: do speaking style, personality, and behavior match the established characterisation?
- Fluency: is the paragraph grammatically sound and easy to read?

Suggested scoring scheme:

- Use a 1-5 Likert scale for each criterion.
- Evaluate anonymous outputs from all systems in random order.
- Ask each rater to score the same fixed sample of test cases.

### Evaluation Setup

Recommended comparison set:

- Baseline A: short-context generator without retrieval.
- Baseline B: longer-context generator without retrieval.
- Proposed model: retrieval-augmented generator.

Suggested test protocol:

- Sample 50-100 held-out continuation examples.
- Generate one continuation from each model for each example.
- Use at least 2 human raters if possible.
- Report average scores and variance for each criterion.

### Primary and Secondary Outcomes

The primary outcome should be contextual coherence because it is the main research target.

Secondary outcomes:

- Narrative consistency.
- Character consistency.
- Fluency.

This keeps the paper focused while still showing broader benefits of the proposed method.

### Why This Evaluation Design Fits the Project

- It aligns the evaluation directly with the research question.
- It combines standard NLP metrics with human judgment.
- It gives enough evidence for both a course project and a possible early-stage paper draft.

## Design 4: Training Pipeline and Colab Workflow

### Recommended Stack

Use the following stack for a practical Colab-based workflow:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn for TF-IDF retrieval
- optionally `rank_bm25` for BM25 retrieval

This stack is standard, well documented, and suitable for both coursework and research prototyping.

### Data Preparation Pipeline

The preprocessing workflow should be:

1. Collect public-domain English texts from the chosen narrative world.
2. Clean front matter, end matter, chapter headings, and obvious OCR artefacts.
3. Split the text into paragraphs.
4. Build supervised examples of the form:
   - context paragraphs
   - target next paragraph
   - retrieval pool restricted to previous paragraphs only
5. Save train, validation, and test splits in a reusable format such as JSONL or CSV.

Each sample should also keep metadata such as:

- source book
- paragraph index
- chapter index if available

This metadata will help with retrieval control, analysis, and error inspection.

### Retrieval Pipeline

For the first project version, retrieval should stay simple and stable:

1. Represent all historical paragraphs in the retrieval pool using TF-IDF or BM25.
2. Use the current context as the query.
3. Retrieve the top-k earlier paragraphs.
4. Filter out trivially adjacent paragraphs if needed, so retrieval is not just copying the immediate local context.

This gives a strong and interpretable retrieval baseline without heavy compute.

### Generator Fine-Tuning Pipeline

Recommended first training setup:

- Tokenizer: GPT-2 tokenizer
- Generator: `distilgpt2`
- Max input length: set based on Colab memory, for example 512 or 768 tokens
- Training objective: causal language modelling on the structured input-plus-target sequence

Training format:

- Prefix the model input with explicit section markers such as `[CONTEXT]`, `[RETRIEVED]`, and `[TARGET]`.
- Train the model to generate the target next paragraph after the structured prompt.

Suggested training sequence:

1. Train Baseline A with short local context only.
2. Train Baseline B with a longer local context window only.
3. Train the retrieval-augmented model with the same generator backbone.
4. Compare generation quality and metric scores on the held-out test set.

### Colab Execution Plan

The project should be split into separate Colab notebooks or sections:

- Notebook 1: data collection and preprocessing
- Notebook 2: retrieval index building and inspection
- Notebook 3: generator fine-tuning
- Notebook 4: evaluation and result analysis

This separation is useful because Colab sessions are temporary and can disconnect unexpectedly.

Practical Colab safeguards:

- Save processed datasets to Google Drive.
- Save model checkpoints to Google Drive after each epoch.
- Log hyperparameters and experiment names clearly.
- Keep batch sizes small and use gradient accumulation if memory is limited.
- Prefer shorter experiments first to validate the full pipeline before long runs.

### Baseline Training Budget

A realistic initial budget for Colab:

- Start with `distilgpt2`
- Use a limited subset of the corpus for debugging
- Train for a small number of epochs first
- Expand only after the pipeline produces valid outputs

This reduces the risk of spending hours on runs that fail because of formatting, tokenization, or retrieval bugs.

### Error Analysis Workflow

After training, inspect outputs manually and group failure cases:

- loss of plot continuity
- wrong entity or location references
- character voice drift
- generic or repetitive text
- over-copying from retrieved passages

This analysis will help strengthen both the final report and any later paper draft.

### Why This Pipeline Fits the Project

- It is realistic for a student team using Colab.
- It supports clean experimentation and ablation studies.
- It minimises engineering complexity while keeping the method research-oriented.

## Design 5: Risks, Limitations, and Fallback Plan

### Main Risks

The project is feasible, but several practical risks should be anticipated early.

#### Risk 1: Insufficient Data Volume

If the chosen narrative world does not contain enough text, the model may overfit and generate repetitive or low-quality continuations.

Mitigation:

- Prefer a corpus with multiple books or stories in the same narrative world.
- Start with Sherlock Holmes or another public-domain series with recurring characters.
- If needed, increase the corpus to include more works from the same series rather than switching to unrelated texts.

#### Risk 2: Colab Resource Limits

Training may be interrupted by session timeouts, GPU unavailability, or memory limits.

Mitigation:

- Use `distilgpt2` as the default generator.
- Save checkpoints and processed data to Google Drive frequently.
- Keep the first experiments small and confirm the pipeline before longer runs.
- Use small batch sizes and gradient accumulation when necessary.

#### Risk 3: Weak Gains from Retrieval

Retrieval augmentation may not clearly outperform a strong long-context baseline.

Mitigation:

- Compare against both a short-context and a longer-context baseline.
- Evaluate several values of top-k.
- Inspect retrieval quality manually to confirm the retriever is bringing useful evidence.
- Report negative or mixed findings honestly if gains are small.

#### Risk 4: Evaluation Difficulty

Automatic metrics may fail to reflect narrative quality, and human evaluation can be noisy.

Mitigation:

- Use automatic and human evaluation together.
- Keep the human scoring rubric simple and consistent.
- Use anonymised outputs and random ordering during rating.
- Include qualitative case studies in the report.

#### Risk 5: Copyright and Data Access Problems

Modern commercial novels may be hard to use legally or hard to distribute within the repository.

Mitigation:

- Use public-domain texts.
- Store scripts for data collection and preprocessing rather than uploading large raw datasets when needed.
- Document data sources clearly in the report and repository.

### Research Limitations

Even if the method works well, the project will still have some natural limitations:

- It studies continuation within one narrative world rather than open-domain story generation.
- It uses paragraph-level continuation, not full chapter planning.
- It may improve coherence without solving deeper story planning.
- Human judgments of story quality remain partly subjective.

These limitations are acceptable for a course project and can be framed as future work in a paper draft.

### Fallback Plan

If the full retrieval-augmented setup becomes too heavy or unstable, use the following fallback order:

1. Complete a strong baseline comparison using short versus long context only.
2. Add a simple TF-IDF retrieval module rather than a more complex semantic retriever.
3. Restrict the corpus to one strong narrative world with clean data.
4. Reduce the project goal to demonstrating improved contextual coherence on a smaller but carefully evaluated dataset.

This fallback still gives a valid and well-structured ST456 project.

### Minimum Viable Project

The minimum successful version of the project should include:

- One clean narrative-world dataset.
- One baseline generator without retrieval.
- One retrieval-augmented generator.
- Automatic evaluation.
- A small but structured human evaluation.

This should be treated as the non-negotiable core deliverable.

## Current Design Status

The project concept is now defined at a level suitable for:

- proposal drafting
- implementation planning
- team role allocation
- early literature review and experiment setup

Current decisions:

- Preferred narrative-world dataset: Sherlock Holmes public-domain texts from Project Gutenberg.
- Preferred generator baseline: `distilgpt2`, with `gpt2` as an optional extension.
- Preferred retrieval method for the first version: TF-IDF or BM25.
- Proposal draft has been prepared in `proposal.md`.
- Implementation plan has been prepared in `docs/plans/2026-03-25-novel-continuation-implementation.md`.
