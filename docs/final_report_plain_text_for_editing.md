We study next-paragraph generation in Sherlock Holmes stories using a controlled set of GPT-2 fine-tuning experiments. Given the first k paragraphs of a passage, the model has to produce the next one. We keep the backbone fixed at distilgpt2 and vary prompt format, context length, tuning strategy, and an auxiliary ranking loss. The five main experiments (E1–E5) are evaluated with target-conditioned perplexity, ROUGE-L, BERTScore, and an entity-overlap diagnostic, using three decoding seeds for generation metrics. The pattern is modest but consistent: structured formatting helps a little, four paragraphs of context help more, and E3 is the strongest run overall. LoRA is much weaker than full fine-tuning here, and the ranking objective does not improve on the best baseline. A retrieval variant is included in Appendix A, where it also trails E3.

Neural generators can sound fluent while losing track of what is happening. In a paragraph-continuation task, that weakness is hard to hide: the next paragraph has to follow the scene, the speaker, and the local plot, rather than merely produce plausible Victorian prose. This makes the task a useful test of coherence beyond sentence-level fluency.

The Sherlock Holmes stories are a convenient testbed because the narrative world is stable. Holmes, Watson, the London setting, and the style recur across the texts. When a continuation drifts away from that world, the error is usually visible rather than subtle.

We ask a deliberately narrow question: which model-side choices matter for paragraph continuation in this setting? The main comparison does not rely on retrieval or prompt engineering tricks. It focuses on context serialization, context length, full fine-tuning versus LoRA, and a simple auxiliary ranking loss.

The mainline follows four pairwise comparisons. E1 versus E2 asks whether structured paragraph markers help more than plain concatenation. E2 versus E3 asks whether a longer context window helps. E3 versus E4 compares full fine-tuning with LoRA under the same long-context structured setup. E3 versus E5 tests the auxiliary ranking objective against the strongest no-auxiliary baseline. We also run E5W as a robustness check for the auxiliary weight, but we do not treat it as a separate mainline result.

The results are not dramatic, but they are fairly easy to read. Structured input gives a small gain. Longer context gives the clearest gain, making E3 the strongest mainline model. LoRA performs worse than full fine-tuning, and the auxiliary ranking loss does not improve on that baseline. Retrieval is kept in the appendix because it does not change the conclusion.

This project is closest to work on autoregressive transformers for conditional text generation. These models fit continuation tasks naturally: they generate left to right, and pre-trained GPT-style models can be adapted with ordinary fine-tuning (Vaswani et al., 2017; Radford et al., 2019). The choice is also practical. A small GPT-2 variant is large enough to show differences between settings, but still small enough for repeated Colab runs.

The project also relates to long-form and story generation. Prior work has shown that local fluency does not guarantee global coherence (Fan et al., 2018; Holtzman et al., 2020). Paragraph continuation is a smaller version of the same problem: the output is short, but it still has to respect the preceding discourse.

Parameter-efficient adaptation is another relevant comparison. LoRA updates a small set of low-rank adapters rather than all model weights (Hu et al., 2022). Since this project uses limited compute and a small backbone, it is worth checking whether the cheaper tuning method is enough.

Finally, we look at training objectives beyond standard next-token prediction. If the language-model loss is too local to reward paragraph coherence, a ranking signal may help. E5 tests this idea in a simple form by asking the model to prefer the true next paragraph over sampled alternatives. Retrieval is included only as an appendix comparison.

We formulate the task as conditional next-paragraph generation. Each example contains the previous k paragraphs and the true next paragraph from the same book. Training still uses token-level supervision, but the behavior we care about is paragraph-level: the continuation should fit the scene, preserve the narrative state, and stay close to the source style.

All mainline experiments use distilgpt2 as the shared backbone. This keeps the comparison controlled and keeps the compute manageable. The model is small enough for repeated Colab runs, but not so small that every setting collapses to the same behavior. By holding the backbone fixed across E1–E5, we can attribute most differences to the design choice being tested.

The first design choice is prompt format. The plain setting concatenates context paragraphs directly. The structured setting adds explicit paragraph markers. The hope is that these markers make discourse boundaries easier for the model to read, especially when several paragraphs are provided. We test this with E1 versus E2.

The second choice is context length. The short setting uses k=2 paragraphs; the long setting uses k=4. More context should help with scene and character state, but it also leaves less room under the fixed token budget. E2 and E3 isolate this trade-off.

The third comparison is full fine-tuning versus LoRA. Full fine-tuning updates all model weights. LoRA trains low-rank adapters while leaving the base model mostly fixed. E3 and E4 test whether that efficiency trade-off works for this small literary dataset.

The final comparison changes the objective. The baseline uses the standard causal language-model loss on the target continuation. The auxiliary setting adds a ranking loss, so the model is also trained to score the true next paragraph above sampled negatives. E3 and E5 test whether that extra signal helps.

The implementation uses PyTorch, Hugging Face Transformers, and PEFT. The course permits either TensorFlow or PyTorch; we used PyTorch because TensorFlow GPU support was not reliable across all group members' local machines. To keep the setup consistent, every reported training, generation, and evaluation run was executed in Google Colab.

We use four public-domain Sherlock Holmes texts from Project Gutenberg: The Adventures of Sherlock Holmes, The Memoirs of Sherlock Holmes, The Hound of the Baskervilles, and The Return of Sherlock Holmes. The preprocessing pipeline removes boilerplate material, segments each book into paragraphs, filters out unsuitable paragraphs, and converts the cleaned corpus into supervised (context, target) continuation examples. Each example consists of the previous k paragraphs and the next paragraph from the same book.

The processed dataset contains 5285 training examples, 660 validation examples, and 663 test examples. All reported numerical results use this fixed split.

Table: Dataset split summary for the fixed paragraph-continuation split.

Split | Samples
Train | 5285
Validation | 660
Test | 663

The main experiment line is designed to isolate one design factor at a time, as shown in Table 2.

Table: Main experiment matrix.

ID | Goal | Setup
E1 | Baseline | k=2, plain, full, no aux
E2 | Structure | k=2, structured, full, no aux
E3 | Context length | k=4, structured, full, no aux
E4 | Fine-tuning | k=4, structured, LoRA, no aux
E5 | Objective | k=4, structured, full, ranking

An additional run, E5W, repeats E5 with a larger auxiliary weight. This run is not treated as part of the core mainline narrative. Instead, it functions as a robustness check for the auxiliary setting and supports the final E5 selection using validation-side model comparison.

Across the mainline experiments, we keep the core hyperparameters fixed: model = distilgpt2, epochs = 3, batch size = 2, gradient accumulation steps = 4, learning rate = 5 × 10^-5, warmup steps = 20, maximum sequence length = 512, and training seed = 42. Holding these settings constant helps keep the pairwise comparisons interpretable. Model selection is based on validation performance rather than test-set outcomes. For the E5 versus E5W comparison, we use validation main loss rather than test metrics to decide which auxiliary-weight variant should represent the mainline auxiliary configuration.

Evaluation uses automatic metrics: target-conditioned perplexity, ROUGE-L (Lin, 2004), and BERTScore F1 (Zhang et al., 2020). We additionally report entity overlap as a diagnostic metric. It is useful for rough consistency inspection, but it is not treated as the sole basis for major claims because it is a lightweight heuristic rather than a full semantic evaluation.

Generation-based metrics are evaluated with three fixed random seeds (13, 42, and 2026). We report the mean for all metrics and the standard deviation for generation-based metrics. Perplexity is deterministic for a fixed checkpoint and is therefore reported as a mean or single-value quantity across seeds. This evaluation protocol is intended to distinguish stable model differences from small amounts of sampling variance.

The submitted evaluation is limited to automatic metrics; human judgments would be a useful extension but are outside the scope of this project. Retrieval is reported only in Appendix A, and the main claims of the paper rest on E1–E5.

Table 3 summarizes the mainline results. Test metrics are reported on the full held-out test set of 663 samples.

Table: Mainline results on the held-out test set. Perplexity is deterministic for a fixed checkpoint, while ROUGE-L, BERTScore F1, and entity overlap are reported as mean ± standard deviation over three decoding seeds.

Experiment | Validation Main Loss | Validation PPL | Test PPL | ROUGE-L | BERTScore F1 | Entity Overlap
E1 | 3.1008 | 25.0981 | 27.8239 | 0.1105 ± 0.0009 | 0.8339 ± 0.0001 | 0.1432 ± 0.0078
E2 | 3.1001 | 25.1064 | 27.8098 | 0.1105 ± 0.0006 | 0.8342 ± 0.0002 | 0.1401 ± 0.0055
E3 | 3.0867 | 25.0427 | 27.6488 | 0.1105 ± 0.0009 | 0.8342 ± 0.0003 | 0.1233 ± 0.0021
E4 | 3.2964 | 30.3667 | 32.6113 | 0.1102 ± 0.0009 | 0.8329 ± 0.0004 | 0.1288 ± 0.0051
E5 | 3.1010 | 25.6218 | 28.3570 | 0.1106 ± 0.0008 | 0.8343 ± 0.0001 | 0.1230 ± 0.0007

For robustness checking, E5W achieves a validation main loss of 3.1059, validation perplexity of 25.8079, test perplexity of 28.5527, ROUGE-L of 0.1118 ± 0.0009, BERTScore F1 of 0.8345 ± 0.0003, and entity overlap of 0.1257 ± 0.0033. Because it does not improve the validation-side selection criterion relative to E5, it is not used as the main auxiliary result in Table 3.

E2 is close to E1, with a small validation-side gain. ROUGE-L is almost unchanged at 0.1105, BERTScore moves from 0.8339 to 0.8342, and validation main loss drops from 3.1008 to 3.1001. That is enough to say the structured format helps a little. It is not enough to claim a large change in quality.

E3 is the best model in the mainline. It has the lowest validation main loss (3.0867), the lowest validation perplexity (25.0427), and the lowest test perplexity (27.6488). ROUGE-L and BERTScore stay close to E2, so the gain shows up more clearly in the likelihood-based metrics than in surface-overlap metrics. Even so, the direction is consistent enough to support the longer context setting.

This is the cleanest result in the paper. Under the same long structured setup, LoRA falls well behind full fine-tuning. Validation main loss rises from 3.0867 to 3.2964, test perplexity jumps from 27.6488 to 32.6113, and BERTScore drops as well. In this task, with this backbone, full tuning is the better choice.

E5 does not beat E3. Its validation main loss is higher (3.1010 vs. 3.0867), its validation perplexity is higher (25.6218 vs. 25.0427), and its test perplexity is higher (28.3570 vs. 27.6488). ROUGE-L and BERTScore barely move, while entity overlap is about the same. The ranking loss does not help in its current form.

We also compare E5 with E5W, which uses a larger auxiliary weight. E5W is slightly worse on validation main loss (3.1059 vs. 3.1010). The difference is small, so E5W still works as a robustness check. It does not change the conclusion.

Space constraints keep full generated passages out of the main text. Manual inspection of sampled continuations followed the same broad pattern as the metrics: E3 usually stayed closest to the preceding scene, E4 was more prone to generic continuations, and E5 did not show a clear coherence gain over E3. We therefore treat qualitative inspection as supporting evidence rather than as a separate headline result.

This study has several limits. First, the evaluation is mostly automatic; we do not yet have human judgments of narrative quality. Second, the experiments use one literary domain and one small backbone, so the conclusions should not be read as general claims about story generation. A final caveat is more specific: the best long-context setting still truncates a noticeable share of examples under the 512-token budget. Token-budget inspection gives truncation rates of about 24.7% on the training split and 17.3% on the evaluation split for the k=4 structured setting. This does not overturn the E3 result, but it makes the longer-context gain less clean than it first appears.

We ran a controlled set of paragraph-continuation experiments with a fixed distilgpt2 backbone. The main result is simple: structured formatting helps slightly, longer context helps more, and E3 is the strongest mainline system. LoRA is clearly worse than full fine-tuning in this setting. The auxiliary ranking loss also fails to improve on the best baseline.

For this task and model scale, the most useful change is giving the model a better view of the preceding text. Switching to parameter-efficient adaptation, or adding the simple ranking loss used here, does not pay off.

The next steps are straightforward. Human evaluation would say more about narrative quality than ROUGE or BERTScore can. Larger backbones may also change the trade-offs, especially for LoRA. The appendix retrieval result suggests one narrower lesson as well: retrieval may still help, but not with the simple setup tested here.

All members contributed to project discussion, experiment planning, and final review. The main responsibility split was as follows:

• Zhangrui Dong: dataset preparation, preprocessing, split construction, and part of the model training work.
• Zhengyao Zhao: model development, training pipeline implementation, experiment configuration, and Colab training runs.
• Zhengrong Gao: report writing, methodology and result discussion, LaTeX polishing, and final submission packaging.
• Haotian Wang: automatic evaluation, metric computation, result aggregation, and consistency checks for the reported numbers.

Retrieval is treated as an appendix comparison rather than a main contribution. We ran it separately and compared it with the strongest completed mainline model, E3. The retrieval variant was weaker. The appendix system uses a lightweight TF-IDF retriever over earlier passages, following a classical sparse retrieval setup (Salton and Buckley, 1988).

Table: Appendix retrieval comparison against the strongest mainline model.

Model | Val Loss | Val PPL | Test PPL | ROUGE | BERT
E3 | 3.0867 | 25.0427 | 27.6488 | 0.1105 ± 0.0009 | 0.8342 ± 0.0003
Retrieval | 3.1752 | 27.3757 | 29.7175 | 0.1105 | 0.8325

The retrieval run is weaker than E3 on both likelihood-based and overlap-based metrics. We therefore keep it as an appendix result and leave the mainline interpretation unchanged.

The submitted code package contains the implementation details needed to reproduce the experiments:

• dataset download, preprocessing, and split-building scripts;
• training configurations for E1–E5, E5W, and retrieval;
• generation and automatic-evaluation scripts;
• saved metric summaries from the Colab runs;
• the one-click Colab notebook used for the official run workflow.

References

• Fan, Angela; Lewis, Mike; Dauphin, Yann (2018). Hierarchical Neural Story Generation. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics.
• Holtzman, Ari; Buys, Jan; Du, Li; Forbes, Maxwell; Choi, Yejin (2020). The Curious Case of Neural Text Degeneration. International Conference on Learning Representations.
• Hu, Edward J.; Shen, Yelong; Wallis, Phillip; Allen-Zhu, Zeyuan; Li, Yuanzhi; Wang, Shean; Wang, Lu; Chen, Weizhu (2022). LoRA: Low-Rank Adaptation of Large Language Models. International Conference on Learning Representations.
• Lin, Chin-Yew (2004). ROUGE: A Package for Automatic Evaluation of Summaries. Text Summarization Branches Out.
• Radford, Alec; Wu, Jeffrey; Child, Rewon; Luan, David; Amodei, Dario; Sutskever, Ilya (2019). Language Models are Unsupervised Multitask Learners. OpenAI Technical Report.
• Salton, Gerard; Buckley, Christopher (1988). Term-Weighting Approaches in Automatic Text Retrieval. Information Processing & Management.
• Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N.; Kaiser, Lukasz; Polosukhin, Illia (2017). Attention Is All You Need. Advances in Neural Information Processing Systems.
• Zhang, Tianyi; Kishore, Varsha; Wu, Felix; Weinberger, Kilian Q.; Artzi, Yoav (2020). BERTScore: Evaluating Text Generation with BERT. International Conference on Learning Representations.
