Next-Paragraph Generation in Sherlock Holmes Texts.
We conduct a series of controlled GPT-2 fine-tuning experiments with the goal of generating the next paragraph in Sherlock Holmes passages based on their first k paragraphs. Our backbone remains distilgpt2, while we experiment with prompt format, context length, tuning regime, and an auxiliary ranking objective. The results of our five major experiments (E1-E5) are evaluated in terms of perplexity conditioned on the target, ROUGE-L, BERTScore, and an entity-overlap diagnostic. The metrics are computed with decoding done via three seeds.

The observed trends are rather mild yet clear. Formatted prompt tends to be beneficial to some degree, longer context is more helpful than the shorter one, and E3 shows itself to be the best of all runs. Full fine-tuning outperforms LoRA in a significant way. Meanwhile, the ranking loss provides no advantage over other approaches.

Fluent but coherent? Coherence is an essential property of neural generators, which tend to sound quite fluent despite being unable to maintain the topic, character identity, etc. This aspect becomes apparent in the course of a paragraph continuation: the generated paragraph should align with the scene, the speaker, and the plotline, rather than just generate fluent Victorian prose.

This task is a convenient setup for assessing coherence because the narrative universe is relatively constant in the Sherlock Holmes stories: Holmes himself, Watson, the London environment, and the style are recurring elements. A continuation deviating from this universe will likely make sense.

Which model-side choices make a difference in our problem? The key experiment does not hinge upon any external data sources or prompt tricks; we are interested in evaluating effects of context formatting, context length, full fine-tuning versus LoRA, and a ranking auxiliary objective.

Mainline Results
The mainline proceeds through four experiments. Experiment 1 (E1) vs Experiment 2 (E2) compares structured paragraph indicators versus unstructured concatenation of paragraphs. Experiment 2 (E2) vs Experiment 3 (E3) measures the effect of a larger context window size. Experiment 3 (E3) vs Experiment 4 (E4) compares full finetuning with LoRA within the same structured, long-context setting. Finally, Experiment 3 (E3) vs Experiment 5 (E5) compares the auxiliary ranking objective to the strongest no-auxiliary baseline. Experiment 5 (E5W) with a different auxiliary weight is considered in an appendix for the sake of robustness analysis, but we do not report the experiment as a distinct result from the mainline.

These results are modest in magnitude but fairly clear-cut. The use of structured input yields a small improvement. The increase in context window size delivers a significant performance gain, which makes E3 the leading mainline configuration. LoRA is inferior to full finetuning, and the additional auxiliary ranking objective does not improve the result further. As always, retrieval was placed in the appendix, since it does not change the findings.

This research has several precedents in the literature on conditional autoregressive text generation. First of all, autoregressive transformers fit the continuation task well, since they generate left-to-right text, and a pre-trained GPT-style model can be adjusted via fine-tuning (Vaswani et al., 2017; Radford et al., 2019). In addition, the approach was chosen for practical reasons – a small GPT-2 architecture is sufficiently large to yield distinctions between conditions, but still small enough to perform repeated Colab runs.

Furthermore, this research can be viewed as connected to prior work on long-form and story generation. Existing papers have demonstrated that a fluent text might not necessarily be coherent (Fan et al., 2018; Holtzman et al., 2020). This phenomenon applies to our task as well: the generated text is shorter in scope, but it still has to match the preceding discourse.

Another area of relevance concerns parameter-efficient model adaptation. Instead of adjusting all parameters, the LoRA technique updates a few adapters of lower rank (Hu et al., 2022). In this project, a small backbone and constrained computation limit allow us to compare this efficient tuning procedure to standard finetuning.

Finally, the topic of this paper touches upon alternative training methods for autoregressive transformers beyond next-token prediction. Indeed, in our case, the language-model loss might be too localized to ensure paragraph coherence; therefore, an additional ranking signal should be provided to guide model training.

In this work, we consider conditional next-paragraph generation. For each training example, there are k prior paragraphs from the book, followed by the true next paragraph. The training signals remain at the token level, but the behavior of interest is at the paragraph level: the continuation must be appropriate to the scene, maintain the story state, and match the original style.

For all the experimental comparisons below, we use distilgpt2 as the shared backbone architecture. On the one hand, this choice ensures fair comparison; on the other, the small size allows us to run multiple experiments in Google Colab. Thus, we have a large number of settings, but not so many that different architectures are required. By using the same backbone for E1-E5, we control for any variance due to model differences, allowing us to more reliably attribute variance to the experimental conditions.

The first design variable is the input format. The plain setting simply concatenates the paragraphs from the input context. In the structured version, additional formatting cues (like paragraph delimiters) are added to the input text. The hypothesis is that the additional cues will be helpful for the model to identify the structure even with multiple paragraphs. This is tested in E1 vs. E2.

The next design choice is context length. Short setting (k=2) gives fewer paragraphs as input, while long setting (k=4) provides more. With more context information, the model should better capture the story state, but there would also be less room within the fixed-length input sequence. Thus, E2 vs. E3 explores this trade-off.

The next experiment compares full fine-tuning with LoRA fine-tuning. While the former updates all model parameters during training, in the latter only the parameters in the additional adapters are tuned. E3 vs. E4 tests the feasibility of this optimization for the specific case of a relatively small literary corpus.

Finally, in the last comparison, we change the objective. The standard objective is causal language modeling applied to the continuation paragraph. The auxiliary setting introduces an auxiliary ranking objective that encourages the model to rate the true paragraph as higher than sampled negative samples. E3 vs. E5 explores the effectiveness of the ranking loss signal.

The implementation relies on PyTorch, Hugging Face Transformers, and PEFT. The class allows us to use either TensorFlow or PyTorch; we use PyTorch due to TensorFlow's unreliable GPU support on all group members' local machines. To maintain consistency, all training, generation, and evaluation runs reported below were performed in Google Colab.

We use four free Sherlock Holmes novels available on Project Gutenberg: The Adventures of Sherlock Holmes, The Memoirs of Sherlock Holmes, The Hound of the Baskervilles, and The Return of Sherlock Holmes. The preprocessing pipeline strips all boilerplate content, breaks each novel into paragraphs, removes unfit paragraphs, and encodes the resulting data in supervised continuation examples. Each example is a pair containing k preceding paragraphs and the following paragraph from the same novel.

The resulting dataset consists of 5285 training samples, 660 validation samples, and 663 test samples. All numerical results reported below employ this fixed split.

Dataset split overview for the fixed paragraph continuation split.

Split | Samples
Train | 5285
Validation | 660
Test | 663

The main experiment line is planned to single out one design factor at a time as illustrated by Table 2 below.

Main experiment line.

Experiment ID | Goal | Experiment setup
E1 | Baseline | k=2, plain, full, no aux
E2 | Structure | k=2, structured, full, no aux
E3 | Context length | k=2, structured, full, no aux
E4 | Fine-tuning | k=2, structured, LoRA, no aux
E5 | Objective | k=2, structured, full, ranking

In addition to the above, we perform one more experiment E5W that repeats E5 but with a larger auxiliary weight parameter. We do not consider this run as part of the mainline experiment narrative. Instead, it serves as a robustness study of the auxiliary case and provides a basis for selecting the E5 variant based on validation model comparison.

Throughout the main experiment line, we maintain fixed core hyperparameters:

Model: distilgpt2
Epochs: 3
Batch size: 2
Gradient accumulation steps: 4
Learning rate: 5 × 10^-5
Warmup steps: 20
Max length: 512
Training seed: 42

Using a fixed core configuration simplifies pairwise comparisons. Our model choice is made based on the validation performance and not the test set scores. In particular, when comparing E5 and E5W, we rely on validation main loss to select the auxiliary weight variant.

Evaluations use automatic metrics such as target-conditioned perplexity, ROUGE-L (Lin, 2004), and BERTScore F1 (Zhang et al., 2020). We also evaluate using entity overlap as an additional diagnostic metric. While useful for a cursory sanity check on the model's consistency, this metric is not used to make major conclusions since it represents a light heuristic compared to semantic evaluation.

For generation-based metrics, evaluations are performed using a total of three fixed random seeds (13, 42, and 2026). For the former class, we report both the mean and standard deviation across all three seeds. Perplexity being a deterministic measure for a fixed checkpoint, we report its values either in the mean or a single figure form. Such an evaluation procedure aims to discern model differences that are consistent across runs and exclude sampling variance noise.

Human evaluation is not considered in this submission and represents a valuable addition, however outside of the scope of our project. Retrieval results are reported solely in Appendix A; E1-E5 constitute the paper's main conclusions.

Below we summarize the results for the main experimental setting. Metrics are computed on a held-out test set of size 663.

Table 3: Results for the held-out test set evaluation. Perplexity is deterministic for a fixed checkpoint, ROUGE-L, BERTScore F1, and entity overlap are presented as mean ± standard deviation across three different seeds.

Experiment | Validation Main Loss | Validation PPL | Test PPL | ROUGE-L | BERTScore F1 | Entity Overlap
E1 | 3.1008 | 25.0981 | 27.8239 | 0.1105 ± 0.0009 | 0.8339 ± 0.0001 | 0.1432 ± 0.0078
E2 | 3.1001 | 25.1064 | 27.8098 | 0.1105 ± 0.0006 | 0.8342 ± 0.0002 | 0.1401 ± 0.0055
E3 | 3.0867 | 25.0427 | 27.6488 | 0.1105 ± 0.0009 | 0.8342 ± 0.0003 | 0.1233 ± 0.0021
E4 | 3.2964 | 30.3667 | 32.6113 | 0.1102 ± 0.0009 | 0.8329 ± 0.0004 | 0.1288 ± 0.0051
E5 | 3.1010 | 25.6218 | 28.3570 | 0.1106 ± 0.0008 | 0.8343 ± 0.0001 | 0.1230 ± 0.0007

For robustness checks, E5W yields a validation main loss of 3.1059, validation perplexity of 25.8079, test perplexity of 28.5527, ROUGE-L of 0.1118 ± 0.0009, BERTScore F1 of 0.8345 ± 0.0003, and entity overlap of 0.1257 ± 0.0033. Since it does not make an improvement in the validation-side selection criterion compared to E5, it is not used as the main auxiliary result in Table 3.

The performance of E2 is similar to that of E1. There is only a slight improvement on the validation side where ROUGE-L does not change much, moving from 0.1105 to 0.1118, and where BERTScore shifts from 0.8339 to 0.8342. The improvement in the likelihood criteria, however, is obvious: validation main loss changes from 3.1008 to 3.1001. While there is room to claim improvement, it cannot be considered significant enough.

The best-performing model within the mainline of results is E3, yielding the smallest validation main loss (3.0867), validation perplexity (25.0427), and test perplexity (27.6488). The other metrics are very close to E2's, hence the gain is more visible when measuring likelihood metrics as opposed to surface-overlap measures. Still, it is obvious enough to warrant the setting of longer context.

This is perhaps the most solid finding of the entire paper. Under the same structure with a longer input text, LoRA lags well behind FT in all metrics. The validation main loss goes up from 3.0867 to 3.2964, test perplexity increases from 27.6488 to 32.6113, and BERTScore drops accordingly. Given the task and backbone model used here, FT proves more efficient.

E5 does not perform better than E3: its validation main loss is higher (3.1010 vs. 3.0867), its validation perplexity is higher (25.6218 vs. 25.0427), and its test perplexity is higher (28.3570 vs. 27.6488). There is no significant difference between ROUGE-L and BERTScore scores, and entity overlap remains roughly the same. E5 is not a better setting because its ranking loss is not helpful.

Finally, we compare E5 against the larger auxiliary weight version, E5W. The latter performs slightly worse on validation main loss with respect to E5 (3.1059 vs. 3.1010), hence it still holds as a robustness check option. It neither negates nor modifies the previous findings.

Due to the space constraint, complete continuations are not included in the main body. However, our manual analysis of samples continued following the same trend: E3 was the most coherent model, whereas E4 showed more tendencies toward generic continuations; E5 demonstrated the highest degree of incoherence relative to other models. Therefore, qualitative inspection can be considered additional confirmation, not a headline finding itself.

There are several limitations of this study. First of all, it is mostly based on automated assessment. Human judgment on narrative coherence is unavailable at the moment. Secondly, this paper relies on the literary domain and a limited-size backbone model, thus no generalizing claims can be made. Finally, this particular limitation is specific to this paper: the best setting among longer contexts still truncates quite a few examples because of token budget constraints. According to token-budget inspection results, the k=4 structured settings truncate about 24.7% and 17.3% of examples on training and evaluation splits respectively.

We conducted a series of paragraph continuation experiments under the same experimental settings with a fixed backbone of distilgpt2. As our key finding, structural formatting leads to slight gains, context window expansion leads to more gains, and E3 outperforms other baselines as a mainline method. Moreover, LoRA is inferior to full fine-tuning in this setting, and the additional ranking loss is not helpful.

For this problem and model size, the most effective modification for the model is expanding its perception of the previous text. Changing to parameter-efficient tuning or introducing the proposed ranking loss is not beneficial.

There is no doubt that we will conduct some human evaluations in future works, which will further verify our hypothesis. In addition, larger backbones may lead to different results, especially for LoRA. The retrieval result in the appendix provides another piece of evidence: retrieval might still be beneficial, but our retrieval mechanism cannot produce desirable effects.

We would like to emphasize the role each member played in the entire project, from project discussion and experimental design to reporting and reviewing the final result. Specifically, the major tasks assigned to each member were:

• Zhangrui Dong: Dataset preparation, pre-processing, data splitting, and a portion of model training work.
• Zhengyao Zhao: Model building, training pipeline designing, experiment setting, and Colab training.
• Zhengrong Gao: Report writing, methodology and result discussion, LaTex document editing, and paper submission.
• Haotian Wang: Automatic evaluation, metric calculation, result aggregation, and verification of the reported numbers.

In our work, retrieval is regarded as an appendix experiment, not a mainline contribution. We conducted it separately and compared it with the most powerful finished mainline model, i.e., E3. Our retrieval version performed worse than E3. For the retrieval model, a light-weight TF-IDF-based retriever on earlier passages was utilized, which is a classic sparse retrieval approach (Salton and Buckley, 1988).

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
