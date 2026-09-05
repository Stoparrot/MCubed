# Grammar, small transformers and operational engrams

Research note, 2026-09-05. This is a targeted literature review, not an exhaustive survey or a reproduction of the cited results. Paper pages/abstracts and selected primary-source passages were checked. Years below use original publication/submission dates, not search-engine crawl dates. New experiments have not started.

## What the evidence changes in our plan

1. Synthetic grammars give us known rules and controlled counterexamples, but their results must transfer to independent English material.
2. A developmental curriculum is a hypothesis. Compare it with the same examples shuffled and interleaved under the same token budget.
3. Recognition, grammatical generation, compositional meaning, discourse coherence and factual knowledge are different targets.
4. A feature visible to a probe need not be used by the model. Use held-out interventions and collateral-damage controls before calling it a causal mechanism.
5. None of the sources reviewed establishes a universal minimum parameter count for English grammar or independently removable English “engram” modules.

## Developmental foundations and data efficiency

| Primary source | What it contributes | Implication and limit for M³ |
|---|---|---|
| Jean Berko (1958), [The Child's Learning of English Morphology](https://www.mpi.nl/publications/item2281723/childs-learning-english-morphology); [original paper](https://pure.mpg.de/pubman/item/item_2281723/component/file_2281722/Berko) | Tests productive morphology with invented words. | Use fresh nonce roots to separate regular inflection from memorized word forms. Our written cloze task is an adaptation, not a replication of children's spoken task. |
| Bloom et al. (1975), [Structure and Variation in Child Language](https://childes.talkbank.org/access/Eng-NA/0docs/Bloom1975.pdf) | Development involves interacting form, meaning and individual variation; discusses cumulative construction complexity. | Order broad dependencies from simpler phrases to linked clauses. Do not present our numbered curriculum as a universal child acquisition order or assign invented acquisition ages. |
| Warstadt et al., [Findings of the BabyLM Challenge](https://arxiv.org/abs/2504.08165) (2025 arXiv posting, reporting the first challenge) | Studies pretraining with limited linguistic input; many curriculum submissions did not provide clear gains. | Include a shuffled-data control. Data efficiency is not evidence of a minimum model size; results from masked encoders do not directly establish autoregressive generation quality. |
| Hu et al. (2024), [Findings of the Second BabyLM Challenge](https://arxiv.org/abs/2412.05149) | 10M/100M-word tracks, varied objectives and corpora; compute remains associated with results. | Count distinct data, repeated tokens and compute separately. Small data does not imply small training cost. |
| Opper, Morrison and Siddharth (2023), [On the effect of curriculum learning with developmental data for grammar acquisition](https://arxiv.org/abs/2311.00128v2) | In their BabyBERTa experiments, source exposure, especially speech-derived corpora, explains much of the apparent curriculum benefit. | Match per-source and per-concept token exposure, not just training steps or example counts. Test multiple adult English genres as well. |
| Zhou et al. (2026), [What Exactly do Children Receive in Language Acquisition?](https://arxiv.org/abs/2603.02082v2) | Analyzes questions and relative clauses in 57 English CHILDES corpora with validated construction labels. | Gives a recent empirical basis for separating question/relative-clause types. A parser's labels and child-directed input frequencies are not universal ground truth for learning order. |

## Measuring grammar and composition

| Primary source | Result or method | Implication and limit for M³ |
|---|---|---|
| Warstadt et al. (2019), [BLiMP](https://arxiv.org/abs/1912.00582) | 67 sets of expert-designed grammatical minimal pairs; contrasts target syntax, morphology and related semantics. | Use positive/negative likelihood comparisons, plus separate generation tests. Favoring a grammatical option does not prove the model can generate fluent paragraphs. Do not train on benchmark pairs. |
| Kim and Linzen (2020), [COGS](https://arxiv.org/abs/2010.05465) | Models can succeed on familiar examples while failing unfamiliar combinations of familiar words and structures. | Hold out lexical roles and structural combinations. COGS is semantic parsing, not a direct score of free-text English quality. |
| Allen-Zhu and Li (2023; revised 2025), [Physics of Language Models, Part 1: Learning Hierarchical Language Structures](https://arxiv.org/abs/2305.13673v4) | Autoregressive transformers learn synthetic hierarchical CFGs; analyses relate learned representations to parsing computations. | A known synthetic grammar is a useful microscope. These controlled languages and model settings do not establish the smallest model for natural English. |
| Ahuja et al. (2024; revised 2025), [Learning Syntax Without Planting Trees](https://arxiv.org/abs/2404.16367v3) | Training objective and dataset structure affect hierarchical generalization; pruning exposes subnetworks with different generalization behavior. | Start with our existing causal LM objective and create tests where hierarchical and linear shortcuts disagree. A pruned circuit is not automatically an efficient standalone model. |

## Compact representation and causal mechanisms

| Primary source | Result or method | Implication and limit for M³ |
|---|---|---|
| Weiss, Goldberg and Yahav (2021), [Thinking Like Transformers](https://arxiv.org/abs/2106.06981) | RASP expresses sequence computations in a language related to transformer operations. | Use formal-language toy tasks to understand constructive capacity and dependencies. A hand-designed solution's expressibility does not guarantee it can be learned efficiently. |
| Lindner et al. (2023), [Tracr: Compiled Transformers as a Laboratory for Interpretability](https://arxiv.org/abs/2301.05062) | Compiles programs into transformers with known internal structure. | An optional calibration tool for interventions, separate from the trained M³ baseline. Compilation is not a learning result. |
| Lepori, Serre and Pavlick (2023; revised 2025), [Uncovering Intermediate Variables in Transformers using Circuit Probing](https://arxiv.org/abs/2311.04354v3) | Uses parameter-level circuits and ablation, including subject–verb agreement and reflexive anaphora in GPT-2. | Closely matches our operational engram question: does a component causally support a specified variable? It does not show a universally isolated grammar module. |
| Hall Maudslay and Cotterell (2021), [Do Syntactic Probes Probe Syntax?](https://arxiv.org/abs/2106.02559) | Probe performance changes when semantic cues are disrupted with grammatical nonsense sentences. | Test nonce/meaning-shuffled material and random-feature controls; a successful probe alone is weak evidence of internalized grammar. |
| Allen-Zhu and Li (2024), [Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405); Morris et al. (2025), [How much do language models memorize?](https://arxiv.org/abs/2505.24832v3) | Estimate storage under different controlled definitions: roughly 2 versus 3.6 bits per parameter in their respective settings. | These are different experimental measures, not a universal conversion from parameters to English concepts. Measure compression, memorization and generalization separately. |
| Cheng et al. (2026), [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372) | Introduces a module named **Engram** using conditional n-gram memory, evaluated at large scale. | This name overlaps ours, but the method is a lookup-memory architecture, not a measurement of the smallest grammar representation. Do not import it merely because of the name. Count its memory tables in total storage if investigated later. |

## Grammar versus a world model

Li et al., [Emergent World Representations](https://arxiv.org/abs/2210.13382v5) (2022; ICLR 2023), use a controlled Othello task and interventions on learned board-state representations. This motivates a later text-described simulator with exact state transitions, hidden states and counterfactual tests. A learned game state is not evidence of a general physical world model.

Bender and Koller, [Climbing towards NLU](https://aclanthology.org/2020.acl-main.463/) (2020), argue for distinguishing linguistic form from grounded meaning. For M³, this is a reason to state what our text-only evaluations can establish, not a claim that text cannot support useful representations.

Our inference: dictionary definitions can help lexical knowledge, but require examples, argument structure, inflections and sense distinctions. Wikipedia can supply factual/expository text, but next-token loss alone cannot establish fact consistency, causal prediction, state tracking or perception. Those need separate evaluations and, eventually, interactions or observations. For a tiny deployed system, external retrieval may be preferable to memorizing every fact; report index/storage and retrieval latency as part of that system's cost.

## Terminology to use in reports

- **Behavioral acquisition:** robust performance on a frozen family of unseen examples and controlled out-of-distribution tests.
- **Operational Engram-μ(C):** smallest tested complete model that passes capability C under fixed vocabulary, context, data, budget and evaluation. Report an upper bound within the searched family.
- **Candidate causal circuit:** an identified component whose controlled intervention changes target behavior on held-out cases with measured effects on other behaviors.
- **Representation dimension:** dimensions used by a chosen readout or compression method, with that method's assumptions. This is not the number of parameters needed to learn the capability.
- **Model composition:** retained performance when capabilities are trained and used together. It is not arithmetic addition of independently trained weight matrices.

Recent interpretability additions, including August 2026 tiny-transformer work, are reviewed in [the model microscope proposal](INTERPRETABILITY.md).
