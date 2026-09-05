# A model microscope for M³

Research update, 2026-09-05. This adds recent small-model interpretability work to [the grammar proposal](EXPERIMENT_003.md). No new model, probe or interpretability method has been trained or installed. This is a targeted review of primary sources, including recent research notes; it is not an exhaustive survey or independent replication.

## What the fMRI analogy gets right

We can inspect activations at every layer and token, rather than infer neural activity indirectly from blood flow. We can also save exact checkpoints and replace internal activations during a forward pass. This provides unusually fine access to a model's computation.

The hard part is interpretation. A visible response to the word “cup” might encode spelling, token frequency, a syntactic role, a location, or several of these together. A concept can span many coordinates; one coordinate can contribute to multiple concepts. A colorful activation plot is therefore a measurement display, not proof of a learned concept.

Our target remains induction from examples. Probes and simulator annotations are analysis tools outside the trained language model; we do not give it explicit rules or request deduction traces.

## Recent advances worth considering

| Date and primary source | What is new/useful | Fit and limitation for M³ |
|---|---|---|
| **21 Aug 2026:** Turner, Wu and Batson, [Characterizing interference weights in a tiny language model](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html) | Analyzes a one-layer, 2.9M-parameter transformer through decomposed interactions; distinguishes effects on predictions from effects on loss. | Highly relevant scale. Large weights are not necessarily useful mechanisms. The explicit virtual-weight representation expands to about 331M weights; interpretability sparsity is not automatically deployment compression. Its normalization-free architecture differs from ours. |
| **6 Jul 2026:** Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) | Introduces the J-lens, a Jacobian-based readout and intervention method for representations related to verbalizable content. | A promising later comparison with simple readouts. It surfaces a subset of representations, not everything the model encodes. Its usefulness for our very small base model is untested; do not infer human-like awareness from a lens display. |
| **18 May 2026:** Chanin, [Are Sparse Autoencoder Benchmarks Reliable?](https://arxiv.org/abs/2605.18229) | Audits SAE metrics using reseeding, synthetic ground truth and training trajectories; finds serious reliability problems in some popular metrics. | Use multiple controls and our known-world ground truth. A benchmark score or readable feature label alone is insufficient evidence. This recent audit is itself a research result requiring scrutiny. |
| **28 Jan 2026:** Liu et al., [Concept Component Analysis](https://arxiv.org/abs/2601.20420v2) | Proposes latent-variable assumptions and an unmixing framework for extracting concept-related representations. | Useful experimental alternative to SAEs. Its guarantees depend on modeling assumptions; a discovered component is not automatically a causally used or uniquely identifiable concept. |
| **12 Nov 2025; revised Dec:** Christensen and Riggs, [Decomposition of Small Transformer Models](https://arxiv.org/abs/2511.08854v2) | Extends Stochastic Parameter Decomposition to transformers, including a toy induction circuit and concept-related components in GPT-2-small. | One of the closest matches to searching for small causal mechanisms. “GPT-2-small” is still much larger than our planned sub-million models. Not evidence that independently learned concepts can be merged without interference. |
| **25 Jun 2025:** Bushnaq, Braun and Sharkey, [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790) | Uses stochastic causal importance to make parameter-space decomposition more practical than earlier APD methods. | Candidate for later circuit extraction. Extra decomposition training and representation size must be counted separately from the deployed model. |
| **2025:** Anthropic, [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Uses sparse replacement components and attribution graphs to generate hypotheses about internal computations, with intervention-based checks. | Relevant visual design for the dashboard. Graphs depend on approximations; explain replacement error, omitted paths and intervention validation. They are not a complete transcript of model reasoning. |
| **31 Jan 2025:** Paulo, Shabalin and Belrose, [Transcoders Beat Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2501.18823) | Compares sparse models of component input→output behavior with activation reconstruction, including skip transcoders. | Transcoders may expose computation more clearly than an activation-only dictionary. Results do not guarantee fidelity or usefulness on M³'s squared-ReLU/RMS model. |
| **12 Mar 2025:** Karvonen et al., [SAEBench](https://arxiv.org/abs/2503.09532) | Evaluates sparse autoencoders on several practical and representational measures rather than reconstruction alone. | Useful evaluation ideas, read together with the 2026 reliability audit. We should not optimize an SAE against a single attractive metric. |

Earlier directly relevant anchors: [circuit probing for agreement and reflexives](https://arxiv.org/abs/2311.04354v3), [Jabberwocky probe controls](https://arxiv.org/abs/2106.02559), [Othello state representations and interventions](https://arxiv.org/abs/2210.13382v5), and [Tracr's known-ground-truth transformers](https://arxiv.org/abs/2301.05062). See the grammar research note for their scope. The newest sources above are not all arXiv papers; the Anthropic items are primary technical research publications with checked publication dates.

## What would count as evidence that a concept is internalized?

Use an evidence ladder, not a binary “concept found” indicator:

1. **Behavior:** new examples, new surface forms and held-out combinations work reliably. Distinguish lexical familiarity, grammar and world coherence.
2. **Readout:** a deliberately small probe can recover a defined variable from held-out activations. Compare with input-only, random-feature, untrained-model and shuffled-label controls.
3. **Causal use:** replacing a relevant activation with one from a matched counterfactual example changes the output in the predicted direction.
4. **Specificity:** the intervention changes the intended variable while preserving unrelated identities, grammaticality and other learned capabilities. Compare norm-matched random perturbations and irrelevant patches.
5. **Transfer and retention:** effects persist across held-out words, trajectory families, checkpoints and independently trained seeds. Correlated paraphrases are one family for uncertainty estimates.

Failure to find a linear readout does not prove absence of a representation. Successful decoding does not prove the model uses that information. Ablation can have widespread effects or place activations outside their normal distribution; matched swaps and multiple controls reduce, but do not eliminate, that problem. Do not call an intervention a complete mechanistic explanation without testing alternatives.

## First practical experiment: carried versus left behind

Construct matched W0 histories with the same names, similar lengths and controlled word counts. One history ends with the cup carried into the hall; another ends with the cup put down before Ava leaves. Query the cup's location. Include paraphrases and alternative orderings of irrelevant state descriptions to break superficial cues.

Record the residual stream after each layer and each event boundary; initially also record embedding outputs and block/MLP outputs. Fit readouts for person location, object location and carrying state using **analysis-training families only**. Read from the query-prefix position before the answer is generated: never include answer tokens or future states in the feature vector.

On held-out matched pairs, patch a candidate object-state representation from one run into the other. Does the cup answer switch appropriately? Does Ava's own location remain correct? Does grammar remain intact? Patch in both directions, include unrelated-state and same-state controls, and repeat across lexically different families. A person-location change that also changes the cup answer everywhere may reflect a conflated representation rather than separate entity state.

Start at whole-layer/token sites, then narrow to heads or subspaces when evidence warrants it. Exhaustive small-model interventions are preferable to a complicated approximate circuit detector at this stage. We should not collect every attention matrix from every training step: start with selected checkpoints and a bounded diagnostic set.

For spelling versus concept separation, test the same label in different roles, different labels with the same demonstrated affordance, and the same referent mentioned through different unambiguous expressions. Give any new label the declared exposure budget; do not assume synonym knowledge a tiny model has never encountered.

## Dashboard addition, after baseline success

A proposed “Model microscope” page should show:

- Checkpoint, dataset and analysis-suite hashes; selected example and controlled counterpart.
- Token × layer view of probe/readout values, with uncertainty and held-out status.
- Original, counterfactual and patched output probabilities and generated text.
- Intervention location, affected dimensions/components, and preservation/failure of other variables.
- Acquisition curve across checkpoints and exposures to a new entity.
- Optional later feature/circuit graph with reconstruction error and confirmed/unconfirmed edges distinguished.

Label evidence as “correlated,” “decodable,” “causally supported,” or “not established.” Do not label arbitrary neurons “the cup engram.” Record all tested intervention candidates and choose them on development families to avoid cherry-picking a compelling image.

Use lightweight PyTorch hooks against the exact current model first. A library conversion is optional and must reproduce original logits before interpretation; RoPE, QK normalization and normalization placement matter. Trained SAEs/transcoders belong to their checkpoint and distribution; downloaded dictionaries for other models are not interchangeable.

## Connection to making the model smaller

Interpretability can suggest where capacity is shared or unused. It cannot directly translate a sparse visualization into removable weights. Confirm a proposed reduction by creating a smaller dense model or genuinely supported sparse representation, measuring actual artifact size/runtime, and rerunning all held-out grammar/coherence/retention tests. Count retraining, distillation and analysis compute. A low-dimensional probe reading a six-state variable is not a lower bound on the network needed to learn the text task.

Recommended sequence: passing W0 model → direct recordings and matched activation patches → simple probes and checkpoint trajectories → optional transcoder or parameter decomposition. No external model API or paid interpretability service is required for the first two steps.

## Post-training inference and an optional larger verifier

After each concept/word training stage, run the same versioned inference suite and compare checkpoints before/after acquisition. Include isolated-word exposure as a lexical diagnostic, short controlled contexts, and unseen combinations. A word alone does not identify its current referent, sense or world state.

For W0, the first activation decoder should be a small readout for the six known states or their component variables. Supervision for that readout comes from the simulator, and belongs only to analysis training data. Render its prediction as inspectable text such as “cup location: hall.” Compare that prediction with the true state and with actual generated answers. The displayed sentence is the readout's output, not a quotation of the model's internal thoughts.

A larger verifier can later describe or review evidence, but it must not be the primary evidence that the tiny model has internalized a concept. A high-capacity decoder may reconstruct a plausible explanation from its own prior knowledge or from input leakage, much as an impressive reconstruction can conceal how much comes from the reconstruction system rather than the measured signal.

Required controls for any such decoder: no-activation/input-only baseline, random or untrained-model activations, shuffled activations/labels, held-out trajectory and lexical families, and an explicit capacity limit or capacity-matched comparison. If the decoder performs equally well without trained-model activations, the claimed evidence fails. A feature's descriptive label must be checked through new examples and interventions, not merely accepted because it sounds plausible. Large-model judgments should be blinded to candidate identity and logged with model version, prompt, agreement and adjudicated errors; no API spending or data transmission is implied by this proposal.

Next-token logits already expose the model's predicted distribution exactly for a given context; no larger model is needed to infer them. Longer continuations require rollouts, because subsequent activations depend on the tokens actually chosen. Record fixed-seed and multiple-seed rollouts, uncertainty and error rates. A lens/readout cannot promise one inevitable future output.

## Affecting predictions versus helping predictions

In the [August 21 study](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html), the connections analyzed are **virtual weights** between components in an expanded representation, not simply individual raw transformer weights. The distinction is between changing the output distribution and improving its fit to the data.

For an illustrative W0 question whose correct answer is “hall,” suppose removing a connection changes the probability of that answer:

| With connection | Without connection | Interpretation on this example |
|---:|---:|---|
| 80% | 55% | The connection helps the correct answer. |
| 55% | 80% | The connection hurts the correct answer. |
| 80% | 80%, but wrong alternatives change | It can affect the distribution without helping this answer. |

These probabilities are invented examples, not measured M³ results. Compare loss with and without the intervention across held-out contexts: `loss_without − loss_with > 0` indicates average helpfulness. A connection can help one context and hurt another. Raw magnitude, activation brightness and attention strength do not settle this question. Evaluate collateral effects before proposing removal, then measure an actually smaller model if compression is the objective.
