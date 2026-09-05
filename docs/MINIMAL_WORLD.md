# W0 — a minimum viable coherent world

Proposal, 2026-09-05. The model learns text examples inductively. This specification defines the data creator and verifier; it is not model input and is not a symbolic reasoning module attached to inference. It is a deliberately tiny positive control, not a claim to be the mathematically smallest possible coherent world.

## Six states, ten valid transitions

One person (Ava), one object (a cup), two places (the kitchen and the hall).

- Person location: kitchen or hall.
- Object status: in the kitchen, in the hall, or carried by Ava.
- A carried object's physical location is Ava's location.
- Actions: move to the other place; pick up the cup when co-located and not carrying it; put down the cup when carrying it.
- No creation/destruction, hidden events, automatic object movement, simultaneous actions, ambiguous pronouns, belief states, planning requests or proof tasks.

There are 2 × 3 = **6 reachable states**. The agent can move from every state; pick-up is possible in two states and put-down in two states: **10 valid directed transitions**. There are 410 legal traces of lengths 1–6 across all six possible initial states, counted with repeated states/cycles allowed. These combinatorial counts were checked by enumeration; no model was trained.

Example training text:

> Ava is in the kitchen. The cup is in the kitchen. Ava picks up the cup. Ava moves to the hall. Ava puts down the cup. Ava moves to the kitchen. The cup is in the hall.

Alternative observed examples can end in a natural question and answer:

> Where is the cup? The cup is in the hall.

Only normal surface text goes to the model. State, action, query and answer annotations stay in the dataset/verifier. Include both declarative continuations and example-based QA; do not expect an otherwise untrained base model to understand a new instruction format at evaluation time.

This world is too small to establish generic knowledge of cups, people or English. It is useful precisely because failures are identifiable. The words can initially act as labels; later lexical transfer tests determine whether the learned behavior extends to new labels and entities.

## Separate verifiers

| Verifier | Checks | Explicitly outside its claim |
|---|---|---|
| Grammar verifier | Generated output is a valid sentence in the declared W0 English fragment; reference forms and agreement are well formed. | It does not certify all English or the truth of a sentence. |
| Coherence verifier | Parse output into a proposition/event, replay the supplied history, and check truth or action validity against the exact final state. | It does not certify physical understanding outside this simulated world. |
| Task completion verifier | The requested entity/location is supplied; output is not empty, generic, repetitive, contradictory or unrelated. | Grammatical filler is not a success. |

Also report **joint success**: all applicable verifiers pass on the same example. Count parser rejection separately from verified falsehood. A rigid realizer/parser shares assumptions, so include independently authored valid paraphrases and invalid cases, plus manual review of boundary cases. Do not silently count a valid out-of-fragment English sentence as “bad English”; report unsupported form.

Use ordinary unconstrained decoding. The verifier scores after generation and does not repair answers. A simulator allowed to answer on the model's behalf would trivialize the experiment.

## Data quantity and distribution

“Enough” is an outcome of a learning curve, not a chosen file size. The world has only 410 short traces: producing millions of paraphrases does not create millions of independent experiences.

Proposed data construction:

1. Enumerate all six states and ten transitions for a coverage audit. Keep every transition available in training.
2. Use lengths 1–4 for core training. Partition complete initial-state/action-trace families before rendering; put all paraphrases of a family in one split. Include length-0 state descriptions as a distinct component.
3. Reserve some length-4 trajectories for development, independent ones for the sealed test, and lengths 5–6 for longer-history tests. Lengths 7–8 form a separate extrapolation stress test. All whole episodes must fit the fixed 256-token context; overflow is rejected rather than truncated.
4. For a diagnostic finite-state baseline, transitions may all be visible. The neural test concerns novel sequences/renderings and maintained references, not unseen laws of motion.
5. Target 25% state descriptions, 25% single-transition examples, 40% multi-transition histories, 10% redundant-observation/distractor histories, **by sampled tokens**. Distractors initially restate irrelevant true facts; they introduce no extra entities or hidden dynamics.
6. Balance queried person/object, final kitchen/hall locations, initial states and action types where feasible. Log the actual joint distribution and any impossible cells. In paired evaluation, valid/invalid alternatives are equally frequent; invalid text is not included as positive LM training data.
7. Start with roughly 2,000 distinct rendered training episodes, cap the initial bank at 10,000, and report how many semantic trajectories and surface families those episodes actually represent. Build nested, coverage-audited subsets at 100, 250, 500, 1,000, 2,000 and 10,000 surface episodes. If a subset cannot cover the ten transitions, report that limitation explicitly.

The exact split IDs, renderer families and proportions must be frozen after audit. Reversing the order of sentences is not automatically a semantics-preserving augmentation. Reordering independent state descriptions is valid; reordering events often changes the world.

The tokenizer stays the existing 2,048-token tokenizer for the first controlled comparison. Do not fit a tokenizer on held-out examples. A dedicated tiny/byte tokenizer is a later independent compression experiment, with total bytes and sequence lengths counted.

## Model-size hypothesis on this Mac

The existing architecture has no learned normalization parameters. Its count is `12 × layers × width² + 2 × vocabulary × width`, including untied input/output matrices.

| Candidate | Layers × width | Parameters | FP32 weight MiB | Role |
|---|---:|---:|---:|---|
| W0-small | 2 × 64 | 360,448 | 1.375 | First small-model hypothesis |
| W0-deeper | 4 × 64 | 458,752 | 1.750 | Tests depth at small width |
| W0-wider | 2 × 128 | 917,504 | 3.500 | Tests width versus depth |
| Existing small | 4 × 128 | 1,310,720 | 5.000 | Larger positive control |
| Existing medium | 6 × 256 | 5,767,168 | 22.000 | Diagnostic fallback only |

Four attention heads, FP32, compilation off, context 256; same model operations and optimizer family initially. These counts are arithmetic for proposed configurations, not trained results.

**Working hypothesis:** 0.36–0.92M parameters should be a plausible range for W0's restricted language and state continuation. Start with 0.36M and the 1.31M positive control. This is a prior to test, not a probability-calibrated forecast, a lower bound, or a claim that this size can produce general English.

The six-state engine itself is much smaller than any transformer above. Its existence prevents us from calling a transformer result a universal minimum representation. Our objective is a small learned text model that generalizes across specified observations, not the smallest program implementing a known simulator.

### Measured timing anchors versus estimates

At 256-token context and 4,096 sampled tokens/optimizer step on this Mac:

- 1.31M model: **9.44 seconds per million sampled tokens**, averaged over the completed 40.96M-token run including its recorded evaluations/saves.
- 5.77M model: **30.26 seconds per million**, measured the same way.

Linear extrapolation at the **same old workload** gives:

| Sampled token budget | 1.31M time estimate | 5.77M time estimate |
|---:|---:|---:|
| 1M | 9.4 s | 30.3 s |
| 4M | 37.7 s | 121.0 s |
| 16M | 151.0 s | 484.2 s |

These are not measured W0 timings. Episode-preserving batches, padding, different evaluation frequency, startup, MPS overhead and thermal/other-app load change them. In particular, do not extrapolate the 0.36M model's time by parameter ratio. Benchmark its actual batch format once implementation is ready. Interpretation runs and data creation need separate budgets.

Use checkpoints at 0.25M, 1M, 4M and 16M *actual non-padding training tokens* within a preregistered schedule. Compare matched budgets first; a declared stopping rule can then estimate time-to-target. Never extend a completed learning-rate schedule while calling it an unchanged run. Track both unique episodes and repeated exposures.

The first screen is two sizes × three seeds, one fixed mixture. Add the two intermediate sizes only if they help bracket a successful candidate. Compare curriculum ordering after fixing a workable data distribution, with equal examples/exposure. Proposed initial training cap: 30 local GPU minutes, not yet started; stop earlier on a clear valid result or verifier/data failure. No cloud spend.

There is no meaningful single “parameters / training time” optimum: that ratio can reward a model simply for training longer. Find the set of successful models for which neither model size nor time can be improved without worsening the other. Choose the smallest one within an agreed time budget, or the fastest within an agreed storage budget. Report inference latency separately. No existing run has met W0's acceptance gate, so an optimum is currently unknown.

## Growing the world and measuring a new entity

After W0 passes, change one axis at a time:

1. **Rename:** change “cup” to a new label with the same observed behavior. Tests label binding and lexical transfer.
2. **Additional object:** retain the cup and introduce a ball with the same affordances. Tests separate identities, interference and reference. With one carrying slot, two objects and two places there are 16 reachable states, not an unrestricted 18.
3. **New type/affordance:** introduce an object that cannot be picked up, with observed demonstrations. This requires evidence of a new behavior, not just another name.
4. **More places, a second person, ownership and transfers:** separate versions and gates.
5. **Partial observation:** add missing information and “unknown” only after fully observed state tracking works. Preserve the distinction between a false proposition and an unobserved one.

For each new-entity experiment, restore the **same parent checkpoint** and use nested demonstration counts `0, 1, 2, 4, 8, 16, 32, 64, 128`. Balance demonstrations over available behaviors; include random/coverage-based selection as distinct arms. Record demonstrations, occurrences of the new name, distinct trajectories, total training tokens, gradient updates and wall time. Repeating one example 1,000 times is one distinct example and 1,000 exposures.

Separate learning from demonstrations in the prompt (no weight update) from learning through training. Count any introduction or definition as evidence. Distinguish a new label for a known type from a genuinely new affordance; no method can infer an arbitrary unseen property from its name alone. Test unseen states/contexts and retained performance on old entities. Report the smallest tested successful sample count with uncertainty across seeds and entity identities, plus failed lower counts, rather than a universal minimum.

A dictionary and Wikipedia are later sources of lexical/factual examples. They are unnecessary for establishing this first inductive competence and would obscure its failure modes. The 33-component grammar plan broadens English forms separately from these controlled world extensions.
