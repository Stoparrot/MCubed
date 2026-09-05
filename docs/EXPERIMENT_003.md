# Experiment 003 proposal — inductive grammar and a minimal coherent world

Draft for review, 2026-09-05. **No new training has started.** The user stopped experiment 002 and requested a research-informed proposal before proceeding. Model training should learn from examples; we will not feed it grammar rules, deduction traces, parser labels or simulator state. Explicit definitions belong in the data creator and verifiers.

Read [research and sources](GRAMMAR_RESEARCH.md) and [the six-state world specification](MINIMAL_WORLD.md). The curriculum below is a bank of testable hypotheses, not a claim about children's universal acquisition order. Some later entries concern meaning or discourse and have their own verifier. They are not counted as grammar accuracy.

## What we have actually learned

| Local run | Parameters, including embeddings | Sampled training tokens | Measured runtime | Best sampled validation loss |
|---|---:|---:|---:|---:|
| baseline-001 | 1,310,720 | 4,096,000 | 51.92 s | 3.5619 |
| coherence-small-001 | 1,310,720 | 40,960,000 | 386.47 s | 2.3893 |
| coherence-medium-001 | 5,767,168 | 40,960,000 | 1,239.43 s | 1.9823 |
| coherence-expanded-001 | 5,767,168 | 2,273,280 additional | 69.01 s | 2.0454 at initialization |

The expanded run was stopped by user request at step 555. Its cumulative lineage has 43,233,280 tokens. Its evaluation windows differ because the training seed also controls evaluation sampling; **2.0454 is not a directly matched comparison with 1.9823**. It did not establish whether expanded data helps. Keep its STOP file in place.

More training and greater capacity reduced loss in the completed comparisons. Saved outputs still contain serious character shifts, repetition and inconsistent events. These runs have one seed each and do not identify an optimal capacity. The larger run began with an untracked dashboard directory; exact training-source hashes were saved, and the configuration/training code was committed. The other three runs recorded clean working trees. Real test loss remains unevaluated.

The TinyStories tokenizer and vocabulary remain fixed references. A 20k-story prefix and 256-token windows are limited inputs; random windows can omit antecedents and initial states. Therefore the world experiment must keep each complete episode inside the model context. Changing that batching policy is explicit and belongs in this new experiment.

## What we mean by success

The goal is induction that transfers: coherent output on unfamiliar instances and combinations, under a stated language/world scope. We do not require the model to verbalize rules, perform proofs or output reasoning chains. Behavioral tests cannot establish whether its internal solution is equivalent to a compressed template.

Three independent checks:

1. **Grammar:** form and agreement, independent of whether an event is true. A grammatical false statement is a grammar success and coherence failure.
2. **World coherence:** statements and continuations agree with the small world's state and event history. A true but malformed sentence fails grammar separately.
3. **Entity acquisition:** new names, new coexisting entities, and new entity types are separate learning problems. Measure data needed for each without silently changing the other two criteria.

Generated text must be checked without a grammar-constrained decoder. Otherwise the decoder, rather than the learned model, could be producing the grammaticality. Empty output, generic filler and endlessly repeated true statements must not pass. Assess required information and completion length as well as correctness.

## 33 proposed training iterations, from simple constructions to composition

Each row means an independently versioned data component and an evaluation slice. Train on positive examples of the target plus its prerequisites. Negatives, metadata and explanations are reserved for the verifier unless an explicitly separate training-objective experiment is defined. Prerequisites below reference G numbers. Examples illustrate the construction, not final train or test items.

| ID | Component and prerequisites | Hypothesis about generalization from examples | Decisive held-out test |
|---|---|---|---|
| G01 | Lexical category slots; none | Distributional exposure supports placing familiar words in new noun/verb contexts. | Hold out word–slot combinations; compare with a unigram/bigram baseline. Do not mistake unseen-word ignorance for syntax failure. |
| G02 | Simple subject–verb clauses; 01 | Short events transfer across familiar participants and verbs. | “Ava sleeps.” New subject–verb combinations; reverse lexeme frequencies. |
| G03 | Subject–verb–object; 02 | Word order preserves participant roles across unfamiliar pairings. | “Ava moves the cup.” Exchange agent/patient in reversible events; score form and meaning separately. |
| G04 | Determiner + noun; 01 | Singular count-noun phrases transfer beyond memorized phrases. | “a cup / the cup.” Hold out determiner–noun pairings; noun countability is controlled. |
| G05 | Adjective placement; 04 | Modifiers compose with new nouns. | “the red cup.” Hold out adjective–noun combinations; compare accepted versus displaced modifiers, not subjective adjective-order preferences. |
| G06 | Productive regular plurals; 04 | Number marking transfers to new regular nouns and nonce roots. | “one dax / two daxes.” Use independently specified orthographic classes; no unseen irregular expectations. |
| G07 | Copular clauses; 02,06 | Person/number and predicate slots combine. | “The cups are red.” Cross noun number with unseen predicates. |
| G08 | Progressive aspect; 07 | Auxiliary + verb form transfers across actions. | “Ava is moving.” Hold out verb–auxiliary combinations and plural subjects. |
| G09 | Locative phrases; 04,07 | Location phrases compose without confusing entity and place. | “The cup is in the room.” Unseen entity–place pairs; grammar and simulated location scored separately. |
| G10 | Possessive noun phrases; 04 | Owner–object associations transfer across lexical pairs. | “Ava's cup.” Reverse ownership and hold out owners in selected phrase positions. |
| G11 | Pronoun case; 03 | Subject/object position supports the appropriate pronoun form. | “She sees him.” Role-swapped contexts with balanced pronoun frequency. |
| G12 | Present-tense agreement; 02,06 | Agreement generalizes across subjects and predicates. | “The cup falls / the cups fall.” Cross lexical families and number. |
| G13 | Regular past tense; 02 | Past morphology transfers beyond encountered inflected forms. | Familiar and nonce regular verbs in new contexts; control spelling alternations. |
| G14 | Irregular inflection and productive defaults; 06,13 | Observed exceptions coexist with productive regular forms. | Test taught irregular forms in new sentences and unseen regular roots. No claim that unseen irregulars are predictable. |
| G15 | Negation and do-support; 12 | Polarity changes preserve participants and valid verb forms. | “Ava does not move the cup.” Balanced positive/negative contexts; grammar versus truth separately. |
| G16 | Modal + bare verb; 02 | Auxiliary combinations transfer to new predicates. | “Ava can move.” Hold out modal–verb pairs; actual ability belongs to a later world extension. |
| G17 | Yes/no question formation; 07,08,15 | Question forms generalize beyond one memorized opener. | “Is Ava moving?” Novel subjects/actions; use natural example prompts, not unseen transformation instructions. |
| G18 | Wh-questions; 03,09,17 | Question type aligns with the requested participant or location. | “Where is the cup?” / “What does Ava move?” Unseen role/question combinations. |
| G19 | Coordination; 03,12 | Two familiar units can combine without losing agreement or referents. | “Ava and Bo move.” Unseen coordinated subjects/clauses; compare with their component sentences. |
| G20 | Ditransitive argument frames; 03,10 | Verb-specific argument patterns transfer to new participants. | “Ava gives Bo a cup.” Hold out role combinations while respecting each verb's licensed frames. |
| G21 | Count and mass noun contexts; 04,06 | Quantifier selection generalizes within learned noun classes. | “many cups / much water.” Separate taught lexical class from grammatical combination. |
| G22 | Comparatives; 05,07 | Comparative forms compose with new entity pairs. | “The cup is larger than the box.” Form checked separately from world comparisons; extension needs size attributes. |
| G23 | Tense/aspect combinations; 08,13,14 | Temporal forms preserve a stated event interval. | “Ava has moved the cup.” Hold out verb–aspect pairs; use contexts with unambiguous intended timing. |
| G24 | Temporal/causal subordination; 19,23 | Linked clauses preserve event order and supplied causes. | “After Ava moves, Bo waits.” Reverse clause order without changing chronology; causal truth needs its own environment. |
| G25 | Sentential complements; 03,19 | Embedded clauses retain their own participants. | “Ava says that Bo waits.” New matrix/embedded combinations; do not infer beliefs from speech automatically. |
| G26 | Subject relative clauses; 03,12 | A noun can be modified by a familiar event. | “The person who moves the cup waits.” Novel noun–clause pairings. |
| G27 | Object relative clauses; 26 | Nonlocal role assignment survives different surface orders. | “The cup that Ava moves is red.” Match vocabulary to subject-relative controls. |
| G28 | Agreement across intervening nouns; 12,26,27 | Subject information survives conflicting nearby number cues. | “The cup near the boxes is red.” Test distractor number, longer distances and separately deeper structures. |
| G29 | Reflexive reference; 11,25 | Reflexive forms depend on the appropriate local participant. | “Ava sees herself.” Longer clauses with competing antecedents and lexical controls. |
| G30 | Negation/quantifier scope; 15,21,25 | The model distinguishes scoped meanings in finite, unambiguous scenes. | “Not every cup is red” versus “No cup is red.” Test truth conditions; do not collapse grammaticality and semantics. |
| G31 | Cross-sentence entity reference; 10,11,18 | Familiar entities retain identity across sentence boundaries. | Full names first, then uniquely resolvable pronouns; deliberate ambiguity must permit abstention later. |
| G32 | Event persistence and coherent continuation; 09,18,31 | Text learned from valid trajectories supports consistent continuations on unseen trajectories. | Six-state world: after a carried object is put down and the person leaves, the object stays. See world spec. No physical object-permanence claim. |
| G33 | Combined constructions and domain transfer; selected predecessors | Acquired components coexist in unfamiliar combinations and non-story prose. | Withhold entire construction combinations and adult registers: descriptions, instructions, neutral dialogue, short factual exposition. Separate failure of grammar, content and coherence. |

The broad progression is developmentally motivated by increasing phrase/clause dependencies, productive morphology and later complex constructions. It is an engineering order, not Brown's morpheme sequence. Grammar/meaning develop interactively; children do not finish a syntax module before learning word meanings. Relevant foundations and modern results are linked in the research note.

## How we combine components without assuming additive “engrams”

For a chosen set of components, keep the same examples, source proportions and sampled-token budget across:

- Joint mixture from the start.
- Simple-to-complex ordering with replay of prior components.
- Reversed or counterbalanced ordering with the same overall exposure, for a small representative subset.
- Single-component plus prerequisites, as a diagnostic reference.

A full 33! order search is unnecessary. Start with G02–G09 and the restricted W0 locative/question constructions. Expand only after a positive control passes. Do not train 33 isolated models and assume their weight matrices can be added together. Measure retention after each stage and performance on withheld combinations.

For the pilot, propose 50% current-stage tokens and 50% balanced replay in the curriculum arm; construct the joint arm from the resulting exact total multiset. Keep final whole-mixture consolidation equal. The actual frozen manifest must contain exact token weights and exposure counts; sequence counts alone are insufficient.

## Synthetic data creator: required design before implementation

Use ordinary Python functions, a typed lexicon and a surface realizer. Simulator events and grammatical features are generator metadata; the model receives only rendered text. A learned/LLM data creator is unnecessary for the initial controlled world. Later paraphrases need a recorded teacher identity, prompts, raw outputs, validator decisions, cost and data terms; reject changes that alter the intended event.

Each record needs: stable record/family ID; concept IDs; generator/config revision; parent version; split; lexeme IDs; construction-family ID; token/word count; text; private-to-evaluation annotations; world version, initial state and action trace when relevant. Each manifest needs source/generator hashes, tokenizer hash, seeds, unique counts, sampled exposure counts, split/grouping rules, per-component distribution and archive checksums.

Reserve evaluation families before rendering. Exact text deduplication is insufficient: paraphrases of the same held-out trajectory or semantic skeleton belong to the same split. Verify entity-role distributions and label balance; prevent length, punctuation or tokenization from predicting the answer. Balance realistic and nonce cases separately. Syntactically valid nonsense is useful for isolating form, and must not contaminate coherence training as if it were a true event.

Data components and their mixtures are separate immutable versions. A mixture manifest references component hashes and proportions. Fixes make new versions. Keep data archives local, with Git metadata/tokenizer/specs and dashboard links. Never silently modify the TinyStories versions. No generated corpus has been released for this proposal.

## Verifier and evaluation contract

- Use minimal-pair likelihood as one measure; for identical prefixes, score the complete alternative continuations (including all subtokens) with a fixed protocol. Report length-sensitive cases separately. Minimal pairs remain evaluation-only.
- Check unconstrained generations using an independent parser/verifier for the controlled fragment; keep verifier tests hand-authored and adversarial, not just round-trips through the generator.
- Hold out lexical roles, construction combinations, surface families, semantic trajectories, longer lengths and deeper structures as separate test axes. Provide exposure to a new lexeme in neutral training contexts when measuring grammatical transfer rather than lexical acquisition.
- Test both dev and sealed final sets. Freeze final generators/seeds/family assignments before selection. If final examples inform changes, retire that test version.
- Record per-concept and worst-slice results, coverage, abstentions, repetition, task completion and three training seeds. Use family-level uncertainty because paraphrases are correlated. Do not pool thousands of paraphrases into a falsely narrow confidence interval.
- Proposed fragment gate: >=99% grammar validity and >=99% world-answer accuracy on the canonical dev set, >=95% on every eligible held-out family, plus >=95% joint grammar/coherence/completion success. Retention decrease <=2 percentage points per prior component. Freeze these as scoped engineering targets after verifier audit; they are not a definition of perfect English.
- General English needs an independent reviewed corpus and a broader test inventory. Passing our finite parser cannot certify arbitrary English. Dialect, register and ambiguous-but-valid readings need explicit treatment.

## Capacity and time-to-target

Use the existing nanochat-derived architecture unchanged for the first capacity sweep; see exact candidate counts and Mac-based estimates in `MINIMAL_WORLD.md`. Compare complete model bytes, tokenizer bytes, any auxiliary runtime assets, peak memory, training seconds/tokens and inference latency at a common acceptance threshold. A masked dense tensor does not save runtime or dense storage merely because many weights are zero.

After a passing positive control, reduce depth/width. Separately test tied embeddings and tokenizer alternatives, with their own configs and tokenization-aware evaluation. Compression/distillation and training-from-scratch answer different questions and should have separate lineage and total compute accounting.

## Boundaries for this turn

Current training is stopped, artifacts preserved, and the dashboard remains available. This turn delivers the research, 33 hypotheses, world specification, verifier/data requirements and proposed size/time sweep. It does not train the new curriculum, build a new multimodal architecture, or claim that any proposed small model has passed.
