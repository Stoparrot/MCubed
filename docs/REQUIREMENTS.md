# M³ — initial requirements

M³ (M-cubed, Minimum Multimodal Model) is a small research lab studying how model capacity, training data and computation affect learned capabilities. Its eventual product is a family of simple, small, portable multimodal models. The first milestone is text only.

## What “minimum” means

Engram-μ(x) is a working name for the **smallest tested configuration that meets a preregistered capability threshold under fixed data, training budget and evaluation conditions**. It is an empirical upper bound on sufficient capacity within a searched model family. It is not a biological engram measurement or a proof of the smallest possible model. Parameter count alone misses embeddings, precision, activation memory, context, training compute and latency. Report the tradeoffs rather than collapsing everything into one number.

Developmental psychology should motivate the question and controls. Childlike vocabulary is not a simulation of child development. Text about a hidden object is not direct evidence of perceptual object permanence. Before a concept study, define observable behavior, a chance baseline, distractors, held-out templates and counterbalanced examples. A single task score does not establish general understanding.

## Responsibilities

| Responsibility | Owner | Output |
|---|---|---|
| Hypothesis, concepts, metrics, budgets, parameter sweeps | Engram-μ | Frozen experiment specification and decision |
| Architecture, trainer, debugging, inference, profiling | Model creator | Readable code, checks, artifacts |
| Environment, containers, device checks, costs | Model IT | Verified runtime and dated cost estimate |
| Metrics, artifact comparison and later run controls | Monitoring | Read-only records first; approved launch workflow later |
| Dataset versions, deduplication, quality audit | Data archiver | Immutable manifest and proposed next dataset version |

One Codex workflow can perform these roles initially. Do not add a parameter-tuning agent yet: Engram-μ specifies the search; Model creator executes it. Independent review of evaluation design will be more valuable than another implementation role once concept studies begin.

## Milestones

1. **Pipeline pilot (now):** native Mac, documented nanochat-derived model, bounded TinyStories subset, train/evaluate/checkpoint/resume/sample, documented loss reduction. No promise of coherent generation in 200 steps.
2. **Text baseline:** longer fixed budget, train/validation curves, frozen generation prompts and human rubric, one final held-out test, three seeds when comparing configurations. Tune only against validation.
3. **Capacity study:** change depth/width at fixed tokenizer and context; compare equal-token budgets first, then report extra compute needed to reach the threshold. Include a larger positive control, smaller failing candidates and optimization checks. Failure at a budget is not an impossibility result.
4. **Linux/cloud parity:** run CPU container and NVIDIA smoke tests, benchmark representative hardware, compare time-to-target and total cost. Store source revision and environment with every result.
5. **Dashboard controls:** separate source checkout from run artifacts; approved immutable commit/config/data/quote, launch, stop, inference and audit log. Add one execution worker, concurrency limits and process-state reconciliation before remote controls.
6. **Multimodal concept study:** choose one behavior and paired dataset first; add the smallest modality encoder and explicit interface needed. Compare unimodal controls, shuffled cross-modal pairs and novel combinations. Do not assume concepts live in removable modules or that training order is irrelevant; test order and interference explicitly.

## Success requirements for milestone 1

- Training runs on the Mac GPU and has a CPU fallback with a clear device report.
- Baseline uses the compact nanochat-derived GPT in `mcubed/model.py`, with all deviations recorded in `mcubed/MODEL_PROVENANCE.md`, with a small train-only byte-level BPE vocabulary. This is a new experimental configuration, not a reproduction of the TinyStories paper's published scores.
- Source data comes from a pinned revision. Story-level selected splits are disjoint after whitespace-normalized exact deduplication. Near-duplicate/template leakage remains a limitation requiring audit before capability claims.
- Configs, selected data and tokenizer have reproducible manifests. Models and datasets stay out of Git.
- Training logs finite loss, validation loss, token throughput, elapsed time and estimated compute cost. Validation is sampled consistently; final test is token-weighted across all held-out tokens.
- Checkpoints include weights, optimizer, step, random state and config; interrupted runs can resume without changing the schedule. CPU resume parity is tested. Identical results across CPU/MPS/CUDA are not assumed.
- Time limit and graceful SIGINT/SIGTERM/STOP handling work at optimizer-step boundaries. The time limit can overrun by one step, evaluation and checkpoint IO; it is not a cloud billing enforcement mechanism.
- Inference reloads the saved tokenizer and weights. Report weak text honestly.

## What was missing from the original plan

The main gaps were an operational definition of capacity, evaluation controls, a tokenizer budget, contamination policy, reproducibility records, a positive control, and a distinction between research progress and infrastructure progress. Add a frozen human generation rubric before language-quality claims. Add three seeds and uncertainty before comparing capacities. Retain failures and dataset edits so the search is auditable.

Start with the documented nanochat-derived baseline and standard causal attention using PyTorch's backend dispatch. Add newer attention variants only after a baseline exposes a specific limitation, one change per experiment. Premature architecture novelty makes it harder to identify what caused a result.

## Costs and cloud selection

Local process compute is logged at a supplied hourly rate; the default $0 means no purchased cloud instance, not free electricity or development. Codex usage, storage, downloads, inference and cloud idle time need separate accounting.

At the cloud milestone, Model IT should compare providers initially and weekly, recording timestamp, source URL, region, availability, GPU/VRAM, CPU/RAM, on-demand/spot terms, storage/egress and minimum billing. Rank **estimated total cost to reach the target**, not nominal GPU dollars/hour. Use measured provider throughput; never transfer Mac throughput to a different GPU. Inference should compare latency and cost at stated request volume. Spot interruption and checkpoint overhead matter.

No cloud provider is selected and no recurring cloud-price job is enabled yet. Enable the weekly follow-up after budget, region and acceptable interruption policy are set. Do not spin up instances during price research.

## Sources and interpretation

- [nanochat](https://github.com/karpathy/nanochat): current practical reference, selected after the user asked for a newer Karpathy example. See `MODEL_OPTIONS.md` and the model provenance file. Older nanoGPT is retained only as reference.
- [TinyStories paper](https://arxiv.org/abs/2305.07759): motivates small text models; its findings do not establish a concept-capacity minimum for this setup.
- [Dataset card](https://huggingface.co/datasets/roneneldan/TinyStories): records the source and variants. Retain source metadata; review dataset/model distribution terms before publishing artifacts.
- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) and [Docker GPU support](https://docs.docker.com/desktop/features/gpu/): native macOS MPS and Linux container GPU execution are distinct environments.
