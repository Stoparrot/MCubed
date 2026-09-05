# Experiment 001 — TinyStories pipeline and text baseline

## Question

Can a small nanochat-derived decoder learn held-out TinyStories statistics on this Mac with a reproducible, bounded training pipeline?

## Frozen pilot

| Item | Choice |
|---|---|
| Upstream model | M³ adaptation of nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; see `mcubed/MODEL_PROVENANCE.md` |
| Dataset | Original TinyStories text files, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| Selection | First 20,000 unique train stories; first 1,000 remaining unique valid stories, alternating 500 validation / 500 test |
| Tokenizer | Byte-level BPE, 2,048 tokens, trained on selected train stories only; EOS between stories |
| Model | 4 layers, 4 heads, width 128, context 256; untied embeddings, RoPE, RMS/QK norm, squared ReLU; no bias/dropout |
| Optimizer | AdamW, β=(0.9,0.95), decay 0.1, clip 1; warmup + cosine |
| Effective batch | 8 sequences × 256 tokens × 2 accumulated microbatches = 4,096 tokens/step |
| Pilot budget | 200 steps = 819,200 sampled training tokens; 600-second process limit |
| Runtime | Native MPS, FP32, no compile; CPU correctness checks |
| Random seed | 1337 |
| Initial cost ceiling | No cloud spend. Process time bounded approximately by 10 minutes plus final evaluation/save |

Selection is deterministic and inexpensive but is a prefix, not a representative shuffled sample of the entire corpus. Training windows are uniformly sampled with replacement from the selected token stream. They may cross story boundaries; EOS is a separator, not an attention reset. The model may not see an entire story in its 256-token context. State these limitations in any report.

## Acceptance

1. Correctness checks pass: causal masking, next-token shift, split deduplication, checkpoint resume parity, data-tamper detection.
2. The pilot completes with finite loss and lower held-out validation loss than initialization. This validates learning in the pipeline, not coherence or a cognitive capability.
3. Saved checkpoint reloads and generates text with the exact tokenizer. Report total parameters and size; include both token and output embeddings; the selected model has 1,310,720 total parameters.
4. Record device, source/data hashes, time, throughput, configs and outcome. Keep test data untouched during the pilot.

## Follow-up baseline

`configs/baseline.json`: same architecture and data; 1,000 steps (4,096,000 training tokens), 50 warmup steps, at most 3,600 seconds. It is a separate run with a different learning-rate schedule. Do not extend the pilot checkpoint under a changed schedule and call it the same run.

Before starting a model-size sweep, freeze a quality threshold based on the baseline and an agreed human rubric. Suggested rubric: grammar, local consistency, prompt continuation and repetition, each scored 0–2 on 20 fixed unseen prompts by a reviewer blind to model size. This rubric is a proposal, not a validated benchmark. Record complete outputs and seeds; do not cherry-pick. Human review is initially cheaper and easier to inspect than adding an LLM judge.

After selecting the text baseline on validation, run `mcubed.evaluate` once and archive its test report. Test loss is a different estimator from sampled validation loss; comparisons are meaningful only at the same tokenizer/context/protocol. If test results influence changes, a new untouched test set is required for subsequent confirmation.

## Stop conditions

Stop on nonfinite losses/gradients, time limit, SIGINT/SIGTERM, or a `STOP` file. A finite earlier checkpoint survives numerical failure. A force kill or power loss can lose work since the last evaluation checkpoint. Persist artifacts to a mounted volume in containers. A stopped cloud process does not terminate its billable instance.
