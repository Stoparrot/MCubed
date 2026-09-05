# Experiment 002 — toward coherent story continuation

The user authorized improving text coherence, with discretion over local training, data and parameter count, on 2026-09-05. No cloud spending is involved.

## Objective and development evaluation

Produce short, reasonably grammatical continuations that preserve the characters and events in a supplied story opening. This is story completion, not instruction-following chat. A chat interface can preserve a story history, but conversational competence needs separate data and evaluation.

Before inspecting new samples, freeze the 20 development prompts in `evals/story-prompts.json`: temperature 0.7, top-k 50, seeds 42–61, up to 160 new tokens, stop at EOS. Save every output. These prompts are newly authored but not verified absent from training; they are development material, not a clean benchmark. Do not train on them.

Review each output on grammar, event/character consistency, relevance to its opening, and repetition (0 = major failures, 1 = mixed, 2 = mostly successful). Provisional development target: at least 16/20 outputs score at least 6/8, with no zero for grammar or consistency. Record the full scoring and examples; an assistant review is provisional and needs user confirmation, not a claim of independent human validation. Never claim a capacity minimum from this run.

## Sequence and local budget

1. Run the existing 1,000-step baseline without changing its config, tokenizer, or model. Inspect its validation curve and all fixed-prompt outputs.
2. If it is undertrained and the training/validation gap is small, first increase the training budget while preserving 1.31M parameters and the tokenizer. Use a separate recorded config/run. If needed, warm-start explicitly from a recorded checkpoint as a new experiment; never disguise this as unchanged-schedule resume.
3. Expand the training subset if diversity or overfitting becomes limiting. Freeze and copy the existing validation/test files and tokenizer, and exclude their story hashes from new training stories. Increasing data must not quietly redefine the comparison.
4. Increase width/depth only if the smaller model remains inadequate; record the new count and training budget. Keep the same architecture operations to preserve simplicity.

Allow up to 60 minutes of additional local GPU training for this round, split into bounded runs. Stop earlier once the development target is reached. The existing baseline has its own one-hour upper bound but should take minutes based on the pilot. Do not use the real test split to choose model sizes, data or training duration. Record a final test only after the selected candidate is frozen.

Keep all changes and failures. Report observed coherence, remaining weaknesses, parameter/weight size, total training tokens (including any parent checkpoint), elapsed time and local compute assumptions. A model that only improves loss but fails the rubric has not met the objective.

## First decision

The 1,000-step baseline reached train loss 3.5555 and validation loss 3.5619, with confused characters and events across the saved development samples. The gap is small. Next run: `configs/coherence-small.json`, same data/tokenizer/model, random initialization, 10,000 steps / 40.96M sampled tokens, maximum 20 minutes. Only training duration, evaluation interval and warmup schedule change. This isolates whether more training is enough before expanding capacity.

## Capacity comparison

The small model completed 10,000 steps in 386.47 seconds, train loss 2.3222 / validation 2.3893. Fixed-prompt outputs still lose characters/events and repeat phrases; the development target is not met. Next: six layers, width 256, four heads (5,767,168 parameters), with the same 20k-story data, tokenizer, 256-token context, 40.96M training tokens, optimizer and schedule. Only depth/width increase; the process cap is 30 minutes to allow the larger model to complete the same token budget. This is a larger positive control, not a minimum claim.

## Expanded-data continuation

The 5.77M model completed its 40.96M-token run in 1,239.43 seconds. Final train loss 1.7075 / validation 1.9823. All 20 fixed samples are retained; character confusion, repetition and event contradictions remain common. The widening generalization gap motivates a larger training subset rather than further repetition of only 20k stories.

Next fixed configuration: `configs/coherence-expanded.json`, the same model/context and tokenizer, 200,000 unique training stories from the same pinned source. Copy validation/test files and tokenizer byte-for-byte, exclude their normalized story hashes from training, and register a new manifest. Start from `runs/coherence-medium-001/best.pt` at step 10,000, recording its hash; reset AdamW and use learning rate 0.0003 → 0.00003, warmup 100 steps, seed 1338. The new phase allows 15,000 steps / 61.44M additional tokens, at most 1,800 seconds. Total lineage is at most 102.4M tokens. Completed additional runs so far used 1,625.90 seconds; this phase keeps total measured training below the 60-minute round budget, apart from bounded evaluation/save overhead.

This phase changes data and optimization schedule together and is a practical quality improvement, not a controlled estimate of the effect of data alone. Keep the same 20-prompt generation protocol and rubric. Do not evaluate the real test set unless a final candidate is selected. Archive all data locally; the user declined remote archive upload.
