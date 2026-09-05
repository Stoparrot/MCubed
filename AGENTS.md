# M³ working instructions

## Purpose and current scope

Build the smallest understandable model that passes a defined evaluation within a stated training budget. Current milestone: reproduce a tiny text-only nanochat-derived/TinyStories pipeline on native macOS, then Linux. Read `docs/REQUIREMENTS.md`, `docs/EXPERIMENT_001.md`, and `docs/STATUS.md` before changing experiment behavior. Use `README.md` for commands. GPT-6 Astra is the development collaborator, not the model being trained.

## Work style

Keep instructions and code short, concrete, and composable. Carry authorized work through implementation and verification. Choose ordinary implementation details without repeated confirmation. Explain assumptions. Prefer one process and plain functions over agent frameworks, services, plugins, or abstractions. Roles are responsibilities; do not launch autonomous agents just because roles are named. The experiment designer owns the hypothesis, parameter sweep and acceptance criteria; the model creator implements them.

## Scientific contract

- Freeze the evaluation and budget before changing the model. Never tune on test data.
- Preserve the documented nanochat-derived baseline; architecture changes go in a separate explicit experiment.
- Fit tokenizers on training data only. Data changes create a new version and manifest; never silently repair frozen data.
- Compare parameter counts including embeddings, artifact bytes, runtime, and quality. A failed training run is not proof of insufficient capacity.
- Do not describe lower language-model loss as object permanence, concept localization, or evidence of a universal minimum.
- Record exact configs, code/data/tokenizer hashes, seeds, device, environment, time, and cost assumptions. Report measured and estimated values separately.
- Never claim multimodal support, GPU portability, reproducibility across devices, or coherent language without measurements.

## Engineering contract

- Use `.venv/bin/python`; run commands from the repository root. Setup: `uv venv --python 3.11 .venv`, then `uv pip install --python .venv/bin/python -r requirements.txt`.
- Native Mac uses MPS; Linux NVIDIA uses CUDA; CPU is the correctness fallback. Keep FP32 and compilation disabled for experiment 001.
- Keep source in `mcubed/`, pinned upstream in `vendor/`, configs in `configs/`, artifacts in ignored `data/` and `runs/`.
- Test with `.venv/bin/python -m unittest discover -s tests -v`. Add tests for scientific validity, data boundaries, checkpointing and consequential behavior; avoid tests that only repeat implementation details.
- Local bounded smoke/pilot runs are part of development. Long runs follow the recorded experiment budget. Paid provisioning and changes to cloud spend need a concrete quote, limit and user authorization. This repository does not provision cloud resources.
- Git branches use `codex/`. Preserve repository history and licenses. Do not commit data, weights, secrets or virtual environments.
- Commit reviewed code before a formal experiment. Development runs may record dirty source with hashes and must be labeled accordingly. Future dashboard approval must bind the exact commit, config, dataset and cost estimate.
- Update `docs/STATUS.md` with outcomes and remaining limits when a milestone finishes. Stop testing once relevant checks pass unless a new change warrants another check.
