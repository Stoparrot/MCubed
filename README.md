# M³ — Minimum Multimodal Model

M-cubed studies how small an understandable model can be while still passing a defined evaluation. **The first experiment is text only:** a compact nanochat-derived model trained from scratch on a bounded TinyStories subset.

The first Mac GPU pilot passed: 200 steps in 55 seconds, validation loss 7.624 → 4.930. Text generation works but remains incoherent. See [pilot results](reports/pilot-001.json).

Training is currently stopped. The next proposed study separates grammar, a [six-state coherent world](docs/MINIMAL_WORLD.md), and [causal interpretability](docs/INTERPRETABILITY.md). Review the [33 experiment hypotheses](docs/EXPERIMENT_003.md) and [research note](docs/GRAMMAR_RESEARCH.md) before new training.

Start with [requirements](docs/REQUIREMENTS.md), [experiment 001](docs/EXPERIMENT_001.md), [current status](docs/STATUS.md), and [Codex setup](docs/CODEX.md).

## Run on this Mac

The setup creates a Python 3.11 environment. The verified local pilot already exists at `runs/pilot-001`; use `runs/pilot-002` if repeating it. Run all commands from the repository root. This workspace already has the environment and prepared data; skip their creation when continuing here.

```sh
cd /Users/uw/Desktop/MCubed
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m mcubed.doctor
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m mcubed.prepare
```

Without uv, use `python3.11 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`. The doctor should report `selected: mps` and `backward_ok: true` on Apple Silicon. An explicit unavailable GPU fails with an explanation; `auto` can choose CPU.

The data command streams 20,000 training and 1,000 held-out stories from pinned files, writes a tokenizer and hash manifest, and refuses to overwrite an existing data directory. It does not download the entire corpus. Approximate local dataset size is tens of MB; Python/PyTorch dependencies use substantially more disk.

Run the short pilot (200 steps, at most approximately 10 minutes plus final evaluation/save):

```sh
.venv/bin/python -m mcubed.train --config configs/pilot.json --out runs/pilot-001 --device mps
.venv/bin/python -m mcubed.sample --checkpoint runs/pilot-001/best.pt --device mps
.venv/bin/python -m mcubed.report --run runs/pilot-001
```

Choose a new output directory for every new run. The trainer refuses to overwrite a run. A new random initialization and initial validation measurement are included in each run. The model has 1.31M total parameters; the pilot is a pipeline check and is unlikely to produce coherent stories.

After reviewing the pilot, run the longer baseline (1,000 steps; one-hour process limit):

```sh
.venv/bin/python -m mcubed.train --config configs/baseline.json --out runs/baseline-001 --device mps
.venv/bin/python -m mcubed.sample --checkpoint runs/baseline-001/best.pt --device mps --output runs/baseline-001/sample.json
```

In another terminal, monitor it:

```sh
.venv/bin/python -m mcubed.report --run runs/baseline-001 --watch
```

Press **Ctrl-C in the training terminal** to stop at an optimizer-step boundary and save. Alternatively, `touch runs/baseline-001/STOP`. Remove only that STOP file before resuming. Resume requires the original config, data, source code, device and hourly rate:

```sh
.venv/bin/python -m mcubed.train --config configs/baseline.json --out runs/baseline-001 --device mps --resume
```

Elapsed time is cumulative across resumes; resuming does not reset the time budget. To change the budget or learning-rate schedule, make a new configuration and new run. SIGKILL/power loss recovers only to the last checkpoint.

After selecting the final baseline using validation, score the test set once:

```sh
.venv/bin/python -m mcubed.evaluate --checkpoint runs/baseline-001/best.pt --device mps --output runs/baseline-001/test.json
```

Do not use this command during pilot debugging or a parameter search. The test report refuses overwrite.

## What is saved

| Artifact | Purpose |
|---|---|
| `data/.../manifest.json` | Source revision, selection rules, split counts and hashes |
| `data/.../tokenizer.json` | Tokenizer trained only on selected training stories |
| `runs/.../environment.json` | Source revision/hashes, device, versions and cost assumptions |
| `runs/.../metrics.jsonl` | Training/validation loss, perplexity, throughput, timing and estimated cost |
| `runs/.../best.pt` | Best checkpoint selected on validation |
| `runs/.../last.pt` | Latest resumable checkpoint, including optimizer and RNG |
| `runs/.../summary.json` | Completion/stop reason, quality change, model size and totals |
| `runs/.../failure.json` | Failure record, if a run errors |

## Dashboard and versioned data

```sh
.venv/bin/python -m dashboard.server --device cpu --port 8765
```

Open http://127.0.0.1:8765 for experiment curves, saved development samples, local inference, story chat and a paginated dataset browser. See [dashboard and Linux setup](docs/DASHBOARD.md). Training launch/stop approvals and cloud management remain future work.

Dataset manifests, the tokenizer and archive checksums are versioned in [datasets](datasets/README.md). The exact compressed archive stays local in `data/archives/`; it is not uploaded or committed. Transfer that archive separately when moving to Linux, then restore with the documented checksum-verifying command.

To continue training under a new schedule, choose a new config and run directory and pass `--init-from runs/PARENT/best.pt`. This loads weights only and resets the optimizer; parent hashes and cumulative training tokens are recorded. `--resume` instead retains the original optimizer and unchanged schedule.

The present CLI does not enforce GitHub approvals; use it as a local development tool.

`--hourly-usd RATE` on the trainer logs process time × supplied rate. The default zero means no purchased cloud compute. It excludes electricity, Codex usage, storage, network and VM idle time. Report estimates exclude evaluation and startup overhead and apply only to measured hardware/configuration. They are not provider quotes or spending controls.

## Containers

Native macOS is the MPS path. Docker Desktop runs Linux containers; these Dockerfiles do not expose Apple Metal. The CPU container can check basic portability:

```sh
docker build -t mcubed:cpu .
docker run --rm mcubed:cpu
docker run --rm -v "$PWD/data:/workspace/data:ro" -v "$PWD/runs:/workspace/runs" mcubed:cpu python -m mcubed.train --out runs/container-pilot --device cpu
```

On a Linux **x86-64** NVIDIA host with compatible drivers and NVIDIA Container Toolkit:

```sh
docker build --platform linux/amd64 -f Dockerfile.cuda -t mcubed:cuda .
docker run --rm --gpus all mcubed:cuda
docker run --rm --gpus all -v "$PWD/data:/workspace/data:ro" -v "$PWD/runs:/workspace/runs" mcubed:cuda python -m mcubed.train --out runs/cuda-pilot --device cuda
```

The CPU image is verified on Linux ARM64 in Docker Desktop: build, device check, all eight correctness tests, training, checkpoint persistence and inference passed. It also loads the earlier Mac-trained checkpoint. The CUDA definition passes Docker build checks and its base digest resolves, but its full build and GPU execution still require verification on Linux x86-64 with NVIDIA hardware. Both base images are pinned by digest; the CPU image explicitly installs CPU-only PyTorch. See [verification details](docs/DOCKER_VERIFICATION.md). No cloud resources are provisioned; stopping a training process does not stop cloud billing.

## GitHub and provenance

Repository: [Stoparrot/MCubed](https://github.com/Stoparrot/MCubed). Source/configuration are committed; data and checkpoints are ignored. Keep artifacts in backed-up storage separate from Git and reference them by hashes in result reports.

The model is an M³ adaptation of current nanochat; see [model provenance](mcubed/MODEL_PROVENANCE.md) and [reference comparison](docs/MODEL_OPTIONS.md). Upstream MIT notices are retained in `vendor/`. The M³ repository uses the MIT license. TinyStories provenance is recorded separately and data is not redistributed here.
