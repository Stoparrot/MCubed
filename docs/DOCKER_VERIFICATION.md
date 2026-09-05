# Docker verification — 2026-09-05

The CPU definition is verified on Docker Desktop 4.39.0 / Docker Engine 28.0.1, Linux ARM64 on this Apple Silicon Mac. The CUDA definition has passed static build checks only; its full build and GPU runtime remain unverified.

| Check | CPU result | CUDA result |
|---|---|---|
| Base image resolves | Passed; pinned multi-platform digest | Passed; pinned x86-64 image digest |
| Image build and dependency consistency | Passed, including `pip check` | Not executed |
| Docker build checks | Definition built successfully | Passed, zero warnings |
| Device forward/backward | Passed on CPU | Requires NVIDIA host |
| Five correctness tests | All passed in container | Requires NVIDIA host |
| Training and persisted checkpoint | Five steps completed | Requires NVIDIA host |
| Inference from container checkpoint | Passed | Requires NVIDIA host |
| Inference from earlier Mac/MPS checkpoint | Passed | Requires NVIDIA host |

## Changes made during verification

- Pinned the Python and PyTorch base images by their resolved digests.
- CPU definition explicitly installs PyTorch from its CPU wheel index before the remaining requirements. This avoids pulling CUDA dependencies when the CPU definition is used on x86-64.
- CUDA definition explicitly selects `linux/amd64` through `CUDA_PLATFORM`; the chosen upstream image has no ARM64 variant. Merely using this definition on an Apple Silicon Mac does not provide NVIDIA GPU access.
- Both definitions include M³'s MIT license and check installed dependency consistency. Upstream license files are included under `vendor/`.

The CPU runtime used Python 3.11.16 and PyTorch 2.8.0+cpu. Runs used four CPUs, a 4 GiB memory limit, and no container network access. Existing data was mounted read-only. The five-step training check used a separate output directory, `runs/docker-smoke-001`, and preserved the model architecture while reducing the step/evaluation/batch budget. Full smoke configuration and results are recorded in `reports/docker-verification.json`. It is a functional check, not an experiment score or throughput comparison. The real held-out test split was not evaluated.

## Repeat the essential CPU checks

From the repository root, with Docker Desktop running:

```sh
docker build -t mcubed:cpu .
docker run --rm --network none --cpus 4 --memory 4g mcubed:cpu
docker run --rm --network none --cpus 4 --memory 4g \
  --mount "type=bind,src=$PWD/tests,dst=/tests,readonly" \
  mcubed:cpu python -m unittest discover -s /tests -v
```

Tests exercise causal masking, next-token targets, split deduplication, interrupted/resumed training, checkpoint equivalence, data-tamper detection and synthetic held-out scoring. Tests are mounted for verification and are not bundled into the runtime image.

Use the README commands for a complete pilot in a new run directory. Mount data read-only and run outputs read/write. The runtime image has source hashes but no `.git` directory; its run metadata therefore does not infer a Git commit. Pair the run with its image digest and the source commit used to build it. Transitive Python packages are recorded per run; base-image pinning alone does not lock every dependency.

## Remaining NVIDIA verification

Static check (works on this Mac):

```sh
docker build --check -f Dockerfile.cuda .
```

On a Linux x86-64 NVIDIA host, build the full image, run the doctor with `--gpus all`, then run a bounded training/inference check using `--device cuda`. Require `cuda: true` and successful backward computation. Checkpoint generation and dependency installation have not been tested in that image. No paid host was provisioned for this verification.
