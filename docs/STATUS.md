# M³ status

Updated: 2026-09-05. Initial local setup and pilot complete.

- Workspace: `/Users/uw/Desktop/MCubed`; remote: `https://github.com/Stoparrot/MCubed` (public).
- Mac: Apple Silicon, 24 GiB memory, macOS 26.6.2. Project Python 3.11.11, PyTorch 2.8.0. Native MPS forward/backward verified outside the Codex sandbox. CPU fallback verified.
- Data prepared: 20,000 train stories / 4,908,839 tokens; 500 validation stories / 106,549 tokens; 500 test stories / 107,916 tokens. Tokenizer vocabulary 2,048. Test data has not been evaluated.
- Five correctness tests pass, including exact CPU stop/resume parity and causal masking.
- Docker Desktop engine is running. CPU image built and verified on Linux ARM64: device check, all five tests, five-step training, checkpoint persistence and inference, including loading the Mac-trained checkpoint. CUDA definition passes build checks with its explicit x86-64 target; full CUDA build and NVIDIA execution remain unverified. Both base images are pinned. See `docs/DOCKER_VERIFICATION.md`.
- Model switched before the first pilot to a compact nanochat-derived GPT; historical nanoGPT remains reference-only. M³ license changed to MIT at the user's request.
- Pilot completed on MPS: 200 steps / 819,200 sampled tokens in 54.585 seconds. Validation loss 7.623897 → 4.930347. Total parameters 1,310,720; FP32 weights 5 MiB; resumable checkpoint 15,760,021 bytes.
- Source used: `d0d2b4260a9c2edea8c97da6c1708cd7164051d5`, clean working tree. Full local artifacts: `runs/pilot-001/`; portable aggregate report: `reports/pilot-001.json`.
- Checkpoint reload and generation verified on both CPU and MPS. Output contains recognizable words but is not coherent. Different generation lengths/devices were used; these are functional checks, not a comparative latency benchmark.
- Held-out evaluation code verified with synthetic uniform logits, including the final partial window. Real test stories remain untouched.
- Next: review the pilot and run `configs/baseline.json` as a new run (`runs/baseline-001`). Freeze prompts/rubric before language-quality or capacity comparisons.
- No cloud provider selected, no cloud spend authorized, no recurring job enabled, no browser dashboard implemented. These are staged requirements after the text baseline.
