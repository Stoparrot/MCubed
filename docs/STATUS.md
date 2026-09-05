# M³ status

Updated: 2026-09-05. Initial setup in progress.

- Workspace: `/Users/uw/Desktop/MCubed`; remote: `https://github.com/Stoparrot/MCubed` (public).
- Mac: Apple Silicon, 24 GiB memory, macOS 26.6.2. Project Python 3.11.11, PyTorch 2.8.0. Native MPS forward/backward verified outside the Codex sandbox. CPU fallback verified.
- Data prepared: 20,000 train stories / 4,908,839 tokens; 500 validation stories / 106,549 tokens; 500 test stories / 107,916 tokens. Tokenizer vocabulary 2,048. Test data has not been evaluated.
- Four correctness tests pass, including exact CPU stop/resume parity and causal masking.
- Docker CLI installed; engine is stopped. CPU/CUDA container execution has not been verified.
- Model switched before the first pilot to a compact nanochat-derived GPT; historical nanoGPT remains reference-only. M³ license changed to MIT at the user's request.
- Pending: commit source, run bounded MPS pilot, reload/generate from checkpoint, record results.
- No cloud provider selected, no cloud spend authorized, no recurring job enabled, no browser dashboard implemented. These are staged requirements after the text baseline.
