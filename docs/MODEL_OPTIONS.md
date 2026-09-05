# Karpathy model references — checked 2026-09-05

| Project | Purpose | Fit for M³ |
|---|---|---|
| [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) | February 12, 2026; roughly 200 lines of dependency-free scalar Python for training and inference | Best educational simplicity reference. Scalar autograd is much slower than batched tensor operations; use a clearly labeled PyTorch adaptation for TinyStories scale. |
| [nanochat](https://github.com/karpathy/nanochat) | Current nanoGPT successor; end-to-end LLM harness | Practical current model reference. A small subset needs explicit documentation of differences from upstream. |
| [autoresearch](https://github.com/karpathy/autoresearch) | Created March 6, 2026; automatic experiments on simplified nanochat | Newest relevant project by creation date. Upstream currently requires one NVIDIA GPU; use as a workflow reference later. |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Earlier compact GPT implementation | Marked deprecated upstream; retained locally as a historical baseline while choosing its replacement. |

nanochat master resolved to `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (commit dated July 3, 2026). autoresearch master resolved to `228791fb499afffb54b46200aca536f79142f117` (March 26, 2026). Repository push timestamps can differ from the latest default-branch commit, so they are not used as model release dates.

Recommendation: a compact nanochat-derived model for GPU training and microgpt as a guide to keep its code understandable. The current nanochat model also includes several experimental features; importing its whole harness would exceed this milestone's scope. Any simplified version must be named as an M³ adaptation, not an exact nanochat reproduction.

## MPS initialization

MPS stands for Metal Performance Shaders. PyTorch uses the `mps` device to execute tensor operations on an Apple GPU. Initialization means setting up device access, memory and the runtime's first-use resources. It is distinct from initializing model weights and does not train the model. The doctor already verified a simple forward/backward computation on this Mac's GPU.

Source: [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html).
