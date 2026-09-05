# Model provenance

`model.py` is an M³ adaptation of the GPT in [Karpathy nanochat](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py), commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (July 3, 2026). Upstream's MIT license is retained at `vendor/nanochat/LICENSE`.

Retained: pre-normalized residual transformer blocks, rotary position encoding (base 10,000, negative-angle convention), RMS normalization, Q/K normalization, untied token and output embeddings, squared ReLU MLP, bias-free linear layers, normalized embeddings, zero residual-output initialization.

Simplified: joint QKV projection, standard multi-head full causal attention through PyTorch SDPA, FP32, RMS epsilon fixed at 1e-5, AdamW on all learned matrices, no KV cache, context-window regeneration during inference.

Omitted: Muon optimizer, grouped-query/sliding-window attention, explicit Flash Attention kernels, mixed precision, distributed training, value embeddings/gates, learned residual/embedding mixing, smear/backout, attention sharpening, logit soft-capping, vocabulary padding, meta-device initialization, chat post-training and task evaluation harness. These omissions are deliberate scope choices, not demonstrated improvements. No nanochat leaderboard performance is claimed.

The original nanoGPT model remains unchanged in `vendor/nanogpt/` only as a historical reference; runtime imports use `mcubed/model.py` exclusively. The architecture changed before the first pilot, following the user's request to find a current Karpathy example. Dataset, optimizer schedule and pilot budget remain fixed.
