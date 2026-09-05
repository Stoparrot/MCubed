"""Small nanochat-derived GPT. See MODEL_PROVENANCE.md for retained/omitted features.

Adapted by M³ contributors from Karpathy nanochat (MIT), pinned July 2026 source.
FP32, full causal attention, ordinary AdamW; no external kernels or platform code.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


def norm(x):
    # Explicit epsilon keeps the same FP32 definition across backends.
    return F.rms_norm(x, (x.size(-1),), eps=1e-5)


def rotate(x, cos, sin):
    half = x.size(-1) // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos + b * sin, -a * sin + b * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = [t.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
                   for t in self.qkv(x).chunk(3, dim=-1)]
        q, k = norm(rotate(q, cos, sin)), norm(rotate(k, cos, sin))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = Attention(c)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(norm(x), cos, sin)
        return x + self.proj(F.relu(self.fc(norm(x))).square())


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        if c.n_embd % c.n_head or (c.n_embd // c.n_head) % 2:
            raise ValueError('Heads must divide width and head dimension must be even for RoPE.')
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        dim = c.n_embd // c.n_head
        frequencies = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = torch.outer(torch.arange(c.block_size, dtype=torch.float32), frequencies)
        self.register_buffer('cos', angles.cos()[None, None], persistent=False)
        self.register_buffer('sin', angles.sin()[None, None], persistent=False)
        # nanochat-style initialization: residual branch outputs initially zero.
        nn.init.normal_(self.embedding.weight, std=0.8)
        nn.init.normal_(self.lm_head.weight, std=0.001)
        bound = math.sqrt(3 / c.n_embd)
        for block in self.blocks:
            nn.init.uniform_(block.attn.qkv.weight, -bound, bound)
            nn.init.uniform_(block.fc.weight, -bound, bound)
            nn.init.zeros_(block.attn.proj.weight)
            nn.init.zeros_(block.proj.weight)

    def forward(self, ids, targets=None):
        length = ids.size(1)
        if not 0 < length <= self.config.block_size:
            raise ValueError('Input length must be positive and within block_size.')
        x = norm(self.embedding(ids))
        for block in self.blocks:
            x = block(x, self.cos[:, :, :length], self.sin[:, :, :length])
        x = norm(x)
        logits = self.lm_head(x if targets is not None else x[:, [-1], :])
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas,
                                 weight_decay=weight_decay)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.config.block_size:])
            logits = logits[:, -1] / temperature
            if top_k is not None:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float('-inf'))
            token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, token), dim=1)
        return ids
