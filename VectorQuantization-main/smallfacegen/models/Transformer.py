"""
GPT-style decoder-only Transformer for learning the prior distribution
over the discrete codebook indices produced by the VQ encoder.

Usage:
  - Input : (B, T) LongTensor of codebook indices (flattened spatial map)
  - Output: (B, T, vocab_size) logits for next-token prediction
  - At inference, call .generate() autoregressively to sample new sequences,
    then reshape to (H, W) and pass through the VQ decoder to get images.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, n_embd: int, n_head: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Fixed causal mask (lower-triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, seq_len, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # project and split into q, k, v
        q, k, v = self.qkv_proj(x).split(C, dim=-1)                        # each (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)        # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale                       # (B, nh, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)         # (B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward (GPT-style)."""

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, n_embd: int, n_head: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, seq_len, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ff   = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ── Main model ─────────────────────────────────────────────────────────────────

class GPTDecoder(nn.Module):
    """
    Decoder-only GPT for learning a prior over VQ-VAE discrete codes.

    Args:
        vocab_size : codebook size (n_e in VQVAE config)
        seq_len    : total number of tokens per image  (H_lat * W_lat)
        n_embd     : embedding / hidden dimension
        n_head     : number of attention heads
        n_layer    : number of transformer blocks
        dropout    : dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.vocab_size = vocab_size

        self.tok_emb  = nn.Embedding(vocab_size, n_embd)
        self.pos_emb  = nn.Embedding(seq_len, n_embd)
        self.drop     = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList([
            TransformerBlock(n_embd, n_head, seq_len, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f     = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()
        print(f"GPTDecoder: {self._count_params() / 1e6:.2f}M parameters")

    # ── init ──────────────────────────────────────────────────────────────────

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) LongTensor of token indices, T <= seq_len
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.seq_len, f"Sequence too long: {T} > {self.seq_len}"

        positions = torch.arange(T, device=idx.device).unsqueeze(0)         # (1, T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))          # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)                                               # (B, T, vocab_size)

    # ── autoregressive sampling ────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,            # (B, T_prompt) starting tokens, can be empty
        n_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample n_new_tokens tokens.

        Returns:
            (B, T_prompt + n_new_tokens) LongTensor
        """
        self.eval()
        idx = prompt.clone()
        for _ in range(n_new_tokens):
            # crop context to the last seq_len tokens
            idx_ctx = idx[:, -self.seq_len:]
            logits  = self(idx_ctx)             # (B, T, vocab_size)
            logits  = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold      = top_values[:, -1].unsqueeze(-1)
                logits         = logits.masked_fill(logits < threshold, float("-inf"))

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)            # (B, 1)
            idx        = torch.cat([idx, next_token], dim=1)

        return idx


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 128×128 image → 4× downsampled → 32×32 latent → 1024 tokens
    SEQ_LEN = 32 * 32
    model = GPTDecoder(
        vocab_size=512,
        seq_len=SEQ_LEN,
        n_embd=256,
        n_head=8,
        n_layer=6,
        dropout=0.1,
    )
    idx = torch.randint(0, 512, (2, SEQ_LEN))
    logits = model(idx)
    print("Forward logits shape:", logits.shape)    # (2, 1024, 512)

    prompt    = torch.zeros(1, 1, dtype=torch.long)
    generated = model.generate(prompt, n_new_tokens=SEQ_LEN - 1, top_k=100)
    print("Generated sequence shape:", generated.shape)  # (1, 1024)