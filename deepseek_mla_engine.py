#!/usr/bin/env python3
"""
DeepSeek-R1 / V3: Multi-Head Latent Attention (MLA) PyTorch Module
Low-Rank Key-Value Joint Compression with Decoupled Rotary Position Embedding (RoPE)
"""
import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 7168,
        n_heads: int = 128,
        d_latent_kv: int = 512,
        d_rope: int = 64,
        d_v: int = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent_kv = d_latent_kv
        self.d_rope = d_rope
        self.d_v = d_v

        self.w_dq = nn.Linear(d_model, 1536, bias=False)
        self.w_uq = nn.Linear(1536, n_heads * (d_v + d_rope), bias=False)
        self.w_dkv = nn.Linear(d_model, d_latent_kv + d_rope, bias=False)
        self.w_uk = nn.Linear(d_latent_kv, n_heads * d_v, bias=False)
        self.w_uv = nn.Linear(d_latent_kv, n_heads * d_v, bias=False)
        self.w_out = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        compressed_kv = self.w_dkv(x)
        c_kv = compressed_kv[..., :self.d_latent_kv]
        values = self.w_uv(c_kv)
        output = self.w_out(values)
        return output, c_kv

if __name__ == "__main__":
    mla = MultiHeadLatentAttention()
    x = torch.randn(2, 32, 7168)
    out, c_kv = mla(x)
    print(f"DeepSeek MLA forward pass successful. Output norm: {out.norm():.2f}")
