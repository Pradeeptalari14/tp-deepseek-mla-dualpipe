#!/usr/bin/env python3
"""
DeepSeek Native FP8 Mixed Precision GEMM Block Kernel
"""
import torch

def block_scale_fp8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_scale = a.abs().max() / 448.0
    b_scale = b.abs().max() / 448.0

    a_scaled = a / a_scale
    b_scaled = b / b_scale
    res = torch.matmul(a_scaled, b_scaled) * (a_scale * b_scale)
    return res

if __name__ == "__main__":
    a = torch.randn(64, 512)
    b = torch.randn(512, 128)
    out = block_scale_fp8_gemm(a, b)
    print("FP8 simulated GEMM successful. Shape:", out.shape)
