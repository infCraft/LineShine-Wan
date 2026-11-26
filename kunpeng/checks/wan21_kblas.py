#!/usr/bin/env python3
"""Exercise FP32 and BF16 GEMM for Wan2.1 KBLAS binding validation."""

import torch


for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
    left = torch.ones((256, 256), dtype=dtype)
    right = torch.ones((256, 256), dtype=dtype)
    result = left @ right
    expected = torch.full_like(result, 256)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    print(f"{label}_gemm_mean={float(result.float().mean())}")
print("WAN21_KBLAS_MATMUL_OK")
