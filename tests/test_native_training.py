"""Regression coverage for native-kernel selection and general training shapes."""
import os
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.train.modulation import modulate
from src.train.selective_ffn import recompute, native_supported


class NativeFallbackTests(unittest.TestCase):
    def test_batched_modulation_uses_every_sample_condition(self):
        x = torch.randn(2, 7, 8, requires_grad=True)
        offset = torch.randn(2, 1, 8, requires_grad=True)
        scale = torch.randn(2, 1, 8, requires_grad=True)
        expected = x * (1 + scale) + offset
        # A library path deliberately exists in the environment; batch two
        # must not reach the batch-one native ABI or load this path.
        with patch.dict(os.environ, {"WAN21_SVE_MODULATION_LIB": "/missing/native.so"}):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = modulate(x, offset, scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grads = torch.autograd.grad(actual.square().sum(), (x, offset, scale))
        refs = torch.autograd.grad(expected.square().sum(), (x, offset, scale))
        for grad, ref in zip(grads, refs):
            torch.testing.assert_close(grad, ref, rtol=0, atol=0)

    def test_ffn_fallback_matches_forward_and_backward(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = nn.Sequential(nn.Linear(8, 16), nn.GELU(approximate="tanh"),
                                      nn.Linear(16, 8)).to(dtype)
                x = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
                expected = model(x)
                model.forward = types.MethodType(recompute, model)
                with patch.dict(os.environ, {"WAN21_NATIVE_OPS": "1",
                                             "WAN21_XH_LINEAR_OP": "/missing/native.so",
                                             "WAN21_SVE_GELU_LIB": "/missing/gelu.so"}):
                    actual = model(x)
                params = (x, *model.parameters())
                grads = torch.autograd.grad(actual.float().square().sum(), params)
                refs = torch.autograd.grad(expected.float().square().sum(), params)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for grad, ref in zip(grads, refs):
                    torch.testing.assert_close(grad, ref, rtol=0, atol=0)

    def test_native_ffn_requires_video_shape_and_bf16_state(self):
        model = nn.Sequential(nn.Linear(1536, 8960), nn.GELU(),
                              nn.Linear(8960, 1536)).bfloat16()
        with patch.dict(os.environ, {"WAN21_NATIVE_OPS": "1",
                                     "WAN21_XH_LINEAR_OP": "/not-loaded.so",
                                     "WAN21_SVE_GELU_LIB": "/not-loaded.so"}):
            self.assertTrue(native_supported(model, torch.empty(1, 3328, 1536)))
            self.assertFalse(native_supported(model, torch.empty(1, 256, 1536)))
            self.assertFalse(native_supported(model, torch.empty(2, 3328, 1536)))
            model.float()
            self.assertFalse(native_supported(model, torch.empty(1, 3328, 1536)))


@unittest.skipUnless(os.environ.get("WAN21_TEST_NATIVE_KERNELS") == "1",
                     "requires the compute-node native kernel environment")
class NativeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(36)
        from src.train.xh_linear import load_native
        load_native()

    def assert_relative_error(self, actual, expected, limit=0.03):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
        self.assertLess(float(error), limit)

    def test_linear_forward_and_all_gradients(self):
        from src.train.xh_linear import xh_linear
        torch.manual_seed(42)
        x = (torch.randn(1, 512, 1536) * 0.1).bfloat16().requires_grad_()
        w = (torch.randn(1536, 1536) * 0.02).bfloat16().requires_grad_()
        bias = (torch.randn(1536) * 0.01).bfloat16().requires_grad_()
        actual = xh_linear(x, w, bias)
        reference = nn.functional.linear(x.float(), w.float(), bias.float())
        grad = (torch.randn_like(actual.float()) * 0.01).bfloat16()
        actual_grads = torch.autograd.grad(actual, (x, w, bias), grad)
        reference_grads = torch.autograd.grad(reference, (x, w, bias), grad.float())
        self.assert_relative_error(actual, reference)
        for value, expected in zip(actual_grads, reference_grads):
            self.assert_relative_error(value, expected)

    def test_native_cross_attention_forward_and_backward(self):
        from src.train.wan21_attention_native import attention
        torch.manual_seed(43)
        q = (torch.randn(1, 3328, 12, 128) * 0.1).bfloat16().requires_grad_()
        k = (torch.randn(1, 512, 12, 128) * 0.1).bfloat16().requires_grad_()
        v = (torch.randn_like(k.float()) * 0.1).bfloat16().requires_grad_()
        actual = attention(q, k, v)
        reference = nn.functional.scaled_dot_product_attention(
            q.float().transpose(1, 2), k.float().transpose(1, 2),
            v.float().transpose(1, 2)).transpose(1, 2)
        grad = (torch.randn_like(actual.float()) * 0.01).bfloat16()
        actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
        reference_grads = torch.autograd.grad(reference, (q, k, v), grad.float())
        self.assert_relative_error(actual, reference)
        for value, expected in zip(actual_grads, reference_grads):
            self.assert_relative_error(value, expected, limit=0.05)


if __name__ == "__main__":
    unittest.main()
