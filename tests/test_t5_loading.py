"""Check meta loading, requested precision and padding removal on a tiny T5."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.train.model_factory import add_wan_to_path

add_wan_to_path()
from wan.modules import t5


def tiny_encoder(**kwargs):
    with torch.device(kwargs["device"]):
        model = t5.T5Encoder(vocab=32, dim=16, dim_attn=16, dim_ffn=32,
                            num_heads=4, num_layers=1, num_buckets=8, dropout=0.0)
        return model.to(dtype=kwargs["dtype"])


class Tokenizer:
    def __init__(self, **kwargs):
        pass

    def __call__(self, texts, **kwargs):
        ids = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0]])
        return ids, ids.ne(0).long()


class T5LoadingTests(unittest.TestCase):
    def test_requested_dtype_and_unpadded_outputs(self):
        torch.manual_seed(41)
        source = tiny_encoder(device="cpu", dtype=torch.bfloat16).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.pt"
            torch.save(source.state_dict(), checkpoint)
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(dtype=dtype), patch.object(t5, "umt5_xxl", tiny_encoder), \
                        patch.object(t5, "HuggingfaceTokenizer", Tokenizer):
                    encoder = t5.T5EncoderModel(text_len=6, dtype=dtype,
                                               device=torch.device("cpu"),
                                               checkpoint_path=str(checkpoint), tokenizer_path="unused")
                    self.assertTrue(all(p.dtype == dtype and not p.is_meta and not p.requires_grad
                                        for p in encoder.model.parameters()))
                    ids, mask = encoder.tokenizer(["a", "b"])
                    with torch.no_grad():
                        reference = encoder.model(ids, mask)
                        actual = encoder(["a", "b"], torch.device("cpu"))
                    for i, length in enumerate((3, 2)):
                        self.assertEqual(tuple(actual[i].shape), (length, 16))
                        torch.testing.assert_close(actual[i], reference[i, :length],
                                                   rtol=1e-4 if dtype == torch.float32 else 0.02,
                                                   atol=1e-5 if dtype == torch.float32 else 0.02)


if __name__ == "__main__":
    unittest.main()
