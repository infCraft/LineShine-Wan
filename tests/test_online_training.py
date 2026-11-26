"""Raw-input regression checks; all generated media lives in TemporaryDirectory."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.data.online_training import RawMedia
from src.data.raw_index import SynchronizedBucketBatchSampler
from src.video_io import write_video


class OnlineDataTests(unittest.TestCase):
    def make_source(self, root):
        Image.new("RGB", (48, 32), (255, 0, 0)).save(root / "image.png")
        rows = [{"path": "image.png", "type": "image", "caption": f"red image {i}",
                 "sample_id": str(i)} for i in range(24)]
        manifest = root / "images.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        source = RawMedia(manifest, image_size=32, image_bucket=None, video_frames=5,
                          video_fps=4.0, seed=2026)
        return source

    def test_raw_pixels_and_caption_without_latent_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            key, caption, pixels = source.read(("32/image_1", 3))
            self.assertEqual((key, caption), ("3", "red image 3"))
            self.assertEqual(tuple(pixels.shape), (3, 1, 32, 32))
            self.assertTrue(torch.equal(pixels[0], torch.ones_like(pixels[0])))
            self.assertTrue(torch.equal(pixels[1:], -torch.ones_like(pixels[1:])))
            self.assertEqual({p.name for p in root.iterdir()}, {"image.png", "images.jsonl"})

    def test_raw_video_uses_requested_frame_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = torch.linspace(-1, 1, 9)[None, :, None, None].expand(3, 9, 32, 32)
            write_video(root / "clip.mp4", video, fps=4)
            manifest = root / "videos.jsonl"
            manifest.write_text(json.dumps({"path": "clip.mp4", "type": "video", "caption": "grey clip",
                                            "clip_start_sec": 1}) + "\n")
            source = RawMedia(manifest, image_size=32, image_bucket=None, video_frames=5,
                              video_fps=4.0, seed=2026)
            _, caption, pixels = source.read(("32/video_5", 0))
            self.assertEqual(caption, "grey clip")
            self.assertEqual(tuple(pixels.shape), (3, 5, 32, 32))
            self.assertTrue(torch.isfinite(pixels).all())
            self.assertLess(abs(pixels[:, 0].mean().item()), 0.05)
            self.assertGreater(pixels[:, -1].mean().item(), 0.9)
            short = RawMedia(manifest, image_size=32, image_bucket=None, video_frames=13,
                             video_fps=4.0, seed=2026, training_task="t2v")
            with self.assertRaisesRegex(ValueError, "exceeds.*duration"):
                short.read(("32/video_13", 0))

    def test_openvid_path_caption_and_explicit_path_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "official"
            videos.mkdir()
            write_video(videos / "clip.mp4", torch.zeros(3, 9, 32, 32), fps=4)
            manifest = root / "openvid.jsonl"
            row = {"local_path": "/unavailable/clip.mp4", "video": "clip.mp4",
                   "caption_clean": "clean video caption", "raw_caption": "unused raw caption",
                   "sample_id": "clip_seg001", "clip_start_sec": 1, "ffprobe": {"unused": "metadata"}}
            manifest.write_text(json.dumps(row) + "\n")
            options = dict(image_size=32, image_bucket=None, video_frames=5, video_fps=4.0,
                           seed=2026, video_root=videos, training_task="t2v")
            source = RawMedia(manifest, **options)
            key, caption, pixels = source.read(("32/video_5", 0))
            self.assertEqual((key, caption), ("clip_seg001", "clean video caption"))
            self.assertEqual(tuple(pixels.shape), (3, 5, 32, 32))
            self.assertNotIn("ffprobe", source.rows["32/video_5"][0])
            row.update(path="official/clip.mp4", caption="explicit caption", video="missing.mp4")
            manifest.write_text(json.dumps(row) + "\n")
            source = RawMedia(manifest, **options)
            self.assertEqual(source.read(("32/video_5", 0))[1], "explicit caption")

    def test_streamed_manifest_keeps_existing_t2i_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "images.jsonl"
            raw = b'\n{"path":"image.png","caption":"caption","type":"image"}\r\n'
            manifest.write_bytes(raw)
            expected = hashlib.sha256(json.dumps(
                [hashlib.sha256(raw).hexdigest(), 32, None, 5, 4.0], sort_keys=True).encode()).hexdigest()
            with patch.object(Path, "read_bytes", side_effect=AssertionError("must stream JSONL")):
                source = RawMedia(manifest, image_size=32, image_bucket=None, video_frames=5,
                                  video_fps=4.0, seed=2026)
            self.assertEqual(source.catalog_fingerprint, expected)

    def test_t2v_rejects_images_and_invalid_video_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            options = dict(image_size=32, image_bucket=None, video_frames=5, video_fps=4.0,
                           seed=2026, training_task="t2v")
            for changes in ({"video_frames": 1}, {"video_frames": 6}, {"video_fps": 0},
                            {"video_fps": float("nan")}, {"image_size": 0}, {"image_size": 31}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    RawMedia(source.path, **dict(options, **changes))
            with self.assertRaisesRegex(ValueError, "unsupported raw media type"):
                RawMedia(source.path, **options)

    def test_video_resume_rejects_changed_root_and_frame_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "videos.jsonl"
            rows = [{"video": "clip.mp4", "caption_clean": "caption", "sample_id": str(i),
                     "clip_start_sec": i * 3} for i in range(24)]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            options = dict(image_size=32, image_bucket=None, video_frames=5, video_fps=4.0,
                           seed=2026, training_task="t2v", video_root=root)
            source = RawMedia(manifest, **options)
            sampler = SynchronizedBucketBatchSampler(source, batch_size=1, num_replicas=4, rank=2)
            sequence = list(sampler)
            sampler.advance(2)
            restored = SynchronizedBucketBatchSampler(source, batch_size=1, num_replicas=4, rank=2)
            restored.load_state_dict(sampler.state_dict())
            self.assertEqual(list(restored), sequence[2:])
            for changes in ({"video_root": root / "different"}, {"video_frames": 9}, {"video_fps": 8.0}):
                changed = RawMedia(manifest, **dict(options, **changes))
                other = SynchronizedBucketBatchSampler(changed, batch_size=1, num_replicas=4, rank=2)
                with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, "catalog_fingerprint"):
                    other.load_state_dict(sampler.state_dict())

    def test_missing_or_corrupt_video_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "videos.jsonl"
            manifest.write_text(json.dumps({"path": "bad.mp4", "caption": "caption"}) + "\n")
            source = RawMedia(manifest, image_size=32, image_bucket=None, video_frames=5,
                              video_fps=4.0, seed=2026, training_task="t2v")
            for payload in (None, b"not a video"):
                if payload is not None:
                    (root / "bad.mp4").write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises((OSError, RuntimeError, ValueError)):
                    source.read(("32/video_5", 0))

    def test_disjoint_rank_batches_and_resume_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            samplers = [SynchronizedBucketBatchSampler(source, batch_size=1, num_replicas=4,
                                                       rank=r, seed=2026) for r in range(4)]
            sequences = [list(sampler) for sampler in samplers]
            indices = [request[1] for sequence in sequences for batch in sequence for request in batch]
            self.assertEqual(sorted(indices), list(range(24)))
            samplers[2].advance(2)
            state = samplers[2].state_dict()
            restored = SynchronizedBucketBatchSampler(source, batch_size=1, num_replicas=4, rank=2, seed=2026)
            restored.load_state_dict(state)
            self.assertEqual(list(restored), sequences[2][2:])
            source.catalog_fingerprint = "changed"
            with self.assertRaisesRegex(RuntimeError, "catalog_fingerprint"):
                restored.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
