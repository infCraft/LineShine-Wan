"""Exercise the real shell entrypoints with an isolated scheduler stub."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class OnlineSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.scripts = Path(__file__).resolve().parents[1] / "kunpeng"
        stub = self.root / "dsub"
        stub.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        stub.chmod(0o755)
        for name, kind, path in (("t2v", "video", "clips/a.mp4"), ("t2i", "image", "images/a.jpg")):
            (self.root / f"{name}.jsonl").write_text(json.dumps(dict(type=kind, path=path, caption="sample")) + "\n")
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith("WAN21_")}
        self.env.update(WAN21_DSUB=str(stub), WAN21_QUEUE="example",
                        WAN21_ENCODER_WEIGHTS=str(self.root), WAN21_BLAS_CHECK=str(stub),
                        WAN21_ENV_ACTIVATE=str(self.root / "t2v.jsonl"))

    def submit(self, task, overrides=None, mode="scratch", data=None):
        return subprocess.run([str(self.scripts / f"submit_wan21_{task}_multinode.sh"),
                               data or f"{task}.jsonl", "run", "3", mode], cwd=self.root,
                              env=dict(self.env, **(overrides or {})), capture_output=True, text=True)

    def test_required_site_settings(self):
        for key in ("WAN21_ENV_ACTIVATE", "WAN21_QUEUE", "WAN21_BLAS_CHECK", "WAN21_ENCODER_WEIGHTS"):
            result = self.submit("t2i", {key: ""})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(key, result.stderr)
            self.assertFalse(result.stdout)

    def test_resume_and_default_video_manifest(self):
        manifest = self.root / "t2v.jsonl"
        self.assertNotEqual(self.submit("t2v", data="default").returncode, 0)
        self.assertNotEqual(self.submit("t2v", mode="resume").returncode, 0)
        step = self.root / "run/checkpoints/step_00000002"
        step.mkdir(parents=True)
        (step / "COMPLETED").touch()
        self.assertNotEqual(self.submit("t2v").returncode, 0)
        result = self.submit("t2v", {"WAN21_VIDEO_MANIFEST": str(manifest)}, "resume", "default")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertIn("WAN21_TRAIN_START_MODE=resume", args)
        self.assertIn(f"WAN21_TRAIN_DATA={manifest}", args)

    def test_video_and_image_resource_and_precision_defaults(self):
        for task, ranks, per_node, memory in (("t2v", "16", "4", "108"), ("t2i", "64", "16", "27")):
            with self.subTest(task=task):
                result = self.submit(task)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)
                self.assertEqual(args[args.index("-q") + 1], "example")
                self.assertIn(f"WAN21_TRAIN_DATA={self.root / (task + '.jsonl')}", args)
                self.assertIn(f"WAN21_TRAIN_RUN_DIR={self.root / 'run'}", args)
                self.assertIn(f"WAN21_BLAS_CHECK={self.env['WAN21_BLAS_CHECK']}", args)
                self.assertTrue(Path(args[-1]).is_file())
                for flag, expected in (("-N", ranks), ("-rpn", per_node), ("-nn", "4"),
                                       ("-R", f"cpu=38;mem={memory}G"), ("--mpi", "hmpi")):
                    self.assertEqual(args[args.index(flag) + 1], expected)
                for setting in ("WAN21_TRAIN_PRECISION=bf16", "WAN21_VIDEO_FRAMES=49",
                                "WAN21_VIDEO_FPS=16", f"WAN21_TRAIN_TASK={task}",
                                "WAN21_TRAIN_BATCH_SIZE=1", "WAN21_TRAIN_GRAD_ACCUM=1"):
                    self.assertIn(setting, args)
                self.assertIn("WAN21_STORED_PRECISION=fp32", args)
                self.assertIn("WAN21_TEXT_PRECISION=fp32", args)
                self.assertIn("WAN21_NATIVE_OPS=1", args)
                self.assertFalse(any(value.startswith("WAN21_CACHE_ROOT=") for value in args))

    def test_bf16_hierarchical_mode_and_online_only_input(self):
        result = self.submit("t2v", {"WAN21_PACK_RPN": "16", "WAN21_STORED_PRECISION": "bf16"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WAN21_HIERARCHICAL_COMM=1", json.loads(result.stdout))
        for overrides in ({"WAN21_CACHE_ROOT": "/unused/cache"},
                          {"WAN21_STORED_PRECISION": "bf16", "WAN21_TRAIN_PRECISION": "fp32"},
                          {"WAN21_PACK_RPN": "4", "WAN21_HIERARCHICAL_COMM": "1"}):
            result = self.submit("t2v", overrides)
            self.assertNotEqual(result.returncode, 0)

    def test_rank_override_keeps_node_memory_budget(self):
        result = self.submit("t2v", {"WAN21_PACK_RPN": "8"})
        args = json.loads(result.stdout)
        self.assertEqual(args[args.index("-R") + 1], "cpu=38;mem=54G")
        self.assertEqual(args[args.index("-N") + 1], "32")

    def test_hbm_requests_full_node_without_ddr_numa_restriction(self):
        result = self.submit("t2v", {"WAN21_STORED_PRECISION": "bf16"})
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args[args.index("-N") + 1], "4")
        self.assertEqual(args[args.index("-R") + 1], "cpu=608;mem=400G")
        self.assertNotIn("-a", args)
        for override in ({"WAN21_FFN_HBM": "0"}, {"WAN21_VIDEO_FRAMES": "5"},
                         {"WAN21_NATIVE_OPS": "0"}, {"WAN21_TRAIN_BATCH_SIZE": "2"}):
            result = self.submit("t2v", {"WAN21_STORED_PRECISION": "bf16", **override})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("-a", json.loads(result.stdout))

    def test_invalid_video_shape_is_rejected_before_submission(self):
        for setting in ({"WAN21_VIDEO_FRAMES": "1"}, {"WAN21_VIDEO_FRAMES": "48"},
                        {"WAN21_VIDEO_FPS": "nan"}, {"WAN21_VIDEO_FPS": "0"},
                        {"WAN21_IMAGE_SIZE": "0"}, {"WAN21_PACK_RPN": "2"}):
            with self.subTest(setting=setting):
                result = self.submit("t2v", setting)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(result.stdout)


if __name__ == "__main__":
    unittest.main()
