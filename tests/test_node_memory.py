"""Memory observations remain explicit about missing PSS and root cgroups."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.train.node_memory import NodeMemoryMonitor, memory_snapshot


class NodeMemoryTests(unittest.TestCase):
    def test_real_process_sampling_and_monitor_shutdown(self):
        record = memory_snapshot([os.getpid()])
        self.assertGreater(record["node_kib"]["MemTotal"], 0)
        self.assertEqual(record["rank_pss_sum_kib"], record["rank_pss_kib"][str(os.getpid())])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            with NodeMemoryMonitor(path, [os.getpid()], min(os.sched_getaffinity(0))) as monitor:
                pass
            self.assertFalse(monitor.thread.is_alive())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("MemAvailable", rows[0]["node_kib"])

    def test_unreadable_pss_is_unknown_instead_of_partial_sum(self):
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == "smaps_rollup":
                raise PermissionError("test inaccessible PSS")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            record = memory_snapshot([os.getpid()])
        self.assertIsNone(record["rank_pss_sum_kib"])
        self.assertEqual(record["pss_errors"][str(os.getpid())], "PermissionError")


if __name__ == "__main__":
    unittest.main()
