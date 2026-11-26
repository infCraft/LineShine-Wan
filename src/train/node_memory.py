"""Optional node memory observations; RSS is never summed across shared ranks."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path


def read_kib_fields(path, names):
    fields = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        if key in names:
            fields[key] = int(value.split()[0])
    return fields


def cgroup_memory(pid):
    records = []
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if controllers and "memory" not in controllers.split(","):
            continue
        # A root cgroup is a node-wide observation, not a job memory limit.
        root = Path("/sys/fs/cgroup") / ("memory" if controllers else "")
        path = root / relative.lstrip("/")
        names = (("memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes")
                 if controllers else ("memory.current", "memory.peak", "memory.max", "memory.events"))
        record = {"path": str(path), "scope": "root" if relative == "/" else "process_cgroup"}
        for name in names:
            try:
                record[name] = (path / name).read_text().strip()
            except OSError as error:
                record[name] = None
                record.setdefault("read_errors", {})[name] = type(error).__name__
        records.append(record)
    return records


def memory_snapshot(pids):
    result = {"time_unix": time.time(), "hostname": socket.gethostname(),
              "node_kib": read_kib_fields(Path("/proc/meminfo"),
                                          {"MemTotal", "MemAvailable", "MemFree", "Cached", "Shmem"}),
              "rank_pss_kib": {}, "cgroups": []}
    seen_cgroups = set()
    for pid in pids:
        try:
            result["rank_pss_kib"][str(pid)] = read_kib_fields(
                Path(f"/proc/{pid}/smaps_rollup"), {"Pss"})["Pss"]
        except (OSError, KeyError) as error:
            result["rank_pss_kib"][str(pid)] = None
            result.setdefault("pss_errors", {})[str(pid)] = type(error).__name__
        try:
            for record in cgroup_memory(pid):
                if record["path"] not in seen_cgroups:
                    seen_cgroups.add(record["path"])
                    result["cgroups"].append(record)
        except OSError as error:
            result.setdefault("cgroup_errors", {})[str(pid)] = type(error).__name__
    values = list(result["rank_pss_kib"].values())
    result["rank_pss_sum_kib"] = sum(values) if all(v is not None for v in values) else None
    return result


class NodeMemoryMonitor:
    """One sampler per node; process PSS excludes unlisted child processes."""
    def __init__(self, output, pids, cpu, interval=5.0):
        self.output, self.pids, self.cpu, self.interval = output, pids, cpu, interval
        self.stop = threading.Event()
        self.error = None

    def __enter__(self):
        self.handle = self.output.open("a", buffering=1)
        self.thread = threading.Thread(target=self._run, name="node-memory", daemon=True)
        self.thread.start()
        return self

    def _run(self):
        try:
            os.sched_setaffinity(0, {self.cpu})
            while True:
                self.handle.write(json.dumps(memory_snapshot(self.pids), sort_keys=True) + "\n")
                if self.stop.wait(self.interval):
                    break
        except Exception as error:
            self.error = error
            print(f"NODE_MEMORY_MONITOR_FAILED {type(error).__name__}: {error}", flush=True)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.thread.join()
        self.handle.close()
        if self.error is not None and exc_type is None:
            raise RuntimeError("node memory monitor failed") from self.error
