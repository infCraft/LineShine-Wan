import json
import zipfile
from pathlib import Path

from src.data.shared_openvid import manifest_candidates, scan_shared_parts, select_smoke_rows


def test_scan_shared_parts_ignores_hidden_and_temp(tmp_path: Path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "OpenVid_part1.zip").write_bytes(b"zip")
    (shared / "OpenVid_part2_partaa").write_bytes(b"a")
    (shared / "OpenVid_part2_partab").write_bytes(b"b")
    (shared / ".OpenVid_part3.zip").write_bytes(b"hidden")
    temp = shared / "._____temp"
    temp.mkdir()
    (temp / "OpenVid_part4.zip").write_bytes(b"temp")

    parts = scan_shared_parts(shared)

    assert sorted(parts) == ["OpenVid_part1.zip", "OpenVid_part2.zip"]
    assert parts["OpenVid_part1.zip"]["format"] == "zip"
    assert parts["OpenVid_part2.zip"]["format"] == "split"


def test_manifest_candidates_and_smoke_selection(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"video": "a.mp4", "part": "OpenVid_part1.zip", "source_id": "s1", "quality_score": 0.1},
        {"video": "b.mp4", "part": "OpenVid_part1.zip", "source_id": "s2", "quality_score": 0.9},
        {"video": "c.mp4", "part": "OpenVid_part9.zip", "source_id": "s3", "quality_score": 1.0},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    shared = {"OpenVid_part1.zip": {"part": "OpenVid_part1.zip", "part_num": 1, "format": "zip", "bytes": 3, "files": []}}

    candidates = manifest_candidates(manifest, shared)
    smoke = select_smoke_rows(candidates, part="OpenVid_part1.zip", limit=1, require_unique_source=True)

    assert [r["video"] for r in candidates] == ["b.mp4", "a.mp4"]
    assert smoke[0]["video"] == "b.mp4"

