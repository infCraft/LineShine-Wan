from src.data.make_clip_segments import expand_rows, segment_count


def test_segment_count_uses_non_overlapping_3s_windows_with_cap():
    assert segment_count(2.99, clip_duration=3.0, max_segments=8, safety_margin=0.0) == 0
    assert segment_count(3.0, clip_duration=3.0, max_segments=8, safety_margin=0.0) == 1
    assert segment_count(6.0, clip_duration=3.0, max_segments=8, safety_margin=0.0) == 2
    assert segment_count(80.0, clip_duration=3.0, max_segments=8, safety_margin=0.0) == 8


def test_expand_rows_sets_unique_segment_sample_ids():
    rows = [
        {
            "sample_id": "sample_a",
            "video": "a.mp4",
            "seconds": 10.0,
            "caption_clean": "a test caption",
            "local_path": "/tmp/a.mp4",
        }
    ]

    segments, skipped, summary = expand_rows(
        rows,
        clip_duration=3.0,
        max_segments=8,
        safety_margin=0.0,
        split="train_stage1_segments",
    )

    assert skipped == []
    assert [row["sample_id"] for row in segments] == ["sample_a_seg000", "sample_a_seg001", "sample_a_seg002"]
    assert [row["clip_start_sec"] for row in segments] == [0.0, 3.0, 6.0]
    assert all(row["segment_source_sample_id"] == "sample_a" for row in segments)
    assert summary["segment_rows"] == 3
