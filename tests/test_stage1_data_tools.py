from collections import Counter

from src.data.audit_cache_bucket import visual_token_count
from src.data.stage1_expand import select_extra_rows


def test_visual_token_count_matches_wan_single_bucket():
    assert visual_token_count((16, 13, 32, 32), (1, 2, 2)) == 3328
    assert visual_token_count((16, 13, 31, 32), (1, 2, 2)) is None


def test_select_extra_rows_excludes_samples_and_sources():
    candidates = [
        {"sample_id": "a", "video": "a.mp4", "source_id": "s1", "quality_score": 1.0},
        {"sample_id": "b", "video": "b.mp4", "source_id": "s2", "quality_score": 0.9},
        {"sample_id": "c", "video": "c.mp4", "source_id": "s3", "quality_score": 0.8},
    ]

    rows, skipped = select_extra_rows(
        candidates,
        exclude_sample_ids={"a"},
        exclude_source_ids={"s2"},
        split="train_stage1_extra",
        max_count=None,
    )

    assert [row["sample_id"] for row in rows] == ["c"]
    assert rows[0]["split"] == "train_stage1_extra"
    assert skipped == Counter({"excluded_sample_id": 1, "excluded_source_id": 1})


def test_select_extra_rows_honors_max_count():
    candidates = [
        {"sample_id": "a", "video": "a.mp4", "source_id": "s1"},
        {"sample_id": "b", "video": "b.mp4", "source_id": "s2"},
        {"sample_id": "c", "video": "c.mp4", "source_id": "s3"},
    ]

    rows, skipped = select_extra_rows(
        candidates,
        exclude_sample_ids=set(),
        exclude_source_ids=set(),
        split="train_stage1_extra",
        max_count=2,
    )

    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert skipped == Counter()
