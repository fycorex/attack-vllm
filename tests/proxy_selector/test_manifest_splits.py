import pytest

from proxy_selector.manifests import NegativeCandidate, assert_disjoint_image_ids, select_hard_negatives


def _candidate(image_id: str, objects: set[str]) -> NegativeCandidate:
    return NegativeCandidate(image_id, "object", "cat", frozenset(objects), frozenset({"what", "object"}))


def test_split_overlap_is_rejected() -> None:
    assert_disjoint_image_ids({"gallery": ["1"], "dev": ["2"], "test": ["3"]})
    with pytest.raises(ValueError, match="overlaps"):
        assert_disjoint_image_ids({"gallery": ["1"], "dev": ["1"]})


def test_hard_negative_ranking_is_deterministic_and_embedding_free() -> None:
    candidates = [_candidate("b", {"dog"}), _candidate("a", {"dog"}), _candidate("c", {"car"})]
    selected = select_hard_negatives(
        candidates,
        category="object",
        target_answer="horse",
        reference_coco_categories={"dog"},
        reference_vqa_tokens={"what"},
        excluded_image_ids=set(),
        count=2,
    )
    assert [candidate.image_id for candidate in selected] == ["a", "b"]
