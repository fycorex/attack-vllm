"""Deterministic split validation and proxy-independent hard-negative ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class NegativeCandidate:
    image_id: str
    category: str
    canonical_answer: str
    coco_categories: frozenset[str]
    vqa_tokens: frozenset[str]


def assert_disjoint_image_ids(splits: Mapping[str, Iterable[str]]) -> None:
    """Raise with the first overlap; no split can reuse gallery/source/target IDs."""
    seen: dict[str, str] = {}
    for split, identifiers in splits.items():
        for image_id in identifiers:
            previous = seen.setdefault(str(image_id), split)
            if previous != split:
                raise ValueError(f"Image ID {image_id!r} overlaps {previous!r} and {split!r}.")


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def select_hard_negatives(
    candidates: Sequence[NegativeCandidate],
    *,
    category: str,
    target_answer: str,
    reference_coco_categories: Iterable[str],
    reference_vqa_tokens: Iterable[str],
    excluded_image_ids: Iterable[str],
    count: int = 8,
) -> list[NegativeCandidate]:
    """Select metadata-similar negatives without consulting proxy embeddings."""
    excluded = {str(item) for item in excluded_image_ids}
    reference_coco = frozenset(reference_coco_categories)
    reference_vqa = frozenset(reference_vqa_tokens)
    eligible = [
        candidate for candidate in candidates
        if candidate.category == category
        and candidate.canonical_answer != target_answer
        and candidate.image_id not in excluded
    ]
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -(_jaccard(candidate.coco_categories, reference_coco) + _jaccard(candidate.vqa_tokens, reference_vqa)),
            candidate.image_id,
        ),
    )
    if len(ranked) < count:
        raise ValueError(f"Need {count} hard negatives but only found {len(ranked)} eligible candidates.")
    return ranked[:count]
