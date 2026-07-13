from __future__ import annotations

import math
import random
from typing import Iterable


def average_ranks(values: Iterable[float]) -> list[float]:
    values = list(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def pearson(first: Iterable[float], second: Iterable[float]) -> float | None:
    first, second = list(first), list(second)
    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean, second_mean = sum(first) / len(first), sum(second) / len(second)
    first_centered = [value - first_mean for value in first]
    second_centered = [value - second_mean for value in second]
    denominator = math.sqrt(sum(value * value for value in first_centered) * sum(value * value for value in second_centered))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(first_centered, second_centered)) / denominator


def spearman(first: Iterable[float], second: Iterable[float]) -> float | None:
    first, second = list(first), list(second)
    return pearson(average_ranks(first), average_ranks(second))


def bootstrap_spearman(first: list[float], second: list[float], samples: int = 2000, seed: int = 0) -> dict:
    estimate = spearman(first, second)
    if estimate is None or len(first) < 3:
        return {"spearman": estimate, "ci95": [None, None], "pairs": len(first)}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        indices = [rng.randrange(len(first)) for _ in first]
        value = spearman([first[index] for index in indices], [second[index] for index in indices])
        if value is not None:
            draws.append(value)
    draws.sort()
    interval = [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]] if draws else [None, None]
    return {"spearman": estimate, "ci95": interval, "pairs": len(first)}


def clustered_bootstrap_spearman(first: list[float], second: list[float], cluster_ids: list[str],
                                 samples: int = 2000, seed: int = 0) -> dict:
    if not (len(first) == len(second) == len(cluster_ids)):
        raise ValueError("first, second, and cluster_ids must have equal lengths")
    estimate = spearman(first, second)
    clusters: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        clusters.setdefault(str(cluster_id), []).append(index)
    cluster_names = sorted(clusters)
    if estimate is None or len(cluster_names) < 3:
        return {"spearman": estimate, "ci95": [None, None], "pairs": len(first),
                "clusters": len(cluster_names)}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sampled_clusters = [rng.choice(cluster_names) for _ in cluster_names]
        indices = [index for name in sampled_clusters for index in clusters[name]]
        value = spearman([first[index] for index in indices], [second[index] for index in indices])
        if value is not None:
            draws.append(value)
    draws.sort()
    interval = [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]] if draws else [None, None]
    return {"spearman": estimate, "ci95": interval, "pairs": len(first),
            "clusters": len(cluster_names)}
