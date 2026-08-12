"""
entropy_utils.py
================
The actual "FedEntropy" idea, isolated from all the Flower/PyTorch
plumbing so it's easy to read and to unit-test on its own.

Background: under non-IID data, some clients' local models become
strongly biased toward the one or two classes they happen to hold. If the
server blindly averages every client's weights every round, those biased
updates drag the global model off balance. FedEntropy's fix: have each
client also report a "soft label" -- its average predicted class
distribution on its own data. Before aggregating, the server greedily
removes whichever clients' soft labels would make the *combined*
(weighted-average) distribution more skewed, keeping only the subset that
maximizes the combined distribution's entropy (i.e. is most balanced
across classes).
"""

import math
from typing import List, Sequence

import numpy as np


def shannon_entropy(distribution: Sequence[float], eps: float = 1e-12) -> float:
    """Shannon entropy (base 2, in bits) of a probability distribution."""
    return -sum(p * math.log2(p + eps) for p in distribution if p > eps)


def weighted_average_distribution(
    distributions: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    """Combines several probability distributions into one, weighted by `weights`."""
    weights = np.asarray(weights, dtype=np.float64)
    stacked = np.stack(distributions)
    return (stacked * weights[:, None]).sum(axis=0) / weights.sum()


def select_clients_by_entropy(
    soft_labels: List[np.ndarray], weights: List[float]
) -> List[int]:
    """
    Greedy backward-elimination client filtering.

    Starting from "keep everyone", repeatedly try removing each remaining
    client and check whether doing so would raise the entropy of the
    combined (weighted-average) soft-label distribution. If the best such
    removal improves entropy, actually remove that client and repeat;
    otherwise stop. This is the same greedy idea as `check_entropy_sub` in
    the original codebase, rewritten to be O(n^2) instead of doing full
    re-indexing tricks, and to read as ordinary Python.

    Args:
        soft_labels: one probability-distribution vector per client.
        weights: one weight (typically: number of local samples) per client.

    Returns:
        Indices (into `soft_labels`/`weights`) of the clients to KEEP.
    """
    keep = list(range(len(soft_labels)))

    def aggregate_entropy(indices: List[int]) -> float:
        combined = weighted_average_distribution(
            [soft_labels[i] for i in indices], [weights[i] for i in indices]
        )
        return shannon_entropy(combined)

    current_entropy = aggregate_entropy(keep)

    improved = True
    while improved and len(keep) > 1:
        improved = False
        best_entropy = current_entropy
        best_removal = None

        for idx in keep:
            trial = [i for i in keep if i != idx]
            trial_entropy = aggregate_entropy(trial)
            if trial_entropy > best_entropy:
                best_entropy = trial_entropy
                best_removal = idx

        if best_removal is not None:
            keep.remove(best_removal)
            current_entropy = best_entropy
            improved = True

    return keep
