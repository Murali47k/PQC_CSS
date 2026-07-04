"""c_select: client scoring functions.

Each scorer takes `client_metrics`: dict[node_id -> dict[metric_name -> value]]
(the latest metrics each client reported) and returns dict[node_id -> score],
higher score = more likely to be selected. `strategy.py` sorts by this score
and keeps the top-K clients for the next round.

Includes the proposed PQC-aware scorer plus common baselines from the
literature so they can be benchmarked against each other with the exact
same Flower/task.py pipeline.
"""

from __future__ import annotations

import random


def _minmax_normalize(values: dict[int, float]) -> dict[int, float]:
    """Scale a metric to [0, 1] across currently-known clients."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def compute_pci(client_metrics: dict[int, dict]) -> dict[int, float]:
    """PQC Cost Index: a single normalized number summarizing crypto burden.

    Combines (normalized) KEM keygen/encaps/decaps time, sign/verify time,
    and bytes transmitted for PQC material. All sub-components are weighted
    equally by default -- this can be tuned/ablated in experiments.
    """
    keys = [
        "kem_keygen_ms",
        "kem_encaps_ms",
        "kem_decaps_ms",
        "dsa_sign_ms",
        "dsa_verify_ms",
        "pqc_total_bytes",
    ]
    node_ids = list(client_metrics.keys())
    normalized_per_key = {}
    for key in keys:
        raw = {nid: client_metrics[nid].get(key, 0.0) for nid in node_ids}
        normalized_per_key[key] = _minmax_normalize(raw)

    pci = {}
    for nid in node_ids:
        pci[nid] = sum(normalized_per_key[key][nid] for key in keys) / len(keys)
    return pci


# --------------------------------------------------------------------------
# Proposed method
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "utility": 0.35,
    "trust": 0.20,
    "reliability": 0.15,
    "pci": 0.15,
    "comm_cost": 0.10,
    "dropout": 0.05,
}


def score_pqc_aware(client_metrics: dict[int, dict], weights: dict | None = None) -> dict[int, float]:
    """Composite score = learning utility & trust minus PQC/communication/dropout cost.

    score = w_u*utility + w_t*trust + w_r*reliability
            - w_p*PCI - w_c*comm_cost - w_d*dropout_risk
    """
    weights = weights or DEFAULT_WEIGHTS
    node_ids = list(client_metrics.keys())

    utility = _minmax_normalize({nid: client_metrics[nid].get("data_size", 0.0) for nid in node_ids})
    trust = _minmax_normalize({nid: client_metrics[nid].get("trust", 1.0) for nid in node_ids})
    reliability = _minmax_normalize(
        {nid: 1.0 - client_metrics[nid].get("dropout_probability", 0.0) for nid in node_ids}
    )
    comm_cost = _minmax_normalize({nid: client_metrics[nid].get("model_bytes", 0.0) for nid in node_ids})
    dropout = _minmax_normalize({nid: client_metrics[nid].get("dropout_probability", 0.0) for nid in node_ids})
    pci = compute_pci(client_metrics)

    return {
        nid: (
            weights["utility"] * utility.get(nid, 0.0)
            + weights["trust"] * trust.get(nid, 0.0)
            + weights["reliability"] * reliability.get(nid, 0.0)
            - weights["pci"] * pci.get(nid, 0.0)
            - weights["comm_cost"] * comm_cost.get(nid, 0.0)
            - weights["dropout"] * dropout.get(nid, 0.0)
        )
        for nid in node_ids
    }


# --------------------------------------------------------------------------
# Baselines (for comparison, not PQC-aware)
# --------------------------------------------------------------------------

def score_random(client_metrics: dict[int, dict]) -> dict[int, float]:
    return {nid: random.random() for nid in client_metrics}


def score_data_size(client_metrics: dict[int, dict]) -> dict[int, float]:
    return {nid: m.get("data_size", 0.0) for nid, m in client_metrics.items()}


def score_loss_based(client_metrics: dict[int, dict]) -> dict[int, float]:
    """Higher local loss = selected first (assumed to have more to contribute)."""
    return {nid: m.get("train_loss", 0.0) for nid, m in client_metrics.items()}


def score_resource_aware(client_metrics: dict[int, dict]) -> dict[int, float]:
    cpu = _minmax_normalize({nid: m.get("cpu_score", 0.0) for nid, m in client_metrics.items()})
    bw = _minmax_normalize({nid: m.get("bandwidth_mbps", 0.0) for nid, m in client_metrics.items()})
    return {nid: cpu.get(nid, 0.0) * bw.get(nid, 0.0) for nid in client_metrics}


def score_trust_based(client_metrics: dict[int, dict]) -> dict[int, float]:
    return {nid: m.get("trust", 0.0) for nid, m in client_metrics.items()}


SCORERS = {
    "pqc_aware": score_pqc_aware,
    "random": score_random,
    "data_size": score_data_size,
    "loss_based": score_loss_based,
    "resource_aware": score_resource_aware,
    "trust_based": score_trust_based,
}