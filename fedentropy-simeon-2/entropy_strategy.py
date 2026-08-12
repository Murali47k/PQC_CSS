"""
entropy_strategy.py
====================
A Flower Strategy that adds FedEntropy's filtering step on top of
standard FedAvg.

This is the main payoff of using Flower: we don't have to reimplement
client sampling, communication, retries, or weighted parameter
averaging. We only override `aggregate_fit` to decide *which* of the
round's client results get handed to FedAvg's (inherited, unmodified)
averaging logic.
"""

from typing import List, Tuple, Union

import numpy as np
from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from entropy_utils import select_clients_by_entropy


class FedEntropyStrategy(FedAvg):
    """
    Extends FedAvg with a soft-label-based client filter.

    Each client's `fit()` result includes a "soft_label" string in its
    metrics (see client.py). Before aggregating, we:
      1. Parse every client's soft label back into a probability vector.
      2. Run the greedy entropy-maximizing filter (entropy_utils.py) to
         decide which clients to keep.
      3. Hand only the kept clients' results to FedAvg.aggregate_fit,
         which does the actual weighted-average of model weights.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        if not results:
            return None, {}

        soft_labels = []
        weights = []
        for _, fit_res in results:
            soft_label_str = fit_res.metrics["soft_label"]
            soft_label = np.array([float(v) for v in soft_label_str.split(",")])
            soft_labels.append(soft_label)
            weights.append(fit_res.num_examples)

        keep_indices = select_clients_by_entropy(soft_labels, weights)
        kept_results = [results[i] for i in keep_indices]

        num_dropped = len(results) - len(kept_results)
        print(
            f"[Round {server_round}] FedEntropy filter: kept "
            f"{len(kept_results)}/{len(results)} clients "
            f"({num_dropped} dropped as likely to bias the aggregation)."
        )

        # Delegate the actual weighted-average aggregation to FedAvg --
        # this is the "let Flower do the heavy lifting" part.
        return super().aggregate_fit(server_round, kept_results, failures)
