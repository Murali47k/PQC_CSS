"""
strategy.py
-----------
FedEntOptStrategy: a Flower `Strategy` (subclassing FedAvg) that implements
Algorithm 1 of the FedEntOpt paper.

In the paper, every client uploads its label-count vector l^(k) to the
server exactly once, before training starts, at negligible communication
cost. In this simulation, we mirror that behaviour by computing every
client's label-count vector directly from the partitioning we ourselves
generated (see task.compute_label_counts) and handing it to the strategy at
construction time -- this is equivalent to "collect all label vectors in a
setup message" but skips implementing that extra message round-trip.

Only `configure_fit` is overridden: it runs the entropy-maximizing greedy
selection with a FIFO exclusion buffer (Algorithm 1) instead of FedAvg's
uniform random sampling. Aggregation (`aggregate_fit`) is left untouched, so
FedEntOpt still performs standard size-weighted FedAvg aggregation over
whichever clients it selected -- exactly as described in the paper.
"""
from typing import Dict, List

import numpy as np
from flwr.common import FitIns, GetPropertiesIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


class FedEntOptStrategy(FedAvg):
    """Entropy-based client selection (Algorithm 1) + FedAvg aggregation."""

    def __init__(
        self,
        label_counts: Dict[str, np.ndarray],
        num_classes: int,
        clients_per_round: int,
        buffer_frac: float = 0.5,
        selection_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_counts = label_counts
        self.num_classes = num_classes
        self.M = clients_per_round
        self.all_cids: List[str] = list(label_counts.keys())
        K = len(self.all_cids)

        # 0 < Q <= K - M, matching the paper's constraint on buffer size.
        q = max(1, int(round(buffer_frac * K)))
        self.Q = min(q, max(1, K - self.M))

        self.buffer: List[str] = []  # FIFO exclusion buffer B (holds partition ids)
        self.rng = np.random.default_rng(selection_seed)

        # Flower's simulation backend hands out an opaque random `cid` per
        # ClientProxy that has no relation to our partition ids, so we
        # resolve cid -> partition_id once (via get_properties) and cache it.
        self._cid_to_pid: Dict[str, str] = {}

    def _resolve_partition_ids(self, proxies: Dict[str, ClientProxy]) -> Dict[str, str]:
        unresolved = [cid for cid in proxies if cid not in self._cid_to_pid]
        for cid in unresolved:
            res = proxies[cid].get_properties(
                GetPropertiesIns(config={}), timeout=None, group_id=None
            )
            self._cid_to_pid[cid] = str(res.properties.get("partition_id", cid))
        return self._cid_to_pid

    def _select_clients(self, available_cids: List[str]) -> List[str]:
        """Algorithm 1 from the paper."""
        pool = set(available_cids)
        # Keep only buffered clients that are actually connected this round.
        self.buffer = [c for c in self.buffer if c in pool]

        L = np.zeros(self.num_classes)
        selected: List[str] = []

        candidates = [c for c in available_cids if c not in self.buffer]
        if not candidates:
            candidates = list(available_cids)

        # i = 1: uniform random first pick
        first = str(self.rng.choice(candidates))
        selected.append(first)
        L += self.label_counts.get(first, np.zeros(self.num_classes))
        self.buffer.append(first)
        while len(self.buffer) > self.Q:
            self.buffer.pop(0)

        # i = 2 .. M: greedy entropy maximization
        for _ in range(1, min(self.M, len(available_cids))):
            candidates = [
                c for c in available_cids if c not in self.buffer and c not in selected
            ]
            if not candidates:
                candidates = [c for c in available_cids if c not in selected]
            if not candidates:
                break

            best_c, best_h = None, -1.0
            for c in candidates:
                cand_vec = L + self.label_counts.get(c, np.zeros(self.num_classes))
                total = cand_vec.sum()
                h = _entropy_bits(cand_vec / total) if total > 0 else 0.0
                if h > best_h:
                    best_h, best_c = h, c

            selected.append(best_c)
            L += self.label_counts.get(best_c, np.zeros(self.num_classes))
            self.buffer.append(best_c)
            while len(self.buffer) > self.Q:
                self.buffer.pop(0)

        return selected

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        # Make sure every simulated client has registered with the manager.
        client_manager.wait_for(num_clients=len(self.all_cids))
        all_proxies: Dict[str, ClientProxy] = client_manager.all()

        cid_to_pid = self._resolve_partition_ids(all_proxies)
        pid_to_proxy = {cid_to_pid[cid]: proxy for cid, proxy in all_proxies.items()}
        available_pids = [pid for pid in self.all_cids if pid in pid_to_proxy]

        selected_pids = self._select_clients(available_pids)

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        return [(pid_to_proxy[pid], fit_ins) for pid in selected_pids]


def weighted_average(metrics: List[tuple]) -> Dict[str, float]:
    """Aggregate client-reported metrics, weighted by number of examples."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    out: Dict[str, float] = {}
    keys = metrics[0][1].keys() if metrics else []
    for k in keys:
        out[k] = sum(n * m[k] for n, m in metrics) / total
    return out
