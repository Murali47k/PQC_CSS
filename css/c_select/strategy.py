"""c_select: PQC-aware client-selection Strategy for Flower's ServerApp API.

`ScoredSelectionStrategy` extends FedAvg and only overrides *how nodes are
chosen* each round (`configure_train`) and *how their reported metrics are
remembered* (`aggregate_train`). Everything else (weight aggregation,
evaluation, etc.) is inherited unchanged from FedAvg, so results stay
directly comparable across scoring functions -- only the selection policy
changes between "our method" and each baseline.
"""

from collections.abc import Iterable
from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.fedavg import sample_nodes

from c_select.selection import SCORERS


class ScoredSelectionStrategy(FedAvg):
    """FedAvg with pluggable top-K client scoring instead of pure random sampling.

    Parameters
    ----------
    scorer_name : one of c_select.selection.SCORERS ("pqc_aware", "random",
        "data_size", "loss_based", "resource_aware", "trust_based").
    weights : optional override of the PQC-aware score's term weights.
    (all other args are forwarded to FedAvg, e.g. fraction_train, min_train_nodes)
    """

    def __init__(self, scorer_name: str = "pqc_aware", weights: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if scorer_name not in SCORERS:
            raise ValueError(f"Unknown scorer_name '{scorer_name}'. Options: {list(SCORERS)}")
        self.scorer_name = scorer_name
        self.scorer = SCORERS[scorer_name]
        self.weights = weights
        self.client_metrics: dict[int, dict] = {}  # node_id -> last reported metrics

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.fraction_train == 0.0:
            return []

        all_node_ids = list(grid.get_node_ids())
        num_nodes = max(int(len(all_node_ids) * self.fraction_train), self.min_train_nodes)

        known = {nid: m for nid, m in self.client_metrics.items() if nid in all_node_ids}
        if len(known) < num_nodes:
            # Not enough history yet (e.g. round 1): fall back to random sampling.
            node_ids, _ = sample_nodes(grid, self.min_available_nodes, num_nodes)
            log(INFO, "configure_train (round %s): insufficient metric history, random-sampled %s nodes",
                server_round, len(node_ids))
        else:
            kwargs = {"weights": self.weights} if self.scorer_name == "pqc_aware" else {}
            scores = self.scorer(known, **kwargs)
            node_ids = sorted(scores, key=scores.get, reverse=True)[:num_nodes]
            log(INFO, "configure_train (round %s): scorer=%s selected %s/%s nodes",
                server_round, self.scorer_name, len(node_ids), len(all_node_ids))

        config["server-round"] = server_round
        from flwr.app import RecordDict
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        for msg in replies:
            if msg.has_content():
                node_id = msg.metadata.src_node_id
                metrics = dict(msg.content["metrics"]) if "metrics" in msg.content else {}
                self.client_metrics[node_id] = metrics
        return super().aggregate_train(server_round, replies)