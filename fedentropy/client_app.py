"""
client_app.py
-------------
Defines the Flower client used by every simulated client, regardless of
which strategy (FedAvg / FedEntOpt) the server is running. Each client
loads its own CIFAR-10 partition on demand via flwr-datasets (cached by
`task._get_fds`, so the underlying HuggingFace dataset is only pulled
once per process) -- there's no manual download step and no `./data`
folder involved.

The server-side strategy decides *which* clients train each round; this
file only defines *what a client does once selected*: local SGD for
`local_epochs` epochs.
"""
from typing import Dict, List, Tuple

import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from task import (
    build_resnet18_cifar,
    get_parameters,
    load_partition_loaders,
    set_parameters,
    test,
    train,
)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        partition: str,
        alpha: float,
        classes_per_client: int,
        seed: int,
    ):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.partition = partition
        self.alpha = alpha
        self.classes_per_client = classes_per_client
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_resnet18_cifar()
        self.trainloader = None  # lazily loaded on first use

    def _ensure_data(self):
        if self.trainloader is None:
            self.trainloader, _ = load_partition_loaders(
                self.partition_id,
                self.num_partitions,
                self.batch_size,
                partition=self.partition,
                alpha=self.alpha,
                classes_per_client=self.classes_per_client,
                seed=self.seed,
            )

    def get_properties(self, config: Dict) -> Dict:
        # Flower's simulation backend assigns each ClientProxy a random
        # opaque `cid`, so the server-side strategy cannot know in advance
        # which physical data partition a given proxy corresponds to. This
        # mirrors the one-time "upload your label counts" step of FedEntOpt:
        # the strategy asks each client once, via get_properties, which
        # partition it owns, and caches the answer for the rest of the run.
        return {"partition_id": str(self.partition_id)}

    def get_parameters(self, config: Dict) -> List:
        return get_parameters(self.model)

    def fit(self, parameters: List, config: Dict) -> Tuple[List, int, Dict]:
        self._ensure_data()
        set_parameters(self.model, parameters)
        avg_loss = train(
            self.model,
            self.trainloader,
            epochs=self.local_epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
        )
        return get_parameters(self.model), len(self.trainloader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters: List, config: Dict) -> Tuple[float, int, Dict]:
        # Centralized evaluation (server-side, on the held-out test set) is
        # used for all reported metrics, so client-side evaluate is unused
        # in this experiment but implemented for API completeness.
        self._ensure_data()
        set_parameters(self.model, parameters)
        loss, acc = test(self.model, self.trainloader, self.device)
        return loss, len(self.trainloader.dataset), {"accuracy": acc}


def make_client_fn(
    num_partitions: int,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    partition: str,
    alpha: float,
    classes_per_client: int,
    seed: int,
):
    """Build the `client_fn` closure that Flower calls once per client."""

    def client_fn(context: Context) -> Client:
        partition_id = int(context.node_config.get("partition-id", 0))
        return FlowerClient(
            partition_id,
            num_partitions,
            local_epochs,
            lr,
            weight_decay,
            batch_size,
            partition,
            alpha,
            classes_per_client,
            seed,
        ).to_client()

    return client_fn


def build_client_app(
    num_partitions: int,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    partition: str,
    alpha: float,
    classes_per_client: int,
    seed: int,
) -> ClientApp:
    return ClientApp(
        client_fn=make_client_fn(
            num_partitions,
            local_epochs,
            lr,
            weight_decay,
            batch_size,
            partition,
            alpha,
            classes_per_client,
            seed,
        )
    )
