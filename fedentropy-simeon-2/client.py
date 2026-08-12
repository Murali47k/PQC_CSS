"""
client.py
=========
The Flower client: represents a single federated learning participant.

Each round, a client:
  1. Receives the current global model weights from the server.
  2. Trains locally on its own private data partition for a few epochs.
  3. Computes a "soft label" summarizing its local predictions.
  4. Sends back its updated weights, its number of local samples, and the
     soft label (via the `metrics` dict, encoded as a comma-separated
     string -- Flower's metrics values must be simple scalars, so we
     serialize the vector to a string and parse it back out on the
     server side, see entropy_strategy.py).

Note what is NOT here: any FedEntropy-specific logic. The client doesn't
know or care whether it will be filtered out -- it just trains normally
and reports honestly. All the filtering happens on the server.
"""

from torch.utils.data import DataLoader
from flwr.client import NumPyClient

from common import get_weights, set_weights, train, compute_soft_label
from model import SimpleCNN


class FedEntropyClient(NumPyClient):
    def __init__(
        self,
        train_loader: DataLoader,
        num_classes: int = 10,
        local_epochs: int = 1,
        lr: float = 0.05,
        device: str = "cpu",
    ):
        self.net = SimpleCNN(num_classes=num_classes)
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device

    def get_parameters(self, config):
        return get_weights(self.net)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.train_loader, self.local_epochs, self.lr, self.device)

        soft_label = compute_soft_label(self.net, self.train_loader, self.num_classes, self.device)
        metrics = {"soft_label": ",".join(f"{p:.6f}" for p in soft_label)}

        return get_weights(self.net), len(self.train_loader.dataset), metrics
