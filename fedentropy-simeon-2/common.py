"""
common.py
=========
Small PyTorch helper functions shared by the client (client.py) and the
server's centralized evaluation (main.py). Keeping them here avoids
duplicating training/evaluation code in multiple places.
"""

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_weights(net: nn.Module) -> List[np.ndarray]:
    """Extracts a model's parameters as a list of numpy arrays (Flower's wire format)."""
    return [value.cpu().numpy() for value in net.state_dict().values()]


def set_weights(net: nn.Module, weights: List[np.ndarray]) -> None:
    """Loads a list of numpy arrays (as produced by get_weights) back into a model."""
    params_dict = zip(net.state_dict().keys(), weights)
    state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net: nn.Module, loader: DataLoader, epochs: int, lr: float, device: str = "cpu") -> None:
    """
    Plain local SGD training. This mirrors `Local_FedAvg.train()` in the
    original codebase: nothing algorithm-specific happens here, all of
    FedEntropy's logic lives in what happens to the result afterward
    (see entropy_strategy.py).
    """
    net.to(device)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def evaluate(net: nn.Module, loader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
    """Returns (average_loss, accuracy) of `net` on `loader`."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def compute_soft_label(net: nn.Module, loader: DataLoader, num_classes: int, device: str = "cpu") -> np.ndarray:
    """
    Runs the just-trained local model over the client's own training data
    and averages the predicted class-probability vectors into one vector.

    This "soft label" is a cheap, privacy-preserving summary of what the
    model currently believes about the local class distribution -- the
    server never sees the raw data, only this vector. The server uses
    soft labels (not raw data) to decide which clients' updates are
    likely to unbalance the aggregated model. That decision process is
    the core idea of FedEntropy (see entropy_utils.py).
    """
    net.to(device)
    net.eval()
    total_probs = torch.zeros(num_classes)
    num_samples = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            probs = torch.softmax(net(images), dim=1).sum(dim=0).cpu()
            total_probs += probs
            num_samples += images.size(0)

    return (total_probs / num_samples).numpy()
