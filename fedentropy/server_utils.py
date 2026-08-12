"""
server_utils.py
----------------
Centralized evaluation: after each aggregation round, the server loads the
new global parameters into a fresh model and evaluates it on the full
CIFAR-10 test set (loaded once via flwr-datasets / HuggingFace `datasets`,
no manual download). This is what both strategies are scored on, so FedAvg
and FedEntOpt are compared on an identical, apples-to-apples metric.
"""
from typing import Dict

import torch

from task import build_resnet18_cifar, load_centralized_testloader, set_parameters, test


def make_evaluate_fn(device: torch.device, batch_size: int = 128):
    testloader = load_centralized_testloader(batch_size=batch_size)
    model = build_resnet18_cifar()

    def evaluate_fn(server_round: int, parameters, config: Dict):
        set_parameters(model, parameters)
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn
