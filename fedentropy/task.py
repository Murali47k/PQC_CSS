"""
task.py
-------
Shared building blocks for the FedEntOpt vs. FedAvg comparison:
  * a CIFAR-10-sized ResNet-18
  * label-skew federated partitioning via flwr-datasets (no manual
    download / no local ./data folder -- HuggingFace's `datasets` caches
    the CIFAR-10 arrow files in ~/.cache/huggingface on first use, exactly
    like your IID example)
  * local train / test loops
  * (de)serialization helpers between PyTorch state_dicts and Flower NDArrays

This mirrors the experimental setup in "Optimizing Federated Learning by
Entropy-Based Client Selection" (FedEntOpt), simplified to a single dataset
(CIFAR-10) and a single architecture (ResNet-18) as requested.
"""
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, ShardPartitioner
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

NUM_CLASSES = 10
DATASET_NAME = "uoft-cs/cifar10"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_resnet18_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 adapted for 32x32 CIFAR-style inputs.

    Standard ImageNet ResNet-18 downsamples too aggressively for 32x32
    images, so we shrink the stem: a 3x3 stride-1 conv instead of the
    7x7 stride-2 conv, and we drop the initial max-pool.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_parameters(model: nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# --------------------------------------------------------------------------
# Data (flwr-datasets: no manual download, no ./data folder)
# --------------------------------------------------------------------------
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to a partition loaded from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


_fds: Optional[FederatedDataset] = None  # module-level cache, like your example
_fds_key = None


def _get_fds(
    num_partitions: int, partition: str, alpha: float, classes_per_client: int, seed: int
) -> FederatedDataset:
    """Build (and cache) the FederatedDataset with the requested label-skew
    partitioner. Only rebuilt if the partitioning config actually changes,
    mirroring the `global fds` caching pattern from a plain IID setup."""
    global _fds, _fds_key
    key = (num_partitions, partition, alpha, classes_per_client, seed)
    if _fds is None or _fds_key != key:
        if partition == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif partition == "dirichlet":
            # Distribution-based label skew, Dir(alpha) -- FedEntOpt paper's
            # "most challenging" Dir(0.1) setting by default.
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha,
                min_partition_size=10,
                self_balancing=False,
                seed=seed,
            )
        elif partition == "shard":
            # Quantity-based label skew -- the paper's "C = j" setting:
            # every client sees samples from only `classes_per_client`
            # distinct classes.
            partitioner = ShardPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                num_shards_per_partition=classes_per_client,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown partition strategy: {partition}")

        _fds = FederatedDataset(
            dataset=DATASET_NAME,
            partitioners={"train": partitioner},
            seed=seed,
        )
        _fds_key = key
    return _fds


def load_partition_loaders(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partition: str = "dirichlet",
    alpha: float = 0.1,
    classes_per_client: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, np.ndarray]:
    """Load one client's CIFAR-10 partition and its label-count vector.

    Returns (trainloader, label_counts) where label_counts is used by
    FedEntOpt exactly like the paper's one-time l^(k) upload.
    """
    fds = _get_fds(num_partitions, partition, alpha, classes_per_client, seed)
    part = fds.load_partition(partition_id, split="train")

    labels = np.array(part["label"])
    label_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    uniq, cnt = np.unique(labels, return_counts=True)
    label_counts[uniq] += cnt

    part = part.with_transform(apply_transforms)
    trainloader = DataLoader(part, batch_size=batch_size, shuffle=True)
    return trainloader, label_counts


def compute_all_label_counts(
    num_partitions: int,
    partition: str = "dirichlet",
    alpha: float = 0.1,
    classes_per_client: int = 2,
    seed: int = 42,
):
    """Compute every client's label-count vector l^(k) up front.

    This is the server-side equivalent of "collect the one-time label
    upload from every client before round 1" (Section IV of the paper) --
    we just read it directly off the partitioner instead of round-tripping
    a message, since we generated the partitioning ourselves.
    Returns a dict keyed by partition id as a string, e.g. {"0": array(...), ...}.
    """
    fds = _get_fds(num_partitions, partition, alpha, classes_per_client, seed)
    label_counts = {}
    for pid in range(num_partitions):
        part = fds.load_partition(pid, split="train")
        labels = np.array(part["label"])
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        if len(labels) > 0:
            uniq, cnt = np.unique(labels, return_counts=True)
            counts[uniq] += cnt
        label_counts[str(pid)] = counts
    return label_counts


def load_centralized_testloader(batch_size: int = 128) -> DataLoader:
    """Load the full CIFAR-10 test split for server-side (centralized)
    evaluation -- identical for both FedAvg and FedEntOpt runs."""
    test_dataset = load_dataset(DATASET_NAME, split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=batch_size)


# --------------------------------------------------------------------------
# Train / test
# --------------------------------------------------------------------------
def train(net, trainloader, epochs: int, lr: float, weight_decay: float, device) -> float:
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device) -> Tuple[float, float]:
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
