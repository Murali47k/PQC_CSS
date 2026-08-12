"""
data.py
=======
Loads a dataset and splits it into non-IID shards across simulated clients.

By default this tries to download the real MNIST dataset via torchvision.
If that fails (e.g. no internet access in a sandboxed environment), it
automatically falls back to a small synthetic image dataset with the same
shape (1x28x28, 10 classes), so the rest of the pipeline can still be run
and inspected fully offline. When you run this on a normal machine with
internet access, it will use real MNIST automatically.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

NUM_CLASSES = 10
IMAGE_SIZE = 28


class SyntheticDigits(Dataset):
    """
    A tiny, dependency-free stand-in for MNIST.

    Each class is assigned a fixed position on a circle within the image;
    every sample is a soft Gaussian "blob" at its class's position, plus
    pixel noise. Unlike pure random-noise templates, this gives the image
    real spatial structure, so a small CNN can actually learn to tell the
    classes apart within a few epochs -- keeping the demo both offline
    and genuinely instructive (you should see accuracy rise over rounds).
    """

    def __init__(self, num_samples: int, seed: int = 0, noise_std: float = 0.15, blob_sigma: float = 4.0):
        rng = np.random.RandomState(seed)

        # One fixed (row, col) center per class, spread evenly around a circle.
        angles = np.linspace(0, 2 * np.pi, NUM_CLASSES, endpoint=False)
        margin = IMAGE_SIZE * 0.3
        centers = np.stack(
            [
                IMAGE_SIZE / 2 + margin * np.sin(angles),  # row
                IMAGE_SIZE / 2 + margin * np.cos(angles),  # col
            ],
            axis=1,
        )

        rows, cols = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE), indexing="ij")

        self.targets = rng.randint(0, NUM_CLASSES, size=num_samples)
        images = np.zeros((num_samples, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for i, label in enumerate(self.targets):
            center_row, center_col = centers[label]
            blob = np.exp(-((rows - center_row) ** 2 + (cols - center_col) ** 2) / (2 * blob_sigma**2))
            images[i] = blob

        noise = rng.randn(num_samples, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32) * noise_std
        self.images = np.clip(images + noise, 0.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        # .copy() avoids a PyTorch warning when this dataset is passed through
        # Ray's shared-memory object store during simulation, which can hand
        # back read-only numpy arrays.
        image = torch.from_numpy(self.images[idx].copy()).unsqueeze(0)  # shape: (1, 28, 28)
        label = int(self.targets[idx])
        return image, label


def load_datasets() -> Tuple[Dataset, Dataset]:
    """Returns (train_dataset, test_dataset). Tries real MNIST, falls back to synthetic data."""
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
        print("Loaded real MNIST dataset.")
        return train_ds, test_ds
    except Exception as exc:  # no internet, blocked mirror, etc.
        print(f"Could not download MNIST ({exc}).")
        print("Falling back to a small synthetic offline dataset instead.")
        train_ds = SyntheticDigits(num_samples=6000, seed=0)
        test_ds = SyntheticDigits(num_samples=1000, seed=1)
        return train_ds, test_ds


def _get_targets(dataset: Dataset) -> np.ndarray:
    """Extracts a plain numpy array of integer labels from a torchvision or synthetic dataset."""
    targets = dataset.targets
    if torch.is_tensor(targets):
        return targets.numpy()
    return np.array(targets)


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.3,
    seed: int = 42,
    min_partition_size: int = 10,
) -> List[List[int]]:
    """
    Splits dataset indices across `num_clients` clients using a Dirichlet
    distribution over class proportions -- the standard way to create a
    label-skewed non-IID partition for federated learning benchmarks.

    - Small `alpha` (e.g. 0.1) -> very skewed clients, each dominated by a
      couple of classes. This is the "hard", interesting case for FL.
    - Large `alpha` (e.g. 100)  -> close to IID (every client looks similar).

    Returns a list of length `num_clients`; each entry is a list of
    dataset indices belonging to that client.
    """
    labels = _get_targets(dataset)
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    rng = np.random.RandomState(seed)

    while True:  # retry until every client has a reasonable amount of data
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # Prevent any client that's already over its "fair share" of
            # the whole dataset from grabbing even more, then renormalize.
            proportions = np.array(
                [
                    p * (len(client_indices[i]) < num_samples / num_clients)
                    for i, p in enumerate(proportions)
                ]
            )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

            for i, shard in enumerate(np.split(class_indices, split_points)):
                client_indices[i].extend(shard.tolist())

        if min(len(idxs) for idxs in client_indices) >= min_partition_size:
            break  # every client has enough data - accept this partition

    for idxs in client_indices:
        rng.shuffle(idxs)

    return client_indices


def make_client_subset(dataset: Dataset, indices: List[int]) -> Subset:
    """Wraps a list of indices as a PyTorch Subset, ready to hand to a DataLoader."""
    return Subset(dataset, indices)
