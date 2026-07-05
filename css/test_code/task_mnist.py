"""c_select: A Flower / PyTorch app (MNIST version)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

class Net(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self):
        super().__init__()

        # MNIST images are grayscale -> 1 input channel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Cache the dataset
fds = None

# MNIST normalization
pytorch_transforms = Compose(
    [
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
    ]
)


def apply_transforms(batch):
    """Apply transforms to MNIST images."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load one IID partition of MNIST."""
    global fds

    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)

    partition_train_test = partition.train_test_split(
        test_size=0.2,
        seed=42,
    )

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
    )

    return trainloader, testloader


def load_centralized_dataset():
    """Load the complete MNIST test set."""
    test_dataset = load_dataset(
        "ylecun/mnist",
        split="test",
    )

    dataset = test_dataset.with_transform(apply_transforms)

    return DataLoader(
        dataset,
        batch_size=128,
    )


def train(net, trainloader, epochs, lr, device):
    """Train the model."""
    net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=0.9,
    )

    net.train()

    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / (epochs * len(trainloader))


def test(net, testloader, device):
    """Evaluate the model."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    correct = 0
    loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss /= len(testloader)
    return loss, accuracy