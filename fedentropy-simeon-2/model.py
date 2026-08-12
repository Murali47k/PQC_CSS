"""
model.py
========
A tiny CNN for 28x28 grayscale images (MNIST-shaped), loosely modeled on
the classic LeNet used in the original FedEntropy code, but simplified.

Kept deliberately small so that a full federated round -- many clients
each training this network -- runs in seconds on a laptop CPU. Swap this
out for a bigger model (VGG, ResNet, ...) once the pipeline itself is
understood; nothing else in the project needs to change.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)   # 28x28 -> 24x24
        self.pool = nn.MaxPool2d(2, 2)                 # 24x24 -> 12x12
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)   # 12x12 -> 8x8
        # a second pooling step (in forward) takes 8x8 -> 4x4
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)          # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)                 # raw logits (no softmax here)
