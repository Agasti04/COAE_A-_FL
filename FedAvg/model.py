"""
model.py

Defines the PyTorch model used for mood prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoodNet(nn.Module):
    """
    A simple feedforward neural network for multi-class mood classification.

    Architecture:
    - Input layer: size = number of common features
    - Hidden layer: size = hidden_dim (default 64)
    - Output layer: size = number of classes (e.g. 3 moods)
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: input tensor of shape (batch_size, input_dim)

        Returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
