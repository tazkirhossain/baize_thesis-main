"""Character CNN architecture supplied by the user on 22 October 2026.

Expects floating one-hot character features (batch, sequence_length, characters).
Encoding, local training and federated aggregation are separate stages.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CharacterCNNConfig:
    """Vocabulary size excludes extra_characters; all clients must agree on order."""

    number_of_characters: int
    extra_characters: str = ""
    dropout_input: float = 0.0


class CharacterLevelCNN(nn.Module):
    def __init__(self, args, number_of_classes):
        super().__init__()
        if type(args.number_of_characters) is not int or args.number_of_characters < 1:
            raise ValueError("number_of_characters must be a positive integer")
        if type(number_of_classes) is not int or number_of_classes < 2:
            raise ValueError("number_of_classes must be an integer of at least two")
        if not isinstance(args.extra_characters, str):
            raise ValueError("extra_characters must be a string")
        if not math.isfinite(args.dropout_input) or not 0 <= args.dropout_input <= 1:
            raise ValueError("dropout_input must be between zero and one")
        self.n_characters = args.number_of_characters + len(args.extra_characters)
        self.number_of_classes = number_of_classes
        self.dropout_input = nn.Dropout1d(args.dropout_input)

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.n_characters, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, number_of_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        if x.ndim != 3 or x.shape[2] != self.n_characters:
            raise ValueError(f"expected (batch, sequence_length, {self.n_characters})")
        if x.shape[0] < 1 or x.shape[1] < 4:
            raise ValueError("nonempty batch and sequence length >= 4 are required")
        if not torch.is_floating_point(x):
            raise TypeError("expected floating one-hot character features, not token IDs")
        # Dropout1d expects channels in dimension 1, just like Conv1d.
        x = x.transpose(1, 2)
        x = self.dropout_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        residual = x
        x = self.conv3(x) + residual
        residual = x
        x = self.conv4(x) + residual
        residual = x
        x = self.conv5(x) + residual
        residual = x
        x = self.conv6(x) + residual
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.adaptive_pool(x)
        return self.fc(x.flatten(1))  # Raw logits; CrossEntropyLoss applies softmax.
