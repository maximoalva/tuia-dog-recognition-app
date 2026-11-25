"""Modelo B, CNN propia entrenada desde cero.

La clase vive en un modulo y no en la notebook para que la app pueda cargar el checkpoint
con torch.load (necesita importar la clase con el mismo nombre y modulo).
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, convs: int) -> nn.Sequential:
    """Convoluciones 3x3 con BatchNorm y ReLU, y un MaxPool que divide el tamaño por 2."""
    layers: list[nn.Module] = []
    for i in range(convs):
        # Sin bias porque BatchNorm ya agrega su propio desplazamiento.
        layers += [
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CNNCustom(nn.Module):
    """Bloques convolucionales, global average pooling, embedding y clasificador."""

    def __init__(self, num_classes: int, channels: tuple[int, ...] = (32, 64, 128, 256), convs_per_block: int = 1,
                 embedding_dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        blocks, in_channels = [], 3
        for out_channels in channels:
            blocks.append(conv_block(in_channels, out_channels, convs_per_block))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(nn.Linear(in_channels, embedding_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding de la capa previa al clasificador, el que usa la busqueda por similitud."""
        return self.embedding(torch.flatten(self.pool(self.features(x)), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.embed(x)))
