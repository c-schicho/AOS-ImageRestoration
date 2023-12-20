from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from model.restormer import Restormer


@dataclass
class AosRestorationConfig:
    num_blocks: List[int]
    num_heads: List[int]
    channels: List[int]
    num_refinement: int
    expansion_factor: float


class AOSRestoration(nn.Module):

    def __init__(self, config: AosRestorationConfig):
        super(AOSRestoration, self).__init__()
        self.model = Restormer(
            config.num_blocks,
            config.num_heads,
            config.channels,
            config.num_refinement,
            config.expansion_factor
        )
        self.out = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.model(x))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.state_dict(destination=torch.load(path))
