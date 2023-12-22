import torch
import torch.nn as nn

from model.restormer import Restormer
from utils import Config


class AosRestorationConfig:

    def __init__(self, config: Config):
        self.num_blocks = config.num_blocks
        self.num_heads = config.num_heads
        self.channels = config.channels
        self.num_refinement = config.num_refinement
        self.expansion_factor = config.expansion_factor
        self.num_focal_planes = config.num_focal_planes


class AOSRestoration(nn.Module):

    def __init__(self, config: AosRestorationConfig):
        super(AOSRestoration, self).__init__()
        self.model = Restormer(
            config.num_blocks,
            config.num_heads,
            config.channels,
            config.num_refinement,
            config.expansion_factor,
            config.num_focal_planes
        )
        self.out = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=config.num_focal_planes, out_channels=1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.model(x))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.state_dict(destination=torch.load(path))

    @staticmethod
    def get_model_from_config(config: Config):
        aos_restoration_config = AosRestorationConfig(config)
        model = AOSRestoration(aos_restoration_config)

        if config.model_file:
            model.load(config.model_file)

        return model
