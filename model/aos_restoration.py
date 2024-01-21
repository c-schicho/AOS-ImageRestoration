import torch
import torch.nn as nn

from model.restormer import Restormer
from utils import Config
from torch.nn.functional import conv2d


class AosRestorationConfig:

    def __init__(self, config: Config):
        self.num_blocks = config.num_blocks
        self.num_heads = config.num_heads
        self.channels = config.channels
        self.num_refinement = config.num_refinement
        self.expansion_factor = config.expansion_factor
        self.num_focal_planes = config.num_focal_planes
        self.skip_mode = config.skip_mode


class AOSRestoration(nn.Module):

    def __init__(self, config: AosRestorationConfig):
        super(AOSRestoration, self).__init__()
        self.model = Restormer(
            config.num_blocks,
            config.num_heads,
            config.channels,
            config.num_refinement,
            config.expansion_factor,
            config.num_focal_planes + 1,
            config.skip_mode
        )
        self.out = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=config.num_focal_planes + 1, out_channels=1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.kernel_gx = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=torch.float32
        )
        self.weights_gx = self.kernel_gx.view(1, 1, 3, 3).repeat(1, config.num_focal_planes, 1, 1).cuda()
        self.kernel_gy = torch.tensor(
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ],
            dtype=torch.float32
        )
        self.weights_gy = self.kernel_gy.view(1, 1, 3, 3).repeat(1, config.num_focal_planes, 1, 1).cuda()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grads = conv2d(x, self.weights_gx, padding=1) * conv2d(x, self.weights_gy, padding=1)
        inputs = torch.cat((x, grads), dim=1)
        return self.out(self.model(inputs))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def get_model_from_config(config: Config):
        aos_restoration_config = AosRestorationConfig(config)
        model = AOSRestoration(aos_restoration_config)

        if config.model_file:
            model.load(config.model_file)
            print('model checkpoint loaded successfully')

        return model
