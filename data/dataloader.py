from typing import Tuple, List
import torch
import torchvision.transforms as transforms
from data.transform import invert_channels


def get_default_train_transform(patch_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(patch_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x if torch.rand(1).item() > 0.5 else invert_channels(x)),
        transforms.ToTensor()
    ])


def get_default_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
