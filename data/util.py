import PIL
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from typing import Tuple

from tranforms.transforms import InvertChannels
from datasets.AOSDataset import AOSDataset
from dataloaders.AOSDataloader import AOSDataloader
from torch.utils.data import DataLoader


G_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.RandomCrop(patch_size), # insert patch size here
    transforms.Lambda(lambda x: x if torch.rand(1).item() > 0.5 else InvertChannels()(x)),
    transforms.ToTensor()
])


def get_aos_loaders(train_ratio: float = 0.8,
                    train_batch_size: int = 32,
                    test_batch_size: int = 32,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    seed: int = 42,
                    patch_size = [512, 512],
                    dataset_folder:str =  "../focal_dataset") -> Tuple[DataLoader, DataLoader]:
    """
    Get PyTorch DataLoaders for training and testing AOSDataset with image restoration transforms.

    Args:
        train_ratio (float): The ratio of the dataset to be used for training.
        train_batch_size (int): The batch size for the training DataLoader.
        test_batch_size (int): The batch size for the testing DataLoader.
        shuffle (bool): If True, the data is shuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
        transform (torchvision.transforms.Compose): A composition of transforms applied to the input data.
        target_transform (torchvision.transforms.Compose): A composition of transforms applied to the target data.
        seed (int): The random seed for the generator. If None, no seed is set.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoader for the training set and DataLoader for the testing set.
    """    

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(patch_size,  pad_if_needed=True),
        transforms.Lambda(lambda x: x if torch.rand(1).item() > 0.5 else InvertChannels()(x)),
        transforms.ToTensor()
    ])

    # Create AOSDataset instance with specified transforms
    dataset = AOSDataset(dataset_folder,
                         transform=transform)

    # Create train and test loaders using AOSDataloader
    train_loader, test_loader = AOSDataloader(dataset,
                                               train_ratio=train_ratio,
                                               train_batch_size=train_batch_size,
                                               test_batch_size=test_batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               seed=seed)

    return train_loader, test_loader
