import torch
from AOSDataset import AOSDataset
from AOSDataloader import AOSDataloader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from typing import Tuple

G_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomInvert(p=0.5), # Randomly inverst the colors of an image with an probability of 0.5

    # Makes sence but we need mean and std
    #transforms.Normalize(mean=0, std=1),

    # I would definitly try these 
    #transforms.RandomAdjustSharpness(sharpness_factor[, p]) # could be usefull.
    #transforms.RandomAutocontrast([p])
    #transforms.RandomEqualize([p])

    # I think that could maybe be beneficial but I really doubt it. 
    #transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    #transforms.GaussianBlur(kernel_size=5),

    # I wouldn't use the ones below 
    #transforms.RandomRotation(degrees=30),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

# I would use the same transform for both the features and the targets!
G_DEFAULT_TARGET_TRANSFORM = G_DEFAULT_TRANSFORM

def get_aos_loaders(train_ratio: float = 0.8,
                    train_batch_size: int = 32,
                    test_batch_size: int = 32,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    transform: torchvision.transforms.Compose = G_DEFAULT_TRANSFORM,
                    target_transform: torchvision.transforms.Compose = G_DEFAULT_TARGET_TRANSFORM,
                    seed: int = 42,
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

    # Create AOSDataset instance with specified transforms
    dataset = AOSDataset(dataset_folder,
                         transform=transform,
                         target_transform=target_transform)

    # Create train and test loaders using AOSDataloader
    train_loader, test_loader = AOSDataloader(dataset,
                                               train_ratio=train_ratio,
                                               train_batch_size=train_batch_size,
                                               test_batch_size=test_batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               seed=seed)

    return train_loader, test_loader
