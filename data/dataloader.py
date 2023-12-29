from typing import Tuple, List

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

from data.dataset import AOSDataset
from data.transform import invert_channels


def aos_dataloader(
        dataset: Dataset,
        train_ratio: float = 0.8,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None
) -> (DataLoader, DataLoader):
    """
    Args:
        dataset (Dataset): The input dataset.
        train_ratio (float): The ratio of the dataset to be used for training.
        train_batch_size (int): The batch size for the training DataLoader.
        test_batch_size (int): The batch size for the testing DataLoader.
        shuffle (bool): If True, the data is shuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
        collate_fn: Specifies how to collate multiple samples into a batch.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        drop_last (bool): If True, drops the last batch if its size is less than batch_size.
        timeout (float): If positive, the data loader will timeout after the specified seconds.
        worker_init_fn: If not None, this will be called on each worker subprocess with the worker id as input.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the testing set.
    """

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin_memory, drop_last=drop_last,
                              timeout=timeout, worker_init_fn=worker_init_fn)

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle,
                             num_workers=num_workers, collate_fn=collate_fn,
                             pin_memory=pin_memory, drop_last=drop_last,
                             timeout=timeout, worker_init_fn=worker_init_fn)

    return train_loader, test_loader


def get_aos_loaders(
        train_ratio: float = 0.8,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        patch_size: Tuple[int, int] = (512, 512),
        dataset_folder: str = "../focal_dataset"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get PyTorch DataLoaders for training and testing AOSDataset with image restoration transforms.

    Args:
        train_ratio (float): The ratio of the dataset to be used for training.
        train_batch_size (int): The batch size for the training DataLoader.
        test_batch_size (int): The batch size for the testing DataLoader.
        shuffle (bool): If True, the data is shuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoader for the training set and DataLoader for the testing set.
    """

    transform = __get_default_train_transform(patch_size)
    dataset = AOSDataset(dataset_folder, transform=transform)

    train_loader, test_loader = aos_dataloader(
        dataset,
        train_ratio=train_ratio,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return train_loader, test_loader


def get_single_aos_loader(
        dataset_folder: str,
        batch_size: int,
        patch_size: int,
        dataset_len: int,
        num_workers: int = 1,
        focal_planes: List[int] = [10, 50, 150],
        use_train_transform: bool = True,
        shuffle: bool = True
) -> DataLoader:
    transform = (
        __get_default_train_transform((patch_size, patch_size)) if use_train_transform else __get_default_test_transform()
    )
    dataset = AOSDataset(dataset_folder, transform=transform, maximum_datasize=dataset_len, focal_stack=focal_planes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def __get_default_train_transform(patch_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x if torch.rand(1).item() > 0.5 else invert_channels(x)),
        transforms.RandomCrop(patch_size, pad_if_needed=True),
        transforms.ToTensor()
    ])


def __get_default_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
