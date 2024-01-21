from typing import List

from torch.utils.data import DataLoader

from data.dataset import AOSDataset


def get_aos_loader(
        dataset_folder: str,
        batch_size: int,
        patch_size: int,
        dataset_len: int,
        num_workers: int = 1,
        focal_planes: List[int] = [10, 50, 150],
        use_train_transform: bool = True,
        shuffle: bool = True
) -> DataLoader:
    dataset = AOSDataset(
        dataset_folder,
        maximum_datasize=dataset_len,
        focal_stack=focal_planes,
        patch_size=(patch_size, patch_size),
        use_train_transform=use_train_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
