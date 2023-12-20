import torch
from torch.utils.data import Dataset, DataLoader, random_split

def AOSDataloader(dataset: Dataset,
                  train_ratio: float = 0.8,
                  train_batch_size: int = 32,
                  test_batch_size: int = 32,
                  shuffle: bool = True,
                  seed: int = 42,
                  num_workers: int = 0,
                  collate_fn = None,
                  pin_memory: bool = False,
                  drop_last: bool = False,
                  timeout: float = 0,
                  worker_init_fn = None) -> (DataLoader, DataLoader):
    """
    Args:
        dataset (Dataset): The input dataset.
        train_ratio (float): The ratio of the dataset to be used for training.
        train_batch_size (int): The batch size for the training DataLoader.
        test_batch_size (int): The batch size for the testing DataLoader.
        shuffle (bool): If True, the data is shuffled at every epoch.
        seed (int): The random seed for the generator. If None, no seed is set.
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
    # Set the random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Calculate sizes for train and test sets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Use random_split to split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for train set
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin_memory, drop_last=drop_last,
                              timeout=timeout, worker_init_fn=worker_init_fn)

    # Create DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle,
                             num_workers=num_workers, collate_fn=collate_fn,
                             pin_memory=pin_memory, drop_last=drop_last,
                             timeout=timeout, worker_init_fn=worker_init_fn)

    return train_loader, test_loader


