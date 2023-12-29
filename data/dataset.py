import os
import glob
from typing import Dict, List
from itertools import groupby

import numpy as np
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset


class AOSDataset(Dataset):
    """
    A custom dataset class for handling AOS data.

    Args:
        folder (str): List of folder paths containing data.
        transform (torchvision.transforms.Compose, optional): Transform for input data. Defaults to transforms.Compose([transforms.ToTensor()]).
        relative_path (bool, optional): Whether input folders are specified as relative paths. Defaults to True.
        maximum_datasize (int, optional): Maximum number of samples to load. Defaults to None.
    """

    def __init__(
            self,
            folder: str,
            transform: torchvision.transforms.Compose = transforms.Compose([transforms.ToTensor()]),
            relative_path: bool = True,
            maximum_datasize: int = None,
            focal_stack: list[int] = [10, 50, 150]
    ):
        """
        Initializes the AOSDataset.

        Args:
            folder (str): List of folder paths containing data.
            transform (torchvision.transforms.Compose, optional): Transform for input data. Defaults to transforms.Compose([transforms.ToTensor()]).
            relative_path (bool, optional): Whether input folders are specified as relative paths. Defaults to True.
            maximum_datasize (int, optional): Maximum number of samples to load. Defaults to None, which loads all images
        """
        if maximum_datasize is not None and maximum_datasize <= 0:
            raise ValueError("maximum_datasize must be greater than 0 or None")

        self.folder = os.path.join(os.getcwd(), folder) if relative_path else folder
        self.maximum_datasize = maximum_datasize
        self.transform = transform
        self.focal_stack = focal_stack
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, List[str]]]:
        """
        Load data from the specified root folder and its sub-folders.

        Returns:
            List[Dict[str, List[str]]]: List of dictionaries containing 'label' and 'training_data' paths.
        """
        all_images = glob.glob(os.path.join(self.folder, "**", "*.png"), recursive=True)
        all_images.sort(key=self.__get_unique_id)
        grouped_images = {
            key: list(values) for key, values in groupby(all_images, lambda path: self.__get_unique_id(path))
        }

        data = []
        focal_plane_search_strings = [f"{focal_plane:03d}" for focal_plane in self.focal_stack]

        for key, files in grouped_images.items():
            ground_truth_paths = list(filter(lambda file: 'GT' in file, files))
            input_paths = list(filter(lambda file: file.split('_')[-2] in focal_plane_search_strings, files))

            if len(ground_truth_paths) != 1 or len(input_paths) != len(self.focal_stack):
                continue

            input_paths.sort(key=lambda path: int(path.split('_')[-2]))

            data.append(
                {
                    'ground_truth_path': ground_truth_paths[0],
                    'input_paths': input_paths
                }
            )

            if self.maximum_datasize is not None and len(data) == self.maximum_datasize:
                break

        return data

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing input features and labels.
        """
        # Fetch file information from the data
        data_dict = self.data[idx]
        feature_filenames = data_dict['input_paths']
        label_filename = data_dict['ground_truth_path']

        # Load images and labels dynamically
        labels = io.imread(label_filename)

        # Load the feature images
        features = np.zeros((labels.shape[0], labels.shape[1], len(feature_filenames)))

        for i, file in enumerate(feature_filenames):
            image = io.imread(os.path.join(file))

            # In this dataset each image strangly has the same value in all three 
            # channels -> I only use the first entry and do not check for the others
            # for execution time reasons.
            features[:, :, i] = image[:, :, 0]

        # Apply the same transform to both features and labels
        images = np.concatenate((features, labels[:, :, np.newaxis]), axis=-1)

        images = images.astype('uint8')
        images = self.transform(images)

        labels = images[-1][None, ...]
        features = images[:-1]

        return features, labels

    @staticmethod
    def __get_unique_id(path: str):
        return ''.join(os.path.split(path)[-1].split('_')[:2])
