import os
from typing import Dict, List

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
        data = []

        # Collect all file paths in the root folder and its sub-folders
        all_files = []
        for root, dirs, files in os.walk(self.folder):
            all_files.extend([os.path.join(root, file) for file in files])

        # Find all unique datapoints.
        unique_ids = sorted({int(parts[-5]) for filename in all_files if
                             filename.endswith(".png") and (parts := filename.split('_')) and len(parts) >= 2 and parts[
                                 -5].isdigit()})

        for id in unique_ids:
            label_filename = f"0_{id}_GT_pose_0_thermal.png"
            label_filepath = next((file for file in all_files if label_filename in file), None)

            feature_filenames = [f"0_{id}_integral_focal_{i:03d}_cm.png" for i in self.focal_stack]
            feature_filepaths = [next((file for file in all_files if feature_filename in file), None) for
                                 feature_filename in feature_filenames]

            # Check if both label and all training data files exist
            if label_filepath in all_files and all(training_file in all_files for training_file in feature_filepaths):
                data.append({
                    'label': label_filepath,
                    'training_data': feature_filepaths
                })

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
        feature_filenames = data_dict['training_data']
        label_filename = data_dict['label']

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
