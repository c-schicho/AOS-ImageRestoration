from typing import Dict, List
import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

class AOSDataset(Dataset):
    """
    A custom dataset class for handling AOS data.

    Args:
        folders (list[str]): List of folder paths containing data.
        transform (torchvision.transforms.Compose, optional): Transform for input data. Defaults to transforms.Compose([transforms.ToTensor()]).
        target_transform (torchvision.transforms.Compose, optional): Transform for labels. Defaults to transforms.Compose([transforms.ToTensor()]).
        relative_path (bool, optional): Whether input folders are specified as relative paths. Defaults to True.
        maximum_datasize (int, optional): Maximum number of samples to load. Defaults to None.
    """
    def __init__(self, 
                folders: list[str], 
                transform: torchvision.transforms.Compose = transforms.Compose([transforms.ToTensor()]),
                target_transform: torchvision.transforms.Compose = transforms.Compose([transforms.ToTensor()]), 
                relative_path: bool = True,
                maximum_datasize: int = None):
        """
        Initializes the AOSDataset.

        Args:
            folders (list[str]): List of folder paths containing data.
            transform (torchvision.transforms.Compose, optional): Transform for input data. Defaults to transforms.Compose([transforms.ToTensor()]).
            target_transform (torchvision.transforms.Compose, optional): Transform for labels. Defaults to transforms.Compose([transforms.ToTensor()]).
            relative_path (bool, optional): Whether input folders are specified as relative paths. Defaults to True.
            maximum_datasize (int, optional): Maximum number of samples to load. Defaults to None.
        """
        if maximum_datasize is not None and maximum_datasize <= 0:
            raise ValueError("maximum_datasize must be greater than 0 or None")

        current_directory = os.getcwd()
        self.folders = [os.path.join(os.getcwd(), folder) if relative_path else folder for folder in folders]
        self.maximum_datasize = maximum_datasize
        self.transform = transform
        self.target_transform = target_transform
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, List[str]]]:
        """
        Load data from specified folders.

        Returns:
            list[dict]: List of dictionaries containing 'label' and 'training_data' paths.
        """

        # The labels are named as follows 
        # 0_{id}_GT_pose_0_thermal.png

        # The features are named as follows#
        # 0_{id}_pose_{i}_thermal.png
        
        data = []

        for folder in self.folders:
            files = os.listdir(folder)

            # Find all unique datapoints. 
            unique_ids = sorted({int(parts[1]) for filename in files if filename.endswith(".png") and (parts := filename.split('_')) and len(parts) >= 2 and parts[1].isdigit()})

            for id in unique_ids:
                label_filename = f"0_{id}_GT_pose_0_thermal.png"
                label_filepath = os.path.join(folder, label_filename)

                feature_filenames = [f"0_{id}_pose_{i}_thermal.png" for i in range(11)]
                feature_filepaths = [os.path.join(folder, filename) for filename in feature_filenames]


                # Check if both label and all training data files exist
                if label_filename in files and all(training_file in files for training_file in feature_filenames):
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
            features[:,:,i] = io.imread(os.path.join(file))

        if self.transform:
            features = self.transform(features)

        if self.target_transform:
            labels = self.target_transform(labels)

        return features, labels
