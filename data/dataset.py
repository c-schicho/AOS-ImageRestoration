import glob
import os
import random
from itertools import groupby
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms.functional import crop, hflip, vflip


class AOSInferenceDataset(Dataset):

    def __init__(self, path: str, focal_stack: list[int] = [10, 50, 150]):
        """
        Args:
            path (str): Path to the folder containing the images.
            focal_stack (list[int], optional): List of focal planes to use. Defaults to [10, 50, 150].
        """
        self.path = path
        self.focal_stack = focal_stack
        self.data = self._load_data()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def _load_data(self) -> List[List[str]]:
        all_images = glob.glob(os.path.join(self.path, "**", "*.png"), recursive=True)
        all_images.sort(key=self.__get_unique_id)
        grouped_images = {
            key: list(values) for key, values in groupby(all_images, lambda path: self.__get_unique_id(path))
        }

        data = []
        focal_plane_search_strings = [f"{focal_plane:03d}" for focal_plane in self.focal_stack]

        for key, files in grouped_images.items():
            input_paths = list(filter(lambda file: file.split('_')[-2] in focal_plane_search_strings, files))

            if len(input_paths) != len(self.focal_stack):
                print("Warning: Incomplete focal stack found. Skipping image.")
                continue

            input_paths.sort(key=lambda path: int(path.split('_')[-2]))
            data.append(input_paths)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        feature_filenames = self.data[idx]
        image_shape = io.imread(feature_filenames[0]).shape
        h, w = image_shape[0], image_shape[1]
        features = np.zeros((h, w, len(feature_filenames)))

        for i, file in enumerate(feature_filenames):
            image = io.imread(os.path.join(file))
            if len(image.shape) == 3 and image.shape[2] > 1:
                features[:, :, i] = image[:, :, 0]
            elif len(image.shape) == 2:
                features[:, :, i] = image
            else:
                print("Warning: Image has an unexpected shape. Skipping image.")

        features = self.transform(features.astype('uint8'))
        return features

    @staticmethod
    def __get_unique_id(path: str):
        return ''.join(os.path.split(path)[-1].split('_')[:2])


class AOSDataset(Dataset):

    def __init__(
            self,
            folder: str,
            relative_path: bool = True,
            maximum_datasize: int = None,
            focal_stack: list[int] = [10, 50, 150],
            patch_size: Union[tuple[int, int], None] = None,
            use_train_transform: bool = True
    ):
        """
        Args:
            folder (str): Path to the folder containing the images.
            relative_path (bool, optional): Whether input folders are specified as relative paths. Defaults to True.
            maximum_datasize (int, optional): Maximum number of samples to load. Defaults to None, which loads all images,
            focal_stack (list[int], optional): List of focal planes to use. Defaults to [10, 50, 150].
            patch_size (Union[tuple[int, int], None], optional): Size of the patches to extract from the images. Defaults to None.
            use_train_transform (bool, optional): Whether to use the train transform. Defaults to True.
        """
        if maximum_datasize is not None and maximum_datasize <= 0:
            raise ValueError("maximum_datasize must be greater than 0 or None")

        self.folder = os.path.join(os.getcwd(), folder) if relative_path else folder
        self.maximum_datasize = maximum_datasize
        self.focal_stack = focal_stack
        self.data = self._load_data()
        self.patch_size = patch_size
        self.use_train_transform = use_train_transform

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing input features and ground_truth.
        """
        # Fetch file information from the data
        data_dict = self.data[idx]
        label_filename = data_dict['ground_truth_path']
        ground_truth = io.imread(label_filename)

        feature_filenames = data_dict['input_paths']
        features = np.zeros((ground_truth.shape[0], ground_truth.shape[1], len(feature_filenames)))

        for i, file in enumerate(feature_filenames):
            image = io.imread(os.path.join(file))
            # In this dataset each image strangly has the same value in all three 
            # channels -> I only use the first entry and do not check for the others
            # for execution time reasons.
            features[:, :, i] = image[:, :, 0]

        images = transforms.ToPILImage()(
            np.concatenate((features, ground_truth[:, :, np.newaxis]), axis=-1).astype('uint8')
        )
        if self.use_train_transform:
            mask = transforms.ToPILImage()(self.__get_loss_weight_mask(ground_truth))
            images, mask = self.__random_flip(images, mask)
        else:
            mask = transforms.ToPILImage()(np.zeros_like(ground_truth))

        if self.patch_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(images, output_size=self.patch_size)
            images = crop(images, i, j, h, w)
            mask = crop(mask, i, j, h, w)

        images = transforms.ToTensor()(images)
        mask = transforms.ToTensor()(mask)

        features = images[:-1]
        ground_truth = images[-1][None, ...]

        return features, ground_truth, mask

    @staticmethod
    def __random_flip(images: Image, mask: Image) -> tuple[Image, Image]:
        if random.random() > 0.5:
            images = hflip(images)
            mask = hflip(mask)

        if random.random() > 0.5:
            images = vflip(images)
            mask = vflip(mask)

        return images, mask

    @staticmethod
    def __get_loss_weight_mask(ground_truth: np.ndarray) -> np.ndarray:
        ground_truth = cv2.medianBlur(ground_truth, 5)
        _, thresh = cv2.threshold(ground_truth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask = np.ones_like(ground_truth) * 255
        cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(0, 1, 0), thickness=cv2.FILLED)

        # a person takes usually 2-5% of the image
        if mask.mean() > 30:
            # if the mean is too high, we might have found the inverse mask
            mask = 255 - mask

        if mask.mean() > 30:
            # if both mask versions have a high mean, we also consider environmental objects / structures
            # therefore, we fall back to an unweighted mask.
            mask = mask * 0

        return cv2.dilate(mask, np.ones((5, 5)))

    @staticmethod
    def __get_unique_id(path: str):
        return ''.join(os.path.split(path)[-1].split('_')[:2])
