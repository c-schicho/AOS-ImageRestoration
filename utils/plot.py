from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from IPython.display import display
from PIL import Image
from torch.utils.data import DataLoader

from data import AOSDataset


def plot_dataset(dataset: AOSDataset, n_images: int = 1, shuffle: bool = True):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    plot_dataloader(dataloader, n_images)


def plot_dataloader(dataloader: DataLoader, n_images: int = 1):
    for n in range(n_images):
        feature, label = next(iter(dataloader))
        batches, feature_channels, x_size, y_size = feature.shape
        _, label_channels, _, _ = label.shape

        n_images_to_show = feature_channels + label_channels

        # paste together
        # features
        feature_images = feature.detach().cpu()
        xs = feature_images.view(-1, 1, x_size, y_size)  # unflatten
        x_im = __data_to_image(*xs)

        # labels
        label = label.detach().cpu()
        xl = label[0, 0].view(-1, 1, x_size, y_size)  # unflatten
        x_re = __data_to_image(*xl)

        im = Image.new('L', (n_images_to_show * x_size, y_size))
        im.paste(x_im, (y_size, 0))
        im.paste(x_re, (0, 0))
        display(im, metadata={'width': '100%'})


def plot_model_comparison(results_df: pd.DataFrame, fig_size: Tuple[int, int] = (15, 7)):
    """
    expects a dataframe with the following columns: [run_id, MSE, PSNR, SSIM]
    """

    x = np.arange(3)  # the label locations
    width = 0.8 / len(results_df)  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(figsize=fig_size)
    for index, row in results_df.iterrows():
        a = [np.round(row['MSE'], 5), np.round(row['PSNR'], 2), np.round(row['SSIM'], 3)]
        measurement = pd.Series(a)
        attribute = row['run_id']
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_title('Model Comparison by Metrics')
    ax.set_xlabel('Metric')
    ax.set_xticks(x + (width * ((len(results_df) - 1) / 2)))
    ax.set_xticklabels(["MSE", "PSNR", "SSIM"])
    ax.legend(loc='upper left', ncol=3)
    ax.set_yscale('log')
    ax.set_ylim(10 ** (-5), results_df['PSNR'].max() + 1_000)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()


def __data_to_image(*data: torch.Tensor) -> Image:
    """
    Convert multiple tensors to one big image.

    Parameters
    ----------
    data0, data1, ... dataN : torch.Tensor
        One or more tensors to be merged into a single image.

    Returns
    -------
    image : Image
        PIL image with all the tensors next to each other.
    """
    # concatenate all data
    big_pic = torch.cat([x for x in data], dim=-1)

    to_image = transforms.Compose([
        transforms.ToPILImage()
    ])

    return to_image(big_pic)
