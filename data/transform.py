import PIL
import numpy as np
import torch


def invert_channels(img):
    """
    Invert the values of each channel of an image.

    Args:
        img (PIL.Image.Image or torch.Tensor): Input image.

    Returns:
        PIL.Image.Image or torch.Tensor: Inverted image.
    """
    if isinstance(img, torch.Tensor):
        inverted_img = 1 - img
    else:
        inverted_img = img.copy()
        inverted_img = PIL.Image.fromarray(255 - np.array(inverted_img))

    return inverted_img
