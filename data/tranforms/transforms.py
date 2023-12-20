import PIL
import numpy as np
import torch


class InvertChannels:
    def __call__(self, img):
        """
        Invert the values of each channel of an image.

        Args:
            img (PIL.Image.Image or torch.Tensor): Input image.

        Returns:
            PIL.Image.Image or torch.Tensor: Inverted image.
        """
        if isinstance(img, torch.Tensor):
            # If input is a torch tensor
            inverted_img = 1 - img
        else:
            # If input is a PIL image
            inverted_img = img.copy()
            inverted_img = PIL.Image.fromarray(255 - np.array(inverted_img))

        return inverted_img