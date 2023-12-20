from PIL import Image
import AOSDataset
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from IPython.display import display

def plot_dataset(dataset: AOSDataset, N_images: int = 1, shuffle: bool = True):
    
    """
    Plot the label image and 11 feature images side by side.

    Parameters:
    """

    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    for n in range(N_images):

        feature, label = next(iter(dataloader))
        batches, feature_channels, x_size, y_size = feature.shape
        _, label_channels, _,_ = label.shape

        n_images_to_show =  feature_channels+label_channels

        # paste together
        # features
        feature_images = feature.detach().cpu()
        xs = feature_images.view(-1, 1, x_size, y_size)  # unflatten
        x_im = data_to_image(*xs)
        
        # labels
        label = label.detach().cpu()
        xl = label[0,0].view(-1, 1, x_size, y_size)  # unflatten
        x_re = data_to_image(*xl)
        
        im = Image.new('L', (n_images_to_show * x_size, y_size))
        im.paste(x_im, (y_size, 0))
        im.paste(x_re, (0, 0))
        display(im, metadata={'width': '100%'})


def data_to_image(*data: torch.Tensor, 
                  means: tuple = (0, ), 
                  stds: tuple = (1., )) -> Image:
    """
    Convert multiple tensors to one big image.
    
    Parameters
    ----------
    data0, data1, ... dataN : torch.Tensor
        One or more tensors to be merged into a single image.
    means : tuple or torch.Tensor, optional
        Original mean of the image before normalisation.
    stds : tuple or torch.Tensor, optional
        Original standard deviation of the image before normalisation.

    Returns
    -------
    image : Image
        PIL image with all of the tensors next to each other.
    """
    # concatenate all data
    big_pic = torch.cat([x for x in data], dim=-1)
    
    means = torch.tensor(means)
    stds = torch.tensor(stds)
    to_image = transforms.Compose([
        # inverts normalisation of image
        #transforms.Normalize(-means / stds, 1. / stds),
        #transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.ToPILImage()
    ])
    
    return to_image(big_pic)