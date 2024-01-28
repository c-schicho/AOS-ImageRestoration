import os
import time

import matplotlib.pyplot as plt
import torch
from pytorch_msssim import ssim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
from torchvision.utils import save_image
from tqdm import tqdm

from data import AOSSubmissionDataset
from model import AOSRestoration
from utils import Config, get_device


def generate_images(config: Config):
    n_iter = 0

    model = AOSRestoration.get_model_from_config(config).to(get_device())
    model.eval()

    data_path = os.path.join(config.data_path, config.data_name, "inference")
    dataset = AOSSubmissionDataset(data_path, config.focal_planes, False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers
    )

    output_path = os.path.join(config.result_path, "inference")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, initial=1, dynamic_ncols=True)
        for inputs in progress_bar:
            inputs = inputs.to(model.device)
            outputs = model(inputs)

            current_time_in_millis = int(round(time.time() * 1000))
            image_path = os.path.join(output_path, f"{n_iter}_{current_time_in_millis}.png")
            save_image(outputs, image_path)


def generate_images_with_evaluation(config: Config):
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    n_iter = 0

    model = AOSRestoration.get_model_from_config(config).to(get_device())
    model.eval()

    data_path = os.path.join(config.data_path, config.data_name, "test")
    dataset = AOSSubmissionDataset(data_path, config.focal_planes, True)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers
    )

    output_path = os.path.join(config.result_path, "test")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, initial=1, dynamic_ncols=True)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)

            total_psnr += __calculate_psnr(outputs, targets, 1.0)
            total_ssim += ssim(outputs, targets, 1.0).item()
            total_mse += mse_loss(outputs, targets).item()
            n_iter += 1

            current_time_in_millis = int(round(time.time() * 1000))
            image_path = os.path.join(output_path, f"{n_iter}_{current_time_in_millis}.png")
            save_image(outputs, image_path)

    avg_psnr = round(total_psnr / n_iter, 6)
    avg_ssim = round(total_ssim / n_iter, 6)
    avg_mse = round(total_mse / n_iter, 6)

    x = ['PSNR', 'SSIM', 'MSE']
    y = [avg_psnr, avg_ssim, avg_mse]

    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
    plt.savefig(os.path.join(output_path, "evaluation_plot.png"))

    with open(os.path.join(output_path, "evaluation.txt"), "w") as f:
        f.write(f"PSNR: {avg_psnr}\n")
        f.write(f"SSIM: {avg_ssim}\n")
        f.write(f"MSE: {avg_mse}\n")


def __calculate_psnr(outputs: torch.Tensor, targets: torch.Tensor, data_range: float, eps: float = 10e-2) -> float:
    psnr_value = psnr(outputs, targets, data_range)

    if psnr_value == float('inf'):
        # this happens when the images are exactly the same so the mse is equal to 0
        psnr_value = psnr(torch.add(outputs, eps), targets, data_range)

    return psnr_value.item()
