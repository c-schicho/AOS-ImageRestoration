import os
from typing import Tuple

import torch
from pytorch_msssim import ssim
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from data import get_single_aos_loader
from model import AOSRestoration
from utils import Config


def eval(config: Config):
    model = AOSRestoration.get_model_from_config(config)
    eval_model(model, config)


def eval_model(model: AOSRestoration, config: Config) -> Tuple[float, float]:
    total_psnr = 0.0
    total_ssim = 0.0
    n_iter = 0

    test_data_path = os.path.join(config.data_path, config.data_name, "test")
    data_loader = get_single_aos_loader(
        test_data_path,
        config.test_batch_size,
        512,
        100,  # TODO
        config.workers,
        False
    )

    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)

            total_psnr += psnr(outputs, targets)
            total_ssim += ssim(outputs, targets)

            n_iter += 1

            avg_psnr = total_psnr / n_iter
            avg_ssim = total_ssim / n_iter
            progress_bar.set_description(
                f"Test Iter: [{n_iter}/{len(data_loader)}] PSNR: {avg_psnr:2f} SSIM: {avg_ssim:3f}"
            )

    return avg_psnr, avg_ssim
