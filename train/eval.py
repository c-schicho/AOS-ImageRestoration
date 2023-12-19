from typing import Tuple

import torch
from pytorch_msssim import ssim
from tqdm import tqdm

from model import AOSRestoration
from utils import Config


def eval(config: Config):
    model = AOSRestoration.get_model_from_config(config)
    eval_model(model, config)


def eval_model(model: AOSRestoration, config: Config) -> Tuple[float, float]:
    total_psnr = 0.0
    total_ssim = 0.0
    n_iter = 0

    data_loader = get_test_loader(batch_size=config.test_batch_size)  # TODO
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


def psnr(output: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> float:
    output /= data_range
    target /= target
    mse = torch.mean((output - target) ** 2)
    score = -10 * torch.log10(mse)
    return score
