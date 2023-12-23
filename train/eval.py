import os
from typing import Tuple, Union

import torch
from pytorch_msssim import ssim
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from data import get_single_aos_loader
from model import AOSRestoration
from utils import Config


def eval(config: Config):
    writer = SummaryWriter(config.result_path)
    model = (AOSRestoration
             .get_model_from_config(config)
             .cuda())
    model.load(config.model_file)
    print('model checkpoint loaded successfully')
    eval_model(model, config, writer)


def eval_model(
        model: AOSRestoration,
        config: Config,
        writer: Union[SummaryWriter, None] = None,
        step: int = 0,
        n_images: Union[int, None] = None
) -> Tuple[float, float, float]:
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    n_iter = 0

    test_data_path = os.path.join(config.data_path, config.data_name, "test")
    data_loader = get_single_aos_loader(
        test_data_path,
        config.test_batch_size,
        512,
        n_images,
        config.workers,
        config.focal_planes,
        False
    )

    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)

            data_range = __calculate_data_range(outputs)
            total_psnr += __calculate_psnr(outputs, targets, data_range)
            total_ssim += ssim(outputs, targets, data_range)
            total_mse += mse_loss(outputs, targets)

            n_iter += 1

            avg_psnr = total_psnr / n_iter
            avg_ssim = total_ssim / n_iter
            avg_mse = total_mse / n_iter
            progress_bar.set_description(
                f"Test Iter: [{n_iter}/{len(data_loader)}] PSNR: {avg_psnr} SSIM: {avg_ssim} MSE: {avg_mse}"
            )

            if writer is not None and n_iter % 10 == 0:
                writer_step = step + n_iter
                writer.add_images(tag="eval_input_images", img_tensor=inputs.cpu(), global_step=writer_step)
                writer.add_images(tag="eval_output_images", img_tensor=outputs.cpu(), global_step=writer_step)
                writer.add_images(tag="eval_target_images", img_tensor=targets.cpu(), global_step=writer_step)

    if writer is not None:
        writer.add_scalars(
            main_tag="eval_metrics",
            tag_scalar_dict={"psnr": avg_psnr, "ssim": avg_ssim, "mse": avg_mse},
            global_step=n_iter
        )

    return avg_psnr, avg_ssim, avg_mse


def __calculate_data_range(outputs: torch.Tensor, eps: float = 10e-2) -> float:
    data_range = (torch.max(outputs) - torch.min(outputs)).item()

    if data_range == 0:
        data_range = eps

    return data_range


def __calculate_psnr(outputs: torch.Tensor, targets: torch.Tensor, data_range: float, eps: float = 10e-2) -> float:
    psnr_value = psnr(outputs, targets, data_range)

    if psnr_value == float('inf'):
        psnr_value = psnr(torch.add(outputs, eps), targets, data_range)

    return psnr_value.item()
