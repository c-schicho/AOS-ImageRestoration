import os
from typing import Tuple, Union

import torch
from pytorch_msssim import ssim
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
             .load(config.model_file))
    eval_model(model, config, writer)


def eval_model(
        model: AOSRestoration,
        config: Config,
        writer: Union[SummaryWriter, None] = None,
        step: int = 0,
        n_images: Union[int, None] = None
) -> Tuple[float, float]:
    total_psnr = 0.0
    total_ssim = 0.0
    n_iter = 0

    test_data_path = os.path.join(config.data_path, config.data_name, "test")
    data_loader = get_single_aos_loader(
        test_data_path,
        config.test_batch_size,
        512,
        n_images,
        config.workers
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

            if writer is not None and n_iter % 10 == 0:
                writer_step = step + n_iter
                writer.add_images(tag="eval_input_images", img_tensor=inputs.cpu(), global_step=writer_step)
                writer.add_images(tag="eval_output_images", img_tensor=outputs.cpu(), global_step=writer_step)
                writer.add_images(tag="eval_target_images", img_tensor=targets.cpu(), global_step=writer_step)

    return avg_psnr, avg_ssim
