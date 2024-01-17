import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_aos_loader
from model import AOSRestoration
from train.eval import eval_model
from utils import Config, get_device

GRADIENT_AGGREGATION = 8


def train(config: Config):
    train_data_path = os.path.join(config.data_path, config.data_name, "train")

    best_eval_psnr = float('-inf')
    best_eval_ssim = float('-inf')
    n_milestone = 0

    avg_loss = 0  # between evaluations
    n_loss = 0

    device = get_device()
    writer = SummaryWriter(config.result_path)
    model = AOSRestoration.get_model_from_config(config).to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=(config.num_iter // GRADIENT_AGGREGATION) + 1, eta_min=1e-6)

    progress_bar = tqdm(range(1, config.num_iter + 1), initial=1, dynamic_ncols=True)

    for n_iter in progress_bar:
        idx_iter = n_iter - 1
        if n_iter == 1 or idx_iter in config.milestones:
            end_iter = config.milestones[n_milestone] if n_milestone < len(config.milestones) else config.num_iter
            start_iter = config.milestones[n_milestone - 1] if n_milestone > 0 else 0
            length = config.batch_size[n_milestone] * (end_iter - start_iter)
            train_loader = get_aos_loader(
                train_data_path,
                config.batch_size[n_milestone],
                config.patch_size[n_milestone],
                length,
                config.workers,
                config.focal_planes
            )
            train_loader_iter = iter(train_loader)
            n_milestone += 1

        try:
            inputs, targets, masks = next(train_loader_iter)
        except StopIteration:
            # needed if there are not enough datapoints
            train_loader_iter = iter(train_loader)
            inputs, targets, masks = next(train_loader_iter)

        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        model.train()
        outputs = model(inputs)

        loss = __weighted_l1_loss(outputs, targets, masks)
        norm_loss = loss / 16
        norm_loss.backward()

        if n_iter % 16 == 0 or n_iter == config.num_iter:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        avg_loss = (loss.item() + n_loss * avg_loss) / (n_loss + 1)
        n_loss += 1

        progress_bar.set_description(f"Train Iter: [{n_iter}/{config.num_iter}] Loss: {avg_loss:.4f}")

        if n_iter % config.eval_period == 0 or n_iter == config.num_iter:
            writer.add_scalar(tag="train_loss", scalar_value=avg_loss, global_step=n_iter)
            eval_psnr, eval_ssim, _ = eval_model(model, config, writer, n_iter, 100)

            if config.save_each_model or eval_psnr > best_eval_psnr or eval_ssim > best_eval_ssim:
                if eval_psnr > best_eval_psnr:
                    best_eval_psnr = eval_psnr

                if eval_ssim > best_eval_ssim:
                    best_eval_ssim = eval_ssim

                current_time_in_millis = int(round(time.time() * 1000))
                model_path = os.path.join(config.result_path, f"model_{n_iter}_{current_time_in_millis}.pt")
                model.save(model_path)

            avg_loss = 0
            n_loss = 0


def __weighted_l1_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        alpha: float = 0.7
) -> torch.Tensor:
    l1 = torch.abs(inputs - targets)
    weights = ((masks * alpha) + 1)
    return torch.mean(weights * l1)
