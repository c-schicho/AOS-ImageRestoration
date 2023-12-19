import torch

from utils import Config
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.nn.functional import l1_loss
from train.eval import eval_model
from model import AOSRestoration
from torch.utils.tensorboard import SummaryWriter

import os
import time


def train(config: Config):
    best_eval_psnr = float('-inf')
    best_eval_ssim = float('-inf')
    n_milestone = 0

    writer = SummaryWriter(config.result_path)
    model = AOSRestoration.get_model_from_config(config)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.num_iter, eta_min=1e-6)

    progress_bar = tqdm(range(1, config.num_iter + 1), initial=1, dynamic_ncols=True)

    for n_iter in progress_bar:
        idx_iter = n_iter - 1
        if n_iter == 1 or idx_iter in config.milestones:
            end_iter = config.milestones[n_milestone] if n_milestone < len(config.milestones) else config.num_iter
            start_iter = config.milestones[n_milestone - 1] if n_milestone > 0 else 0
            length = config.batch_size[n_milestone] * (end_iter - start_iter)
            train_dataset = None  # TODO
            train_loader = None  # TODO
            n_milestone += 1

        inputs, targets = next(train_loader)
        inputs, targets = inputs.cuda(), targets.cuda()

        model.train()
        outputs = model(inputs)

        loss = l1_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.set_description(f"Train Iter: [{n_iter}/{config.num_iter}] Loss: {loss.item().cpu():.4f}")

        if n_iter % config.eval_period == 0 or n_iter == config.num_iter:
            writer.add_scalar(tag="train_loss", scalar_value=loss.item().cpu(), global_step=n_iter)
            writer.add_images(tag="input_images", img_tensor=inputs.cpu(), global_step=n_iter)
            writer.add_images(tag="output_images", img_tensor=outputs.cpu(), global_step=n_iter)
            writer.add_images(tag="target_images", img_tensor=targets.cpu(), global_step=n_iter)

            eval_psnr, eval_ssim = eval_model(model, config)
            writer.add_scalars(main_tag="eval_metrics", tag_scalar_dict={"psnr": eval_psnr, "ssim": eval_ssim},
                               global_step=n_iter)

            if eval_psnr > best_eval_psnr and eval_ssim > best_eval_ssim:
                current_time_in_millis = int(round(time.time() * 1000))
                model_path = os.path.join(config.result_path, f"model_{n_iter}_{current_time_in_millis}.pt")
                model.save(model_path)
