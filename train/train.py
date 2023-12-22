import os
import time

from torch.nn.functional import l1_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_single_aos_loader
from model import AOSRestoration
from train.eval import eval_model
from utils import Config


def train(config: Config):
    train_data_path = os.path.join(config.data_path, config.data_name, "train")

    best_eval_psnr = float('-inf')
    best_eval_ssim = float('-inf')
    n_milestone = 0

    writer = SummaryWriter(config.result_path)
    model = AOSRestoration.get_model_from_config(config).cuda()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.num_iter, eta_min=1e-6)

    progress_bar = tqdm(range(1, config.num_iter + 1), initial=1, dynamic_ncols=True)

    for n_iter in progress_bar:
        idx_iter = n_iter - 1
        if n_iter == 1 or idx_iter in config.milestones:
            end_iter = config.milestones[n_milestone] if n_milestone < len(config.milestones) else config.num_iter
            start_iter = config.milestones[n_milestone - 1] if n_milestone > 0 else 0
            length = config.batch_size[n_milestone] * (end_iter - start_iter)
            train_loader = iter(
                get_single_aos_loader(
                    train_data_path,
                    config.batch_size[n_milestone],
                    config.patch_size[n_milestone],
                    length,
                    config.workers
                )
            )
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

        progress_bar.set_description(f"Train Iter: [{n_iter}/{config.num_iter}] Loss: {loss.item():.4f}")

        if n_iter % config.eval_period == 0 or n_iter == config.num_iter:
            writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=n_iter)
            writer.add_images(tag="input_images", img_tensor=inputs.cpu(), global_step=n_iter)
            writer.add_images(tag="output_images", img_tensor=outputs.cpu(), global_step=n_iter)
            writer.add_images(tag="target_images", img_tensor=targets.cpu(), global_step=n_iter)

            eval_psnr, eval_ssim = eval_model(model, config)
            writer.add_scalars(main_tag="eval_metrics", tag_scalar_dict={"psnr": eval_psnr, "ssim": eval_ssim},
                               global_step=n_iter)

            if eval_psnr > best_eval_psnr or eval_ssim > best_eval_ssim:
                if eval_psnr > best_eval_psnr:
                    best_eval_psnr = eval_psnr

                if eval_ssim > best_eval_ssim:
                    best_eval_ssim = eval_ssim

                current_time_in_millis = int(round(time.time() * 1000))
                model_path = os.path.join(config.result_path, f"model_{n_iter}_{current_time_in_millis}.pt")
                model.save(model_path)
