import os
import time

from torch.nn.functional import l1_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataloader import get_default_test_transform, get_default_train_transform
from data.dataset import AOSDataset
from model import AOSRestoration
from train.eval import eval_model
from utils import Config


def train(config: Config):

    best_eval_psnr = float('-inf')
    best_eval_ssim = float('-inf')
    n_milestone = 0

    avg_loss = 0  # between evaluations
    n_loss = 0
    epoc = 1

    device = config.torch_device
    writer = SummaryWriter(config.result_path)
    model = AOSRestoration.get_model_from_config(config).to(device)
    model.train() # Set the model into training mode

    # load the data takes a while thus in the beginning only.
    train_data, test_data = AOSDataset(config.data_path,
                                       transform = get_default_test_transform(),
                                       focal_stack = config.focal_planes, 
                                       maximum_datasize=config.maximum_datasize,
                                       ).repeateable_split(int(1/config.test_ratio))
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.num_iter, eta_min=1e-6)

    progress_bar = tqdm(range(1, config.num_iter + 1), initial=1, dynamic_ncols=True)

    for n_iter in progress_bar:
        idx_iter = n_iter - 1

        # Load the data iterator which changes throughout the training. 
        # After the predefined milestones the batch and patch size of the data is adjusted. 
        if n_iter == 1 or idx_iter in config.milestones:
            train_data.transform = get_default_train_transform(config.patch_size[n_milestone])
            train_loader = DataLoader(train_data,
                                    batch_size=config.batch_size[n_milestone],
                                    num_workers=config.workers)
            
            train_loader_iter = iter(train_loader)
            n_milestone += 1
        
        try:
            inputs, targets = next(train_loader_iter)
        except StopIteration:
            # needed if there are not enough datapoints
            # here we computed one epoc
            epoc =+ 1
            train_loader_iter = iter(train_loader)
            inputs, targets = next(train_loader_iter)

        # Train the network 
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = l1_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Calculate the average loss
        avg_loss = (loss.item() + n_loss * avg_loss) / (n_loss + 1)
        n_loss += 1

        # Log the data onto the progressbar
        progress_bar.set_description(f"Train Iter: [{n_iter}/{config.num_iter}] Epoc: {epoc} Loss: {avg_loss:.4f}")

        # Log the data to tensorboard and save the best model
        if n_iter % config.eval_period == 0 or n_iter == config.num_iter:
            writer.add_scalar(tag="train_loss", scalar_value=avg_loss, global_step=n_iter)
            test_evaluations = min(100, len(test_data))
            eval_psnr, eval_ssim, _ = eval_model(model, test_data, config, writer, n_iter, test_evaluations)

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
