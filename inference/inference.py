import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data import AOSInferenceDataset
from model import AOSRestoration
from utils import Config, get_device


def generate_images(config: Config):
    n_iter = 0

    model = AOSRestoration.get_model_from_config(config).to(get_device())
    model.eval()

    data_path = os.path.join(config.data_path, config.data_name, "inference")
    dataset = AOSInferenceDataset(data_path, config.focal_planes)
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
