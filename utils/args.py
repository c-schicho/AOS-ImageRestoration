import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.backends import cudnn


@dataclass
class Config:
    data_path: str
    data_name: str
    result_path: str
    num_blocks: List[int]
    num_heads: List[int]
    num_focal_planes: int
    channels: List[int]
    expansion_factor: float
    num_refinement: int
    num_iter: int
    batch_size: List[int]
    test_batch_size: int
    patch_size: List[int]
    lr: float
    milestones: List[int]
    workers: int
    model_file: str
    train: bool
    eval_period: int
    run_id: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="AOS image restoration using Restormer")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--result_path", type=str, default="result")
    parser.add_argument("--num_blocks", nargs='+', type=int, default=[4, 6, 6, 8],
                        help="number of transformer blocks for each level")
    parser.add_argument("--num_heads", nargs='+', type=int, default=[1, 2, 4, 8],
                        help="number of attention heads for each level")
    parser.add_argument("--num_focal_planes", type=int, default=3)
    parser.add_argument("--channels", nargs='+', type=int, default=[48, 96, 192, 384],
                        help="number of channels for each level")
    parser.add_argument("--expansion_factor", type=float, default=2.66, help="factor of channel expansion for GDFN")
    parser.add_argument("--num_refinement", type=int, default=4, help="number of channels for refinement stage")
    parser.add_argument("--num_iter", type=int, default=300000, help="iterations of training")
    parser.add_argument("--batch_size", nargs='+', type=int, default=[16, 12, 8, 8, 4, 4],
                        help="batch size of loading images for progressive learning")
    parser.add_argument("--test_batch_size", type=int, default=1, help="batch size for evaluation of the model")
    parser.add_argument("--patch_size", nargs='+', type=int, default=[32, 40, 48, 64, 80, 96],
                        help="patch size of each image for progressive learning")
    parser.add_argument("--lr", type=float, default=0.0003, help="initial learning rate")
    parser.add_argument("--milestone", nargs='+', type=int, default=[92_000, 156_000, 204_000, 240_000, 276_000],
                        help="when to change patch size and batch size")
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers")
    parser.add_argument("--seed", type=int, default=-1, help="random seed (-1 for no manual seed)")
    parser.add_argument("--model_file", type=str, default=None, help="path of pre-trained model file")
    parser.add_argument("--train", type=bool, default=True, help="whether to train or test the model")
    parser.add_argument("--eval_period", type=int, default=1_000, help="eval after each num of iterations")
    parser.add_argument("--run_id", type=str, default=f"run_{int(round(time.time() * 1000))}",
                        help="id for the tensorboard results")

    return init_config(parser.parse_args())


def init_config(args) -> Config:
    if not os.path.exists(args.data_path):
        raise ValueError(f"data path [{args.data_path}] does not exist")

    if args.model_file is not None and not os.path.exists(args.model_file):
        raise ValueError(f"model file [{args.model_file}] does not exist")

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(*args)
