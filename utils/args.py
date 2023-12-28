import argparse
import os
import random
import time
from typing import List

import numpy as np
import torch
from torch.backends import cudnn
from utils.device import init_device


class Config:

    def __init__(self, args):
        self.data_path: str = args.data_path
        self.data_name: str = args.data_name
        self.result_path: str = args.result_path
        self.num_blocks: List[int] = args.num_blocks
        self.num_heads: List[int] = args.num_heads
        self.num_focal_planes: int = args.num_focal_planes
        self.focal_planes: List[int] = args.focal_planes
        self.channels: List[int] = args.channels
        self.expansion_factor: float = args.expansion_factor
        self.num_refinement: int = args.num_refinement
        self.num_iter: int = args.num_iter
        self.batch_size: List[int] = args.batch_size
        self.test_batch_size: int = args.test_batch_size
        self.patch_size: List[int] = args.patch_size
        self.lr: float = args.lr
        self.milestones: List[int] = args.milestones
        self.workers: int = args.workers
        self.model_file: str = args.model_file
        self.train: bool = args.train
        self.eval_period: int = args.eval_period
        self.run_id: str = args.run_id
        self.save_each_model: bool = args.save_each_model
        self.torch_device: torch.device = init_device(args.torch_device)
        self.test_ratio: float = 0.05 # fixed ratio gives a fixed subset for the testset. 
        self.num_test_iter: int = args.num_test_iter
        self.maximum_datasize: int = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="AOS image restoration using Restormer")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str, default="data/aos-data")
    parser.add_argument("--result_path", type=str, default="result")
    parser.add_argument("--num_blocks", nargs='+', type=int, default=[4, 6, 6, 8],
                        help="number of transformer blocks for each level")
    parser.add_argument("--num_heads", nargs='+', type=int, default=[1, 2, 4, 8],
                        help="number of attention heads for each level")
    parser.add_argument("--focal_planes", nargs='+', type=int, default=[10, 50, 150], help="focal planes")
    parser.add_argument("--channels", nargs='+', type=int, default=[48, 96, 192, 384],
                        help="number of channels for each level")
    parser.add_argument("--expansion_factor", type=float, default=2.66, help="factor of channel expansion for GDFN")
    parser.add_argument("--num_refinement", type=int, default=4, help="number of channels for refinement stage")
    parser.add_argument("--num_iter", type=int, default=50_000, help="iterations of training")
    parser.add_argument("--batch_size", nargs='+', type=int, default=[12, 8, 4, 4, 2, 2],
                        help="batch size of loading images for progressive learning")
    parser.add_argument("--test_batch_size", type=int, default=1, help="batch size for evaluation of the model")
    parser.add_argument("--patch_size", nargs='+', type=int, default=[48, 56, 72, 96, 120, 144],
                        help="patch size of each image for progressive learning")
    parser.add_argument("--lr", type=float, default=0.0003, help="initial learning rate")
    parser.add_argument("--milestones", nargs='+', type=int, default=[10_000, 20_000, 30_000, 40_000, 50_000], 
                        help="number of epocs to patch size and batch size change")
    parser.add_argument("--workers", type=int, default=0, help="number of data loading workers")
    parser.add_argument("--seed", type=int, default=-1, help="random seed (-1 for no manual seed)")
    parser.add_argument("--model_file", type=str, default="best_model.pt", help="path of pre-trained model file")
    parser.add_argument("--train", type=bool, default=False, help="whether to train or test the model")
    parser.add_argument("--eval_period", type=int, default=500, help="eval after each num of iterations")
    parser.add_argument("--run_id", type=str, default=f"run_{int(round(time.time() * 1000))}",
                        help="id for the tensorboard results")
    parser.add_argument("--save_each_model", type=bool, default=False,
                        help="whether to save each model or only the best")
    parser.add_argument("--torch_device", type=str, default= 'cuda' if torch.cuda.is_available() else 'cpu',
                        help="specify which torch device should be used.")
    #parser.add_argument("--test_ratio", type=float, default=0.05,
    #                    help="Specify the dataset ratio that should be held aside for testing.") # should not be an option cause we would like to fix it
    parser.add_argument("--num_test_iter", type=int, default=100, help="iterations of testing")
    parser.add_argument("--maximum_datasize", type=int, default=None, help="limits the dataset size")
    

    return init_config(parser.parse_args())


def init_config(args) -> Config:
    if not os.path.exists(args.data_path):
        raise ValueError(f"data path [{args.data_path}] does not exist")

    if args.model_file is not None and not os.path.exists(args.model_file):
        raise ValueError(f"model file [{args.model_file}] does not exist")

    results_path = os.path.join(args.result_path, args.run_id)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    setattr(args, "result_path", results_path)
    setattr(args, "num_focal_planes", len(args.focal_planes))

    return Config(args)
