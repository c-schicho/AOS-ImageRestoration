from train import train, eval
from utils import parse_args

if __name__ == "__main__":
    config = parse_args()

    if config.train:
        train(config)
    else:
        eval(config)
