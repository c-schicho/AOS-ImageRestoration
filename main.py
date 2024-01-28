from train import train, test
from utils import parse_args

if __name__ == "__main__":
    config = parse_args()

    if config.train:
        train(config)
    else:
        test(config)
