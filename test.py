from inference import generate_images
from train import train
from utils import parse_args

if __name__ == '__main__':
    config = parse_args()
    config.run_id = "D9_submission"

    if config.train:
        train(config)
    else:
        generate_images(config)
