from inference import generate_images, generate_images_with_evaluation
from train import train, test
from utils import parse_args

if __name__ == '__main__':
    config = parse_args()
    config.run_id = "D9_submission"

    if config.train:
        print("Starting training")
        train(config)
    elif config.test:
        print("Starting generating images with evaluation")
        generate_images_with_evaluation(config)
    else:
        print("Starting generating images")
        generate_images(config)
