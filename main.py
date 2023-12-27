from train import train, eval
from utils import parse_args
import warnings

# Suppress the specific warning about 'aten::sgn.out'
warnings.filterwarnings("ignore", message="The operator 'aten::sgn.out' is not currently supported on the MPS backend*")


if __name__ == "__main__":
    config = parse_args()

    if config.train:
        train(config)
    else:
        if config.model_file is None:
            raise ValueError("no model file provided for testing")

        eval(config)
