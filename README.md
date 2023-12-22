# AOS - Image Restoration

### How to prepare the data

Split the data into two a directory with two sub-folders `test` and `train`. Provide the path to this directory as an
argument when you run the `main.py`.

### How to perform training or evaluation

Run `python main.py` in the root directory. In order to see the possible configuration run `python main.py -h`

### How to see the results of the training

Run `tensorboard --logdir result` in the root directory.