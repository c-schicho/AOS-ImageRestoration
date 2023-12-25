# AOS - Image Restoration

### How to prepare the data

Create a new directory (e.g. `aos-data`). Within this directory create sub-folders which hold the different data
versions (e.g. `focalstack-1`). In each directory which holds the version of the data, you need to create two further
sub-folders `train` and `test` and split the data accordingly. In case you have a different structure like in the
example below, you need to provide the path to the root-data folder (e.g. `aos-data`) as an argument (`--data_path`) to
the `main.py` when running the script. The default values of the script expect the folder structure to be like the
following:

```                           
|-- <project root>     
    |-- data
        |-- aos-data
            |--- focalstack-1
                |-- train
                    <1st focal plane image>.png
                    ...
                    <nth focal plane image>.png
                    <ground truth image>.png
                    ...
                
                |-- test
                    <1st focal plane image>.png
                    ...
                    <nth focal plane image>.png
                    <ground truth image>.png
            ...
                    
    |-- model
    ...
```

### How to perform training or evaluation

Run `python main.py` in the root directory. An example for starting the training of a new model using the folder
structure described above would be  `python --train --data_name focalstack-1`. In order to see all the possible
configurations run `python main.py -h`

### How to see the results of the training

Run `tensorboard --logdir result` in the root directory.