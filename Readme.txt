### FOLDER STRUCTURE ###

In order to run the following script it is important to ensure, that the following folder structure is maintained.
The data directories train, validation and test can be omitted, when the model is only used for inference.

|-- code
    |-- data
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

        |-- validation
            <1st focal plane image>.png
            ...
            <nth focal plane image>.png
            <ground truth image>.png
            ...

        |-- inference
            <1st focal plane image>.png
            ...
            <nth focal plane image>.png
            <ground truth image>.png
            ...

    |-- weights
        model.pt

    ...

    test.py



### FILE NAMINGS ###

Ensure, that the images follow this naming convention:
   # FOCAL IMAGES
   [batch_id]_[image_id]_integral_focal_[focal_plane]_cm.png

   batch_id ... positive integer (leading zeros allowed)
   image_id ... positive integer (leading zeros allowed)
   focal_plane ... [010, 050, 150] representing the focal plane in cm with leading zeros

   ** Important ** batch_id and image_id must be the same for all focal planes of the same image

   # GROUND TRUTH
   [batch_id]_[image_id]_GT_pose_0_thermal.png

   batch_id ... positive integer (leading zeros allowed)
   image_id ... positive integer (leading zeros allowed)

   ** Important ** batch_id and image_id must be the same as for the focal images

The script can also handle images with only one focal image. However, this image must still stick to the naming convention.
But it does not matter which focal plane is used in the name.



### MODEL WEIGHTS ###
The model weights can be downloaded from the following link:
https://drive.google.com/drive/folders/1-MrEd9BurmYb-QWLmZd-muUwH6NDkgYB

Store the model weights with the name model.pt in the model folder shown in the directory structure above.



### DEPENDENCIES ###
To create an environment with all dependencies run the following command in the terminal:

For windows users:
conda env create --file=environment_windows.yml

For linux users:
conda env create --file=environment_linux.yml

For mac users:
conda env create --file=environment_macos.yml

** Important ** The following command assumes, that the conda package manager is installed on your system and that your
 current working directory is the code directory.

 Once the installation is finished, activate the environment with the command provided by the cli



### USAGE ###
If you encounter some performance issues, you can try to reduce the number of workers by changing the value of the
--workers flag.

# IMAGE GENERATION
Put all the focal images in the inference folder shown in the directory structure above. Ensure that the images follow
the naming convention described above.

Use the terminal to navigate to the code directory and run the following command:

python test.py --workers 6 --model_file weights/model.pt

The generated images will be saved in the result folder. Within this folder there will be a sub-folder called
D9_submission which contains another sub-folder called inference. The images will be saved in the inference folder.


# IMAGE GENERATION WITH EVALUATION
Put all the focal images in the test folder shown in the directory structure above. Ensure that the images follow
the naming convention described above. Ensure to provide the ground truth files as well. These are used for the evaluation.

Use the terminal to navigate to the code directory and run the following command:

python test.py --workers 6 --test True --model_file weights/model.pt

The generated images will be saved in the result folder. Within this folder there will be a sub-folder called
D9_submission which contains another sub-folder called test. The images will be saved in the test folder.

** Important ** The script ignores data for which the features (inputs) or targets (ground truth) are missing.


# TRAINING
Put all the respective focal images and ground truths in the train, validation and test folder shown in the directory
structure above. Ensure that the images follow the naming convention described above.

Use the terminal to navigate to the code directory and run the following command:

python test.py --workers 6 --train True --save_each_model True

The generated tensorboard files and the saved model checkpoints will be saved in the D9_submission folder which is
located in the result folder.


# RESOURCES
You can download the used data and model weights from the following links:

Model weights:
https://drive.google.com/file/d/1vyzoRZttrulT6-cXvCR6am01xv36Qeiy/view?usp=drive_link

Test data:
https://drive.google.com/drive/folders/1hPsPrwOISQM6zEoAD9QBCvGl12Fg8nDM?usp=drive_link

Result data:
https://drive.google.com/file/d/1OxFSWgEwu6bUgPsjuY8z5_D5qkWLtLAw/view?usp=drive_link
