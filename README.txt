### FOLDER STRUCTURE ###

In order to run the following script it is important to ensure, that the following folder structure is maintained.
The data directories train and test can be omitted, when the model is only used for inference.

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

        |-- inference
            <1st focal plane image>.png
            ...
            <nth focal plane image>.png
            <ground truth image>.png
            ...

    |-- model
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



### MODEL WEIGHTS ###
The model weights can be downloaded from the following link:
https://drive.google.com/drive/folders/1-MrEd9BurmYb-QWLmZd-muUwH6NDkgYB

Store the model weights with the name model.pt in the model folder shown in the directory structure above.



### DEPENDENCIES ###
To create an environment with all dependencies run the following command in the terminal:

conda env create --file=environment.yml

** Important ** The following command assumes, that the conda package manager is installed on your system and that your
 current working directory is the code directory.



### USAGE ###

# IMAGE GENERATION
Put all the focal images in the inference folder shown in the directory structure above. Ensure that the images follow
the naming convention described above.

Use the terminal to navigate to the code directory and run the following command:

python test.py --model_file model/model.pt

The generated images will be saved in the result folder. Within this folder there will be a sub-folder called
D9_submission which contains another sub-folder called inference. The images will be saved in the inference folder.


# TRAINING
Put all the respective focal images in the train and test folder shown in the directory structure above. Ensure that
the images follow the naming convention described above.

Use the terminal to navigate to the code directory and run the following command:

python test.py --workers 6 --train True --save_each_model True

The generated tensorboard files and the saved model files will be saved in the D9_submission folder which is located in
the result folder.
