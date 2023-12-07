AOS Installation Procedure
---------------------------

** Install visual studio community for windows : https://visualstudio.microsoft.com/vs/community/
Then, install desktop development with c++

** Install visual studio code for windows : https://code.visualstudio.com

** Install python 3.7.9:  https://www.python.org/downloads/release/python-379/      (scroll down to the bottom to find the link : Windows x86-64 executable installer)

** Download or clone the LFR folder from the github page:  https://github.com/JKU-ICG/AOS/tree/stable_release/AOS%20for%20Drone%20Swarms/LFR

** Install extensions from within VS code. Bring up the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of VS Code or the View: Extensions command (Ctrl+Shift+X). For more details :      
   https://code.visualstudio.com/docs/editor/extension-marketplace

	- Install python and Jupyter extensions.

**  Open the Computervision_project.ipynb script ("Make sure that this script is inside the LFR/python folder") in the VS code. Select and set kernel on the top right to python 3.7.9. Open a new terminal (Terminal > New Terminal). 

**  Set the working directory to LFR/python. Now run the command pip install -r requirements.txt in the terminal. This will take a few minutes.

**   Run the command setup_Win.py build_ext --inplace with python as your working directory and make sure that your path is pointing to your python 3.7.9 installation. Open the terminal then write: python setup_Win.py build_ext --inplace

	- It should loo like this: PS D:\LFR\python> & C:/Users/Rakesh/AppData/Local/Programs/Python/Python37/python.exe setup_Win.py build_ext --inplace


** You can now run the AOS integrator script. Please make changes to the script to change path of the images directory and so on. Follow the comments in the script.


** Changing the focal plane in the AOS integrator script: change the value of the variable Focal_plane. By default it is 0 (on the ground). If you want it at, for example, 3m above the ground, you need to use Focal_plane=-3 --> note that distance above the ground need negative values. A good focus range that covers all the target persons is 0..-3 (i.e., from 0 to 3m above ground). But not that a lying person, sitting person, and standing person vary in height.



AOS Training Database 
---------------------

rev_27_10_2023


The training data is split into multiple runs, and each run is split into two batches: batch_DATE_part1.7z and batch_DATE_part2.7z

Each batch contains 5.500 simulations (each run sums up to 11.000 simulations).

After unzipping a batch (https://www.7-zip.org/) the folder batch_DATE_part1OR2/Part1OR2 contains 13 files per simulation (R=run number, S=simulation number):

R_S_pose_0..10_thermal.png 	--> the 11 singe drone images where pose_5 is the center perspective 
R_S_GT_pose_0_thermal.png  	--> is the ground truth image (corresponds to pose_5, but without forest)
R_S_Parameters.txt		--> is the parameter file which contains the applied simulation parameters

IMPORTANT NOTICE: if a simulation does not contain all of its 13 files, then it should not be used. In rare cases, the simulator did not write out all files.

The parameter file is organized as follows (this is an example for 0_1372_Parameters.txt):

---------------------------------------------------------------

img_GT (0, 0, 35, 0.0, 1.57, 0)		--> pose for ground truth image (R_S_pose_0..10_thermal.png), (y,x,z,rot y,rot x,rot z), z=altitude in meters above ground level, (0.0, 1.57, 0)=camera pointing downwards  

img_1 (0, -5, 35, 0.0, 1.57, 0)		--> poses for other perspectives (as above), (y,x,z) in meters, (rot y,rot x,rot z). The x and y values have to be multiplied with (-) sign.   
img_2 (0, -4, 35, 0.0, 1.57, 0)
img_3 (0, -3, 35, 0.0, 1.57, 0)
img_4 (0, -2, 35, 0.0, 1.57, 0)
img_5 (0, -1, 35, 0.0, 1.57, 0)
img_6 (0, 0, 35, 0.0, 1.57, 0)		--> poses for center perspective (equals pose for img_GT) 
img_7 (0, 1, 35, 0.0, 1.57, 0)
img_8 (0, 2, 35, 0.0, 1.57, 0)
img_9 (0, 3, 35, 0.0, 1.57, 0)
img_10 (0, 4, 35, 0.0, 1.57, 0)
img_11 (0, 5, 35, 0.0, 1.57, 0)		--> poses for other perspectives (as above), (x,y,z) in meters, (rot x,rot y,rot z)  

numbers of tree per ha=  100		--> simulated forest density in trees/ha

person shape =  laying							--> simulated person shape (either laying, sitting, idle=standing, or no person), the following 3 person pose parameters are not provided for "no person"
person pose (x,y,z,rot x, rot y, rot z) =  4 -6 0 0 0 -0.0174533 	--> pose of person (x,y,z)=position in meters, (rot x, rot y, rot z)=rotation 
person rotation (z) in radian =  6.265732014659643			--> rotation around z in rad
person rotation (z) in degree =  359.0					--> rotation around z in deg

ambient light =  0.6497614048239921					--> amount of ambient light

azimuth angle of sun light in degrees =  23.0				--> azimuth angle of sun in deg
compass direction of sunlight in degrees =  224.0			--> compass direction of sun in deg

ground surface temperature in kelvin =  286				--> ground surface temperature in kelvin
tree top temperature in kelvin =  306					--> tree top temperature in kelvin

---------------------------------------------------------------

Differences between simulation runs:

For all simulation runs, the simulation parameters were randomly varied in the following ranges:

- numbers of tree per ha=  0,100 (run 20230912 and 20230919) and 0,200 (run 20231027) 	--> Note that only 10% of all simulations have no forest (0 trees/ha).
- person shape = "sitting", "laying", "idle", "no person" 				--> Note that only 10% of all simulations have no person (only forest).
- person pose (x,y) = -10 ... +10m
- person rotation (z) in degree 0 ... 360 (equivalent in rad)
- ambient light =  0.5 ... 1
- azimuth angle of sunlight direction in degree = 0 ... 45 
- compass direction of sunlight in degree = 0 ... 360
- ground surface temperature in kelvin = 260 ... 312
- tree top temperature in kelvin = 295 ... 312

---------------------------------------------------------------

Real data:

You can find real integral image data in the folder real_integrals. 

For integrals with a focal plane at 0m (ground), the file names are encoded as follows: sceneindex_syntehticaperturesize_samplesinsyntehticaperture.
So, 1_5m_11s means: scene 1, 5m synthetic aperture (diameter), sampled with 11 images (at equal distances).

So far, there are no focal stack examples provided. Teams that experiment with focal stacks need to contact us with informtaion about their sampling strategy (step size, equal/unequal steps). We will then produce corresponding forcal stacks of this data on request. Note, that focal stack ranges are fixed to 0..3m ave ground.  

For the data in the Nature_Paper folder, please see details here:
https://www.nature.com/articles/s42256-020-00261-3.epdf?sharing_token=CkVF30c-ohDFg7Bfz7vbXNRgN0jAjWel9jnR3ZoTv0Njw2M16sXA0c1i0-K0I8hyWAyPHw0VoEqSzrkBwYYyW6fhTSE6UR1hLVXodIJxrUXLGCuefrcgODgq7zmQeEDTqcs5bDAPpwteMKEXPcztPtUexI1JTEkxXmS4opWo-LA%3D



