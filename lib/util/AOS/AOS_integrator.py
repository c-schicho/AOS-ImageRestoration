
## Import libraries section ##
import numpy as np
import cv2
import os
import math
from LFR_utils import read_poses_and_images, pose_to_virtualcamera, init_aos, init_window
import LFR_utils as utils
import pyaos
import glm


## path to where the results will be stored 

Download_Location = r'ENTER PATH'    ## Enter path to the directory where you want to save the results.
print(Download_Location)
Integral_Path = os.path.join(Download_Location,'integrals') # Note that your results will be saved to this integrals folder.

# Check if the directory already exists
if not os.path.exists(Integral_Path):
    os.mkdir(Integral_Path)
else:
    print(f"The directory '{Integral_Path}' already exists.")


#############################Start the AOS Renderer###############################################################
w,h,fovDegrees = 512, 512, 50 # # resolution and field of view. This should not be changed.
render_fov = 50

if 'window' not in locals() or window == None:
                                    
    window = pyaos.PyGlfwWindow( w, h, 'AOS' )  
     
aos = pyaos.PyAOS(w,h,fovDegrees) 


set_folder = r'Enter path to your LFR/python directory'          # Enter path to your LFR/python directory
aos.loadDEM( os.path.join(set_folder,'zero_plane.obj'))

####################################################################################################################

#############################Create Poses for Initial Positions###############################################################

# Below are certain functions required to convert the poses to a certain format to be compatabile with the AOS Renderer.

def eul2rotm(theta) :
    s_1 = math.sin(theta[0])
    c_1 = math.cos(theta[0]) 
    s_2 = math.sin(theta[1]) 
    c_2 = math.cos(theta[1]) 
    s_3 = math.sin(theta[2]) 
    c_3 = math.cos(theta[2])
    rotm = np.identity(3)
    rotm[0,0] =  c_1*c_2
    rotm[0,1] =  c_1*s_2*s_3 - s_1*c_3
    rotm[0,2] =  c_1*s_2*c_3 + s_1*s_3

    rotm[1,0] =  s_1*c_2
    rotm[1,1] =  s_1*s_2*s_3 + c_1*c_3
    rotm[1,2] =  s_1*s_2*c_3 - c_1*s_3

    rotm[2,0] = -s_2
    rotm[2,1] =  c_2*s_3
    rotm[2,2] =  c_2*c_3        

    return rotm

def createviewmateuler(eulerang, camLocation):
    
    rotationmat = eul2rotm(eulerang)
    translVec =  np.reshape((-camLocation @ rotationmat),(3,1))
    conjoinedmat = (np.append(np.transpose(rotationmat), translVec, axis=1))
    return conjoinedmat

def divide_by_alpha(rimg2):
        a = np.stack((rimg2[:,:,3],rimg2[:,:,3],rimg2[:,:,3]),axis=-1)
        return rimg2[:,:,:3]/a

def pose_to_virtualcamera(vpose ):
    vp = glm.mat4(*np.array(vpose).transpose().flatten())
    #vp = vpose.copy()
    ivp = glm.inverse(glm.transpose(vp))
    #ivp = glm.inverse(vpose)
    Posvec = glm.vec3(ivp[3])
    Upvec = glm.vec3(ivp[1])
    FrontVec = glm.vec3(ivp[2])
    lookAt = glm.lookAt(Posvec, Posvec + FrontVec, Upvec)
    cameraviewarr = np.asarray(lookAt)
    #print(cameraviewarr)
    return cameraviewarr  



########################## Below we generate the poses for rendering #####################################
# This is based on how renderer is implemented. 

Numberofimages = 11  # Or just the number of images
Focal_plane = 0       # Focal plane is set to the ground so it is zero.

# ref_loc is the reference location or the poses of the images. The poses are the same for the dataset and therefore only the images have to be replaced.

ref_loc = [[5,4,3,2,1,0,-1,-2,-3,-4,-5],[0,0,0,0,0,0,0,0,0,0,0]]   # These are the x and y positions of the images. It is of the form [[x_positions],[y_positions]]

altitude_list = [35,35,35,35,35,35,35,35,35,35,35] # [Z values which is the height]

center_index = 5  # this is important, this will be the pose index at which the integration should happen. For example if you have 5 images, lets say you want to integrate all 5 images to the second image position. Then your center_index is 1 as index starts from zero.

site_poses = []
for i in range(Numberofimages):
    EastCentered = (ref_loc[0][i] - 0.0) #Get MeanEast and Set MeanEast
    NorthCentered = (0.0 - ref_loc[1][i]) #Get MeanNorth and Set MeanNorth
    M = createviewmateuler(np.array([0.0, 0.0, 0.0]),np.array( [ref_loc[0][i], ref_loc[1][i], - altitude_list[i]] ))
    print('m',M)
    ViewMatrix = np.vstack((M, np.array([0.0,0.0,0.0,1.0],dtype=np.float32)))
    print(ViewMatrix)
    camerapose = np.asarray(ViewMatrix.transpose(),dtype=np.float32)
    print(camerapose)
    site_poses.append(camerapose)  # site_poses is a list now containing all the poses of all the images in a certain format that is accecpted by the renderer.
    
    
#############################Read the generated images from the simulator and store in a list ###############################################################

import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

imagelist = []

import glob

for img in sorted(glob.glob(r"Enter path to your images directory" + '/*.png'),key=numericalSort):      # Enter path to the images directory which should contain 11 images.
    n= cv2.imread(img)
    imagelist.append(n)

    
    
aos.clearViews()   # Every time you call the renderer you should use this line to clear the previous views  
for i in range(len(imagelist)):
        aos.addView(imagelist[i], site_poses[i], "DEM BlobTrack")  # Here we are adding images to the renderer one by one.
aos.setDEMTransform([0,0,Focal_plane])

proj_RGBimg = aos.render(pose_to_virtualcamera(site_poses[center_index]), render_fov)
tmp_RGB = divide_by_alpha(proj_RGBimg)
cv2.imwrite(os.path.join( Integral_Path, 'integral.png'), tmp_RGB)   # Final result. Check the integral result in the integrals folder.

