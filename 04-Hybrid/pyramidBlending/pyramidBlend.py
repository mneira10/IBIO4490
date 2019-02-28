#!/usr/bin/python3
# coding: utf-8

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import imsave
import os

def pause():
    input("Press the <ENTER> key to continue.")
    
def gaussian_pyramid(image,i):
    gaussian_pyramid = [image]
    for n in range(0,i):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    
    return gaussian_pyramid
def laplacian_pyramid(gaussian_pyramid,i):
    laplacian_pyramid = []
    
    laplacian_pyramid.append(gaussian_pyramid[i])
    for n in range(i,0,-1):
        image = cv2.pyrUp(gaussian_pyramid[n])
     
        laplacian = cv2.subtract(gaussian_pyramid[n-1],image)
        laplacian_pyramid.append(laplacian)
    
    return laplacian_pyramid
def hybrid_image(image1_laplacian,image2_laplacian,i):
    
    hybrid_image = []
    
    for n in range(0,i+1):
        A = image1_laplacian[n]
        B = image2_laplacian[n]
        x,y,z = A.shape
        if n==0: 
            
            hybrid_image = np.hstack((A[:,0:int(y/2),:],B[:,int(y/2):,:]))   
        else:
            background = cv2.pyrUp(hybrid_image)
            details = np.hstack((A[:,0:int(y/2),:],B[:,int(y/2):,:]))
            hybrid_image = cv2.add(background,details)




    return hybrid_image
## Parameters
name_file_1 = 'naruto.png'
name_file_2 = 'goku.png'
i = 4
## Searching and resizing images
image1 = cv2.imread(name_file_1)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1= cv2.resize(image1, (256, 256))
image2 = cv2.imread(name_file_2)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2= cv2.resize(image2, (256, 256))
## Let the magic happen
image1_gaussian = gaussian_pyramid(image1,i)
image2_gaussian = gaussian_pyramid(image2,i)
image1_laplacian = laplacian_pyramid(image1_gaussian,i)
image2_laplacian = laplacian_pyramid(image2_gaussian,i)
background = hybrid_image(image1_laplacian, image2_laplacian,0)

#Final image
image_hybrid = hybrid_image(image1_laplacian, image2_laplacian,4)

## Plotting
imsave('pyramidBlend.png',image_hybrid)
os.system('display ./pyramidBlend.png')
