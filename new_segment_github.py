# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:20:09 2020

@author: Hamed
"""
#######################
#Dark Object Detection in Ocean
#This code implements an adaptive thresholding method for detecting dark objects
#in ocean using SAR images. The dark objects can be oil spills or look-alikes that exhibit
#similar behavior to oil spills. The adaptive thresholding method analyzes the pixel intensities in the image and
#determines a threshold value dynamically. Pixels with intensities below the threshold
#are classified as potential dark objects. This approach is effective in identifying
#oil spills and similar phenomena in ocean images.
#For more information and detailed implementation, please refer to the research paper:
#A. S. Solberg, C. Brekke, and R. Solberg, "Algorithms for oil spill detection in
#Radarsat and ENVISAT SAR images", Geoscience and Remote Sensing.

# Note that your input image should include backscattering value in db (use: backscatter_dB = 10 * np.log10(backscatter))

#import pandas as pd
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from skimage import morphology
from joblib import Parallel, delayed
from time import time
from astropy.convolution import convolve,Box1DKernel

################### functions

def conv_nan(img,kernel_size):
    # 1D-convolution function (this function can handle nan valu of the image)
    # although convolution in the adaptive thresholding for dark-object detection in ocean is a 2D-convolution, 
    # to improve the speed of the code, we run two 1-D convolution (one over the rows of the image and one over the columns of it), which has the same result as a 2D-convolution
    #
    # img: a line of imge
    kernel=Box1DKernel(kernel_size)
    s=[]
    for i in img:
        s.append(convolve(i, kernel))
    return np.asarray(s)

def adaptiveThresh_par(img,kernel_size,offset):
    #Parallel adaptive thresholding: this code run the 2D adaptive thresholding on the image. It run the algorithm using parallel processing
    # offset is the threshold value used in the adaptive thresholding
    
    ################## Find number of CPU available for parallel processing
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
   
    ############ padding
    # padd the image for convolution
    a=np.argwhere(np.isnan(img))
    shape=img.shape
    off=round(kernel_size/2)
    img=np.pad(img,pad_width=off, mode='edge')## padding
    
    ################################### Convlution

    batch_Size=round(len(img)/num_cpus) # batch size for parallel processing
    dis=Parallel(n_jobs=num_cpus)(delayed(conv_nan)(img[i: i + batch_Size,:],kernel_size)  
                                                 for i in range(0, len(img), batch_Size)) # run 1-D convolution on the rows of image
    del img

    im=np.concatenate( dis, axis=0 )
    im=im.T
    
    del dis
    

    batch_Size=round(len(im)/num_cpus)
    dis=Parallel(n_jobs=num_cpus)(delayed(conv_nan)(im[i: i + batch_Size,:],kernel_size)
                                                 for i in range(0, len(im), batch_Size))  # run 1-D convolution on the columns of image
    im=np.concatenate( dis, axis=0 )
    im=im.T
    
    ################################## unpadding
    unpad=im[off:off+shape[0],off:off+shape[1]]
    unpad[a[:,0],a[:,1]]=np.nan
    del im,dis
    return unpad+offset



#################### prameters
im_name='Khark_incCorrect.tif'   # the name of the input image 
thresh=-1.5 # threshold parameter in the Adaptive Thresholding algorithm
area_thres=1000 # this threshold is used to remove the detected dark object whose area is bellow this threshold
ker_size=[160] # kernel size. You can use multiple kernel size in a list (e.g., [9,15,61,...]). This code will run the algorithm using all the kernel size and then merge the result. I recommend using multiple kernel size
#ker_size=[9,15,61,95,151,171,251,501,901]

st=time()
plt.close('all')

##################### load image
dataset = gdal.Open(im_name)
b = np.array(dataset.GetRasterBand(1).ReadAsArray())# image in db (backscatter_dB = 10 * np.log10(backscatter))

#####################

imm=[]

for i in ker_size: # run for each kernel size
    binary_adaptive=adaptiveThresh_par(b,i,thresh)  
    imm.append(b<=binary_adaptive)
    print (i)

########## merge the results of each kernel with different size 
for i,j in enumerate(imm):
    if i==0:
        s=j.astype('uint8')
    else:
        s=s+j.astype('uint8')
   
 
s=(s>=1) # binary image whtich shows the detected dark objects 


################### remove the detected objects with small area
s=morphology.remove_small_objects(s, min_size=area_thres, connectivity=1, in_place=False)
#####################
print(time()-st)

plt.imshow(s)
plt.figure()
plt.imshow(b)



#np.save(product_loc+'/'+name+'_seg.npy',s)
