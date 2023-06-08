# Dark Object Detection in Ocean using SAR images
#This code implements an adaptive thresholding method for detecting dark objects in ocean using SAR images. The dark objects can be oil spills or look-alikes that exhibit similar behavior to oil spills. The adaptive thresholding method analyzes the pixel intensities in the image and determines a threshold value dynamically. Pixels with intensities below the threshold #are classified as potential dark objects. This approach is effective in identifying oil spills and similar phenomena in ocean images.

For more information and detailed implementation, please refer to the research paper:
- A. S. Solberg, C. Brekke, and R. Solberg, "Algorithms for oil spill detection in #Radarsat and ENVISAT SAR images", Geoscience and Remote Sensing.

## Note: 
- Your input image should include backscattering value in db (use: backscatter_dB = 10 * np.log10(backscatter))
- Due to the large size of SAR data, I was unable to upload an image sample using this code. If you require a sample, please feel free to reach out to me.
