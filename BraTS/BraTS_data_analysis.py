import torch
import torch.onnx as onnx
import torchvision.models as models

import os
import numpy as np

import matplotlib.pyplot as plt
from skimage.transform import rotate

import torch
torch.manual_seed(0)

#import random
#random.seed(0)

import nibabel as nib


np.random.seed(0)


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.utils import data



TRAIN_DATASET_PATH = 'data/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = 'data/MICCAI_BraTS2020_ValidationData/'

test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 25
ax1.imshow(test_image_flair[:,:,test_image_flair.shape[0]//2-slice_w], cmap = 'gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:,:,test_image_t1.shape[0]//2-slice_w], cmap = 'gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:,:,test_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:,:,test_image_t2.shape[0]//2-slice_w], cmap = 'gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:,:,test_mask.shape[0]//2-slice_w])
ax5.set_title('Mask')

# Images shapes
print(test_image_flair.shape)
print(test_image_t1.shape)
print(test_image_t1ce.shape)
print(test_image_t2.shape)
print(test_mask.shape)


#for i in range(0,test_image_flair.shape[2]):
for i in [67, 78, 84]:
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
    slice_w = 0
    ax1.imshow(test_image_flair[:,:,i], cmap = 'gray')
    #ax1.set_title(str(i) + ' Image flair')
    ax1.set_title(' Image flair')
    ax2.imshow(test_image_t1[:,:,i], cmap = 'gray')
    ax2.set_title('Image t1')
    ax3.imshow(test_image_t1ce[:,:,i], cmap = 'gray')
    ax3.set_title('Image t1ce')
    ax4.imshow(test_image_t2[:,:,i], cmap = 'gray')
    ax4.set_title('Image t2')
    ax5.imshow(test_mask[:,:,i], cmap = 'gray')
    ax5.set_title('Mask')



# save mask to csv
#np.savetxt('data1.csv', test_mask[:,:,72], delimiter = ',')


# Skip 50:-50 slices since there is not much to see
from skimage.util import montage 
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_image_t1[50:-50,:,:]), 90, resize=True), cmap ='gray')

#Show segment of tumor for each above slice
# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_mask[60:-60,:,:]), 90, resize=True), cmap ='gray')



n = 7
fig,ax=plt.subplots(n,n,figsize=(16,16))

k=0
for i in range(0, n):
    for j in range(0, n):
        ax[i,j].imshow(rotate(test_image_flair[:,:,k], 90, resize=True), cmap = 'gray')
        ax[i,j].axes.get_xaxis().set_ticks([])
        ax[i,j].axes.get_yaxis().set_ticks([])
        
        k = k + 1


n = 12
fig,ax=plt.subplots(n,n,figsize=(16,16))
plt.subplots_adjust(hspace=0.05, wspace=0)
k=0
for i in range(0, n):
    for j in range(0, n):
        ax[i,j].imshow(test_image_flair[:,:,k], cmap = 'gray')
        ax[i,j].axes.get_xaxis().set_ticks([])
        ax[i,j].axes.get_yaxis().set_ticks([])
        k = k + 1


n = 12
fig,ax=plt.subplots(n,n,figsize=(16,16))
plt.subplots_adjust(hspace=0.05, wspace=0)
k=0
for i in range(0, n):
    for j in range(0, n):
        ax[i,j].imshow(test_mask[:,:,k], cmap = 'gray')
        ax[i,j].axes.get_xaxis().set_ticks([])
        ax[i,j].axes.get_yaxis().set_ticks([])
        k = k + 1
