import os
import numpy as np

import matplotlib.pyplot as plt

import torch
#if not torch.cuda.is_available():
#  raise Exception("GPU not availalbe. CPU training will be too slow.")
#print("device name", torch.cuda.get_device_name(0))

import nibabel as nib

import numpy as np
import helper

from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate


#There is an example image in the nibabel distribution.
images_path = 'data/modulus'
masks_path = 'data/mask'

images_file = os.path.join(images_path, 'PiP_Ox_051_WIP_MB2_EPI_FA90_TE35_23_1_modulus.nii')
masks_file = os.path.join(masks_path, 'PiP_Ox_051_WIP_MB2_EPI_FA90_TE35_23_1_modulupla_mask_t1.nii')

images_file = os.path.join(images_path,
                                   'PiP_Ox_051_WIP_MB2_EPI_FA90_TE35_23_1_modulus.nii')
#Load the file to create a nibabel image object:
imagesLoad = nib.load(images_file).get_fdata()
index = imagesLoad.shape[3]*imagesLoad.shape[2]
input_images = np.memmap('a.array', dtype='float64', mode='w+', shape=(1,256,256,index))
for i in range(0,imagesLoad.shape[3]):
    # Converting from 1 chanel to three chanels
    input_images[0,:,:,(50*i):(50+50*i)] = imagesLoad[:,:,:,i]


target_masks = np.memmap('b.array', dtype='float64', mode='w+', shape=(1,256,256,index))
for i in range(1,22):    
    masks_file = os.path.join(
        masks_path, 'PiP_Ox_051_WIP_MB2_EPI_FA90_TE35_23_1_modulupla_mask_t'+str(i)+'.nii')
    mask = nib.load(masks_file).get_fdata()
    target_masks[0,:,:,(50*(i-1)):(50+50*(i-1))] = mask[:,:,:]

# masked
for j in range(21):
    i = j*50+24
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20, 10))
    #img1 = images.get_fdata()[:,:,i,0] / np.max(images.get_fdata()[:,:,i,0])
    img1 = input_images[0,:,:,i] 
    ax1.imshow(rotate(img1, 90, resize=True), cmap = 'gray')
    ax1.set_title('Image')
    #img2 = masks.get_fdata()[:,:,i] / np.max(masks.get_fdata()[:,:,i])
    img2 = target_masks[0,:,:,i]
    ax2.imshow(rotate(img2, 90, resize=True), cmap = 'gray')
    ax2.set_title('Mask')
    #masked = np.ma.masked_where(img2 == 0, img2)
    #img3 = masks.get_fdata()[:,:,i]
    #img3 = img1.paste(img3, (0, 0), img3)    
    ax3.imshow(rotate(img1, 90, resize=True), cmap = 'gray')
    ax3.imshow(rotate(img2, 90, resize=True), alpha = 0.3)
    ax3.set_title('Masked')

# Sequence of all images, but the same slice.
fig,ax=plt.subplots(4,5,figsize=(16,16))
k=0
for i in range(0, 4):
    for j in range(0, 5):
        w = k*50+24
        img1 = input_images[0,:,:,w]
        img2 = target_masks[0,:,:,w]
        
        ax[i,j].axis("off")
        #ax[i,j].imshow(image, aspect='auto')
        
        ax[i,j].imshow(rotate(img1, 90, resize=True), cmap = 'gray', aspect='auto')
        ax[i,j].imshow(rotate(img2, 90, resize=True), alpha = 0.3, aspect='auto')
        
        k = k + 1

plt.subplots_adjust(hspace=0.01, wspace=0.01)

# All slices for just one of the 21 images with mask
fig,ax=plt.subplots(7,7,figsize=(16,16))
k=0
for i in range(0, 7):
    for j in range(0, 7):
        #w = k*50+24
        w=k
        img1 = input_images[0,:,:,w]
        img2 = target_masks[0,:,:,w]
        ax[i,j].axis("off")
        ax[i,j].imshow(rotate(img1, 90, resize=True), cmap = 'gray', aspect='auto')
        ax[i,j].imshow(rotate(img2, 90, resize=True), alpha = 0.3, aspect='auto')
        k = k + 1
plt.subplots_adjust(hspace=0.02, wspace=0.02)

# All slices for just one of the 21 images not mask
fig,ax=plt.subplots(7,7,figsize=(16,16))
k=0
for i in range(0, 7):
    for j in range(0, 7):
        #w = k*50+24
        w=k
        img1 = input_images[0,:,:,w]
        img2 = target_masks[0,:,:,w]
        ax[i,j].axis("off")
        #ax[i,j].imshow(rotate(img1, 90, resize=True), cmap = 'gray', aspect='auto')
        ax[i,j].imshow(rotate(img2, 90, resize=True), cmap = 'gray', aspect='auto')
        k = k + 1
plt.subplots_adjust(hspace=0.02, wspace=0.02)
