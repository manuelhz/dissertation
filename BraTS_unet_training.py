
import torch
import torch.onnx as onnx
import torchvision.models as models

import os
import os.path as path
from tempfile import mkdtemp
import numpy as np

#import matplotlib.pyplot as plt
#from skimage.transform import rotate

import torch
torch.manual_seed(0)

import random
random.seed(0)

import nibabel as nib
from nilearn.image import resample_img


np.random.seed(0)


from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets, models
from torch.utils import data

import torch.nn as nn


import torch.optim as optim
from torch.optim import lr_scheduler
import time
import csv


num_class = 3
num_channel = 1
#slices = 155 # Number of slices per each 3D image (original image)
slices = 78 # Number of slices per each 3D image (resampled image)
class BRaTSDataset(Dataset):        
    def __init__(self, transform=None):
        
        # Input images        
        data_path = 'data/MICCAI_BraTS2020_TrainingData'
        images_path = next(os.walk(data_path))[1]
        images_path.sort()        
        flair_files = []
        seg_files = []
        t1_files = []
        t1ce_files = []
        t2_files = []        
        for p in images_path:
            file_path = os.path.join(data_path, p)
            file_names= os.listdir(file_path)
            file_names.sort()
            flair_files.append(
                os.path.join(
                    file_path, file_names[0]))
            seg_files.append(
                os.path.join(
                    file_path, file_names[1]))
            t1_files.append(
                os.path.join(
                    file_path, file_names[2]))
            t1ce_files.append(
                os.path.join(
                    file_path, file_names[3]))
            t2_files.append(
                os.path.join(
                    file_path, file_names[4]))
        
        #slices = 155 # Number of slices per each 3D image (original image)
        #slices = 78 # Number of slices per each 3D image (resampled image)
        numberOfFiles = len(flair_files)
        
        #numberOfFiles = 7
        numberOfFiles = 100
        
        
        dataShuffle = random.sample(list(range(0,len(flair_files))), k=numberOfFiles)
        indexes = numberOfFiles * slices
        
        #tempFileName1 = path.join(mkdtemp(), 'newfile1.dat')
        #tempFileName2 = path.join(mkdtemp(), 'newfile2.dat')
        tempFileName1 = 'newfile1.dat'
        tempFileName2 = 'newfile2.dat'
        
        #self.input_images = np.memmap('a.array', dtype='float64', mode='w+', shape=(1,240,240,indexes))
        #self.target_masks = np.memmap('b.array', dtype='float64', mode='w+', shape=(1,240,240,indexes))        
        
        self.input_images = np.memmap(tempFileName1, dtype='float64', mode='w+', shape=(num_channel,120,120,indexes))
        #self.input_images = np.zeros((4,240,240,indexes))
        
        self.target_masks = np.memmap(tempFileName2, dtype='float64', mode='w+', shape=(1,120,120,indexes))        
        #self.target_masks = np.zeros((1,240,240,indexes))
        
        for i in range(numberOfFiles):
            self.input_images[0, :, :, i*slices:(slices+i*slices)] =\
                resample_img(nib.load(flair_files[dataShuffle[i]]), target_affine=np.eye(3)*2., interpolation='nearest').get_fdata()[:120,:120,:]
            """self.input_images[1, :, :, i*slices:(slices+i*slices)] =\
                resample_img(nib.load(t1_files[dataShuffle[i]]), target_affine=np.eye(3)*2., interpolation='nearest').get_fdata()[:120,:120,:]
            self.input_images[2, :, :, i*slices:(slices+i*slices)] =\
                resample_img(nib.load(t1ce_files[dataShuffle[i]]), target_affine=np.eye(3)*2., interpolation='nearest').get_fdata()[:120,:120,:]
            self.input_images[3, :, :, i*slices:(slices+i*slices)] =\
                resample_img(nib.load(t2_files[dataShuffle[i]]), target_affine=np.eye(3)*2., interpolation='nearest').get_fdata()[:120,:120,:]"""
            
            self.target_masks[0, :, :, i*slices:(slices+i*slices)] =\
                resample_img(nib.load(seg_files[dataShuffle[i]]), target_affine=np.eye(3)*2., interpolation='nearest').get_fdata()[:120,:120,:]
        self.input_images = np.transpose(self.input_images,(3, 1, 2, 0))
        self.target_masks = np.transpose(self.target_masks, (3, 0, 1, 2))
        
        ### one hot encoding
        self.target_masks[self.target_masks==4] = 3
        self.target_masks = F.one_hot(torch.as_tensor(self.target_masks).long(),4)        
        self.target_masks = torch.transpose(self.target_masks, 1, 4)
        self.target_masks = self.target_masks[:,:,:,:,0]
        self.target_masks = self.target_masks[:,1:4,:,:]
        self.target_masks = self.target_masks.float()
        
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]    
    




########
#          Define UNet
##########################



# Conv2d as in the original implementation has kernel_size = 3,
# but padding = 1 while original implementation padding = 0
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, num_class, num_channel):
        super().__init__()
        # initialize neural network layers
                
        self.dconv_down1 = double_conv(num_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        
        

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, num_class, 1)
        
        
    def forward(self, x):
        # impolements operations on input data
        
        # down convolution part
        conv1 = self.dconv_down1(x) #
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x) #
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x) #
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        
        # up convolution part
        
        # this is doing an upsampling (neearest algorithm) so it is not learning any parameters
        
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        
        x = self.upsample(x)
        
        x = torch.cat([x, conv2], dim=1)
       

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

########
#          Instantiate UNet model
##########################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

model = UNet(num_class, num_channel)
model = model.to(device)

print(model)


from torchsummary import summary
summary(model, input_size=(1, 120, 120))

#################
#                      Define the main training loop
######################################


from collections import defaultdict
import torch.nn.functional as F


checkpoint_path = "checkpoint.pth"

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)
        inputs = inputs.contiguous()
        targets = targets.contiguous()
        intersection = (inputs * targets).sum(dim=2).sum(dim=2)
        dice = (2. * intersection + smooth)/(inputs.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)
        loss = 1 - dice
        return loss.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, pred, target, metrics, bce_weight=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #(binary_cross_entropy_with_logits has a sigmoid)
        #inputs = F.sigmoid(inputs)       
        
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = torch.sigmoid(pred)
        diceloss = DiceLoss()
        dice = diceloss(pred, target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


        
def training_loop(model, optimizer, scheduler):
    time0 = time.time()
    model.train()  # Set model to training mode

    metrics = defaultdict(float)
    epoch_samples = 0
    

    for inputs, labels in trainLoader:
        
        inputs = inputs.float() # change
        inputs = inputs.to(device)
        
        #labels[labels==4] = 3                
        #labels = F.one_hot(labels.to(torch.int64),4)
        #labels = torch.transpose(labels, 1, 4)
        #labels = labels[:,:,:,:,0]
        #labels = labels[:,1:4,:,:]
        labels = labels.float()
        labels = labels.to(device)
        
        

        # Backpropagation
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            calcLoss = DiceBCELoss()            
            loss = calcLoss(outputs, labels, metrics)      
            loss.backward()
            optimizer.step()
            

        # statistics
        epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples, phase='train')
    epoch_loss = metrics['loss'] / epoch_samples
    

    scheduler.step()
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    
    
    for k in metrics.keys():
        metrics[k] = metrics[k] / epoch_samples
    
    return model, metrics

def validation_loop(model, optimizer):
    global best_loss
    model.eval()   # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs, labels in valLoader:
        inputs = inputs.float() # change
        inputs = inputs.to(device)
        #labels[labels==4] = 3                
        #labels = F.one_hot(labels.to(torch.int64),4)
        #labels = torch.transpose(labels, 1, 4)
        #labels = labels[:,:,:,:,0]
        #labels = labels[:,1:4,:,:]
        labels = labels.float()
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):            
            outputs = model(inputs)
            calcLoss = DiceBCELoss()
            loss = calcLoss(outputs, labels, metrics)

        # statistics
        epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples, phase='val')
    epoch_loss = metrics['loss'] / epoch_samples
    
    # save the model weights
    if epoch_loss < best_loss:
        print(f"saving best model to {checkpoint_path}")
        best_loss = epoch_loss
        torch.save(model.state_dict(), checkpoint_path)
    
    model.load_state_dict(torch.load(checkpoint_path))
    for k in metrics.keys():
        metrics[k] = metrics[k] / epoch_samples
    return model, metrics

def test_loop(model):
    global best_loss
    model.eval()   # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs, labels in testLoader:
        inputs = inputs.float()
        inputs = inputs.to(device)
        #labels[labels==4] = 3                
        #labels = F.one_hot(labels.to(torch.int64),4)
        #labels = torch.transpose(labels, 1, 4)
        #labels = labels[:,:,:,:,0]
        #labels = labels[:,1:4,:,:]
        labels = labels.float()
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):            
            outputs = model(inputs)
            calcLoss = DiceBCELoss()
            loss = calcLoss(outputs, labels, metrics)

        # statistics
        epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples, phase='test')
    for k in metrics.keys():
        metrics[k] = metrics[k] / epoch_samples
    
    return metrics

from os.path import exists
class Log:
    def __init__(self, path, header):
        self.path = path
        self.header = header
        file_exists = exists(path)
        if(not file_exists):  
            with open(path, 'w', encoding='UTF8', newline='') as f:
                # create the csv writer
                writer = csv.writer(f)
                # write header to the csv file
                writer.writerow(header)
        else:
            pass
        
    def addRow(self, row):
        with open(self.path, 'a+', encoding='UTF8', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)    
            # write a row to the csv file
            writer.writerow(row)

######################
#                         Training
################################



###########
# Data set
####

transform = transforms.Compose([
    transforms.ToTensor()
])

braTSDataset = BRaTSDataset(transform = transform)

##########
# Cross validation
####

'''
|---- train ------------------------|- val --|- test -|
|---- train ------|- val --|- test -|--- train -------|
|- val --|- test -|--------------------- train -------|
'''

K = 3 # number of folds

import random
random.seed(0)

# definning the size of each split
fold_size = []
for i in range(K-1):
    #div = int(len(braTSDataset)/3) # if does not matter if I take images from same images in different splits
    div = int((len(braTSDataset)/3)/slices)*slices # make each split from different images 
    fold_size.append(div)
fold_size.append(len(braTSDataset) - np.sum(fold_size))
print("fold_size: ", str(fold_size))


# the next line is commented in orther to produce data from different images
#dataShuffle = random.sample(list(range(0,len(braTSDataset))), k=len(braTSDataset))
dataShuffle = range(0,len(braTSDataset))

folds = []

for i in range(0,K):
    i1 = int(np.sum(fold_size[0:i]))
    i2 = int(np.sum(fold_size[0:i+1]))
    folds.append(dataShuffle[i1:i2])

fold_sets = []
for i in range(K):
    fold_sets.append(Subset(braTSDataset, folds[i]))

for k in range(K):
    checkpoint_path = "checkpoint(" + str(k) + ").pth"
    # divide and multiply by the number of slices to make a mulple of number slices
    val_size = int((len(fold_sets[k])*2/3)/slices)*slices
    test_size = len(fold_sets[k]) - val_size
    print("val_size: ", str(val_size), "test_size: ", str(test_size))
    
    val_set = Subset(fold_sets[k], np.arange(0,val_size))
    test_set = Subset(fold_sets[k], np.arange(val_size,len(fold_sets[k])))
    if k == 0:
        train_set = fold_sets[1]
        s = 2
    else:
        train_set = fold_sets[0]
        s = 1
    for i in range(s,K):
        if i == k:
            pass
        else:
            train_set = torch.utils.data.ConcatDataset([train_set, fold_sets[i]])
    
    batch_size = 25
    
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    valLoader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(num_class, num_channel).to(device)
    header = ['fold', 'epoch', 'phase', 'time', 'bce', 'dice', 'loss']
    csvPath = 'log.csv'
    log = Log(csvPath, header)


    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=51, gamma=0.1)
    
    num_epochs = 50
    
    best_loss = 1e10

    for epoch in range(num_epochs):
        since_0 = time.time()
        print("Fold ", k, end = " ")
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model, metrics = training_loop(model, optimizer_ft, exp_lr_scheduler)    
        newRow = [k, epoch, 'train', time.time()-since_0, metrics['bce'], metrics['dice'], metrics['loss']]
        log.addRow(newRow)
        since_1 = time.time()
        model, metrics = validation_loop(model, optimizer_ft)
        newRow = [k, epoch, 'val', time.time()-since_1, metrics['bce'], metrics['dice'], metrics['loss']]
        log.addRow(newRow)
        time_elapsed = time.time() - since_0
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    since = time.time()
    metrics = test_loop(model)    
    newRow = [k, 0, 'test', time.time()-since, metrics['bce'], metrics['dice'], metrics['loss']]
    log.addRow(newRow)





