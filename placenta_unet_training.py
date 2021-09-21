import os
import numpy as np

#import matplotlib.pyplot as plt

import torch
torch.manual_seed(0)

#import random
#random.seed(0)

import nibabel as nib


np.random.seed(0)

#import helper

#from skimage.util import montage
#import skimage.transform as skTrans
#from skimage.transform import rotate

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets, models
from torch.utils import data

class PlacentaDataset(Dataset):
        
    def __init__(self, transform=None):
        images_path = 'data/modulus'
        masks_path = 'data/mask'
        images_file = os.path.join(images_path,
                                   'PiP_Ox_051_WIP_MB2_EPI_FA90_TE35_23_1_modulus.nii')
        #Load the file to create a nibabel image object:
        imagesLoad = nib.load(images_file).get_fdata()
        index = imagesLoad.shape[3]*imagesLoad.shape[2]

        self.input_images = np.memmap('a.array', dtype='float64', mode='w+', shape=(1,256,256,index))
        for i in range(0,imagesLoad.shape[3]):
            # Converting from 1 chanel to three chanels
            self.input_images[0,:,:,(50*i):(50+50*i)] = imagesLoad[:,:,:,i]
            #self.input_images[1,:,:,(50*i):(50+50*i)] = imagesLoad[:,:,:,i]
            #self.input_images[2,:,:,(50*i):(50+50*i)] = imagesLoad[:,:,:,i]
        # shape must be 25,3,192,192
        self.input_images = np.transpose(self.input_images,(3, 1, 2, 0))
        
        self.target_masks = np.memmap('b.array', dtype='float64', mode='w+', shape=(1,256,256,index))
        for i in range(1,22):    
            masks_file = os.path.join(
                masks_path, 'PiP_Ox_051_WIP_MB2_EPI_FA90_TE35_23_1_modulupla_mask_t'+str(i)+'.nii')
            mask = nib.load(masks_file).get_fdata()
            
            self.target_masks[0,:,:,(50*(i-1)):(50+50*(i-1))] = mask[:,:,:]
            #self.target_masks[1,:,:,(50*(i-1)):(50+50*(i-1))] = mask[:,:,:]
            #self.target_masks[2,:,:,(50*(i-1)):(50+50*(i-1))] = mask[:,:,:]
        self.target_masks = np.transpose(self.target_masks, (3, 0, 1, 2))
        
        #reduce data for code testing
        '''self.input_images = self.input_images[0:100]
        self.target_masks = self.target_masks[0:100]'''
        
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

import torch.nn as nn

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

    def __init__(self, n_class):
        super().__init__()
        # initialize neural network layers
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        
        

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
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

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

model = UNet(1)
model = model.to(device)

print(model)


from torchsummary import summary
summary(model, input_size=(1, 256, 256))

#################
#                      Define the main training loop
######################################


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

from collections import defaultdict
import torch.nn.functional as F




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
    model.train()  # Set model to training mode

    metrics = defaultdict(float)
    epoch_samples = 0

    for inputs, labels in trainLoader:
        inputs = inputs.float() # change
        inputs = inputs.to(device)
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
        inputs = inputs.float() # change
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):            
            outputs = model(inputs)
            calcLoss = DiceBCELoss()
            loss = calcLoss(outputs, labels, metrics)

        # statistics
        epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples, phase='test')
    epoch_loss = metrics['loss'] / epoch_samples
    for k in metrics.keys():
        metrics[k] = metrics[k] / epoch_samples
    
    return metrics

from os.path import exists
class Log:
    def __init__(self, path, header):
        self.path = path
        self.header = header
        file_exists = exists(self.path)
        if(not file_exists): 
            with open(path, 'w', encoding='UTF8', newline='') as f:
                # create the csv writer
                writer = csv.writer(f)
                # write header to the csv file
                writer.writerow(header)
    def addRow(self, row):
        with open(self.path, 'a+', encoding='UTF8', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)    
            # write a row to the csv file
            writer.writerow(row)



######################
#                         Training
################################

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import csv

####################
#Data set
#####

transform = transforms.Compose([
    transforms.ToTensor()
])

placentaDataset = PlacentaDataset(transform = transform)

'''
|---- train ------------------------|- val --|- test -|
|---- train ------|- val --|- test -|--- train -------|
|- val --|- test -|--------------------- train -------|
'''
K = 3 # number of folds
training_size = int(len(placentaDataset)*(K-1)/K)

validation_size = int((len(placentaDataset) - training_size)*2/3)

test_size = len(placentaDataset) - training_size - validation_size

"""train_set, val_set, test_set = data.random_split(
    placentaDataset, [len(placentaDataset)-validation_size-test_size,
                      validation_size, test_size])"""


import random
random.seed(0)
    
    
fold_size = []
for i in range(K-1):
    fold_size.append(int(len(placentaDataset)/3))
fold_size.append(len(placentaDataset) - np.sum(fold_size))

np.arange(0, len(placentaDataset))
dataShuffle = random.sample(list(range(0,len(placentaDataset))), k=len(placentaDataset))


folds = []

for i in range(0,K):
    i1 = int(np.sum(fold_size[0:i]))
    i2 = int(np.sum(fold_size[0:i+1]))
    folds.append(dataShuffle[i1:i2])

fold_sets = []
for i in range(K):
    fold_sets.append(Subset(placentaDataset, folds[i]))

for k in range(K):
    checkpoint_path = "checkpoint(" + str(k) + ").pth"
    val_size = int(len(fold_sets[k])*2/3)
    test_size = len(fold_sets[k]) - val_size
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

    num_class = 1
    model = UNet(num_class).to(device)
    header = ['fold', 'epoch', 'phase', 'time', 'bce', 'dice', 'loss']
    csvPath = 'log.csv'
    log = Log(csvPath, header)


    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    
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
