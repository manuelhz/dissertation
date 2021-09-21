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

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import csv
import random
random.seed(0)

class PlacentaDataset(Dataset):
        
    def __init__(self, transform=None):
        images_path = '../data/modulus'
        masks_path = '../data/mask'
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
        
        # reduce data for code testing
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
#          Define UNet++
##########################



def double_conv2(in_ch, out_ch):
    return nn.Sequential(        
        nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),  # True means cover the origin input
        nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
        )

def up4(x1, x2, x3, x4):
    x = torch.cat([x4, x3, x2, x1], dim=1)
    return x        

def up3(x1, x2, x3):
    x = torch.cat([x3, x2, x1], dim=1)
    return x    

def up(x1, x2):
    # x1--up , x2 ---down        
    diffX = x1.size()[2] - x2.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (
        diffY // 2, diffY - diffY // 2,
        diffX // 2, diffX - diffX // 2,))
    x = torch.cat([x2, x1], dim=1)    
    return x

def down(in_ch, out_ch):    
    return nn.Sequential(
        
        # double conv
        nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),  # True means cover the origin input
        nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
        )

def double_conv_in(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 5, padding=2),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),  # True means cover the origin input
        nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)       
        )


cc = 64


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetPlusPlus, self).__init__()
        self.inconv = double_conv_in(n_channels, cc)
        self.down1 = down(cc, 2 * cc)
        self.down2 = down(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        #self.up1 = up(12 * cc, 4 * cc)
        self.conv1 = double_conv2(12 * cc, 4 * cc)
        
        #self.up20 = up(6 * cc, 2 * cc)
        self.conv20 = double_conv2(6 * cc, 2 * cc)
        
        #self.up2 = up3(8 * cc, 2 * cc)
        self.conv2 = double_conv2(8 * cc, 2 * cc)
        
        #self.up30 = up(3 * cc, cc)
        self.conv30 = double_conv2(3 * cc, cc)
        
        #self.up31 = up3(4 * cc, cc)
        self.conv31 = double_conv2(4 * cc, cc)
        
        #self.up3 = up4(5 * cc, cc)
        self.conv3 = double_conv2(5 * cc, cc)
        
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outconv0 = nn.Conv2d(cc, n_classes, 1)
        self.outconv1 = nn.Conv2d(cc, n_classes, 1)
        self.outconv2 = nn.Conv2d(cc, n_classes, 1)        

    def forward(self, x):        
        
        x1 = self.inconv(x)
        x = self.maxpool(x1)
        
        x2 = self.down1(x)
        x = self.maxpool2(x2)
        
        x3 = self.down2(x)
        x = self.maxpool2(x3)
        
        
        x4 = self.down3(x)
        #x = self.maxpool2(x4)
        
        x4 = self.upsample(x4)
        x = up(x4, x3) # up1
        x = self.conv1(x)
        
        x3 = self.upsample(x3)
        x21 = up(x3, x2) # up20
        x21 = self.conv20(x21)
        
        x = self.upsample(x)
        x = up3(x, x21, x2) # up2
        x = self.conv2(x)
        
        
        x2 = self.upsample(x2)
        x11 = up(x2, x1) # up30
        x11 = self.conv30(x11)
        
        x21 = self.upsample(x21)
        x12 = up3(x21, x11, x1) # up31
        x12 = self.conv31(x12)
        
        x = self.upsample(x)
        x = up4(x, x12, x11, x1) # up3
        x = self.conv3(x)        
        
        #output 0 1 2
        x = self.upsample2(x)
        y2 = self.outconv2(x)
        
        x11 = self.upsample2(x11)
        y0 = self.outconv0(x11)
        
        x12 = self.upsample2(x12)
        y1 = self.outconv1(x12)
        return y0, y1, y2
        





########
#          Instantiate UNet model
##########################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

deep_supervision = True
model = UNetPlusPlus(1,1)
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
    
def training_loop(model, optimizer, scheduler, deep_supervision):    
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
            if deep_supervision == True:
                loss = 0
                for output in outputs:
                    loss += calcLoss(output, labels, metrics)
                    # statistics
                    epoch_samples += inputs.size(0)
                loss /= len(outputs)
            else:
                loss = calcLoss(outputs[-1], labels, metrics)
                # statistics
                epoch_samples += inputs.size(0)
            
            loss.backward()
            optimizer.step()

        

    print_metrics(metrics, epoch_samples, phase='train')
    for k in metrics.keys():
        metrics[k] = metrics[k] / epoch_samples
    epoch_loss = metrics['loss'] / epoch_samples

    scheduler.step()
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
        
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
            loss = calcLoss(outputs[-1], labels, metrics)

        # statistics
        epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples, phase='val')
    epoch_loss = metrics['loss'] / epoch_samples
    for k in metrics.keys():
        metrics[k] = metrics[k] / epoch_samples
    
    
    # save the model weights
    if epoch_loss < best_loss:
        print(f"saving best model to {checkpoint_path}")
        best_loss = epoch_loss
        torch.save(model.state_dict(), checkpoint_path)
    
    model.load_state_dict(torch.load(checkpoint_path))
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
            outputs = model(inputs)[-1]
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



#############
###  Data set
#

transform = transforms.Compose([
    transforms.ToTensor()
])

placentaDataset = PlacentaDataset(transform = transform)

'''
|---- train ------------------------|- val --|- test -|
|---- train ------|- val --|- test -|--- train -------|
|- val --|- test -|--------------------- train -------|
'''
'''
The log class must be different to the version without cross validation
'''
K = 3 # number of folds

    
    
fold_size = []
for i in range(K-1):
    fold_size.append(int(len(placentaDataset)/3))
fold_size.append(len(placentaDataset) - np.sum(fold_size))

np.arange(0, len(placentaDataset))
dataShuffle = random.sample(list(range(0,len(placentaDataset))), k=len(placentaDataset))

folds = [] # where to store index for data sets
for i in range(0,K):
    i1 = int(np.sum(fold_size[0:i]))
    i2 = int(np.sum(fold_size[0:i+1]))
    folds.append(dataShuffle[i1:i2])

fold_sets = [] # store the three folds
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
    num_channels = 1
    deep_supervision = True
    model = UNetPlusPlus(num_channels, num_class).to(device)
    header = ['fold', 'epoch', 'phase', 'time', 'bce', 'dice', 'loss']
    csvPath = 'log.csv'
    log = Log(csvPath, header)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=51, gamma=0.1)
    
    num_epochs = 50
    
    best_loss = 1e10

    for epoch in range(num_epochs):
        since_0 = time.time()
        print("Fold ", k, end = " ")
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model, metrics = training_loop(model, optimizer_ft, exp_lr_scheduler, deep_supervision)    
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
