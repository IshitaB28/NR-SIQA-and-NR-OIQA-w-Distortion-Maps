import matplotlib.pyplot as plt
import pandas as pd
import re
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
import random
random.seed(64)

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import KFold
from scipy.io import loadmat
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy import signal

from mlp_mixer_pytorch import MLPMixer


class QNet_CNN(nn.Module): #Proposed QNet_CNN
    def __init__(self):
        super(QNet_CNN, self).__init__()
        
        # For Distorted Image
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn1_ = nn.BatchNorm2d(32)
        self.bn2_ = nn.BatchNorm2d(32)
        self.bn3_ = nn.BatchNorm2d(64)
        # self.fc1_ = nn.Linear(256, 128) # for patch size 16
        self.fc1_ = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_ = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_ = nn.Linear(16384, 128) # for patch size 128

        # For Distortion Map
        self.conv1D = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2D = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3D = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4D = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn1_D = nn.BatchNorm2d(32)
        self.bn2_D = nn.BatchNorm2d(32)
        self.bn3_D = nn.BatchNorm2d(64)
        # self.fc1_D = nn.Linear(256, 128) # for patch size 16
        self.fc1_D = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_D = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_D = nn.Linear(16384, 128) # for patch size 128

        self.fc1_1 = nn.Linear(128 * 2, 1)  # for concatenated image and distortion map features
        
    def forward(self, x ,xD):

        # x: Distorted omni image
        # xD: Distortion Map

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        xD = xD.view(-1, xD.size(-3), xD.size(-2), xD.size(-1))
        
        Out_x = F.relu(self.conv1(x)) 
        Out_x = F.max_pool2d(Out_x, (2, 2), stride=2)
        Out_x = self.bn1_(Out_x)

        Out_x = F.relu(self.conv2(Out_x))
        Out_x = F.max_pool2d(Out_x, (2, 2), stride=2)
        Out_x = self.bn2_(Out_x)


        Out_x = F.relu(self.conv3(Out_x))
        Out_x = F.relu(self.conv4(Out_x))

        Out_x = F.max_pool2d(Out_x, (2, 2), stride=2)
        Out_x = self.bn3_(Out_x)

        
        Out_x = Out_x.view(-1, self.num_flat_features(Out_x))
        Out_x = F.relu(self.fc1_(Out_x))
        
        Out_D = F.relu(self.conv1D(xD))
        Out_D = F.max_pool2d(Out_D, (2, 2), stride=2)
        Out_D = self.bn1_D(Out_D)

        Out_D = F.relu(self.conv2D(Out_D))
        Out_D = F.max_pool2d(Out_D, (2, 2), stride=2)
        Out_D = self.bn2_D(Out_D)
      
        Out_D = F.relu(self.conv3D(Out_D))
        Out_D = F.relu(self.conv4D(Out_D))
        

        Out_D = F.max_pool2d(Out_D, (2, 2), stride=2)
        Out_D = self.bn3_D(Out_D)
        
        Out_D = Out_D.view(-1, self.num_flat_features(Out_D))
        Out_D = F.relu(self.fc1_D(Out_D))

        cat_total = F.relu(torch.cat((Out_x, Out_D), dim=1)) # concatenated image and distortion map features

        Global_Quality = F.relu(self.fc1_1(cat_total))
        return Global_Quality


    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class QNet_Mixer(nn.Module):
    def __init__(self, image_size = (32, 32), channels = 1, patch_size = 8, dim = 128, depth = 8, num_classes = 1):
        super(QNet_Mixer, self).__init__()
        self.d_mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
        self.d_mixer_model = torch.nn.Sequential(*list(self.d_mixer_model.children())[:-1]) # for image

        self.i_mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
        self.i_mixer_model = torch.nn.Sequential(*list(self.i_mixer_model.children())[:-1]) # for distortion map

        self.fc_d = nn.Linear(dim, 128)
        self.fc_i = nn.Linear(dim, 128)

        self.fc = nn.Linear(128*2, num_classes)


    def forward(self, x, i):

        # x: distortion maps
        # i: omni image

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        i = i.view(-1, i.size(-3), i.size(-2), i.size(-1))


        x = self.d_mixer_model(x)
        i = self.i_mixer_model(i)
        
        
        x = self.fc_d(x)
        i = self.fc_i(i)
        
        
        f = torch.cat((x, i), dim=1) # concatenated image and distortion map features

        return self.fc(f)
    


class QNet_CNNWOD(nn.Module):
    def __init__(self):
        super(QNet_CNNWOD, self).__init__()
        
        # For Distorted Image
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn1_ = nn.BatchNorm2d(32)
        self.bn2_ = nn.BatchNorm2d(32)
        self.bn3_ = nn.BatchNorm2d(64)
        # self.fc1_ = nn.Linear(256, 128) # for patch size16
        self.fc1_ = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_ = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_ = nn.Linear(16384, 128) # for patch size 128

        self.fc1_1 = nn.Linear(128, 1)
        
    def forward(self, x):

        # x: Distorted omni image
        # xD: Distortion Map

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        
        Out_x = F.relu(self.conv1(x)) 
        Out_x = F.max_pool2d(Out_x, (2, 2), stride=2)
        Out_x = self.bn1_(Out_x)

        Out_x = F.relu(self.conv2(Out_x))
        Out_x = F.max_pool2d(Out_x, (2, 2), stride=2)
        Out_x = self.bn2_(Out_x)


        Out_x = F.relu(self.conv3(Out_x))
        Out_x = F.relu(self.conv4(Out_x))

        Out_x = F.max_pool2d(Out_x, (2, 2), stride=2)
        Out_x = self.bn3_(Out_x)

        
        Out_x = Out_x.view(-1, self.num_flat_features(Out_x))
        Out_x = F.relu(self.fc1_(Out_x))

        Global_Quality = F.relu(self.fc1_1(Out_x))
        return Global_Quality


    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class QNet_MixerWOD(nn.Module): # Proposed QNet_Mixer model
    def __init__(self, image_size = (32, 32), channels = 1, patch_size = 8, dim = 128, depth = 8, num_classes = 1):
        super(QNet_MixerWOD, self).__init__()
        self.mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
        self.mixer_model = torch.nn.Sequential(*list(self.mixer_model.children())[:-1]) # for image

        self.fc = nn.Linear(dim, num_classes)


    def forward(self, i):

        # i: omni image

        i = i.view(-1, i.size(-3), i.size(-2), i.size(-1))


        x = self.mixer_model(x)
        

        return self.fc(x)
