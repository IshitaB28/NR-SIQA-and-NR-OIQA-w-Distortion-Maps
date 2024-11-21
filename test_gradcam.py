import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from unet_light_model import UNet_Light

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet_CNN(nn.Module):
    def __init__(self):
        super(QNet_CNN, self).__init__()
        self.conv1L = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2L = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3L = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4L = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn1_L = nn.BatchNorm2d(32)
        self.bn2_L = nn.BatchNorm2d(32)
        self.bn3_L = nn.BatchNorm2d(64)
        self.fc1_L = nn.Linear(1024, 128)
        
        self.conv1R = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2R = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3R = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4R = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.bn1_R = nn.BatchNorm2d(32)
        self.bn2_R = nn.BatchNorm2d(32)
        self.bn3_R = nn.BatchNorm2d(64)
        self.fc1_R = nn.Linear(1024, 128)
        
        self.fc1_1 = nn.Linear(128 * 2, 32) 
        
        self.conv1iL = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2iL = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3iL = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4iL = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn1_iL = nn.BatchNorm2d(32)
        self.bn2_iL = nn.BatchNorm2d(32)
        self.bn3_iL = nn.BatchNorm2d(64)
        self.fc1_iL = nn.Linear(1024, 128)
        
        self.conv1iR = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2iR = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3iR = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4iR = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.bn1_iR = nn.BatchNorm2d(32)
        self.bn2_iR = nn.BatchNorm2d(32)
        self.bn3_iR = nn.BatchNorm2d(64)
        self.fc1_iR = nn.Linear(1024, 128)
        
        self.fc1_i1 = nn.Linear(128 * 2, 32)  


        self.fc_glob = nn.Linear(32*2, 1)
        
    def forward(self, inp):

        xL, xR, iL, iR = inp[0], inp[1], inp[2], inp[3]
        

        x_distort_L = xL.view(-1, xL.size(-3), xL.size(-2), xL.size(-1))
        x_distort_R = xR.view(-1, xR.size(-3), xR.size(-2), xR.size(-1))

        i_distort_L = iL.view(-1, iL.size(-3), iL.size(-2), iL.size(-1))
        i_distort_R = iR.view(-1, iR.size(-3), iR.size(-2), iR.size(-1))

        
        Out_L = F.relu(self.conv1L(x_distort_L)) 
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn1_L(Out_L)

        Out_L = F.relu(self.conv2L(Out_L))
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn2_L(Out_L)


        Out_L = F.relu(self.conv3L(Out_L))
        Out_L = F.relu(self.conv4L(Out_L))

        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn3_L(Out_L)

        
        Out_L = Out_L.view(-1, self.num_flat_features(Out_L))
        Out_L = F.relu(self.fc1_L(Out_L))


        Out_iL = F.relu(self.conv1iL(i_distort_L)) 
        Out_iL = F.max_pool2d(Out_iL, (2, 2), stride=2)
        Out_iL = self.bn1_iL(Out_iL)

        Out_iL = F.relu(self.conv2iL(Out_iL))
        Out_iL = F.max_pool2d(Out_iL, (2, 2), stride=2)
        Out_iL = self.bn2_iL(Out_iL)


        Out_iL = F.relu(self.conv3iL(Out_iL))
        Out_iL = F.relu(self.conv4iL(Out_iL))

        Out_iL = F.max_pool2d(Out_iL, (2, 2), stride=2)
        Out_iL = self.bn3_iL(Out_iL)

        
        Out_iL = Out_iL.view(-1, self.num_flat_features(Out_iL))
        Out_iL = F.relu(self.fc1_iL(Out_iL))
        
        Out_R = F.relu(self.conv1R(x_distort_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn1_R(Out_R)

        Out_R = F.relu(self.conv2R(Out_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn2_R(Out_R)
      
        Out_R = F.relu(self.conv3R(Out_R))
        Out_R = F.relu(self.conv4R(Out_R))
        

        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn3_R(Out_R)
        
        Out_R = Out_R.view(-1, self.num_flat_features(Out_R))
        Out_R = F.relu(self.fc1_R(Out_R))

        Out_iR = F.relu(self.conv1iR(i_distort_R))
        Out_iR = F.max_pool2d(Out_iR, (2, 2), stride=2)
        Out_iR = self.bn1_iR(Out_iR)

        Out_iR = F.relu(self.conv2iR(Out_iR))
        Out_iR = F.max_pool2d(Out_iR, (2, 2), stride=2)
        Out_iR = self.bn2_iR(Out_iR)
      
        Out_iR = F.relu(self.conv3iR(Out_iR))
        Out_iR = F.relu(self.conv4iR(Out_iR))
        

        Out_iR = F.max_pool2d(Out_iR, (2, 2), stride=2)
        Out_iR = self.bn3_iR(Out_iR)
        
        Out_iR = Out_iR.view(-1, self.num_flat_features(Out_iR))
        Out_iR = F.relu(self.fc1_iR(Out_iR))

        cat_total = F.relu(torch.cat((Out_L, Out_R), dim=1))

        Global_Quality_1 = F.relu(self.fc1_1(cat_total))

        cat_total_2 = F.relu(torch.cat((Out_iL, Out_iR), dim=1))

        Global_Quality_2 = F.relu(self.fc1_i1(cat_total_2))

        cat_total_g = F.relu(torch.cat((Global_Quality_1, Global_Quality_2), dim=1))

        Global_Quality = F.relu(self.fc_glob(cat_total_g))

        return Global_Quality
    
    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LIVEDataset(Dataset):
    def __init__(self, dataset_path, indices, train = False):
        self.train = train
        self.dataset_path = dataset_path
        self.indices = indices
        self.o_images_left, self.o_images_right, self.d_images_left, self.d_images_right, self.quality_scores = self.load_image_filenames_and_scores()
        print(len(self.o_images_right))

        self.left_distorted_patches = []
        self.right_distorted_patches = []

        self.left_image_patches = []
        self.right_image_patches = []

        self.all_left_patches = ()
        self.all_right_patches = ()

        self.all_left_image_patches = ()
        self.all_right_image_patches = ()
   
        self.label = []
        self.label_L = []
        self.label_R = []

        dist_model = UNet_Light().to(device)
        dist_model.load_state_dict(torch.load("best_unet_light_32_patch_distortion_pristine_all.pth"))


        c = list(zip(self.d_images_left, self.d_images_right, self.quality_scores))

        random.shuffle(c)

        self.d_images_left, self.d_images_right, self.quality_scores = zip(*c)

        trainindex = indices[:int(0.8 * len(indices))]
        testindex = indices[int((1 - 0.2) * len(indices)):]
        train_index, test_index = [], []

        for i in range(len(indices)):
            if(i in trainindex):
                train_index.append(i)
            elif(i in testindex):
                test_index.append(i)


        if self.train:
            self.curr_indices = train_index
        else:
            self.curr_indices = test_index
        for ind in range(len(self.curr_indices)):

            print(self.d_images_left[ind],self.d_images_right[ind])

        for ind in range(len(self.curr_indices)):

            distorted_image = cv2.imread(self.d_images_left[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))
            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patch = patch.unsqueeze(0)
                    patch = patch.to(device)
                    patches_images = patches_images + (patch, )
                    self.all_left_image_patches = self.all_left_image_patches + (patch, )
                    distortion_patch = dist_model(patch).detach().cpu()
                    patches = patches + (distortion_patch[0],)
                    self.all_left_patches = self.all_left_patches + (distortion_patch[0],)
                    self.label.append(self.quality_scores[ind])
                    self.label_L.append(self.quality_scores[ind])
            self.left_distorted_patches.append(patches)
            self.left_image_patches.append(patches_images)

        for ind in range(len(self.curr_indices)):

            distorted_image = cv2.imread(self.d_images_right[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))
            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patch = patch.unsqueeze(0)
                    patch = patch.to(device)
                    patches_images = patches_images + (patch, )
                    self.all_right_image_patches = self.all_right_image_patches + (patch, )
                    distortion_patch = dist_model(patch).detach().cpu()
                    patches = patches + (distortion_patch[0],)
                    self.all_right_patches = self.all_right_patches + (distortion_patch[0],)
                    self.label_R.append(self.quality_scores[ind])
            self.right_distorted_patches.append(patches)
            self.right_image_patches.append(patches_images)

        

    def load_image_filenames_and_scores(self):
        data_mat = loadmat(self.dataset_path+"data.mat")

        all_original_left_images = []
        all_original_right_images = []
        all_distorted_left_images = []
        all_distorted_right_images = []
        all_dmos = []
        find_start = "i"
        find_end = "."

        for i in range(len(data_mat['dmos'])):

            original = data_mat['ref_names'][0][i][0]
            start_index = original.find("r")
            end_index = original.find(find_end)
            original = original[start_index + 8:end_index]

            distorted = data_mat['img_names'][0][i][0]
            dist_fol = distorted
            start_index = distorted.find(find_start)
            end_index = distorted.find(find_end)
            distorted = distorted[start_index:end_index]

            dmos = data_mat['dmos'][i][0]

            original_l = self.dataset_path+"refimgs/"+original+"_l.bmp"
            original_r = self.dataset_path+"refimgs/"+original+"_r.bmp"

            if "blur" in dist_fol:
                distorted_l = self.dataset_path+"blur/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"blur/"+distorted+"_r.bmp"
                all_distorted_left_images.append(distorted_l)
                all_distorted_right_images.append(distorted_r)
                all_original_left_images.append(original_l)        

                original_r = self.dataset_path+"refimgs/"+original+"_r.bmp"

                all_original_right_images.append(original_r)
                all_dmos.append(dmos)
                

            if "ff" in dist_fol:
                distorted_l = self.dataset_path+"ff/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"ff/"+distorted+"_r.bmp"
                all_distorted_left_images.append(distorted_l)
                all_distorted_right_images.append(distorted_r)
                all_original_left_images.append(original_l)        

                original_r = self.dataset_path+"refimgs/"+original+"_r.bmp"                

                all_original_right_images.append(original_r)
                all_dmos.append(dmos)
                
            if "jp2k" in dist_fol:
                distorted_l = self.dataset_path+"jp2k/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"jp2k/"+distorted+"_r.bmp"
                all_distorted_left_images.append(distorted_l)
                all_distorted_right_images.append(distorted_r)
                all_original_left_images.append(original_l)        
                original_r = self.dataset_path+"refimgs/"+original+"_r.bmp"
                

                all_original_right_images.append(original_r)
                all_dmos.append(dmos)
            

            if "jpeg" in dist_fol:
                distorted_l = self.dataset_path+"jpeg/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"jpeg/"+distorted+"_r.bmp"
                all_distorted_left_images.append(distorted_l)
                all_distorted_right_images.append(distorted_r)
                all_original_left_images.append(original_l)        

                original_r = self.dataset_path+"refimgs/"+original+"_r.bmp"
                

                all_original_right_images.append(original_r)
                all_dmos.append(dmos)
                
                
            if "wn" in dist_fol:
                distorted_l = self.dataset_path+"wn/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"wn/"+distorted+"_r.bmp"
                all_distorted_left_images.append(distorted_l)
                all_distorted_right_images.append(distorted_r)

                all_original_left_images.append(original_l)        

                original_r = self.dataset_path+"refimgs/"+original+"_r.bmp"
                

                all_original_right_images.append(original_r)
                all_dmos.append(dmos)
                

                      
            all_original_right_images.append(original_r)
            all_distorted_left_images.append(distorted_l)
            all_distorted_right_images.append(distorted_r)
            all_dmos.append(dmos)

        return all_original_left_images, all_original_right_images, all_distorted_left_images, all_distorted_right_images, all_dmos

    def __len__(self):
        if self.train:
            return len(self.all_left_patches)
        else:
            return len(self.left_distorted_patches)

    def __getitem__(self, index):

        if self.train:
            return self.all_left_patches[index], self.all_right_patches[index], self.label[index], self.label_L[index], self.label_R[index], self.all_left_image_patches[index], self.all_right_image_patches[index]
        else:
            return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], self.left_image_patches[index], self.right_image_patches[index]


class IRCCynDataset(Dataset):
    def __init__(self, dataset_path, indices, train = False, low = True):
        self.train = train
        self.low = low
        self.dataset_path = dataset_path
        self.indices = indices
        self.o_images_left, self.o_images_right, self.d_images_left, self.d_images_right, self.quality_scores = self.load_image_filenames_and_scores()

        self.left_distorted_patches = []
        self.right_distorted_patches = []

        self.left_image_patches = []
        self.right_image_patches = []

        self.all_left_patches = ()
        self.all_right_patches = ()

        self.all_left_image_patches = ()
        self.all_right_image_patches = ()
   
        self.label = []
        self.label_L = []
        self.label_R = []

        dist_model = UNet_Light().to(device)
        dist_model.load_state_dict(torch.load("best_unet_light_32_patch_distortion_pristine_all.pth"))

        
        c = list(zip(self.d_images_left, self.d_images_right, self.quality_scores))

        random.shuffle(c)

        self.d_images_left, self.d_images_right, self.quality_scores = zip(*c)

        trainindex = indices[:int(0.8 * len(indices))]
        testindex = indices[int((1 - 0.2) * len(indices)):]
        train_index, test_index = [], []

        for i in range(len(indices)):
            if(i in trainindex):
                train_index.append(i)
            elif(i in testindex):
                test_index.append(i)

        if self.train:
            self.curr_indices = train_index
        else:
            self.curr_indices = test_index
        for ind in range(len(self.curr_indices)):

            print(self.d_images_left[ind],self.d_images_right[ind])

        for ind in range(len(self.curr_indices)):

            distorted_image = cv2.imread(self.d_images_left[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))
            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patch = patch.unsqueeze(0)
                    patch = patch.to(device)
                    patches_images = patches_images + (patch, )
                    self.all_left_image_patches = self.all_left_image_patches + (patch, )
                    distortion_patch = dist_model(patch).detach().cpu()
                    patches = patches + (distortion_patch[0],)
                    self.all_left_patches = self.all_left_patches + (distortion_patch[0],)
                    self.label.append(self.quality_scores[ind])
                    self.label_L.append(self.quality_scores[ind])
            self.left_distorted_patches.append(patches)
            self.left_image_patches.append(patches_images)

        for ind in range(len(self.curr_indices)):

            distorted_image = cv2.imread(self.d_images_right[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))
            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patch = patch.unsqueeze(0)
                    patch = patch.to(device)
                    patches_images = patches_images + (patch, )
                    self.all_right_image_patches = self.all_right_image_patches + (patch, )
                    distortion_patch = dist_model(patch).detach().cpu()
                    patches = patches + (distortion_patch[0],)
                    self.all_right_patches = self.all_right_patches + (distortion_patch[0],)
                    self.label_R.append(self.quality_scores[ind])
            self.right_distorted_patches.append(patches)
            self.right_image_patches.append(patches_images)

        

    def load_image_filenames_and_scores(self):
        mos = pd.read_csv(self.dataset_path+"dmos.csv", header=None)

        all_original_left_images = []
        all_original_right_images = []
        all_distorted_left_images = []
        all_distorted_right_images = []
        all_dmos = []

        for i in range(len(mos)):

            original = mos.loc[i, 1]
            original_number = original[8:-4]
            right_or = "CLIPright"+original_number+".bmp"

            distorted = mos.loc[i, 0]
            distorted_number = distorted[8:-4]
            right_dis = "CLIPright"+distorted_number+".bmp"

            dmos = mos.loc[i, 2]

            original_l = self.dataset_path+"images/Left/"+original
            original_r = self.dataset_path+"images/Right/"+right_or

            distorted_l = self.dataset_path+"images/Left/"+distorted
            distorted_r = self.dataset_path+"images/Right/"+right_dis


            # the mean dmos was found to be 20

            if self.low == True:
                if dmos<20:
                    all_original_left_images.append(original_l)
                    all_original_right_images.append(original_r)
                    all_distorted_left_images.append(distorted_l)
                    all_distorted_right_images.append(distorted_r)
                    all_dmos.append(dmos)
            else:
                if dmos>=20:
                    all_original_left_images.append(original_l)
                    all_original_right_images.append(original_r)
                    all_distorted_left_images.append(distorted_l)
                    all_distorted_right_images.append(distorted_r)
                    all_dmos.append(dmos)



        return all_original_left_images, all_original_right_images, all_distorted_left_images, all_distorted_right_images, all_dmos

    def __len__(self):
        if self.train:
            return len(self.all_left_patches)
        else:
            return len(self.left_distorted_patches)

    def __getitem__(self, index):

        if self.train:
            return self.all_left_patches[index], self.all_right_patches[index], self.label[index], self.label_L[index], self.label_R[index], self.all_left_image_patches[index], self.all_right_image_patches[index]
        else:
            return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], self.left_image_patches[index], self.right_image_patches[index]


# Uncomment the required dataset
indices = list(range(0, 97)) #IRCCyn
test_dataset = IRCCynDataset(dataset_path="path-to-irccyn-dataset", indices=indices, train=False, low = True) # low dmos - > high quality
# test_dataset = IRCCynDataset(dataset_path="path-to-irccyn-dataset", indices=indices, train=False, low = False)

# indices = list(range(0, 365)) #Live 1
# test_dataset = LIVEDataset(dataset_path="path-to-live-1-dataset", indices=indices, train=False)
# test_loader = DataLoader(test_dataset, batch_size=1)


test_loader = DataLoader(test_dataset, batch_size=1)
print("Dataset loaded")

quality_model = QNet_CNN().to(device)
quality_model.load_state_dict(torch.load("path-to-best-save-model-for-irccyn.pth"))

cam = GradCAM(model=quality_model, target_layers=[quality_model.bn1_iL, quality_model.bn2_iL, quality_model.bn3_iL])


k = 0
for left_patches, right_patches, target_score, left_image_patches, right_image_patches in test_loader:

    left_patches = torch.stack(left_patches, dim=1)[0]
    right_patches = torch.stack(right_patches, dim=1)[0]

    left_image_patches = torch.stack(left_image_patches, dim=1)[0]
    right_image_patches = torch.stack(right_image_patches, dim=1)[0]
    
    left_patches, right_patches, target_score, left_image_patches, right_image_patches = left_patches.to(device), right_patches.to(device), target_score.to(device), left_image_patches.to(device), right_image_patches.to(device)

    left_image_patches = left_image_patches.view(-1, left_image_patches.size(-3), left_image_patches.size(-2), left_image_patches.size(-1))
    right_image_patches = right_image_patches.view(-1, right_image_patches.size(-3), right_image_patches.size(-2), right_image_patches.size(-1))

    inp = torch.cat((left_patches, right_patches, left_image_patches, right_image_patches), dim = 0)
    inp = inp.view(4, 64, 1, 32, 32) # for patch size 32

    grayscale_cam = cam(input_tensor=inp, targets=None)
    
    lps = inp[0, :, :, :]
    rps = inp[1, :, :, :]
    lips = inp[2, :, :, :]
    rips = inp[3, :, :, :]


    vis_left_patches = []
    vis_right_patches = []
    vis_left_img_patches = []
    vis_right_img_patches = []

    for i in range(64):
        
        img = lips[i, :, :].permute(1, 2, 0).detach().cpu().numpy()
        img = cv2.normalize(img, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = np.clip(img, 0, 1)
        img = np.reshape(img, (32, 32, 1))
        gcam = grayscale_cam[i, :, :]
        viz = show_cam_on_image(img, gcam)
        vis_left_patches.append(viz)

    reconstructed_image_dis_left = np.zeros((256, 256, 3), dtype=np.float32)
    index = 0
    for x in range(0, 256 - 32 + 1, 32):
        for y in range(0, 256 - 32 + 1, 32):
            # print(x, y, index)
            reconstructed_image_dis_left[x:x+32, y:y+32] = vis_left_patches[index]
            index+=1
    reconstructed_image_dis_left = reconstructed_image_dis_left.astype("uint8")
    plt.imshow(reconstructed_image_dis_left)
    plt.colorbar()
    plt.savefig("save-name"+str(k)+".png")
    plt.clf()
    k+=1