import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
import random
random.seed(64)

from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms.functional import to_tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold
from scipy.io import loadmat
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr, kendalltau, pearsonr

from stereo_models import *
from load_dataset import *
from unet_light_model import UNet_Light

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def errfun(t, x, y, signslope):
    return np.sum((y - fitfun(t, x, signslope))**2)

def fitfun(t, x, signslope):
    return t[0] * logistic(t[1], (x - t[2])) + t[3] * x + t[4]

def logistic(t, x):
    return 0.5 - 1 / (1 + np.exp(t * x))

def log_reg(x, a, b, c, d, e):

    u = 0.5 - 1 / (1 + np.exp(b * (x - c)))

    return a * u + d*x+e


class LIVEPhase2DatasetD(Dataset):
    def __init__(self, dataset_path, indices, train = False):
        self.train = train
        self.dataset_path = dataset_path
        self.indices = indices
        self.images_original, self.images_distorted, self.quality_scores = self.load_image_filenames_and_scores()

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


        c = list(zip(self.images_distorted, self.quality_scores))

        random.shuffle(c)

        self.images_distorted, self.quality_scores = zip(*c)

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

            distorted_image = cv2.imread(self.images_distorted[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image_left = distorted_image[:, :distorted_image.shape[1]//2]
            distorted_image_right = distorted_image[:, distorted_image.shape[1]//2:]
            distorted_image_left = cv2.resize(distorted_image_left, (256, 256))
            distorted_image_right = cv2.resize(distorted_image_right, (256, 256))

            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image_left.shape[1], distorted_image_left.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_left[i:i+patch_size, j:j+patch_size]
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

            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image_right.shape[1], distorted_image_right.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_right[i:i+patch_size, j:j+patch_size]
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
        
        data_mat = loadmat(self.dataset_path+"3DDmosRelease.mat")

        all_original = []
        all_distorted = []
        all_dmos = []

        for i in range(len(data_mat['Dmos'])):

            original = data_mat['RefFilename'][i][0][0]
            distorted = data_mat['StiFilename'][i][0][0]
            dmos = data_mat['Dmos'][i][0]

            original = self.dataset_path+"Reference/"+original
            distorted = self.dataset_path+"Stimuli/"+distorted

            all_original.append(original)     
            all_distorted.append(distorted)   

            all_dmos.append(dmos)

        scaled_scores = [(x - min(all_dmos)) * 100 / (max(all_dmos)-min(all_dmos)) for x in all_dmos]

        return all_original, all_distorted, scaled_scores


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
        

class WIVCID(Dataset):
    def __init__(self, dataset_path, indices, train = False):
        self.train = train
        self.dataset_path = dataset_path
        self.indices = indices
        self.images_original, self.images_distorted, self.quality_scores, self.left_quality_scores, self.right_quality_scores = self.load_image_filenames_and_scores()

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

        c = list(zip(self.images_distorted, self.quality_scores, self.left_quality_scores, self.right_quality_scores))

        random.shuffle(c)

        self.images_distorted, self.quality_scores, self.left_quality_scores, self.right_quality_scores = zip(*c)

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

            distorted_image = cv2.imread(self.images_distorted[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image_left = distorted_image[:, :distorted_image.shape[1]//2]
            distorted_image_right = distorted_image[:, distorted_image.shape[1]//2:]
            distorted_image_left = cv2.resize(distorted_image_left, (256, 256))
            distorted_image_right = cv2.resize(distorted_image_right, (256, 256))

            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image_left.shape[1], distorted_image_left.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_left[i:i+patch_size, j:j+patch_size]
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
                    self.label_L.append(self.left_quality_scores[ind])
            self.left_distorted_patches.append(patches)
            self.left_image_patches.append(patches_images)

            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image_right.shape[1], distorted_image_right.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_right[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patch = patch.unsqueeze(0)
                    patch = patch.to(device)
                    patches_images = patches_images + (patch, )
                    self.all_right_image_patches = self.all_right_image_patches + (patch,)
                    distortion_patch = dist_model(patch).detach().cpu()
                    patches = patches + (distortion_patch[0],)
                    self.all_right_patches = self.all_right_patches + (distortion_patch[0],)
                    self.label_R.append(self.right_quality_scores[ind])
            self.right_distorted_patches.append(patches)
            self.right_image_patches.append(patches_images)


    def load_image_filenames_and_scores(self):
        
        mos = pd.read_csv(self.dataset_path+"mos.csv", header = None)

        # print(data_mat.keys())
        all_original = []
        all_distorted = []
        all_mos = []
        all_left_mos = []
        all_right_mos = []

        for i in range(len(mos)):

            if i%55 == 0:
                omos = mos.loc[i, 2]

            original = mos.loc[i, 1]+".png"

            distorted = mos.loc[i, 0]+".png"

            moscore = omos - mos.loc[i, 2]

            mosleft = mos.loc[i, 3]

            mosright = mos.loc[i, 4]

            original = self.dataset_path+"Session3DIQ/"+original

            distorted = self.dataset_path+"Session3DIQ/"+distorted

            all_original.append(original)
            all_distorted.append(distorted)
            all_mos.append(moscore)
            all_left_mos.append(mosleft)
            all_right_mos.append(mosright)


        scaled_scores = [(x - min(all_mos)) * 100 / (max(all_mos)-min(all_mos)) for x in all_mos]

        return all_original, all_distorted, scaled_scores, all_left_mos, all_right_mos


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
        

class WIVCIID(Dataset):
    def __init__(self, dataset_path, indices, train = False):
        self.train = train
        self.dataset_path = dataset_path
        self.indices = indices
        self.images_original, self.images_distorted, self.quality_scores, self.left_quality_scores, self.right_quality_scores = self.load_image_filenames_and_scores()

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
        
        c = list(zip(self.images_distorted, self.quality_scores, self.left_quality_scores, self.right_quality_scores))

        random.shuffle(c)

        self.images_distorted, self.quality_scores, self.left_quality_scores, self.right_quality_scores = zip(*c)

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

            distorted_image = cv2.imread(self.images_distorted[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image_left = distorted_image[:, :distorted_image.shape[1]//2]
            distorted_image_right = distorted_image[:, distorted_image.shape[1]//2:]
            distorted_image_left = cv2.resize(distorted_image_left, (256, 256))
            distorted_image_right = cv2.resize(distorted_image_right, (256, 256))

            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image_left.shape[1], distorted_image_left.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_left[i:i+patch_size, j:j+patch_size]
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
                    self.label_L.append(self.left_quality_scores[ind])
            self.left_distorted_patches.append(patches)
            self.left_image_patches.append(patches_images)

            patch_size=32
            stride=32
            patches = ()
            patches_images = ()
            w, h = distorted_image_right.shape[1], distorted_image_right.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_right[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patch = patch.unsqueeze(0)
                    patch = patch.to(device)
                    patches_images = patches_images + (patch, )
                    self.all_right_image_patches = self.all_right_image_patches + (patch, )
                    distortion_patch = dist_model(patch).detach().cpu()
                    patches = patches + (distortion_patch[0],)
                    self.all_right_patches = self.all_right_patches + (distortion_patch[0],)
                    self.label_R.append(self.right_quality_scores[ind])
            self.right_distorted_patches.append(patches)
            self.right_image_patches.append(patches_images)

        

    def load_image_filenames_and_scores(self):
        
        mos = pd.read_csv(self.dataset_path+"mos.csv", header = None)

        all_original = []
        all_distorted = []
        all_mos = []
        all_left_mos = []
        all_right_mos = []

        for i in range(len(mos)):

            if i%46 == 0:
                omos = mos.loc[i, 2]

            original = mos.loc[i, 1]+".png"

            distorted = mos.loc[i, 0]+".png"

            moscore = omos - mos.loc[i, 2]

            mosleft = mos.loc[i, 3]

            mosright = mos.loc[i, 4]

            original = self.dataset_path+"Session3DIQ/"+original

            distorted = self.dataset_path+"Session3DIQ/"+distorted

            all_original.append(original)
            all_distorted.append(distorted)
            all_mos.append(moscore)
            all_left_mos.append(mosleft)
            all_right_mos.append(mosright)

        scaled_scores = [(x - min(all_mos)) * 100 / (max(all_mos)-min(all_mos)) for x in all_mos]

        return all_original, all_distorted, scaled_scores, all_left_mos, all_right_mos


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
        
        
class IRCCynDatasetD(Dataset):
    def __init__(self, dataset_path, indices, train = False):
        self.train = train
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

            all_original_left_images.append(original_l)
            all_original_right_images.append(original_r)
            all_distorted_left_images.append(distorted_l)
            all_distorted_right_images.append(distorted_r)
            all_dmos.append(dmos)


        scaled_scores = [(x - min(all_dmos)) * 100 / (max(all_dmos)-min(all_dmos)) for x in all_dmos]
        return all_original_left_images, all_original_right_images, all_distorted_left_images, all_distorted_right_images, scaled_scores

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


# Training

quality_model = QNet().to(device)
print("#params", sum(x.numel() for x in quality_model.parameters()))
optimizer = torch.optim.SGD(quality_model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)
torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)
criterion = nn.L1Loss()

seed = 64
batch_size = 2
num_epochs = 200

indices = list(range(0, 360)) #Live 2
train_dataset_l2 = LIVEPhase2DatasetD(dataset_path="path-to-Live-phase-2-dataset", indices=indices, train=True)

# indices = list(range(0, 330)) #WIVC 1
# train_dataset_w1 = WIVCID(dataset_path="path-to-WIVCI-3D/", indices=indices, train=True)

indices = list(range(0, 460)) #WIVC 2
train_dataset_w2 = WIVCIID(dataset_path="path-to-WIVCII-3D/", indices=indices, train=True)

# indices = list(range(0, 97)) #IRCCyn
# train_dataset_ir = IRCCynDatasetD(dataset_path="path-to-IRCCyn", indices=indices, train=True)


train_dataset = ConcatDataset([train_dataset_l2, train_dataset_w2])

print("Dataset loaded for training: ", len(train_dataset))

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

min_val_loss = 99999999

for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):

    print(f"Fold {fold+1}/{num_folds}")

    train_sampler = SubsetRandomSampler(train_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_index)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

    for epoch in tqdm(range(num_epochs)):

        epoch_loss = 0.0
        quality_model.train()

        for left_patch, right_patch, target_score, left_target_score, right_target_score, left_image_patch, right_image_patch in train_loader:
            left_patch, right_patch, target_score, left_target_score, right_target_score, left_image_patch, right_image_patch = left_patch.to(device), right_patch.to(device), target_score.to(device), left_target_score.to(device), right_target_score.to(device), left_image_patch.to(device), right_image_patch.to(device)
           
            output_score = quality_model(left_patch, right_patch, left_image_patch, right_image_patch)
            target_score = target_score.unsqueeze(1).float()


            loss = criterion(output_score.float(), target_score.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

        print(f"Epoch loss {epoch + 1}/{num_epochs}, Loss: {epoch_loss/(len(train_loader))}")


        with torch.no_grad():
            val_loss = 0.0
            quality_model.eval()

            for left_patch, right_patch, target_score, left_target_score, right_target_score, left_image_patch, right_image_patch in val_loader:
                left_patch, right_patch, target_score, left_target_score, right_target_score, left_image_patch, right_image_patch = left_patch.to(device), right_patch.to(device), target_score.to(device), left_target_score.to(device), right_target_score.to(device), left_image_patch.to(device), right_image_patch.to(device)

                output_score = quality_model(left_patch, right_patch, left_image_patch, right_image_patch)

                target_score = target_score.unsqueeze(1).float()
                
                loss = criterion(output_score.float(), target_score.float())


                val_loss += loss.item()

            print(f"Validation Loss: {val_loss/(len(val_loader))}")

            if val_loss<min_val_loss:
                min_val_loss=val_loss
                torch.save(quality_model.state_dict(), "best-save-double-model.pth")
                print("SAVED!")

        torch.save(quality_model.state_dict(), "last-save-double-model.pth")

    break


# Testing:

indices = list(range(0, 365)) #Live 1
test_dataset = LIVEPhase1Dataset(dataset_path="/path-to-live-1-dataset", indices=indices, train=False)

# indices = list(range(0, 360)) #Live 2
# test_dataset = LIVEPhase2Dataset(dataset_path="path-to-live-2-dataset", indices=indices, train=False)

# indices = list(range(0, 330)) #WIVC 1
# test_dataset = WIVCI(dataset_path="path-to-wivc-1-dataset", indices=indices, train=False)

# indices = list(range(0, 460)) #WIVC 2
# test_dataset = WIVCII(dataset_path="path-to-wivc-2-dataset", indices=indices, train=False)

# indices = list(range(0, 97)) #IRCCyn
# test_dataset = IRCCynDataset(dataset_path="path-to-irccyn-dataset", indices=indices, train=False)

print("Dataset loaded for testing: ", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=1)

quality_model = QNet().to(device)
quality_model.load_state_dict(torch.load("best-save-double-model.pth")) 

target_scores = []
output_scores = []

with torch.no_grad():
    
    quality_model.eval()

    for left_patches, right_patches, target_score, left_image_patches, right_image_patches in test_loader:

        left_patches = torch.stack(left_patches, dim=1)[0]
        right_patches = torch.stack(right_patches, dim=1)[0]

        left_image_patches = torch.stack(left_image_patches, dim=1)[0]
        right_image_patches = torch.stack(right_image_patches, dim=1)[0]
        
        left_patches, right_patches, target_score, left_image_patches, right_image_patches = left_patches.to(device), right_patches.to(device), target_score.to(device), left_image_patches.to(device), right_image_patches.to(device)

        output_score = quality_model(left_patches, right_patches, left_image_patches, right_image_patches).mean()
        
        target_score = target_score.unsqueeze(1).float()

        output_scores.append(output_score.item())
        target_scores.append(target_score.item())

    output_scores = np.array(output_scores)
    target_scores = np.array(target_scores)

print("Before Logistic Fitting: ")


p, _ = pearsonr(output_scores, target_scores)
s, _ = spearmanr(output_scores, target_scores)
k, _ = kendalltau(output_scores, target_scores)

print("Pearson Coefficient: ", p)
print("Spearman Coefficient: ", s)
print("Kendall Coefficient: ", k)

print("Nelder Mead fitting: ")

correlation = np.corrcoef(output_scores, target_scores)[0, 1]
if correlation > 0:
    t = [0.0, 0.0, float(np.mean(output_scores)), 0.0, -1.0]
    t[0] = np.abs(np.max(target_scores) - np.min(target_scores))
    t[1] = 1 / np.std(output_scores)
    t[4] = -1
    signslope = 1.0
else:
    t = [0.0, 0.0, float(np.mean(output_scores)), 0.0, -1.0]
    t[0] = -np.abs(np.max(target_scores) - np.min(target_scores))
    t[1] = 1 / np.std(output_scores)
    t[4] = -1
    signslope = -1.0

result = minimize(errfun, t, args=(output_scores, target_scores, signslope), method='Nelder-Mead', options={'disp': False})
optimal_t = result.x

test_outputs_fit = fitfun(optimal_t, output_scores, signslope)
p, _ = pearsonr(test_outputs_fit, target_scores)
s, _ = spearmanr(test_outputs_fit, target_scores)
k, _ = kendalltau(test_outputs_fit, target_scores)

print("Pearson Coefficient: ", p)
print("Spearman Coefficient: ", s)
print("Kendall Coefficient: ", k)

plt.figure()
plt.scatter(test_outputs_fit, target_scores)
plt.xlabel("Predicted Scores", fontsize=14)
plt.ylabel("Ground Truth", fontsize=14)

print("After Logistic Fitting: ")
popt, pcov = curve_fit(log_reg, output_scores, target_scores, p0 = [max(target_scores), min(target_scores), np.mean(output_scores), 1, 0], maxfev=100000)

test_op_curve_fit = log_reg(output_scores, *popt)

p, _ = pearsonr(test_op_curve_fit, target_scores)
s, _ = spearmanr(test_op_curve_fit, target_scores)
k, _ = kendalltau(test_op_curve_fit, target_scores)

print("Pearson Coefficient: ", p)
print("Spearman Coefficient: ", s)
print("Kendall Coefficient: ", k)

plt.figure()
plt.scatter(test_op_curve_fit, target_scores)
plt.xlabel("Predicted Scores", fontsize=14)
plt.ylabel("Ground Truth", fontsize=14)
plt.savefig("scatter-plot.png")