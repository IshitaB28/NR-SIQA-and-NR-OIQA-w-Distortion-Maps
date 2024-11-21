import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re
# import lpips

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy import signal

from unet_light_model import UNet_Light
from load_dataset import *


def cal_ssim(img1, img2):
    
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T
     
    M,N = np.shape(img1)

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    img1 = np.float64(img1)
    img2 = np.float64(img2)
 
    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim,ssim_map

def log_reg(x, a, b, c, d, e):

    u = 0.5 - 1 / (1 + np.exp(b * (x - c)))
    return a * u + d*x+e


# The dataset classes here are designed in such a way such that we can display the distorted image, generated distortion map and the original ssim map.
# These functions are different from what will be used for quality assessment
# This script is designed for visual results only


class LIVEPhase1Dataset(Dataset):
    def __init__(self, dataset_path, train = False, transform=None, transformg = None):
        self.train = train
        self.dataset_path = dataset_path
        self.transform = transform
        self.transformg = transformg
        self.o_images_left, self.o_images_right, self.d_images_left, self.d_images_right, self.quality_scores = self.load_image_filenames_and_scores()

        self.left_distorted_patches = []
        self.right_distorted_patches = []
   
        self.label = []


        for ind in range(len(self.d_images_left)):

            distorted_image = cv2.imread(self.d_images_left[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))
            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.left_distorted_patches.append(patches)

        for ind in range(len(self.d_images_right)):

            distorted_image = cv2.imread(self.d_images_right[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))
            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.right_distorted_patches.append(patches)

        

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

            if "blur" in dist_fol:
                distorted_l = self.dataset_path+"blur/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"blur/"+distorted+"_r.bmp"

            elif "ff" in dist_fol:
                distorted_l = self.dataset_path+"ff/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"ff/"+distorted+"_r.bmp"

            elif "jp2k" in dist_fol:
                distorted_l = self.dataset_path+"jp2k/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"jp2k/"+distorted+"_r.bmp"

            elif "jpeg" in dist_fol:
                distorted_l = self.dataset_path+"jpeg/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"jpeg/"+distorted+"_r.bmp"
                
            elif "wn" in dist_fol:
                distorted_l = self.dataset_path+"wn/"+distorted+"_l.bmp"
                distorted_r = self.dataset_path+"wn/"+distorted+"_r.bmp"
                

            all_original_left_images.append(original_l)        

            original_r = self.dataset_path+"refimgs/"+original+"_r.bmp"
            

            all_original_right_images.append(original_r)
            all_distorted_left_images.append(distorted_l)
            all_distorted_right_images.append(distorted_r)
            all_dmos.append(dmos)


        return all_original_left_images, all_original_right_images, all_distorted_left_images, all_distorted_right_images, all_dmos

    def __len__(self):
        return len(self.quality_scores)

    def __getitem__(self, index):

        o_image_left = cv2.imread(self.o_images_left[index], cv2.IMREAD_GRAYSCALE)
        d_image_left = cv2.imread(self.d_images_left[index], cv2.IMREAD_GRAYSCALE)
        _, ssim_left = cal_ssim(o_image_left, d_image_left)
        ssim_left = cv2.resize(ssim_left, (256, 256))
        ssim_left = Image.fromarray(ssim_left)
        ssim_left = to_tensor(ssim_left)

        o_image_right = cv2.imread(self.o_images_right[index], cv2.IMREAD_GRAYSCALE)
        d_image_right = cv2.imread(self.d_images_right[index], cv2.IMREAD_GRAYSCALE)
        _, ssim_right = cal_ssim(o_image_right, d_image_right)
        ssim_right = cv2.resize(ssim_right, (256, 256))
        ssim_right = Image.fromarray(ssim_right)
        ssim_right = to_tensor(ssim_right)

        d_image_left = cv2.imread(self.d_images_left[index], cv2.IMREAD_GRAYSCALE)
        d_image_left = cv2.resize(d_image_left, (256, 256))
        d_image_left = Image.fromarray(d_image_left)
        d_image_left = to_tensor(d_image_left)
        d_image_right = cv2.imread(self.d_images_right[index], cv2.IMREAD_GRAYSCALE)
        d_image_right = cv2.resize(d_image_right, (256, 256))
        d_image_right = Image.fromarray(d_image_right)
        d_image_right = to_tensor(d_image_right)

        return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], ssim_left, ssim_right, d_image_left, d_image_right
    
class LIVEPhase2Dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images_original, self.images_distorted, self.quality_scores = self.load_image_filenames_and_scores()

        self.left_distorted_patches = []
        self.right_distorted_patches = []
   
        self.label = []


        for ind in range(len(self.images_distorted)):

            distorted_image = cv2.imread(self.images_distorted[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image_left = distorted_image[:, :distorted_image.shape[1]//2]
            distorted_image_right = distorted_image[:, distorted_image.shape[1]//2:]
            distorted_image_left = cv2.resize(distorted_image_left, (256, 256))
            distorted_image_right = cv2.resize(distorted_image_right, (256, 256))

            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image_left.shape[1], distorted_image_left.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_left[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.left_distorted_patches.append(patches)

            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image_right.shape[1], distorted_image_right.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_right[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.right_distorted_patches.append(patches)

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
        return all_original, all_distorted, all_dmos

    def __len__(self):
        return len(self.quality_scores)

    def __getitem__(self, index):

        original_image = cv2.imread(self.images_original[index], cv2.IMREAD_GRAYSCALE)
        o_image_left = original_image[:, :original_image.shape[1]//2]
        o_image_right = original_image[:, original_image.shape[1]//2:]

        distorted_image = cv2.imread(self.images_distorted[index], cv2.IMREAD_GRAYSCALE)
        d_image_left = distorted_image[:, :distorted_image.shape[1]//2]
        d_image_right = distorted_image[:, distorted_image.shape[1]//2:]

        _, ssim_left = cal_ssim(o_image_left, d_image_left)
        ssim_left = cv2.resize(ssim_left, (256, 256))
        ssim_left = Image.fromarray(ssim_left)
        ssim_left = to_tensor(ssim_left)

        _, ssim_right = cal_ssim(o_image_right, d_image_right)
        ssim_right = cv2.resize(ssim_right, (256, 256))
        ssim_right = Image.fromarray(ssim_right)
        ssim_right = to_tensor(ssim_right)

        d_image_left = cv2.resize(d_image_left, (256, 256))
        d_image_left = Image.fromarray(d_image_left)
        d_image_left = to_tensor(d_image_left)

        d_image_right = cv2.resize(d_image_right, (256, 256))
        d_image_right = Image.fromarray(d_image_right)
        d_image_right = to_tensor(d_image_right)

        return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], ssim_left, ssim_right, d_image_left, d_image_right

class LiveVRDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images_original, self.images_distorted, self.quality_scores = self.load_image_filenames_and_scores()

        self.left_distorted_patches = []
        self.right_distorted_patches = []
   
        self.label = []


        for ind in range(len(self.images_distorted)):

            distorted_image = cv2.imread(self.images_distorted[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image_left = distorted_image[:distorted_image.shape[0]//2, :]
            distorted_image_right = distorted_image[distorted_image.shape[0]//2:, :]
            distorted_image_left = cv2.resize(distorted_image_left, (256, 256))
            distorted_image_right = cv2.resize(distorted_image_right, (256, 256))

            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image_left.shape[1], distorted_image_left.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_left[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.left_distorted_patches.append(patches)

            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image_right.shape[1], distorted_image_right.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_right[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.right_distorted_patches.append(patches)


    def load_image_filenames_and_scores(self):
        
        data = pd.read_csv(self.dataset_path+"dmos.csv", header = None)
        
        all_original = []
        all_distorted = []
        all_dmos = []
        
        for i in range(len(data)):

            original = data.loc[i, 1]
            distorted = data.loc[i, 0]
            dmos = data.loc[i, 2]

            original = self.dataset_path+"Images/All_Images/"+original
            distorted = self.dataset_path+"Images/All_Images/"+distorted

            all_original.append(original)     
            all_distorted.append(distorted)   

            all_dmos.append(dmos)
        return all_original, all_distorted, all_dmos


    def __len__(self):
        return len(self.quality_scores)

    def __getitem__(self, index):

        original_image = cv2.imread(self.images_original[index], cv2.IMREAD_GRAYSCALE)
        o_image_left = original_image[:original_image.shape[0]//2, :]
        o_image_right = original_image[original_image.shape[0]//2:, :]

        distorted_image = cv2.imread(self.images_distorted[index], cv2.IMREAD_GRAYSCALE)
        d_image_left = distorted_image[:distorted_image.shape[0]//2, :]
        d_image_right = distorted_image[distorted_image.shape[0]//2:, :]

        if not (o_image_left.shape==d_image_left.shape):
            o_image_left = cv2.resize(o_image_left, (d_image_left.shape[1], d_image_left.shape[0]))
            
        _, ssim_left = cal_ssim(o_image_left, d_image_left)
        ssim_left = cv2.resize(ssim_left, (256, 256))
        ssim_left = Image.fromarray(ssim_left)
        ssim_left = to_tensor(ssim_left)


        if not (o_image_right.shape==d_image_right.shape):
            o_image_right = cv2.resize(o_image_right, (d_image_right.shape[1], d_image_right.shape[0]))
            
        _, ssim_right = cal_ssim(o_image_right, d_image_right)
        ssim_right = cv2.resize(ssim_right, (256, 256))
        ssim_right = Image.fromarray(ssim_right)
        ssim_right = to_tensor(ssim_right)

        d_image_left = cv2.resize(d_image_left, (256, 256))
        d_image_left = Image.fromarray(d_image_left)
        d_image_left = to_tensor(d_image_left)

        d_image_right = cv2.resize(d_image_right, (256, 256))
        d_image_right = Image.fromarray(d_image_right)
        d_image_right = to_tensor(d_image_right)

        return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], ssim_left, ssim_right, d_image_left, d_image_right
    
class IRCCynDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.o_images_left, self.o_images_right, self.d_images_left, self.d_images_right, self.quality_scores = self.load_image_filenames_and_scores()

        self.left_distorted_patches = []
        self.right_distorted_patches = []
   
        self.label = []


        for ind in range(len(self.d_images_left)):

            distorted_image = cv2.imread(self.d_images_left[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))
            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.left_distorted_patches.append(patches)

        for ind in range(len(self.d_images_right)):

            distorted_image = cv2.imread(self.d_images_right[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))
            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.right_distorted_patches.append(patches)

        

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


        return all_original_left_images, all_original_right_images, all_distorted_left_images, all_distorted_right_images, all_dmos

    def __len__(self):
        return len(self.quality_scores)

    def __getitem__(self, index):

        o_image_left = cv2.imread(self.o_images_left[index], cv2.IMREAD_GRAYSCALE)
        d_image_left = cv2.imread(self.d_images_left[index], cv2.IMREAD_GRAYSCALE)
        _, ssim_left = cal_ssim(o_image_left, d_image_left)
        ssim_left = cv2.resize(ssim_left, (256, 256))
        ssim_left = Image.fromarray(ssim_left)
        ssim_left = to_tensor(ssim_left)

        o_image_right = cv2.imread(self.o_images_right[index], cv2.IMREAD_GRAYSCALE)
        d_image_right = cv2.imread(self.d_images_right[index], cv2.IMREAD_GRAYSCALE)
        _, ssim_right = cal_ssim(o_image_right, d_image_right)
        ssim_right = cv2.resize(ssim_right, (256, 256))
        ssim_right = Image.fromarray(ssim_right)
        ssim_right = to_tensor(ssim_right)

        d_image_left = cv2.imread(self.d_images_left[index], cv2.IMREAD_GRAYSCALE)
        d_image_left = cv2.resize(d_image_left, (256, 256))
        d_image_left = Image.fromarray(d_image_left)
        d_image_left = to_tensor(d_image_left)
        d_image_right = cv2.imread(self.d_images_right[index], cv2.IMREAD_GRAYSCALE)
        d_image_right = cv2.resize(d_image_right, (256, 256))
        d_image_right = Image.fromarray(d_image_right)
        d_image_right = to_tensor(d_image_right)

        return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], ssim_left, ssim_right, d_image_left, d_image_right
    
         
class WIVCDatasets(Dataset):
    def __init__(self, dataset_path, score_path):
        self.dataset_path = dataset_path
        self.score_path = score_path
        self.images_original, self.images_distorted, self.quality_scores = self.load_image_filenames_and_scores()

        self.left_distorted_patches = []
        self.right_distorted_patches = []
   
        self.label = []


        for ind in range(len(self.images_distorted)):

            distorted_image = cv2.imread(self.images_distorted[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image_left = distorted_image[:, :distorted_image.shape[1]//2]
            distorted_image_right = distorted_image[:, distorted_image.shape[1]//2:]
            distorted_image_left = cv2.resize(distorted_image_left, (256, 256))
            distorted_image_right = cv2.resize(distorted_image_right, (256, 256))

            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image_left.shape[1], distorted_image_left.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_left[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.left_distorted_patches.append(patches)

            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image_right.shape[1], distorted_image_right.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image_right[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.right_distorted_patches.append(patches)

    def sorted_alphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    def load_image_filenames_and_scores(self):
        items_list = self.sorted_alphanumeric(os.listdir(self.dataset_path))
        original = []
        distorted = []
        df = pd.read_excel(self.score_path, header=None)
        all_mos = []
        x = 0
        for itname in items_list:
            if x%55==0:
                o_name = self.dataset_path+itname
            x+=1
            dname = self.dataset_path+itname
            distorted.append(dname)
            original.append(o_name)
            sl = int(itname[1:-4])
            all_mos.append(df[2][sl-1])

        return original, distorted, all_mos


    def __len__(self):
        return len(self.quality_scores)

    def __getitem__(self, index):
        original_image = cv2.imread(self.images_original[index], cv2.IMREAD_GRAYSCALE)
        o_image_left = original_image[:, :original_image.shape[1]//2]
        o_image_right = original_image[:, original_image.shape[1]//2:]

        distorted_image = cv2.imread(self.images_distorted[index], cv2.IMREAD_GRAYSCALE)
        d_image_left = distorted_image[:, :distorted_image.shape[1]//2]
        d_image_right = distorted_image[:, distorted_image.shape[1]//2:]

        _, ssim_left = cal_ssim(o_image_left, d_image_left)
        ssim_left = cv2.resize(ssim_left, (256, 256))
        ssim_left = Image.fromarray(ssim_left)
        ssim_left = to_tensor(ssim_left)
        _, ssim_right = cal_ssim(o_image_right, d_image_right)
        ssim_right = cv2.resize(ssim_right, (256, 256))
        ssim_right = Image.fromarray(ssim_right)
        ssim_right = to_tensor(ssim_right)

        d_image_left = cv2.resize(d_image_left, (256, 256))
        d_image_left = Image.fromarray(d_image_left)
        d_image_left = to_tensor(d_image_left)

        d_image_right = cv2.resize(d_image_right, (256, 256))
        d_image_right = Image.fromarray(d_image_right)
        d_image_right = to_tensor(d_image_right)

        return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], ssim_left, ssim_right, d_image_left, d_image_right
    
class CVIQD_Dataset(Dataset):
    def __init__(self, score_path, img_dir):
        

        self.score_path = score_path
        self.img_dir = img_dir
        
        self.reference_images, self.distorted_images, self.scores = self.load_triplet_data(self.score_path, self.img_dir)

        self.distorted_patches = []
   
        self.label = []


        for ind in range(len(self.distorted_images)):

            distorted_image = cv2.imread(self.distorted_images[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.resize(distorted_image, (256, 256))

            patch_size=32
            stride=32
            patches = ()
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    patches = patches + (patch,)
            self.distorted_patches.append(patches)

    def load_triplet_data(self, score_path, img_dir):      
        data_mat = loadmat(score_path)
        all_distorted_images = []
        all_reference_images = []
        all_mos = []

        j = 33
        for i in range(len(data_mat['CVIQ'])):

            img = data_mat['CVIQ'][i][0][0]
            mos = data_mat['CVIQ'][i][1][0]

            if i<j:
                all_distorted_images.append(img_dir+img)
                all_reference_images.append(img_dir+data_mat['CVIQ'][j][0][0])
                all_mos.append(mos)
            elif i==j:
                all_distorted_images.append(img_dir+img)
                all_reference_images.append(img_dir+data_mat['CVIQ'][j][0][0])
                all_mos.append(mos)
                j+=34

        return all_reference_images, all_distorted_images, all_mos

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):

        original_image = cv2.imread(self.reference_images[index], cv2.IMREAD_GRAYSCALE)

        distorted_image = cv2.imread(self.distorted_images[index], cv2.IMREAD_GRAYSCALE)

        _, ssim = cal_ssim(original_image, distorted_image)
        ssim = cv2.resize(ssim, (256, 256))
        ssim = Image.fromarray(ssim)
        ssim = to_tensor(ssim)

        distorted_image = cv2.resize(distorted_image, (256, 256))
        distorted_image = Image.fromarray(distorted_image)
        distorted_image = to_tensor(distorted_image)

        return self.distorted_patches[index], self.scores[index], ssim, distorted_image
    

seed = 64
batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# all_dataset = LIVEDataset(dataset_path="path-to-live-phase-1-dataset")
# all_dataset = LIVEPhase2Dataset(dataset_path="path-to-live-phase-2-dataset")
# all_dataset = WIVCDataset(dataset_path="path-to-WIVC-1-Session3DIQ", score_path="path-to-WIVC-1-mos")
# all_dataset = WIVCDataset(dataset_path="path-to-WIVC-2-Session3DIQ", score_path="path-to-WIVC-2-mos")
all_dataset = IRCCynDataset(dataset_path="path-to-IRCCyn-dataset")
# all_dataset = LiveVRDataset(dataset_path="path-to-LIVE3DVR-dataset")
# all_dataset = CVIQD_Dataset(score_path="", img_dir="")

print("Data loaded. Length: ", len(all_dataset))

all_loader = DataLoader(all_dataset)

model = UNet_Light().to(device)

model.load_state_dict(torch.load("best_unet_light_32_patch_distortion_pristine_all.pth"))

patch_size = 32

# For stereoscopic databases:

with torch.no_grad():
    model.eval()

    save_index = 1

    for left_patches, right_patches, label, left_reference_ssim, right_reference_ssim, left_distorted, right_distorted in all_loader:
        

        left_patches = torch.stack(left_patches, dim=1)
        right_patches = torch.stack(right_patches, dim=1)


        left_patches, right_patches = left_patches[0].to(device), right_patches[0].to(device)
        

        reconstructed_left_patches = model(left_patches)
        reconstructed_right_patches = model(right_patches)
        

        num_patches_row = (256 - patch_size) // patch_size + 1
        num_patches_col = (256 - patch_size) // patch_size + 1
        


        reconstructed_image_left = np.zeros((256, 256, 1), dtype=np.float32)
        count_array_left = np.zeros((256, 256, 1), dtype=np.float32)

        predicted_quality = 0.0
        left_predicted_quality = 0.0
        right_predicted_quality = 0.0
                
        index = 0
        for i in range(0, 256 - patch_size + 1, patch_size):
            for j in range(0, 256 - patch_size + 1, patch_size):
                lp = reconstructed_left_patches[index].permute(1, 2, 0).cpu().numpy()
                left_predicted_quality+=np.mean(lp)*np.var(lp)
                reconstructed_image_left[i:i+patch_size, j:j+patch_size] = lp
                count_array_left[i:i+patch_size, j:j+patch_size] += 1
                index += 1

        
        reconstructed_image_right = np.zeros((256, 256, 1), dtype=np.float32) 
        count_array_right = np.zeros((256, 256, 1), dtype=np.float32)
        
        index = 0
        for i in range(0, 256 - patch_size + 1, patch_size):
            for j in range(0, 256 - patch_size + 1, patch_size):
                rp = reconstructed_right_patches[index].permute(1, 2, 0).cpu().numpy()
                right_predicted_quality+=np.mean(rp)*np.var(rp)
                reconstructed_image_right[i:i+patch_size, j:j+patch_size] = rp
                count_array_right[i:i+patch_size, j:j+patch_size] += 1
                index += 1
        
        reconstructed_image_left = cv2.normalize(reconstructed_image_left, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        reconstructed_image_left = np.clip(reconstructed_image_left, 0, 1)
        reconstructed_image_left = (255*reconstructed_image_left).astype(np.uint8)

        reconstructed_image_right /= count_array_right
        reconstructed_image_right = cv2.normalize(reconstructed_image_right, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        reconstructed_image_right = np.clip(reconstructed_image_right, 0, 1)
        reconstructed_image_right = (255*reconstructed_image_right).astype(np.uint8)
        
        left_distorted = left_distorted[0].permute(1, 2, 0).cpu().numpy() 
        left_distorted = cv2.normalize(left_distorted, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        left_distorted = np.clip(left_distorted, 0, 1)
        left_distorted = (255*left_distorted).astype(np.uint8)

        right_distorted = right_distorted[0].permute(1, 2, 0).cpu().numpy()
        right_distorted = cv2.normalize(right_distorted, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        right_distorted = np.clip(right_distorted, 0, 1)
        right_distorted = (255*right_distorted).astype(np.uint8)

        left_reference = left_reference_ssim[0].permute(1, 2, 0).cpu().numpy()
        left_reference = cv2.normalize(left_reference, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        left_reference = np.clip(left_reference, 0, 1)
        left_reference = (255*left_reference).astype(np.uint8)

        right_reference = right_reference_ssim[0].permute(1, 2, 0).cpu().numpy()
        right_reference = cv2.normalize(right_reference, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        right_reference = np.clip(right_reference, 0, 1)
        right_reference = (255*right_reference).astype(np.uint8)
        
        distorted_images = np.concatenate((left_distorted, right_distorted), axis=1)
        output_reconstruction = np.concatenate((reconstructed_image_left, reconstructed_image_right), axis=1)
        original_images = np.concatenate((left_reference, right_reference), axis=1)
        to_save = np.concatenate((distorted_images, output_reconstruction, original_images), axis=0)
        
        cv2.imwrite("path-to-save"+str(save_index)+".png", to_save)
        
        save_index+=1


# Uncomment below for 2D omnidircetional database
        
# with torch.no_grad():
#     model.eval()

#     save_index = 1
        
#     for patches, label, reference_ssim, distorted in all_loader:
#         patches = torch.stack(patches, dim = 1)
#         patches = patches[0].to(device)
#         reconstructed_patches = model(patches)

#         reconstructed_image = np.zeros((256, 256, 1), dtype=np.float32)
#         index = 0
#         for i in range(0, 256 - patch_size + 1, patch_size):
#             for j in range(0, 256 - patch_size + 1, patch_size):
#                 p = reconstructed_patches[index].permute(1, 2, 0).cpu().numpy()
#                 predicted_quality+=np.mean(p)*np.var(p)
#                 reconstructed_image[i:i+patch_size, j:j+patch_size] = p
#                 index += 1

#         reconstructed_image_left /= count_array_left
#         reconstructed_image = cv2.normalize(reconstructed_image, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         reconstructed_image = np.clip(reconstructed_image, 0, 1)
#         reconstructed_image = (255*reconstructed_image).astype(np.uint8)

#         distorted = distorted[0].permute(1, 2, 0).cpu().numpy() 
#         distorted = cv2.normalize(distorted, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         distorted = np.clip(distorted, 0, 1)
#         distorted = (255*distorted).astype(np.uint8)

#         reference = reference_ssim[0].permute(1, 2, 0).cpu().numpy()
#         reference = cv2.normalize(reference, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         reference = np.clip(reference, 0, 1)
#         reference = (255*reference).astype(np.uint8)

#         to_save = np.concatenate((distorted, reconstructed_image, reference), axis=1)

#         cv2.imwrite("path-to-save"+str(save_index)+".png", to_save)
            
#         save_index+=1