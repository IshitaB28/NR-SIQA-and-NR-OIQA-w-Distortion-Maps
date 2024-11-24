import pandas as pd
import cv2
from PIL import Image
import random
random.seed(64)

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torch

from scipy.io import loadmat

from unet_light_model import UNet_Light


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LIVEPhase1Dataset(Dataset):
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
        if self.train:
            return len(self.all_left_patches)
        else:
            return len(self.left_distorted_patches)

    def __getitem__(self, index):

        if self.train:
            return self.all_left_patches[index], self.all_right_patches[index], self.label[index], self.label_L[index], self.label_R[index], self.all_left_image_patches[index], self.all_right_image_patches[index]
        else:
            return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], self.left_image_patches[index], self.right_image_patches[index]
        

class LIVEPhase2Dataset(Dataset):
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
        return all_original, all_distorted, all_dmos


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
        

class WIVCI(Dataset):
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

        

        print("Train indices: ", train_index)
        print()
        print("Test indices", test_index)
        print()

        if self.train:
            self.curr_indices = train_index
        else:
            self.curr_indices = test_index
        for ind in range(len(self.curr_indices)):

            print(self.images_distorted[ind])

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

        all_original = []
        all_distorted = []
        all_mos = []
        all_left_mos = []
        all_right_mos = []

        for i in range(len(mos)):

            original = mos.loc[i, 1]+".png"

            distorted = mos.loc[i, 0]+".png"

            moscore = mos.loc[i, 2]

            mosleft = mos.loc[i, 3]

            mosright = mos.loc[i, 4]

            original = self.dataset_path+"Session3DIQ/"+original

            distorted = self.dataset_path+"Session3DIQ/"+distorted

            all_original.append(original)
            all_distorted.append(distorted)
            all_mos.append(moscore)
            all_left_mos.append(mosleft)
            all_right_mos.append(mosright)

        return all_original, all_distorted, all_mos, all_left_mos, all_right_mos


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
        

class WIVCII(Dataset):
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

        # print(data_mat.keys())
        all_original = []
        all_distorted = []
        all_mos = []
        all_left_mos = []
        all_right_mos = []

        for i in range(len(mos)):

            original = mos.loc[i, 1]+".png"

            distorted = mos.loc[i, 0]+".png"

            moscore = mos.loc[i, 2]

            mosleft = mos.loc[i, 3]

            mosright = mos.loc[i, 4]

            original = self.dataset_path+"Session3DIQ/"+original

            distorted = self.dataset_path+"Session3DIQ/"+distorted

            all_original.append(original)
            all_distorted.append(distorted)
            all_mos.append(moscore)
            all_left_mos.append(mosleft)
            all_right_mos.append(mosright)

        return all_original, all_distorted, all_mos, all_left_mos, all_right_mos


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


class LiveVRDataset(Dataset):
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
            distorted_image_left = distorted_image[:distorted_image.shape[0]//2, :]
            distorted_image_right = distorted_image[distorted_image.shape[0]//2:, :]
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
        if self.train:
            return len(self.all_left_patches)
        else:
            return len(self.left_distorted_patches)

    def __getitem__(self, index):

        if self.train:
            return self.all_left_patches[index], self.all_right_patches[index], self.label[index], self.label_L[index], self.label_R[index], self.all_left_image_patches[index], self.all_right_image_patches[index]
        else:
            return self.left_distorted_patches[index], self.right_distorted_patches[index], self.quality_scores[index], self.left_image_patches[index], self.right_image_patches[index]


class CVIQD(Dataset):
    def __init__(self, indices, score_path, img_dir, train=False):
        
        self.train = train

        self.score_path = score_path
        self.img_dir = img_dir
        
        self.reference_images, self.distorted_images, self.scores = self.load_triplet_data(self.score_path, self.img_dir)

        self.distorted_patches = []
        self.image_patches = []

        self.all_patches = ()
        self.all_image_patches = ()
   
        self.label = []

        dist_model = UNet_Light().to(device)
        dist_model.load_state_dict(torch.load("best_unet_light_32_patch_distortion_pristine_all.pth"))


        c = list(zip(self.reference_images, self.distorted_images, self.scores))

        random.shuffle(c)

        self.reference_images, self.distorted_images, self.scores = zip(*c)

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

            distorted_image = cv2.imread(self.distorted_images[ind], cv2.IMREAD_GRAYSCALE)
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
                    self.all_image_patches = self.all_image_patches + (patch, )
                    distortion_patch = dist_model(patch).detach().cpu()
                    patches = patches + (distortion_patch[0],)
                    self.all_patches = self.all_patches + (distortion_patch[0],)
                    self.label.append(self.scores[ind])
            self.distorted_patches.append(patches)
            self.image_patches.append(patches_images)


        

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
        if self.train:
            return len(self.all_patches)
        else:
            return len(self.distorted_patches)

    def __getitem__(self, index):

        if self.train:
            return self.all_patches[index], self.all_image_patches[index], self.label[index]
        else:
            return self.distorted_patches[index], self.image_patches[index], self.scores[index]

class IQA_ODI_Dataset(Dataset):
    def __init__(self, id_path = "/home/ishita-wicon/Documents/QA/iqa-odi/Imp_ID.txt", dmos_path = "/home/ishita-wicon/Documents/QA/iqa-odi/Imp_DMOS.txt", img_dir = "/home/ishita-wicon/Documents/QA/iqa-odi/all_ref_test_img/", train = False):

        self.train = train
        self.img_dir = img_dir

        self.reference_images, self.distorted_images, self.scores = self.load_triplet_data(self.img_dir, dmos_path, id_path)

        self.distorted_patches = []
        self.image_patches = []

        self.all_patches = ()
        self.all_image_patches = ()
   
        self.label = []

        dist_model = UNet_Light().to(device)
        dist_model.load_state_dict(torch.load("best_unet_light_32_patch_distortion_pristine_all.pth"))


        c = list(zip(self.reference_images, self.distorted_images, self.scores))

        random.shuffle(c)

        self.reference_images, self.distorted_images, self.scores = zip(*c)

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

            distorted_image = cv2.imread(self.distorted_images[ind], cv2.IMREAD_GRAYSCALE)
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
                    self.all_image_patches = self.all_image_patches + (patch, )
                    distortion_patch = dist_model(patch).detach().cpu()
                    patches = patches + (distortion_patch[0],)
                    self.all_patches = self.all_patches + (distortion_patch[0],)
                    self.label.append(self.scores[ind])
            self.distorted_patches.append(patches)
            self.image_patches.append(patches_images)

        

    def load_triplet_data(self, img_dir, dmos_path, id_path):
        id_list = []

        with open(id_path, 'r') as file:
            for line in file:
                id_list.append(line)

        indices_con = []

        for i in range(len(id_list)):
            item = id_list[i]

            if "_ERP_" not in item and "_cpp_" not in item and "_cmp_" not in item and "_isp_" not in item and "_ohp_" not in item:
                indices_con.append(i)
            elif "_cpp_" not in item and "_cmp_" not in item:
                indices_con.append(i)

        dmos_list = []

        with open(dmos_path, 'r') as file:
            for line in file:
                dmos_list.append(line)

        final_dmos = []
        final_ref = []
        final_dist = []

        for i in indices_con:
            x = id_list[i]
            x = x[:x.find("\n")]
            
            spl = x.split("_")
            tp = spl[2]
            num = spl[3]
            refn = img_dir+tp+"_"+num+".jpg"
            distn = img_dir+x+".jpg"

            final_ref.append(refn)
            final_dist.append(distn)

            x = dmos_list[i]
            x = x[:x.find("\n")]
            x = float(x)
            final_dmos.append(x)
        
        return final_ref, final_dist, final_dmos

    def __len__(self):
        if self.train:
            return len(self.all_patches)
        else:
            return len(self.distorted_patches)

    def __getitem__(self, index):

        if self.train:
            return self.all_patches[index], self.all_image_patches[index], self.label[index]
        else:
            return self.distorted_patches[index], self.image_patches[index], self.scores[index]
