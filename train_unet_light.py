import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

from sklearn.model_selection import KFold
from scipy import signal

from unet_light_model import UNet_Light

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
    

class LIVEPristineDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.original_image_paths, self.distorted_image_paths = self.load_image_filenames_and_scores()

        print(len(self.original_image_paths), len(self.distorted_image_paths))

        self.original_ssim_patches = ()
        self.distorted_patches = ()

        prev = ""

        for ind in range(len(self.original_image_paths)):
            now = self.original_image_paths[ind]
            if now == prev:
                pass
            else:
                print(prev)
                prev = now
            image = cv2.imread(self.original_image_paths[ind], cv2.IMREAD_GRAYSCALE)
            distorted_image = cv2.imread(self.distorted_image_paths[ind], cv2.IMREAD_GRAYSCALE)
            
            _, ssim_image = cal_ssim(image, distorted_image)
            ssim_image = cv2.resize(ssim_image, (256, 256))
            
            patch_size=32
            stride=32
            w, h = ssim_image.shape[1], ssim_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = ssim_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    self.original_ssim_patches = self.original_ssim_patches + (patch,)

            distorted_image = cv2.resize(distorted_image, (256, 256))
            
            patch_size=32
            stride=32
            w, h = distorted_image.shape[1], distorted_image.shape[0]
            for i in range(0, h-patch_size+1, stride):
                for j in range(0, w-patch_size+1, stride):
                    patch = distorted_image[i:i+patch_size, j:j+patch_size]
                    patch = Image.fromarray(patch)
                    patch = to_tensor(patch)
                    self.distorted_patches = self.distorted_patches + (patch,)

        print(len(self.original_ssim_patches), len(self.distorted_patches))


    def load_image_filenames_and_scores(self):
        ref_folder = self.dataset_path+"pristine/"
        blur_r_folder = self.dataset_path+"blur_r/"
        ff_r_folder = self.dataset_path+"ff_r/"
        jpeg_r_folder = self.dataset_path+"jpeg_r/"
        jp2k_r_folder = self.dataset_path+"jp2k_r/"
        wn_r_folder = self.dataset_path+"wn_r/"

        gen_o = []
        gen_d = []

        for filename in os.listdir(ref_folder):
            image_name = filename[:-4]+"_"
            
            for df in os.listdir(blur_r_folder):
                if image_name in df:
                    gen_o.append(ref_folder+filename)
                    gen_d.append(blur_r_folder+df)
                    
            for df in os.listdir(ff_r_folder):
                if image_name in df:
                    gen_o.append(ref_folder+filename)
                    gen_d.append(ff_r_folder+df)

            for df in os.listdir(jpeg_r_folder):
                if image_name in df:
                    gen_o.append(ref_folder+filename)
                    gen_d.append(jpeg_r_folder+df)

            for df in os.listdir(jp2k_r_folder):
                if image_name in df:
                    gen_o.append(ref_folder+filename)
                    gen_d.append(jp2k_r_folder+df)

            for df in os.listdir(wn_r_folder):
                if image_name in df:
                    gen_o.append(ref_folder+filename)
                    gen_d.append(wn_r_folder+df)

            for df in os.listdir(ref_folder):
                if image_name[:-1]+"." in df:
                    gen_o.append(ref_folder+filename)
                    gen_d.append(ref_folder+df)


        return gen_o, gen_d
    
    def __len__(self):
        return len(self.original_ssim_patches)
    
    def __getitem__(self, index):
        return self.original_ssim_patches[index], self.distorted_patches[index]
    



seed = 64
batch_size = 4

all_dataset = LIVEPristineDataset(dataset_path="path-to-pristine-live")
print("Dataset Loaded. Length: ", len(all_dataset))
torch.save(all_dataset, './patch_wise_32_all_distortion_pristine_all.pt')

# all_dataset = torch.load('./patch_wise_16_all_distortion_pristine_all.pt')


print("Dataset Loaded. Length: ", len(all_dataset))

generator = torch.Generator()
generator.manual_seed(seed)
train_size = int(1 * len(all_dataset))
test_size = len(all_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size], generator=generator)


num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

min_val_loss = 99999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used: ", device)

for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):

    print(f"Fold {fold+1}/{num_folds}")

    train_sampler = SubsetRandomSampler(train_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    model = UNet_Light()

    model.to(device)
    print("#params", sum(x.numel() for x in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)

    num_epochs = 150

    for epoch in tqdm(range(num_epochs)):

        epoch_loss = 0.0
        model.train()
        for reference_patch, distorted_patch in train_loader:

            reference_patch, distorted_patch = reference_patch.to(device), distorted_patch.to(device)
            output_reconstruction = model(distorted_patch)

            reconstruction_loss = nn.MSELoss()(output_reconstruction, reference_patch)

            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()


            epoch_loss += reconstruction_loss.item()
        print(f"Epoch reconstruction loss {epoch + 1}/{num_epochs}, Loss: {epoch_loss/(len(train_loader))}")


        val_sampler = SubsetRandomSampler(val_index)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

        with torch.no_grad():
            val_loss = 0.0
            model.eval()

            for reference_patch, distorted_patch in val_loader:

                reference_patch, distorted_patch = reference_patch.to(device), distorted_patch.to(device)

                output_reconstruction = model(distorted_patch)

                loss = nn.MSELoss()(output_reconstruction, reference_patch)


                val_loss += loss.item()

            print(f"Validation Loss: {val_loss/(len(val_loader))}")

        if val_loss<min_val_loss:
                min_val_loss=val_loss
                torch.save(model.state_dict(), "best_unet_light_32_patch_distortion_pristine_all.pth")
                print("SAVED!")

        torch.save(model.state_dict(), "last-model-save.pth")

        print()

    break