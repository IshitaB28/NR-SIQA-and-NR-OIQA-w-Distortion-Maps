from PIL import Image
import cv2
import random
import time
random.seed(64)

from torchvision.transforms.functional import to_tensor
import torch

from unet_light_model import UNet_Light
from omni2d_models import *


start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quality_model = QNet_CNN().to(device)
# quality_model = QNet_Mixer().to(device)
quality_model.load_state_dict(torch.load("best-model-saved-path.pth"))

dist_model = UNet_Light().to(device)
dist_model.load_state_dict(torch.load("best_unet_light_32_patch_distortion_pristine_all.pth"))

distorted_image = cv2.imread("sample-omni.png", cv2.IMREAD_GRAYSCALE)
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
        distortion_patch = dist_model(patch).detach().cpu()
        patches = patches + (distortion_patch[0],)

patches = torch.stack(patches, dim = 1)[0]
patches_images = torch.stack(patches_images, dim = 1)[0]

patches, patches_images = patches.unsqueeze(1), patches_images.unsqueeze(1)
        
patches, patches_images = patches.to(device), patches_images.to(device)
print(quality_model(patches, patches_images).mean())
print(time.time()-start)

# Comment above and uncomment below for quality assessment using images only

# start = time.time()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# quality_model = QNet_CNNWOD().to(device)
# # quality_model = QNet_MixerWOD().to(device)
# quality_model.load_state_dict(torch.load("best-model-saved-path.pth"))

# distorted_image = cv2.imread("sample-omni.png", cv2.IMREAD_GRAYSCALE)
# distorted_image = cv2.resize(distorted_image, (256, 256))

# patch_size=32
# stride=32
# patches_images = ()

# w, h = distorted_image.shape[1], distorted_image.shape[0]
# for i in range(0, h-patch_size+1, stride):
#     for j in range(0, w-patch_size+1, stride):
#         patch = distorted_image[i:i+patch_size, j:j+patch_size]
#         patch = Image.fromarray(patch)
#         patch = to_tensor(patch)
#         patch = patch.unsqueeze(0)
#         patch = patch.to(device)
#         patches_images = patches_images + (patch, )

# patches_images = torch.stack(patches_images, dim = 1)[0]

# patches_images  = patches_images.unsqueeze(1)

# patches_images = patches_images.to(device)
# print(quality_model(patches_images).mean())
# print(time.time()-start)