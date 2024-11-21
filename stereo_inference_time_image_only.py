from PIL import Image
import cv2
import random
import time
random.seed(64)

from torchvision.transforms.functional import to_tensor
import torch

from unet_light_model import UNet_Light
from stereo_models import *


start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quality_model = QNet_CNNWOD().to(device)
# quality_model = QNet_MixerWOD().to(device)
# quality_model = Zhou().to(device)
quality_model.load_state_dict(torch.load("best-model-saved-path.pth"))

dist_model = UNet_Light().to(device)
dist_model.load_state_dict(torch.load("best_unet_light_32_patch_distortion_pristine_all.pth"))

left_img = cv2.imread("sample-left.bmp", cv2.IMREAD_GRAYSCALE)
left_img = cv2.resize(left_img, (256, 256))

right_img = cv2.imread("sample-right.bmp", cv2.IMREAD_GRAYSCALE)
right_img = cv2.resize(right_img, (256, 256))

patch_size=32
stride=32
left_patches_images = ()
right_patches_images = ()

w, h = left_img.shape[1], left_img.shape[0]
for i in range(0, h-patch_size+1, stride):
    for j in range(0, w-patch_size+1, stride):
        patch = left_img[i:i+patch_size, j:j+patch_size]
        patch = Image.fromarray(patch)
        patch = to_tensor(patch)
        patch = patch.unsqueeze(0)
        patch = patch.to(device)
        left_patches_images = left_patches_images + (patch, )

w, h = right_img.shape[1], right_img.shape[0]
for i in range(0, h-patch_size+1, stride):
    for j in range(0, w-patch_size+1, stride):
        patch = right_img[i:i+patch_size, j:j+patch_size]
        patch = Image.fromarray(patch)
        patch = to_tensor(patch)
        patch = patch.unsqueeze(0)
        patch = patch.to(device)
        right_patches_images = right_patches_images + (patch, )

left_patches_images = torch.stack(left_patches_images, dim=1)[0]
right_patches_images = torch.stack(right_patches_images, dim=1)[0]
left_patches_images, right_patches_images = left_patches_images.unsqueeze(1), right_patches_images.unsqueeze(1)

left_patches_images, right_patches_images = left_patches_images.to(device), right_patches_images.to(device)

print(quality_model(left_patches_images, right_patches_images).mean())

#Uncommment for Messai model
# g, _, _, _ = quality_model(left_patches_images, right_patches_images)
# print(g.mean())

print(time.time()-start)