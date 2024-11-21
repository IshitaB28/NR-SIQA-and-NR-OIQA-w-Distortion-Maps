import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(64)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omni2d_models import *
from load_dataset import CVIQD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def errfun(t, x, y, signslope):
    return np.sum((y - fitfun(t, x, signslope))**2)

def fitfun(t, x, signslope):
    # return t[0] * logistic(t[1], (x - t[2])) + t[3] + signslope * np.exp(t[4]) * x
    return t[0] * logistic(t[1], (x - t[2])) + t[3] * x + t[4]

def logistic(t, x):
    return 0.5 - 1 / (1 + np.exp(t * x))

def log_reg(x, a, b, c, d, e):

    u = 0.5 - 1 / (1 + np.exp(b * (x - c)))

    return a * u + d*x+e

quality_model = QNet_CNN().to(device)
# quality_model = QNet_Mixer().to(device)
# quality_model = QNet_CNNWOD().to(device)
# quality_model = QNet_MixerWOD().to(device)
print("#params", sum(x.numel() for x in quality_model.parameters()))


optimizer = torch.optim.SGD(quality_model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)

torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)
criterion = nn.L1Loss()

seed = 64
batch_size = 2
num_epochs = 2

indices = list(range(0, 544)) #cviqd
train_dataset = CVIQD(indices, score_path="path-to-CVIQ.mat", img_dir="path-to-cviq-images", train=True)

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
        for patch, image_patch, target_score in train_loader:
            
            patch, image_patch, target_score = patch.to(device), image_patch.to(device), target_score.to(device)

            output_score = quality_model(patch, image_patch)
            # output_score = quality_model(image_patch) # for image only models
            
            
            loss = criterion(output_score.float(), target_score.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

        print(f"Epoch loss {epoch + 1}/{num_epochs}, Loss: {epoch_loss/(len(train_loader))}")


        with torch.no_grad():
            val_loss = 0.0
            quality_model.eval()

            for patch, image_patch, target_score in train_loader:
               
                patch, image_patch, target_score = patch.to(device), image_patch.to(device), target_score.to(device)

                output_score = quality_model(patch, image_patch)
                # output_score = quality_model(image_patch) # for image only models
            
                
                loss = criterion(output_score.float(), target_score.float())


                val_loss += loss.item()

            print(f"Validation Loss: {val_loss/(len(val_loader))}")

            if val_loss<min_val_loss:
                min_val_loss=val_loss
                torch.save(quality_model.state_dict(), "best-model-save.pth")
                print("SAVED!")

        torch.save(quality_model.state_dict(), "last-model-save.pth")

    break