import numpy as np
from tqdm import tqdm
import random
random.seed(64)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold

from stereo_models import *
from load_dataset import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


quality_model = QNet_CNN().to(device)
# quality_model = QNet_Mixer().to(device)
# quality_model = ZhouWD().to(device)
# quality_model = QNet_CNNWOD().to(device)
# quality_model = QNet_MixerWOD().to(device)
# quality_model = Zhou().to(device)
print("#params", sum(x.numel() for x in quality_model.parameters()))

optimizer = torch.optim.SGD(quality_model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)
torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)
criterion = nn.L1Loss()

seed = 64
batch_size = 2
num_epochs = 200

indices = list(range(0, 365)) #Live 1
train_dataset = LIVEPhase1Dataset(dataset_path="/path-to-live-1-dataset", indices=indices, train=True)

# indices = list(range(0, 360)) #Live 2
# train_dataset = LIVEPhase2Dataset(dataset_path="path-to-live-2-dataset", indices=indices, train=True)

# indices = list(range(0, 330)) #WIVC 1
# train_dataset = WIVCI(dataset_path="path-to-wivc-1-dataset", indices=indices, train=True)

# indices = list(range(0, 460)) #WIVC 2
# train_dataset = WIVCII(dataset_path="path-to-wivc-2-dataset", indices=indices, train=True)

# indices = list(range(0, 97)) #IRCCyn
# train_dataset = IRCCynDataset(dataset_path="path-to-irccyn-dataset", indices=indices, train=True)

# indices = list(range(0, 449)) #LIVE3DVR
# train_dataset = LiveVRDataset(dataset_path="path-to-LIVE3DVR-dataset", indices=indices, train=True)

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
            # output_score = quality_model(left_image_patch, right_image_patch) # if training models without distortion maps
            
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
                # output_score = quality_model(left_image_patch, right_image_patch) # for models without distortion maps

                target_score = target_score.unsqueeze(1).float()
                
                loss = criterion(output_score.float(), target_score.float())


                val_loss += loss.item()

            print(f"Validation Loss: {val_loss/(len(val_loader))}")

            if val_loss<min_val_loss:
                min_val_loss=val_loss
                torch.save(quality_model.state_dict(), "save-best-model.pth")
                print("SAVED!")

        torch.save(quality_model.state_dict(), "save-last-model.pth")


    break


# Uncomment bleow to train MessaiWD and Messai models

# quality_model = MessaiWD().to(device)
# quality_model = Messai().to(device)

# num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# min_val_loss = 99999999

# for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):

#     print(f"Fold {fold+1}/{num_folds}")

#     train_sampler = SubsetRandomSampler(train_index)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

#     val_sampler = SubsetRandomSampler(val_index)
#     val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

#     for epoch in tqdm(range(num_epochs)):

#         epoch_loss = 0.0
#         quality_model.train()

#         for left_patch, right_patch, target_score, left_target_score, right_target_score, left_image_patch, right_image_patch in train_loader:
#             left_patch, right_patch, target_score, left_target_score, right_target_score, left_image_patch, right_image_patch = left_patch.to(device), right_patch.to(device), target_score.to(device), left_target_score.to(device), right_target_score.to(device), left_image_patch.to(device), right_image_patch.to(device)
            
#             global_score, left_score, right_score, stereo_score = quality_model(left_patch, right_patch, left_image_patch, right_image_patch)
#             #global_score, left_score, right_score, stereo_score = quality_model(left_image_patch, right_image_patch) # if training Messai
#             target_score = target_score.unsqueeze(1).float()
#             left_target_score = left_target_score.unsqueeze(1).float()
#             right_target_score = right_target_score.unsqueeze(1).float()
            
#             loss_G = criterion(global_score, target_score)
#             loss_L = criterion(left_score,   left_target_score)
#             loss_R = criterion(right_score,  right_target_score)
#             loss_S = criterion(stereo_score, target_score)

#             Lamda, sigma, delta, theta = 2, 1, 1, 1
#             loss = ((Lamda * loss_G) + (sigma * loss_L) + (delta * loss_R) + (theta * loss_S))


#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss+=loss.item()

#         print(f"Epoch loss {epoch + 1}/{num_epochs}, Loss: {epoch_loss/(len(train_loader))}")


#         with torch.no_grad():
#             val_loss = 0.0
#             quality_model.eval()

#             for left_patch, right_patch, target_score, left_target_score, right_target_score, left_image_patch, right_image_patch in val_loader:
                
#                 left_patch, right_patch, target_score, left_target_score, right_target_score, left_image_patch, right_image_patch = left_patch.to(device), right_patch.to(device), target_score.to(device), left_target_score.to(device), right_target_score.to(device), left_image_patch.to(device), right_image_patch.to(device)


#                 global_score, left_score, right_score, stereo_score = quality_model(left_patch, right_patch, left_image_patch, right_image_patch)
#                 #global_score, left_score, right_score, stereo_score = quality_model(left_image_patch, right_image_patch) # for Messai
                
#                 target_score = target_score.unsqueeze(1).float()
#                 left_target_score = left_target_score.unsqueeze(1).float()
#                 right_target_score = right_target_score.unsqueeze(1).float()


#                 loss_G = criterion(global_score, target_score)
#                 loss_L = criterion(left_score,   left_target_score)
#                 loss_R = criterion(right_score,  right_target_score)
#                 loss_S = criterion(stereo_score, target_score)

#                 Lamda, sigma, delta, theta = 2, 1, 1, 1
#                 loss = ((Lamda * loss_G) + (sigma * loss_L) + (delta * loss_R) + (theta * loss_S))


#                 val_loss += loss.item()

#             print(f"Validation Loss: {val_loss/(len(val_loader))}")

#             if val_loss<min_val_loss:
#                 min_val_loss=val_loss
#                 torch.save(quality_model.state_dict(), "save-best-model.pth")
#                 print("SAVED!")

#         torch.save(quality_model.state_dict(), "save-last-model.pth")

#         print()

#     break