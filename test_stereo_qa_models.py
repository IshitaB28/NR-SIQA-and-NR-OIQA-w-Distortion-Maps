import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(64)
import torch
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr, kendalltau, pearsonr


from stereo_models import *
from load_dataset import *


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

seed = 64

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

# indices = list(range(0, 449)) #LIVE3DVR
# test_dataset = LiveVRDataset(dataset_path="path-to-LIVE3DVR-dataset", indices=indices, train=False)


print("Dataset loaded for testing: ", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=1)

quality_model = QNet_CNN().to(device)
# quality_model = QNet_Mixer().to(device)
# quality_model = ZhouWD().to(device)
# quality_model = QNet_CNNWOD().to(device)
# quality_model = QNet_MixerWOD().to(device)
# quality_model = Zhou().to(device)
quality_model.load_state_dict(torch.load("best-model-saved-path.pth")) # replace path with the best model trained on other datasets except the test_dataset
                                                                       # for cross dataset experiments

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
        # output_score = quality_model(left_patches, right_patches, left_image_patches, right_image_patches).mean() # for models without distortion maps
        
        target_score = target_score.unsqueeze(1).float()

        output_scores.append(output_score.item())
        target_scores.append(target_score.item())

    output_scores = np.array(output_scores)
    target_scores = np.array(target_scores)

# Uncomment below for Messai Models
    
# quality_model = MessaiWD().to(device)
# quality_model = Messai().to(device)
    
# with torch.no_grad():
    
#     quality_model.eval()

#     for left_patches, right_patches, target_score, left_image_patches, right_image_patches in test_loader:

#         left_patches = torch.stack(left_patches, dim=1)[0]
#         right_patches = torch.stack(right_patches, dim=1)[0]

#         left_image_patches = torch.stack(left_image_patches, dim=1)[0]
#         right_image_patches = torch.stack(right_image_patches, dim=1)[0]
        
#         left_patches, right_patches, target_score, left_image_patches, right_image_patches = left_patches.to(device), right_patches.to(device), target_score.to(device), left_image_patches.to(device), right_image_patches.to(device)

#         global_score, _, _, _ = quality_model(left_patches, right_patches, left_image_patches, right_image_patches)
#         global_score = global_score.mean()
#         target_score = target_score.unsqueeze(1).float()

#         output_scores.append(global_score.item())
#         target_scores.append(target_score.item())

#     output_scores = np.array(output_scores)
#     target_scores = np.array(target_scores)





    print("Before Logistic Fitting: ")


    p, _ = pearsonr(output_scores, target_scores)
    s, _ = spearmanr(output_scores, target_scores)
    k, _ = kendalltau(output_scores, target_scores)
    rms = np.sqrt(((output_scores - target_scores) ** 2).mean())

    print("Pearson Coefficient: ", p)
    print("Spearman Coefficient: ", s)
    print("Kendall Coefficient: ", k)
    print("RMSE: ", rms)

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
    
    rms = np.sqrt(((test_outputs_fit - target_scores) ** 2).mean())

    print("Pearson Coefficient: ", p)
    print("Spearman Coefficient: ", s)
    print("Kendall Coefficient: ", k)
    print("RMSE: ", rms)

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
    rms = np.sqrt(((test_op_curve_fit - target_scores) ** 2).mean())

    print("Pearson Coefficient: ", p)
    print("Spearman Coefficient: ", s)
    print("Kendall Coefficient: ", k)
    print("RMSE: ", rms)

    plt.figure()
    plt.scatter(test_op_curve_fit, target_scores)
    plt.xlabel("Predicted Scores", fontsize=14)
    plt.ylabel("Ground Truth", fontsize=14)
    plt.savefig("scatter-plot.png")


