import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(64)

import torch
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit, minimize
from scipy.stats import spearmanr, kendalltau, pearsonr

from omni2d_models import *
from load_dataset import CVIQD

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


indices = list(range(0, 544)) #cviqd
test_dataset = CVIQD(indices, score_path="path-to-CVIQ.mat", img_dir="path-to-CVIQ-images", train=False)


print("Dataset loaded for testing: ", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=1)


quality_model = QNet_CNN().to(device)
# quality_model = QNet_Mixer().to(device)
# quality_model = QNet_CNNWOD().to(device)
# quality_model = QNet_MixerWOD().to(device)

quality_model.load_state_dict(torch.load("best-model-save.pth"))

target_scores = []
output_scores = []

with torch.no_grad():
    
    quality_model.eval()
    
    for patches, image_patches, target_score in test_loader:

        patches = torch.stack(patches, dim = 1)[0]
        image_patches = torch.stack(image_patches, dim = 1)[0]

        patches, image_patches, target_score = patches.to(device), image_patches.to(device), target_score.to(device)
        output_score = quality_model(patches, image_patches).mean()
        # output_score = quality_model(image_patches).mean() # for image-only models

        output_scores.append(output_score.item())
        target_scores.append(target_score.item())

    output_scores = np.array(output_scores)
    target_scores = np.array(target_scores)


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
