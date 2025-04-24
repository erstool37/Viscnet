"""
README: 

INFO
- based on I3D model https://github.com/google-deepmind/kinetics-i3d, << download weights in this repo/data/checkpoints
- PYTORCH version for I3D model from https://github.com/piergiaj/pytorch-i3d/tree/master
- uses I3D model as video encoder and compares FVD metrics to tune appropriate RPM and viscWEIGHTS for CFD rendering
- Uses Center-cropped 224, 224 videos, be aware

THOUGHTS
- quite skeptical on I3D model performance, for it is trained on Kinetics dataset(captures big human actions)
- code not complete, requires debugging after dataset is ready
"""


import os
import os.path as osp
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import utils.videotransforms as videotransforms
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from datasets.VideoDataset import VideoDataset
import glob
import numpy as np
from I3D import InceptionI3d

DATA_ROOT = 'dataset/realfluid/'
REAL_VIDEO_SUBDIR = 'decay_5s_10fps_impeller/videos'
REAL_PARA_SUBDIR = 'decay_5s_10fps_impeller/parameters'
REAL_SAVE_SUBDIR = 'decay_5s_10fps_impeller/features' # feature save path
CFD_VIDEO_SUBDIR = 'CFDtuning/videos'
CFD_PARA_SUBDIR = 'CFDtuning/parameters'
CFD_SAVE_SUBDIR = 'CFDtuning/features' # feature save path
CHECKPOINT = 'src/weights/RGB'
FVD_SAVE_SUBDIR = 'FVDresults'

FRAME_NUM = 5
TIME = 10

# runs I3D model on the videos and saves the features
def run(max_steps=64e3, split=1.0, batch_size=1):
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    real_video_paths = sorted(glob.glob(osp.join(DATA_ROOT, REAL_VIDEO_SUBDIR, "*.mp4")))
    real_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, REAL_PARA_SUBDIR, "*.json")))

    cfd_video_paths = sorted(glob.glob(osp.join(DATA_ROOT, CFD_VIDEO_SUBDIR, "*.mp4")))
    cfd_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, CFD_PARA_SUBDIR, "*.json")))
    
    real_ds = VideoDataset(real_video_paths, real_para_paths, FRAME_NUM, TIME)
    cfd_ds = VideoDataset(cfd_video_paths, cfd_para_paths, FRAME_NUM, TIME)

    real_dl = torch.utils.data.DataLoader(real_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    cfd_dl = torch.utils.data.DataLoader(cfd_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)    

    dataloaders = {'real': real_dl, 'cfd': cfd_dl}
    datasets = {'real': real_ds, 'cfd': cfd_ds}
    
    # SETUP MODEL
    # if mode == 'flow':
    #     i3d = InceptionI3d(400, in_channels=2)
    # else:

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157) 
    i3d.load_state_dict(torch.load(CHECKPOINT))
    i3d.cuda()
    i3d.train(False)
            
    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0
                    
    for frames, _, names in real_dl:
        b,c,t,h,w = frames.shape
        frames = Variable(frames.cuda(), volatile=True)
        features = i3d.extract_features(frames)
        np.save(osp.join(DATA_ROOT, REAL_SAVE_SUBDIR, names), features.squeeze(0).permute(1,2,3,0).cpu().numpy())
    
    for frames, _, names in cfd_dl:
        b,c,t,h,w = frames.shape
        frames = Variable(frames.cuda(), volatile=True)
        features = i3d.extract_features(frames)
        np.save(osp.join(DATA_ROOT, CFD_SAVE_SUBDIR, names), features.squeeze(0).permute(1,2,3,0).cpu().numpy())

# groups the dataset based on viscosity and rpm, base on the file path lists' index
def group():
    real_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, REAL_PARA_SUBDIR, "*.json")))
    cfd_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, CFD_PARA_SUBDIR, "*.json")))

    # grouping real videos based on viscosity and rpm
    real_groups = {}
    for idx, json_file in enumerate(real_para_paths):
        with open(json_file, 'r') as f:
            data = json.load(f)
        visc = str(data['dynamic_viscosity'])
        rpm = str(data['rpm'])
        if visc not in real_groups:
            real_visc_rpm_groups[visc] = {}
        if rpm not in real_visc_rpm_groups[visc]:
            real_visc_rpm_groups[visc][rpm] = []
        real_visc_rpm_groups[visc][rpm].append(idx)

    # grouping cfd videos based on viscosity and rpm and weight
    cfd_rpm_groups = {}
    cfd_weight_groups = {}
    for idx, json_file in enumerate(cfd_para_paths):
        with open(json_file, 'r') as f:
            data = json.load(f)
        visc = str(data['dynamic_viscosity'])
        rpm = str(data['rpm'])
        weight = str(data['weight'])

        # viscosity > rpm > list
        if visc not in cfd_rpm_groups:
            cfd_visc_rpm_groups[visc] = {}
        if rpm not in cfd_rpm_groups[visc]:
            cfd_rpm_groups[visc][rpm] = []
        cfd_rpm_groups[visc][rpm].append(idx)

        # viscosity → weight > list
        if visc not in cfd_weight_groups:
            cfd_weight_groups[visc] = {}
        if weight not in cfd_weight_groups[visc]:
            cfd_weight_groups[visc][weight] = []
        cfd_weight_groups[visc][weight].append(idx)

    return real_groups, cfd_rpm_groups, cfd_weight_groups

# simple FVD calculator
def FVDcalculator(mu1, sigma1, mu2, sigma2):
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

# FVD calculator for groups
def FVDgrouper(real_groups, cfd_rpm_groups, cfd_weight_groups):
    fvd_rpm = {}
    fvd_weight = {}

    for visc in real_groups:
        # calculate FVD for RPM
        for rpm in real_groups[visc]:
            real_features = np.concatenate([np.load(p).reshape(-1, np.load(p).shape[-1]) for p in real_groups[visc][rpm]], axis=0)
            mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)

            if visc in cfd_rpm_groups and rpm in cfd_rpm_groups[visc]:
                cfd_features = np.concatenate([np.load(p).reshape(-1, np.load(p).shape[-1]) for p in cfd_rpm_groups[visc][rpm]], axis=0)
                mu_cfd, sigma_cfd = np.mean(cfd_features, axis=0), np.cov(cfd_features, rowvar=False)
                fvd_rpm[(visc, rpm)] = FVDcalculator(mu_real, sigma_real, mu_cfd, sigma_cfd)

        # calculate FVD for weight
        all_real_features = []
        for rpm in real_groups[visc]:
            all_real_features.extend([np.load(p).reshape(-1, np.load(p).shape[-1]) for p in real_groups[visc][rpm]])
        all_real_features = np.concatenate(all_real_features, axis=0)
        mu_real_all, sigma_real_all = np.mean(all_real_features, axis=0), np.cov(all_real_features, rowvar=False)

        if visc in cfd_weight_groups:
            for weight in cfd_weight_groups[visc]:
                cfd_features = np.concatenate([np.load(p).reshape(-1, np.load(p).shape[-1]) for p in cfd_weight_groups[visc][weight]], axis=0)
                mu_cfd, sigma_cfd = np.mean(cfd_features, axis=0), np.cov(cfd_features, rowvar=False)
                fvd_weight[(visc, weight)] = FVDcalculator(mu_real_all, sigma_real_all, mu_cfd, sigma_cfd)

    return fvd_rpm, fvd_weight

# Visualize as heatmap
def visualize(fvd_rpm, fvd_weight):
    # Convert fvd_rpm to matrix form for heatmap
    rpm_visc = sorted(set([k[0] for k in fvd_rpm]))
    rpm_vals = sorted(set([k[1] for k in fvd_rpm]))
    rpm_matrix = np.zeros((len(rpm_visc), len(rpm_vals)))

    for i, visc in enumerate(rpm_visc):
        for j, rpm in enumerate(rpm_vals):
            rpm_matrix[i, j] = fvd_rpm.get((visc, rpm), np.nan)

    plt.figure(figsize=(10, 6))
    sns.heatmap(rpm_matrix, annot=True, xticklabels=rpm_vals, yticklabels=rpm_visc, cmap='coolwarm')
    plt.title("FVD: (Viscosity, RPM)")
    plt.xlabel("RPM")
    plt.ylabel("Viscosity")
    plt.tight_layout()
    plt.show()

    # Convert fvd_weight to matrix form for heatmap
    weight_visc = sorted(set([k[0] for k in fvd_weight]))
    weight_vals = sorted(set([k[1] for k in fvd_weight]))
    weight_matrix = np.zeros((len(weight_visc), len(weight_vals)))

    for i, visc in enumerate(weight_visc):
        for j, weight in enumerate(weight_vals):
            weight_matrix[i, j] = fvd_weight.get((visc, weight), np.nan)

    plt.figure(figsize=(10, 6))
    sns.heatmap(weight_matrix, annot=True, xticklabels=weight_vals, yticklabels=weight_visc, cmap='YlGnBu')
    plt.title("FVD: (Viscosity, Weight)")
    plt.xlabel("Weight")
    plt.ylabel("Viscosity")
    plt.tight_layout()
    plt.show()

# Initialize.
run()
visualize(FVDgrouper(group()))