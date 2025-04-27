"""
README: 

INFO
- based on I3D model https://github.com/google-deepmind/kinetics-i3d, << download weights in this repo/data/checkpoints/rgb_imagenet/model.ckpt.data-00000-of-00001
- PYTORCH version for I3D model from https://github.com/piergiaj/pytorch-i3d/tree/master
- uses I3D model as video encoder and compares FVD metrics to tune appropriate RPM and viscWEIGHTS for CFD rendering
- Uses Center-cropped, (224, 224) videos, be aware

THOUGHTS
- quite skeptical on I3D model performance, for it is trained on Kinetics dataset(captures big human actions)
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
from src.utils.videotransforms import CenterCrop
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from src.datasets.VideoDataset_FVD import VideoDataset_FVD
import glob
import numpy as np
from I3D import InceptionI3d
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm

DATA_ROOT = 'dataset/realfluid/'
REAL_SUBDIR = 'decay_5s_10fps_hotplate'
CFD_SUBDIR = 'decay_5s_10fps_mimic_hotplate'
CHECKPOINT = 'src/weights/FVD/rgb_charades.pt'
VIDEO_SUBDIR = 'videos'
PARA_SUBDIR = 'parameters'
SAVE_SUBDIR = 'features'
FVD_SUBDIR = 'FVDresults'

FRAME_NUM = 4.9 # 4.9 currently, plz modify
TIME = 10

# runs I3D model on the videos and saves the features
def feature_extractor(max_steps=64e3, split=1.0, batch_size=2):
    test_transforms = transforms.Compose([CenterCrop(224)])

    real_video_paths = sorted(glob.glob(osp.join(DATA_ROOT, REAL_SUBDIR, VIDEO_SUBDIR, "*.mp4")))
    real_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, REAL_SUBDIR, PARA_SUBDIR, "*.json")))

    cfd_video_paths = sorted(glob.glob(osp.join(DATA_ROOT, CFD_SUBDIR, VIDEO_SUBDIR, "*.mp4")))
    cfd_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, CFD_SUBDIR, PARA_SUBDIR, "*.json")))
    
    real_ds = VideoDataset_FVD(real_video_paths, real_para_paths, FRAME_NUM, TIME)
    cfd_ds = VideoDataset_FVD(cfd_video_paths, cfd_para_paths, FRAME_NUM, TIME)

    real_dl = torch.utils.data.DataLoader(real_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    cfd_dl = torch.utils.data.DataLoader(cfd_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    dataloaders = {'real': real_dl, 'cfd': cfd_dl}
    datasets = {'real': real_ds, 'cfd': cfd_ds}

    print("Real dataset size:", len(real_ds))
    print("CFD dataset size:", len(cfd_ds))
    
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

    # make folder
    if not osp.exists(osp.join(DATA_ROOT, REAL_SUBDIR, SAVE_SUBDIR)):
            os.makedirs(osp.join(DATA_ROOT, REAL_SUBDIR, SAVE_SUBDIR))

    if not osp.exists(osp.join(DATA_ROOT, CFD_SUBDIR, SAVE_SUBDIR)):
            os.makedirs(osp.join(DATA_ROOT, CFD_SUBDIR, SAVE_SUBDIR))

    # save features for real/cfd videos
    for frames, _, names in real_dl:
        b,c,t,h,w = frames.shape
        frames = Variable(frames.cuda(), volatile=True)
        features = i3d.extract_features(frames) # (B, feature, T, H, W), approximately time 8 downsampled, H, W 32 downsampled
        
        for idx in range(len(names)):    
            np.save(osp.join(DATA_ROOT, REAL_SUBDIR, SAVE_SUBDIR, names[idx]), features[idx].squeeze(0).permute(1,2,3,0).cpu().detach().numpy()) 
    print("Real features saved to:", osp.join(DATA_ROOT, REAL_SUBDIR, SAVE_SUBDIR))
    
    for frames, _, names in cfd_dl:
        b,c,t,h,w = frames.shape
        frames = Variable(frames.cuda(), volatile=True)
        features = i3d.extract_features(frames)

        for idx in range(len(names)):
            np.save(osp.join(DATA_ROOT, CFD_SUBDIR, SAVE_SUBDIR, names[idx]), features[idx].squeeze(0).permute(1,2,3,0).cpu().detach().numpy())
    print("CFD features saved to:", osp.join(DATA_ROOT, CFD_SUBDIR, SAVE_SUBDIR))

# group the dataset based on viscosity and rpm and save grouped path as list
def group(): 
    real_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, REAL_SUBDIR, PARA_SUBDIR, "*.json")))
    cfd_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, CFD_SUBDIR, PARA_SUBDIR, "*.json")))
    real_feature_paths = sorted(glob.glob(osp.join(DATA_ROOT, REAL_SUBDIR, SAVE_SUBDIR, "*.npy")))
    cfd_feature_paths = sorted(glob.glob(osp.join(DATA_ROOT, CFD_SUBDIR, SAVE_SUBDIR, "*.npy")))

    # Match lengths 
    min_real = min(len(real_para_paths), len(real_feature_paths))
    min_cfd = min(len(cfd_para_paths), len(cfd_feature_paths))

    real_para_paths = real_para_paths[:min_real]
    real_feature_paths = real_feature_paths[:min_real]
    cfd_para_paths = cfd_para_paths[:min_cfd]
    cfd_feature_paths = cfd_feature_paths[:min_cfd]

    real_groups = {}  
    cfd_rpm_groups = {}  
    cfd_weight_groups = {}  

    for idx, json_file in enumerate(real_para_paths):
        with open(json_file, 'r') as f:
            data = json.load(f)
        visc = str(data['dynamic_viscosity'])
        rpm = str(int(data['RPM']))

        if visc not in real_groups:
            real_groups[visc] = {}
        if rpm not in real_groups[visc]:
            real_groups[visc][rpm] = []
        real_groups[visc][rpm].append(real_feature_paths[idx])

    for idx, json_file in enumerate(cfd_para_paths):
        with open(json_file, 'r') as f:
            data = json.load(f)
        visc = str(data['dynamic_viscosity'])
        rpm = str(int(data['RPM']))
        weight = str(data['magnification'])

        if visc not in cfd_rpm_groups:
            cfd_rpm_groups[visc] = {}
        if rpm not in cfd_rpm_groups[visc]:
            cfd_rpm_groups[visc][rpm] = []
        cfd_rpm_groups[visc][rpm].append(cfd_feature_paths[idx])

        if visc not in cfd_weight_groups:
            cfd_weight_groups[visc] = {}
        if weight not in cfd_weight_groups[visc]:
            cfd_weight_groups[visc][weight] = []
        cfd_weight_groups[visc][weight].append(cfd_feature_paths[idx])

    return real_groups, cfd_rpm_groups, cfd_weight_groups

### FVD calculator helper functions

# find adjacent viscosity among CFD data and link them to the real viscosity (to find the most closest weight and rpm that visualizes )
def match_viscosity(target_visc, candidate_viscs, threshold=0.5):
    target = float(target_visc)
    candidates = [float(v) for v in candidate_viscs]
    diffs = [abs(target - c) for c in candidates]
    if diffs:
        idx = diffs.index(min(diffs))
        if diffs[idx] <= threshold:
            return str(candidates[idx])
    return None

# calculate FVD metric
def calculate_fvd(real_features, cfd_features, num_iter=10):
    real_features = torch.tensor(real_features, dtype=torch.float32, device='cuda')
    cfd_features = torch.tensor(cfd_features, dtype=torch.float32, device='cuda')

    mu_real = torch.mean(real_features, dim=0)
    mu_cfd = torch.mean(cfd_features, dim=0)
    sigma_real = torch.cov(real_features.T)
    sigma_cfd = torch.cov(cfd_features.T)

    diff = mu_real - mu_cfd
    diff_norm = torch.sum(diff * diff)

    cov_product = sigma_real @ sigma_cfd

    # --- Newton–Schulz iteration for matrix square root ---
    norm = cov_product.norm()
    Y = cov_product / norm
    I = torch.eye(Y.size(0), device=Y.device)
    Z = torch.eye(Y.size(0), device=Y.device)

    for _ in range(num_iter):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    covmean = Z * torch.sqrt(norm)
    trace = torch.trace(sigma_real + sigma_cfd - 2.0 * covmean)

    return (diff_norm + trace).item()

# calculate FVD for all pairs of real and CFD data
def calculate_fvd_pairs(real_groups, cfd_groups, mode='rpm'):
    results = {}
    counter = 0

    for visc_real in real_groups:
        for rpm_real in real_groups[visc_real]:
            real_paths = real_groups[visc_real][rpm_real]
            real_features = torch.cat([torch.from_numpy(np.load(p)).reshape(-1, np.load(p).shape[-1]) for p in real_paths], dim=0)

            matched_visc = match_viscosity(visc_real, list(cfd_groups.keys()))
            if matched_visc is None or matched_visc not in cfd_groups:
                continue

            for rpm_cfd in cfd_groups[matched_visc]:
                cfd_paths = cfd_groups[matched_visc][rpm_cfd]
                cfd_features = torch.cat([torch.from_numpy(np.load(p)).reshape(-1, np.load(p).shape[-1]) for p in cfd_paths], dim=0)

                fvd_value = calculate_fvd(real_features, cfd_features)
                if mode == 'rpm':
                    results[(matched_visc, rpm_real, rpm_cfd)] = fvd_value
                else:
                    results[(matched_visc, rpm_real, rpm_cfd)] = fvd_value

                counter += 1
                if counter % 30 == 0:
                    print(f"[Info] {counter} FVD calculations done.")
    return results

# REAL FVD calculator
def FVDgrouper(real_groups, cfd_rpm_groups, cfd_weight_groups):
    fvd_rpm = calculate_fvd_pairs(real_groups, cfd_rpm_groups, mode='rpm')
    fvd_weight = calculate_fvd_pairs(real_groups, cfd_weight_groups, mode='weight')
    return fvd_rpm, fvd_weight

# prints data in terminal
def printer(fvd_data, mode):
    import collections

    data_dict = collections.defaultdict(list)

    # Organize by (viscosity, real rpm or real weight)
    for key, fvd in fvd_data.items():
        if mode == "rpm":
            visc, real_rpm, cfd_rpm = key
            data_dict[(visc, real_rpm)].append((cfd_rpm, fvd))
        elif mode == "weight":
            visc, real_weight, cfd_weight = key
            data_dict[(visc, real_weight)].append((cfd_weight, fvd))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Sort and print
    for (visc, real_param) in sorted(data_dict.keys(), key=lambda x: (float(x[0]), float(x[1]))):
        entries = data_dict[(visc, real_param)]
        entries = sorted(entries, key=lambda x: x[1])  # Sort by FVD ascending

        print(f"\n=== Real Viscosity {visc}, Real {mode.upper()} {real_param} ===")
        for cfd_param, fvd_value in entries:
            print(f"CFD {mode.upper()}: {cfd_param}, FVD: {fvd_value:.4f}")

# draws heat map
def visualize(fvd_rpm, fvd_weight):
    # --- Plot RPM comparison ---
    if len(fvd_rpm) > 0:
        print("\nPlotting FVD vs RPM...")
        rpm_data = []
        for (visc, real_rpm, cfd_rpm), fvd_value in fvd_rpm.items():
            rpm_data.append((f"{visc} (Real {real_rpm})", int(cfd_rpm), fvd_value))

        rpm_df = sns.load_dataset('tips')[:0]  # dummy empty DataFrame
        import pandas as pd
        rpm_df = pd.DataFrame(rpm_data, columns=["Viscosity_RealRPM", "CFD_RPM", "FVD"])

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=rpm_df, x="CFD_RPM", y="FVD", hue="Viscosity_RealRPM", style="Viscosity_RealRPM", s=100)
        plt.title("FVD vs CFD RPM for Different Real Viscosities")
        plt.xlabel("CFD RPM")
        plt.ylabel("FVD")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- Plot Weight comparison ---
    if len(fvd_weight) > 0:
        print("\nPlotting FVD vs Weight...")
        weight_data = []
        for (visc, real_weight, cfd_weight), fvd_value in fvd_weight.items():
            weight_data.append((f"{visc} (Real {real_weight})", float(cfd_weight), fvd_value))

        weight_df = sns.load_dataset('tips')[:0]
        import pandas as pd
        weight_df = pd.DataFrame(weight_data, columns=["Viscosity_RealWeight", "CFD_Weight", "FVD"])

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=weight_df, x="CFD_Weight", y="FVD", hue="Viscosity_RealWeight", style="Viscosity_RealWeight", s=100)
        plt.title("FVD vs CFD Weight for Different Real Viscosities")
        plt.xlabel("CFD Weight")
        plt.ylabel("FVD")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

##### CALL FUNCTIONS #####
# feature_extractor() # save encoded features
real_groups, cfd_rpm_groups, cfd_weight_groups = group()
fvd_rpm, fvd_weight = FVDgrouper(real_groups, cfd_rpm_groups, cfd_weight_groups)
printer(fvd_rpm, 'rpm')
printer(fvd_weight, 'weight')
# visualize(fvd_rpm, fvd_weight)