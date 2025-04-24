import os
import os.path as osp
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
import json

# MODE = rgb, 
parser = argparse.ArgumentParser()

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

# VIDEO CENTER CROPPED AS 224, be aware 
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

def group():
    real_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, REAL_PARA_SUBDIR, "*.json")))
    cfd_para_paths = sorted(glob.glob(osp.join(DATA_ROOT, CFD_PARA_SUBDIR, "*.json")))

    # group real videos by viscosity and rpm
    real_rpm_groups = {}
    real_weight_groups = {}
    for idx, json_file in enumerate(real_para_paths):
        with open(json_file, 'r') as f:
            data = json.load(f)
        rpm = str(data['rpm'])
        weight = str(data['weight'])

        if rpm not in real_rpm_groups:
            real_rpm_groups[rpm] = []
        real_rpm_groups[rpm].append(idx)

        if weight not in real_weight_groups:
            cfd_weight_groups[weight] = []
        real_weight_groups[weight].append(idx)

    # group cfd videos by viscosity, rpm, weight
    cfd_rpm_groups = {}
    cfd_weight_groups = {}
    for idx, json_file in enumerate(cfd_para_paths):
        with open(json_file, 'r') as f:
            data = json.load(f)
        rpm = str(data['rpm'])
        weight = str(data['weight'])

        if rpm not in cfd_rpm_groups:
            cfd_rpm_groups[rpm] = []
        cfd_rpm_groups[rpm].append(idx)

        if weight not in cfd_weight_groups:
            cfd_weight_groups[weight] = []
        cfd_weight_groups[weight].append(idx)

    return real_rpm_groups,real_weight_groups, cfd_rpm_groups, cfd_weight_groups

def FVDcalculator(grouped_feature_paths):
    group_stats = {}
    for key, paths in grouped_feature_paths.items():
        all_features = []
        for path in paths:
            feat = np.load(path)
            feat = feat.reshape(-1, feat.shape[-1])  # flatten over time
            all_features.append(feat)
        all_features = np.concatenate(all_features, axis=0)

        mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features, rowvar=False)
        group_stats[key] = {'mu': mu, 'sigma': sigma}

    return group_stats



run()
