import torch
import torch.nn as nn
from torchvision import models
import datetime
import cv2
import wandb
import argparse
import numpy as np
import os.path as osp
import glob
import sys
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import importlib
import yaml
import json
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from utils import MAPEcalculator, MAPEflowcalculator, MAPEtestcalculator, set_seed, distribution

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
parser.add_argument("-m", "--method", type=str, required=False, default="pca")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
cfg = config["regression"]

METHOD          = args.method
SCALER          = cfg["preprocess"]["scaler"]
DESCALER        = cfg["preprocess"]["descaler"]
TEST_SIZE       = float(cfg["preprocess"]["test_size"])
RAND_STATE      = int(cfg["preprocess"]["random_state"])
FRAME_NUM       = int(cfg["preprocess"]["frame_num"])
TIME            = int(cfg["preprocess"]["time"])
BATCH_SIZE      = int(cfg["train_settings"]["batch_size"])
NUM_WORKERS     = int(cfg["train_settings"]["num_workers"])
NUM_EPOCHS      = int(cfg["train_settings"]["num_epochs"])
SEED            = int(cfg["train_settings"]["seed"])
DATASET         = cfg["train_settings"]["dataset"]
ENCODER         = cfg["model"]["encoder"]["encoder"]
CNN             = cfg["model"]["encoder"]["cnn"]
CNN_TRAIN       = cfg["model"]["encoder"]["cnn_train"]
LSTM_SIZE       = int(cfg["model"]["encoder"]["lstm_size"])
LSTM_LAYERS     = int(cfg["model"]["encoder"]["lstm_layers"])
OUTPUT_SIZE     = int(cfg["model"]["encoder"]["output_size"])
DROP_RATE       = float(cfg["model"]["encoder"]["drop_rate"])
EMBED_SIZE      = int(cfg["model"]["encoder"]["embedding_size"])
FLOW            = cfg["model"]["flow"]["flow"]
FLOW_BOOL       = cfg["model"]["flow"]["flow_bool"]
DIM             = int(cfg["model"]["flow"]["dim"])
CON_DIM         = int(cfg["model"]["flow"]["con_dim"])
HIDDEN_DIM      = int(cfg["model"]["flow"]["hidden_dim"])
NUM_LAYERS      = int(cfg["model"]["flow"]["num_layers"])
CHECKPOINT      = cfg["directories"]["checkpoint"]["inf_checkpoint"]
RPM_CLASS       = int(cfg["preprocess"]["rpm_class"])
repo = cfg["directories"]["data"]
VAL_ROOT        = repo["data_root"]
REAL_ROOT       = repo["real_root"]
TEST_ROOT       = repo["test_root"]
VIDEO_SUBDIR    = repo["video_subdir"]
PARA_SUBDIR     = repo["para_subdir"]
NORM_SUBDIR     = repo["norm_subdir"]

wandb.init(project="viscosity estimation testing", name="inference", reinit=True, resume="never", config= config)

# model load
dataset_module = importlib.import_module(f"datasets.{DATASET}")
encoder_module = importlib.import_module(f"models.{ENCODER}")
flow_module = importlib.import_module(f"models.{FLOW}")

dataset_class = getattr(dataset_module, DATASET)
encoder_class = getattr(encoder_module, ENCODER)
flow_class = getattr(flow_module, FLOW)

encoder = encoder_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, FLOW_BOOL, RPM_CLASS, EMBED_SIZE)
flow = flow_class(DIM, CON_DIM, HIDDEN_DIM, NUM_LAYERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.cuda()
encoder.eval()
encoder.load_state_dict(torch.load(CHECKPOINT))

# Dataset load
if METHOD == "real":
    video_paths = sorted(glob.glob(osp.join(REAL_ROOT, VIDEO_SUBDIR, "*.mp4")))
    para_paths = sorted(glob.glob(osp.join(REAL_ROOT, NORM_SUBDIR, "*.json")))
elif METHOD == "test":

    video_paths = sorted(glob.glob(osp.join(TEST_ROOT, VIDEO_SUBDIR, "*.mp4")))
    para_paths = sorted(glob.glob(osp.join(TEST_ROOT, NORM_SUBDIR, "*.json")))
else:
    val_video_paths = sorted(glob.glob(osp.join(VAL_ROOT, VIDEO_SUBDIR, "*.mp4")))
    val_para_paths = sorted(glob.glob(osp.join(VAL_ROOT, NORM_SUBDIR, "*.json")))
    _, video_paths = train_test_split(val_video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
    _, para_paths = train_test_split(val_para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)

ds = dataset_class(video_paths, para_paths, FRAME_NUM, TIME)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)

# Error Calculation
all_latents, all_visc, all_rpms = [], [], []
with torch.no_grad():
    for frames, parameters, _, rpm_class in dl:
        frames, parameters, rpm_class = frames.to(device), parameters.to(device), rpm_class.to(device)
        outputs = encoder(frames, rpm_class)
        all_latents.append(outputs.squeeze(0).cpu())               
        all_visc.append(parameters[:, 1].item())                 
        all_rpms.append(rpm_class.item())                            

latents = torch.stack(all_latents).numpy()                   
viscs = np.array(all_visc)                                  
rpms = np.array(all_rpms)                                

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
pca = PCA(n_components=2)
latents_pca = pca.fit_transform(latents)  # shape: [N, 2]

# Plot by viscosity
plt.figure(figsize=(8, 6))
plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=viscs, cmap='viridis')
plt.colorbar(label='Viscosity')
plt.title("PCA: Latent Features Colored by Viscosity")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("src/inference/PCA/pca_by_viscosity.jpg", format='jpg')
plt.close()

# Plot by RPM
plt.figure(figsize=(8, 6))
plt.scatter(latents_pca[:, 0], latents_pca[:, 1], c=rpms, cmap='plasma')
plt.colorbar(label='RPM Class')
plt.title("PCA: Latent Features Colored by RPM")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("src/inference/PCA/pca_by_rpm.jpg", format='jpg')
plt.close()