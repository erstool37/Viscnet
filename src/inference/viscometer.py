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
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config0.yaml")
parser.add_argument("-m", "--method", type=str, required=True)
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
WEIGHT          = float(cfg["model"]["encoder"]["embed_weight"])
FLOW            = cfg["model"]["flow"]["flow"]
FLOW_BOOL       = cfg["model"]["flow"]["flow_bool"]
DIM             = int(cfg["model"]["flow"]["dim"])
CON_DIM         = int(cfg["model"]["flow"]["con_dim"])
HIDDEN_DIM      = int(cfg["model"]["flow"]["hidden_dim"])
NUM_LAYERS      = int(cfg["model"]["flow"]["num_layers"])
CHECKPOINT      = cfg["directories"]["checkpoint"]["inf_checkpoint"]
RPM_CLASS       = int(cfg["preprocess"]["rpm_class"])
repo = cfg["directories"]["data"]
VAL_ROOT       = repo["data_root"]
REAL_ROOT       = repo["real_root"]
TEST_ROOT       = repo["test_root"]
VIDEO_SUBDIR    = repo["video_subdir"]
PARA_SUBDIR     = repo["para_subdir"]
NORM_SUBDIR     = repo["norm_subdir"]

wandb.init(project="viscosity estimation testing", name="inferencev0", reinit=True, resume="never", config= config)

# model load
dataset_module = importlib.import_module(f"datasets.{DATASET}")
encoder_module = importlib.import_module(f"models.{ENCODER}")
flow_module = importlib.import_module(f"models.{FLOW}")

dataset_class = getattr(dataset_module, DATASET)
encoder_class = getattr(encoder_module, ENCODER)
flow_class = getattr(flow_module, FLOW)

encoder = encoder_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, FLOW_BOOL, RPM_CLASS, EMBED_SIZE, WEIGHT)
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
errors = []
for frames, parameters, _, rpm in dl:
    frames, parameters, rpm = frames.to(device), parameters.to(device), rpm.to(device)
    outputs = encoder(frames, rpm)
    
    if FLOW_BOOL:
        z, log_det_jacobian = flow(parameters, outputs)
        visc = flow.inverse(z, outputs)
        error = MAPEtestcalculator(visc.detach(), parameters.detach(), DESCALER, METHOD, repo[f"{METHOD}_root"])
    else:
        wandb.log({"xsph estimation": outputs.detach().cpu()}) 
        error = MAPEtestcalculator(outputs.detach(), parameters.detach(), DESCALER, "real", REAL_ROOT)
    errors.append(error.detach().cpu())

errors_tensor = torch.cat(errors, dim=0)
meanerror = errors_tensor.mean(dim=0)  # shape: [3]

distribution(data=errors_tensor[:,0], ref = 0, save_path='src/inference/error/dist_den.png')
distribution(data=errors_tensor[:,1], ref = 0, save_path='src/inference/error/dist_visco.png')
distribution(data=errors_tensor[:,2], ref = 0, save_path='src/inference/error/dist_surf.png')

print(f"density MAPE: {float(meanerror[0]):.2f}%")
print(f"dynamic viscosity MAPE: {float(meanerror[1]):.2f}%")
print(f"surface tension MAPE: {float(meanerror[2]):.2f}%")

# Regression Validation test
"""
unnorm_outputs_list = []
unnorm_para_list = []

for frames, parameters in test_dl:
    frames, parameters = frames.to(device), parameters.to(device)
    outputs = visc_model(frames)
    
    unnorm_outputs = torch.stack([zdescaler(outputs[:, 0], 'density'), zdescaler(outputs[:, 1], 'dynamic_viscosity'), zdescaler(outputs[:, 2], 'surface_tension')], dim=1)  
    unnorm_para = torch.stack([parameters[:, 0], parameters[:, 1], parameters[:, 2]], dim=1)
    
    unnorm_outputs_list.append(unnorm_outputs.detach().cpu()) 
    unnorm_para_list.append(unnorm_para.detach().cpu())

unnorm_outputs_list = torch.cat(unnorm_outputs_list, dim=0)
unnorm_para_list = torch.cat(unnorm_para_list, dim=0)

groups = defaultdict(list)
for idx, item in enumerate(unnorm_para_list):
    key = item[0].item()
    groups[key].append(idx)
grouped_indices = list(groups.values())

grouped_outputs_list = [unnorm_outputs_list[idx] for idx in grouped_indices]
grouped_para_list = [unnorm_para_list[idx] for idx in grouped_indices]

for idx in range(len(grouped_outputs_list)):
    distribution(grouped_outputs_list[idx][:,0], ref = grouped_para_list[idx][0,0].cpu(), save_path=f'test/precision/dist{(idx+1):02d}_den.png')
    distribution(grouped_outputs_list[idx][:,1], ref = grouped_para_list[idx][0,1].cpu(), save_path=f'test/precision/dist{(idx+1):02d}_visco.png')
    distribution(grouped_outputs_list[idx][:,2], ref = grouped_para_list[idx][0,2].cpu(), save_path=f'test/precision/dist{(idx+1):02d}_surf.png')
"""