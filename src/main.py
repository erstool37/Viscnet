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
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import importlib
import yaml
import json
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from utils.utils import MAPEcalculator, MAPEflowcalculator
from utils.setseed import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
cfg = config["regression"]

NAME            = config["name"]
PROJECT         = config["project"]
VER             = config["version"]
SCALER          = cfg["preprocess"]["scaler"]
DESCALER        = cfg["preprocess"]["descaler"]
TEST_SIZE       = float(cfg["preprocess"]["test_size"])
RAND_STATE      = int(cfg["preprocess"]["random_state"])
FRAME_NUM       = int(cfg["preprocess"]["frame_num"])
TIME            = int(cfg["preprocess"]["time"])
RPM_CLASS       = int(cfg["preprocess"]["rpm_class"])
BATCH_SIZE      = int(cfg["train_settings"]["batch_size"])
NUM_WORKERS     = int(cfg["train_settings"]["num_workers"])
NUM_EPOCHS      = int(cfg["train_settings"]["num_epochs"])
SEED            = int(cfg["train_settings"]["seed"])
DATASET         = cfg["train_settings"]["dataset"]
REAL_NUM_EPOCHS = int(cfg["real_model"]["real_num_epochs"])
LR              = float(cfg["optimizer"]["lr"])
ETA_MIN         = float(cfg["optimizer"]["eta_min"])
W_DECAY         = float(cfg["optimizer"]["weight_decay"])
MASK_CHECKPOINT = cfg["directories"]["checkpoint"]["mask_checkpoint"]
CHECKPOINT      = cfg["directories"]["checkpoint"]["checkpoint"]
REAL_CHECKPOINT = cfg["directories"]["checkpoint"]["real_checkpoint"]
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
LOSS            = cfg["loss"]
OPTIM_CLASS     = cfg["optimizer"]["optim_class"]
SCHEDULER_CLASS = cfg["optimizer"]["scheduler_class"]
PATIENCE        = int(cfg["optimizer"]["patience"])
DATA_ROOT       = cfg["directories"]["data"]["data_root"]
VIDEO_SUBDIR    = cfg["directories"]["data"]["video_subdir"]
PARA_SUBDIR     = cfg["directories"]["data"]["para_subdir"]
NORM_SUBDIR     = cfg["directories"]["data"]["norm_subdir"]
REAL_ROOT       = cfg["directories"]["data"]["real_root"]

set_seed(SEED)

dataset_module = importlib.import_module(f"datasets.{DATASET}")
loss_module = importlib.import_module(f"losses.{LOSS}")
encoder_module = importlib.import_module(f"models.{ENCODER}")
flow_module = importlib.import_module(f"models.{FLOW}")
dataset_class = getattr(dataset_module, DATASET)

today = datetime.datetime.now().strftime("%m%d")
checkpoint = f"{CHECKPOINT}{NAME}_{today}_{VER}.pth"
run_name = osp.basename(checkpoint).split(".")[0]

# LOAD DATA
wandb.init(project=PROJECT, name=run_name, reinit=True, resume="never", config= config)

video_paths = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, NORM_SUBDIR, "*.json")))

train_video_paths, val_video_paths = train_test_split(video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
train_para_paths, val_para_paths = train_test_split(para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)

train_ds = dataset_class(train_video_paths, train_para_paths, FRAME_NUM, TIME)
val_ds = dataset_class(val_video_paths, val_para_paths, FRAME_NUM, TIME)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)

# DEFINE MODEL
encoder_class = getattr(encoder_module, ENCODER)
flow_class = getattr(flow_module, FLOW)
criterion_class = getattr(loss_module, LOSS)
optim_class = getattr(optim, OPTIM_CLASS)
scheduler_class = getattr(optim.lr_scheduler, SCHEDULER_CLASS)

encoder = encoder_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, FLOW_BOOL, RPM_CLASS, EMBED_SIZE)
flow = flow_class(DIM, CON_DIM, HIDDEN_DIM, NUM_LAYERS)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder.to(device)
flow.to(device)
criterion = criterion_class(DESCALER, DATA_ROOT)

if FLOW_BOOL:
    optimizer = optim_class(list(encoder.parameters()) + list(flow.parameters()), lr=LR, weight_decay=W_DECAY)
else:
    optimizer = optim_class(encoder.parameters(), lr=LR, weight_decay=W_DECAY)
scheduler = scheduler_class(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

# TRAIN MODEL
best_val_loss = float("inf")
counter = 0
wandb.watch(encoder, criterion, log="all", log_freq=10)

for epoch in range(NUM_EPOCHS):  
    train_losses = []
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training ")  
    encoder.train()
    for frames, parameters, _, rpm_class in tqdm(train_dl):
        frames, parameters, rpm_class = frames.to(device), parameters.to(device), rpm_class.to(device)
        outputs = encoder(frames, rpm_class)

        if FLOW_BOOL:
            z, log_det_jacobian = flow(parameters, outputs) # para=4, outputs=512
            train_loss = criterion(z, log_det_jacobian)
            visc = flow.inverse(z, outputs)
            MAPEflowcalculator(visc.detach(), parameters.detach(), DESCALER, "train", DATA_ROOT)
        else:
            train_loss = criterion(outputs, parameters)
            MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "train", DATA_ROOT)
        
        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (len(train_losses)) % 10 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"train_loss": mean_train_loss})
    train_losses.clear()

    # VALIDATION
    encoder.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
    for frames, parameters, _, rpm_class in tqdm(val_dl):
        frames, parameters, rpm_class = frames.to(device), parameters.to(device), rpm_class.to(device)
        outputs = encoder(frames, rpm_class)

        if FLOW_BOOL:
            z, log_det_jacobian = flow(parameters, outputs)
            val_loss = criterion(z, log_det_jacobian)
            visc = flow.inverse(z, outputs)
            MAPEflowcalculator(visc.detach(), parameters.detach(), DESCALER, "val", DATA_ROOT)
        else:
            val_loss = criterion(outputs, parameters)
            MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "val", DATA_ROOT)
        val_losses.append(val_loss.item())

    mean_val_loss = mean(val_losses)
    val_losses.clear()
    wandb.log({"val_loss": mean_val_loss})

    # PATIENCE
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.7f}")
    val_losses.clear()
wandb.finish()
torch.save(encoder.state_dict(), checkpoint)

# REAL WORLD DATA TRAINING
"""
# train/val dataset split
real_video_paths = sorted(glob.glob(osp.join(REAL_SAVE_ROOT, "*.mp4")))
real_train_paths, real_val_paths = train_test_split(video_paths, test_size=0.2, random_state=42)

real_train_ds = VideoDataset(real_train_paths, frame_limit=FRAME_NUM*TIME)
real_val_ds = VideoDataset(real_val_paths, frame_limit=FRAME_NUM*TIME)

# Load data
real_train_dl = DataLoader(real_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
real_val_dl = DataLoader(real_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Freeze CNN and LSTM layers
encoder.load_state_dict(torch.load(CHECKPOINT))

for param in encoder.cnn.parameters():
    param.requires_grad = False
for param in encoder.lstm.parameters():
    param.requires_grad = False

# Fine Tuning loop definition
encoder.fc = nn.Linear(encoder.fc.in_features, 1).to(device)
optimizer = torch.optim.Adam(encoder.fc.parameters(), lr=1e-4) #train only fc layer
criterion = nn.MSELoss()

# Fine Tuning Loop
num_epochs = REAL_NUM_EPOCHS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoder.to(device)
encoder.train()
for epoch in range(num_epochs):  
    train_losses = []
    print(f"Epoch {epoch+1}/{num_epochs} - Training ")
    for frames, parameters in real_train_dl:
        frames, parameters = frames.to(device), parameters.to(device)
        
        outputs = encoder(frames)

        #loss calculation/optimization
        train_loss = reg_loss(outputs, parameters)
        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # loss print
        if (len(train_losses)) % 50 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"real_train_loss": mean_train_loss})
    train_losses.clear()

    # Validation loss calculation
    encoder.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{num_epochs} - Validation")

    for frames, parameters in real_val_dl:
        frames, parameters = frames.to(device), parameters.to(device)
        outputs = encoder(frames)

        val_loss = reg_loss(outputs, parameters)
        val_losses.append(val_loss.item())
    mean_val_loss = mean(val_losses)
    wandb.log({"real_val_loss": mean_val_loss})

    scheduler.step()
    current_lr = optimizer.get_last_lr[0]
    print(f"Epoch {epoch+1}/{num_epochs} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.5f}")
wandb.finish() 


# Save the model
torch.save(encoder.state_dict(), REAL_CHECKPOINT)

"""