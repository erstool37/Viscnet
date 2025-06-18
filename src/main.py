import torch
import datetime
import wandb
import argparse
import os.path as osp
import glob
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import importlib
import yaml
from torch.utils.data import DataLoader
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
LR              = float(cfg["optimizer"]["lr"])
ETA_MIN         = float(cfg["optimizer"]["eta_min"])
W_DECAY         = float(cfg["optimizer"]["weight_decay"])
CKPT_ROOT       = cfg["directories"]["checkpoint"]["ckpt_root"]
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
LOSS            = cfg["loss"]
OPTIM_CLASS     = cfg["optimizer"]["optim_class"]
SCHEDULER_CLASS = cfg["optimizer"]["scheduler_class"]
PATIENCE        = int(cfg["optimizer"]["patience"])
DATA_ROOT       = cfg["directories"]["data"]["data_root"]
VIDEO_SUBDIR    = cfg["directories"]["data"]["video_subdir"]
PARA_SUBDIR     = cfg["directories"]["data"]["para_subdir"]
NORM_SUBDIR     = cfg["directories"]["data"]["norm_subdir"]
REAL_ROOT       = cfg["directories"]["data"]["real_root"]
REAL_EPOCHS     = int(cfg["real_model"]["real_epochs"])
REAL_LR         = float(cfg["real_model"]["lr"])
REAL_W_DECAY    = float(cfg["real_model"]["weight_decay"]) 

set_seed(SEED)

dataset_module = importlib.import_module(f"datasets.{DATASET}")
loss_module = importlib.import_module(f"losses.{LOSS}")
encoder_module = importlib.import_module(f"models.{ENCODER}")
flow_module = importlib.import_module(f"models.{FLOW}")
dataset_class = getattr(dataset_module, DATASET)

today = datetime.datetime.now().strftime("%m%d")
checkpoint = f"{CKPT_ROOT}{NAME}_{today}_{VER}.pth"
run_name = osp.basename(checkpoint).split(".")[0]

# LOAD DATA
wandb.init(project=PROJECT, name=run_name, reinit=True, resume="never", config= config)

video_paths = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, NORM_SUBDIR, "*.json")))
real_video_paths = sorted(glob.glob(osp.join(REAL_ROOT, VIDEO_SUBDIR, "*.mp4")))
real_para_paths = sorted(glob.glob(osp.join(REAL_ROOT, NORM_SUBDIR, "*.json")))

train_video_paths, val_video_paths = train_test_split(video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
train_para_paths, val_para_paths = train_test_split(para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
real_train_video_paths, real_val_video_paths = train_test_split(real_video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
real_train_para_paths, real_val_para_paths = train_test_split(real_para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)

train_ds = dataset_class(train_video_paths, train_para_paths, FRAME_NUM, TIME)
val_ds = dataset_class(val_video_paths, val_para_paths, FRAME_NUM, TIME)
real_train_ds = dataset_class(real_train_video_paths, real_train_para_paths, FRAME_NUM, TIME)
real_val_ds = dataset_class(real_val_video_paths, real_val_para_paths, FRAME_NUM, TIME)
inf_ds = dataset_class(real_video_paths, real_para_paths, FRAME_NUM, TIME)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
real_train_dl = DataLoader(real_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
real_val_dl = DataLoader(real_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)
inf_dl = DataLoader(inf_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=None, persistent_workers=False)

# DEFINE MODEL
encoder_class = getattr(encoder_module, ENCODER)
flow_class = getattr(flow_module, FLOW)
criterion_class = getattr(loss_module, LOSS)
optim_class = getattr(optim, OPTIM_CLASS)
scheduler_class = getattr(optim.lr_scheduler, SCHEDULER_CLASS)

encoder = encoder_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, FLOW_BOOL) #, RPM_CLASS, EMBED_SIZE, WEIGHT)
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
"""
best_val_loss = float("inf")
counter = 0
wandb.watch(encoder, criterion, log="all", log_freq=10)
for epoch in range(NUM_EPOCHS):  
    train_losses = []
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training ")  
    encoder.train()
    for frames, parameters, _ in tqdm(train_dl):
        frames, parameters = frames.to(device), parameters.to(device)
        outputs = encoder(frames)

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
    for frames, parameters, _ in tqdm(val_dl):
        frames, parameters = frames.to(device), parameters.to(device)
        outputs = encoder(frames)

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
"""
# REAL WORLD calibration

# Load the pretrained weights
checkpoint = "src/weights/decay_5s_10fps_surfdense_testrun_0414_v3.pth"
encoder.load_state_dict(torch.load(checkpoint, weights_only=True))

from utils import logzdescaler
for frames, parameters, _ in tqdm(val_dl):
    frames, parameters = frames.to(device), parameters.to(device)
    outputs = encoder(frames)
    MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "val", DATA_ROOT)
    outputs = logzdescaler(outputs.squeeze(0)[1], "dynamic_viscosity", REAL_ROOT)
    print(outputs)
    wandb.log({"outputs": outputs})

encoder.load_state_dict(torch.load(checkpoint, weights_only=True)) # Beware, cnn, lstm layers must be identical to the checkpoint
for param in encoder.cnn.parameters():
    param.requires_grad = False
for param in encoder.lstm.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(encoder.fc.parameters(), lr=REAL_LR, weight_decay=REAL_W_DECAY) 
scheduler = scheduler_class(optimizer, T_max=REAL_EPOCHS, eta_min=ETA_MIN)
criterion = criterion_class(DESCALER, REAL_ROOT)
            
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder.to(device)

# TRAINING
wandb.watch(encoder.fc, log="all", log_freq=1)
for epoch in range(REAL_EPOCHS):  
    encoder.train()
    train_losses = []
    print(f"Epoch {epoch+1}/{REAL_EPOCHS} - Training ")
    for frames, parameters, _ in tqdm(real_train_dl):
        frames, parameters = frames.to(device), parameters.to(device)
        outputs = encoder(frames)
        parameters = parameters[:, :3]

        MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "real", DATA_ROOT)
        train_loss = criterion(outputs, parameters)
        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (len(train_losses)) % 10 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"real_train_loss": mean_train_loss})
    train_losses.clear()

    # Validation
    encoder.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{REAL_EPOCHS} - Validation")

    for frames, parameters, _ in tqdm(real_val_dl):
        frames, parameters = frames.to(device), parameters.to(device)
        outputs = encoder(frames)
        parameters = parameters[:, :3]
        
        MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "real", REAL_ROOT)
        val_loss = criterion(outputs, parameters)
        val_losses.append(val_loss.item())
    mean_val_loss = mean(val_losses)
    wandb.log({"real_val_loss": mean_val_loss})

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{REAL_EPOCHS} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.8f}")
wandb.finish() 

# Save the model
real_checkpoint = checkpoint.replace(".pth", "_real.pth")
torch.save(encoder.state_dict(), real_checkpoint)