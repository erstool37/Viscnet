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
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from utils import MAPEcalculator, sanity_check_alignment, set_seed, ddp_setup, ddp_cleanup, confusion_matrix, new_plot_error_distribution, viz_attention, energyCalculator, csv_export, plot_error_distribution, reliability_diagram
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning) # filter out warnings from transformer on classificatio head initialzation which is irrelevant

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

PROJECT         = config["project"]
NAME            = config["name"]
VER             = config["version"]
# Basic settings
NUM_WORKERS     = int(config["train_settings"]["num_workers"])
SEED            = int(config["train_settings"]["seed"])
WATCH_BOOL      = bool(config["train_settings"]["watch_bool"])
CLASS_BOOL      = bool(config["train_settings"]["classification"])
TRAIN_BOOL      = bool(config["train_settings"]["train_bool"])
TEST_BOOL       = bool(config["train_settings"]["test_bool"])
ATTN_BOOL       = bool(config["train_settings"]["attn_bool"])
SAN_BOOL        = bool(config["train_settings"]["sanity_check_bool"])
# Dataset and Dataloader
SCALER          = config["dataset"]["preprocess"]["scaler"]
DESCALER        = config["dataset"]["preprocess"]["descaler"]
### For Train
DATA_ROOT_TRAIN = config["dataset"]["train"]["train_root"]
FRAME_NUM       = float(config["dataset"]["train"]["frame_num"])
TIME            = float(config["dataset"]["train"]["time"])
RPM_CLASS       = int(config["dataset"]["train"]["rpm_class"])
AUG_BOOL        = bool(config["dataset"]["train"]["dataloader"]["aug_bool"])
BATCH_SIZE      = int(config["dataset"]["train"]["dataloader"]["batch_size"])
TEST_SIZE       = float(config["dataset"]["train"]["dataloader"]["test_size"])
RAND_STATE      = int(config["dataset"]["train"]["dataloader"]["random_state"])
DATASET         = config["dataset"]["train"]["dataloader"]["dataloader"]
### For Test
DATA_ROOT_TEST  = config["dataset"]["test"]["test_root"]
FRAME_NUM_TEST    = float(config["dataset"]["test"]["frame_num"])
TIME_TEST         = float(config["dataset"]["test"]["time"])
RPM_CLASS_TEST    = int(config["dataset"]["test"]["rpm_class"])
AUG_BOOL_TEST     = bool(config["dataset"]["test"]["dataloader"]["aug_bool"])
BATCH_SIZE_TEST   = int(config["dataset"]["test"]["dataloader"]["batch_size"])
TEST_SIZE_TEST    = float(config["dataset"]["test"]["dataloader"]["test_size"])
RAND_STATE_TEST   = int(config["dataset"]["test"]["dataloader"]["random_state"])
DATASET_TEST      = config["dataset"]["test"]["dataloader"]["dataloader"]
# Model Settings
TRANS_BOOL      = config["model"]["transformer_bool"]
ENCODER         = config["model"]["transformer"]["encoder"]
VISC_CLASS      = config["model"]["transformer"]["class"]
CNN_TRAIN       = bool(config["model"]["cnn"]["cnn_train"])
CNN             = config["model"]["cnn"]["cnn"]
LSTM_SIZE       = int(config["model"]["cnn"]["lstm_size"])
LSTM_LAYERS     = int(config["model"]["cnn"]["lstm_layers"])
OUTPUT_SIZE     = int(config["model"]["cnn"]["output_size"])
DROP_RATE       = float(config["model"]["cnn"]["drop_rate"])
EMBED_SIZE      = int(config["model"]["cnn"]["embedding_size"])
WEIGHT          = float(config["model"]["cnn"]["embed_weight"])
# Train Settings
VAL_TEST_BOOL   = bool(config["train_settings"]["val_test_bool"])
CURR_BOOL       = int(config["training"]["curr_bool"])
CURR_CKPT       = config["training"]["curr_ckpt"]
NUM_EPOCHS      = int(config["training"]["num_epochs"])
LOSS            = config["training"]["loss"]
SMTH_LABEL      = float(config["training"]["label_smoothing"])
OPTIM_CLASS     = config["training"]["optimizer"]["optim_class"]
SCHEDULER_CLASS = config["training"]["optimizer"]["scheduler_class"]
LR              = float(config["training"]["optimizer"]["lr"])
ETA_MIN         = float(config["training"]["optimizer"]["eta_min"])
W_DECAY         = float(config["training"]["optimizer"]["weight_decay"])
PATIENCE        = int(config["training"]["optimizer"]["patience"])
# MISC Settings
CKPT_ROOT       = config["misc_dir"]["ckpt_root"]
VIDEO_SUBDIR    = config["misc_dir"]["video_subdir"]
PARA_SUBDIR     = config["misc_dir"]["para_subdir"]
NORM_SUBDIR     = config["misc_dir"]["norm_subdir"]

# DDP SETUP
rank, world_size, local_rank = ddp_setup()
set_seed(SEED+rank)
print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

# Set name
today = datetime.datetime.now().strftime("%m%d")
checkpoint = f"{CKPT_ROOT}{NAME}_{today}_{VER}.pth"
run_name = osp.basename(checkpoint).split(".")[0]

# Definition
dataset_module = importlib.import_module(f"datasets.{DATASET}")
dataset_class = getattr(dataset_module, DATASET)
criterion_module = importlib.import_module(f"losses.{LOSS}")
criterion_class = getattr(criterion_module, LOSS)
optim_class = getattr(optim, OPTIM_CLASS)
scheduler_class = getattr(optim.lr_scheduler, SCHEDULER_CLASS)
encoder_module = importlib.import_module(f"models.{ENCODER}")
encoder_class = getattr(encoder_module, ENCODER)
device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

# Model Definition
if TRANS_BOOL: 
    encoder = encoder_class(DROP_RATE, OUTPUT_SIZE, CLASS_BOOL, VISC_CLASS).to(device)
else: 
    encoder = encoder_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, RPM_CLASS, EMBED_SIZE, WEIGHT, VISC_CLASS).to(device)
if CURR_BOOL: 
    state_dict = torch.load(osp.join(CKPT_ROOT, CURR_CKPT), weights_only=True)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc")}
    encoder.load_state_dict(state_dict, strict=False)

for param in encoder.parameters():
    param.requires_grad = True

encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
criterion = criterion_class(DESCALER, DATA_ROOT_TRAIN, SMTH_LABEL)
optimizer = optim_class(encoder.parameters(), lr=LR, weight_decay=W_DECAY)
scheduler = scheduler_class(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

# LOAD DATA
video_paths = sorted(glob.glob(osp.join(DATA_ROOT_TRAIN, VIDEO_SUBDIR, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TRAIN, NORM_SUBDIR, "*.json")))

if rank == 0 and SAN_BOOL: sanity_check_alignment(video_paths, para_paths, FRAME_NUM, TIME)

train_video_paths, val_video_paths = train_test_split(video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
train_para_paths, val_para_paths = train_test_split(para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)

train_ds = dataset_class(train_video_paths, train_para_paths, FRAME_NUM, TIME, AUG_BOOL, VISC_CLASS)
val_ds = dataset_class(val_video_paths, val_para_paths, FRAME_NUM, TIME, aug_bool=False, visc_class=VISC_CLASS)

train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS)

test_dataset_module = importlib.import_module(f"datasets.{DATASET_TEST}")
test_dataset_class = getattr(test_dataset_module, DATASET_TEST)
test_video_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, VIDEO_SUBDIR, "*.mp4")))
test_para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, NORM_SUBDIR, "*.json")))
test_ds = test_dataset_class(test_video_paths, test_para_paths, FRAME_NUM, TIME, aug_bool=False, visc_class=VISC_CLASS)
test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=NUM_WORKERS)

# WANDB INITIATE
if TRAIN_BOOL:
    if rank == 0: 
        wandb.init(project=PROJECT, name=run_name, reinit=True, resume="never", config=config)
        if WATCH_BOOL: 
            wandb.watch(encoder, log="all", log_freq=20)

    # TRAIN MODEL
    best_val_loss = float("inf")
    counter = 0
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        train_losses = []
        if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")
        encoder.train()
        train_loader = tqdm(train_dl) if rank == 0 else train_dl
        for frames, parameters, hotvector, names, rpm_idx in train_loader:
            frames, parameters, hotvector, rpm_idx = frames.to(device), parameters.to(device), hotvector.to(device), rpm_idx.to(device).long().squeeze(-1)
            # print(names)
            outputs = encoder(frames, rpm_idx)
            # train_loss = encoder(frames, rpm_idx, y=parameters, mode="nll")
            if CLASS_BOOL: # Classification
                train_loss = criterion(outputs, hotvector)
                # print("hotvector:", hotvector)
            else: # Regression
                print("hi")
                # train_loss = criterion(outputs, parameters)
                # if rank == 0: MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "train", DATA_ROOT_TRAIN)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            with torch.no_grad():
                torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
                avg_train_loss = train_loss / world_size
                train_losses.append(avg_train_loss.item())

            if (len(train_losses)) % 10 == 0:
                mean_train_loss = mean(train_losses)
                train_losses.clear()
                if rank == 0: wandb.log({"train_loss": mean_train_loss})
        train_losses.clear()

    ### VALIDATION
        encoder.eval()
        val_losses = []
        with torch.no_grad():
            if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
            val_loader = tqdm(val_dl) if rank == 0 else val_dl
            test_loader = tqdm(test_dl) if rank == 0 else test_dl
            for frames, parameters, hotvector, name, rpm_idx in test_loader if VAL_TEST_BOOL else val_loader:
                frames, parameters, hotvector, rpm_idx = frames.to(device), parameters.to(device), hotvector.to(device), rpm_idx.to(device, dtype=torch.int).squeeze(-1)
                # print(names)
                outputs = encoder(frames, rpm_idx)
                # val_loss = encoder(frames, rpm_idx, y=parameters, mode="nll")
                if CLASS_BOOL: # Classification
                    val_loss = criterion(outputs, hotvector)
                    # print("hotvector:", hotvector)
                else: # Regression
                    val_loss = criterion(outputs, parameters)
                    if rank == 0: MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "val", DATA_ROOT_TRAIN)
                torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
                avg_val_loss = val_loss / world_size
                val_losses.append(avg_val_loss.item())
        mean_val_loss = mean(val_losses)
        val_losses.clear()
        if rank == 0: wandb.log({"val_loss": mean_val_loss})
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.7f}")
        ### PATIENCE
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            counter = 0
            if rank == 0: 
                torch.save(encoder.module.state_dict(), checkpoint)
                print(f"Model saved to {checkpoint}")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    if rank==0: print("Training complete.")

if TEST_BOOL and rank == 0:
    # Definition
    test_dataset_module = importlib.import_module(f"datasets.{DATASET_TEST}")
    test_dataset_class = getattr(test_dataset_module, DATASET_TEST)
    encoder.eval()
    if not TRAIN_BOOL: checkpoint = osp.join(CKPT_ROOT, CURR_CKPT)
    encoder.module.load_state_dict(torch.load(checkpoint), strict=False)
    # DATASET LOAD
    test_video_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, VIDEO_SUBDIR, "*.mp4")))
    test_para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, NORM_SUBDIR, "*.json")))
    test_ds = test_dataset_class(test_video_paths, test_para_paths, FRAME_NUM, TIME, aug_bool=False, visc_class=VISC_CLASS)
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=NUM_WORKERS, pin_memory=True)
    # TEST LOOP
    errors = []
    preds_local, tgts_local, energy_local, logits_list, tgt_idx_list = [], [], [], [], []
    with torch.no_grad():
        for frames, parameters, hotvector, names, rpm_idx in tqdm(test_dl):
            frames, parameters, hotvector, rpm_idx = frames.to(device), parameters.to(device), hotvector.to(device), rpm_idx.to(device).long().squeeze(-1)
            outputs = encoder(frames, rpm_idx)
            if ATTN_BOOL: viz_attention(encoder, frames, rpm_idx, names, outputs, hotvector, vid_bool=False, xy_bool=False, time_bool=False, npy_bool=True)
            if CLASS_BOOL: # Classification
                logits_list.append(outputs.squeeze(0).detach().cpu().float())   # for reliability
                tgt_idx_list.append(int(hotvector.squeeze(0).argmax(dim=-1)))

                energy_local.extend(energyCalculator(outputs, T=1.0))
                preds_local.extend(outputs.argmax(1).cpu().numpy().tolist())
                tgts_local.extend(hotvector.cpu().numpy().tolist())
            else: # Regression
                preds_local.extend(outputs.detach().cpu().numpy().tolist())
                tgts_local.extend(parameters.detach().cpu().numpy().tolist())
        if CLASS_BOOL:
            logits = torch.stack(logits_list, dim=0)                      # [N,10]
            idx = torch.tensor(tgt_idx_list, dtype=torch.long)            # [N]
            onehot = F.one_hot(idx, num_classes=logits.shape[1]).float()  # [N,10]

            reliability_diagram(logits, onehot, run_name)
            confusion_matrix(run_name, preds_local, tgts_local)
            csv_export(logits, energy_local)
        else:
            # plot_error_distribution(run_name, preds_local, tgts_local, DESCALER, DATA_ROOT_TEST)
            new_plot_error_distribution(run_name, preds_local, tgts_local, DESCALER, DATA_ROOT_TEST)

if rank == 0: wandb.finish()
ddp_cleanup()