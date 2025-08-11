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
from utils.utils import MAPEcalculator, MAPEflowcalculator, sanity_check_alignment
from utils import set_seed, ddp_setup, ddp_cleanup
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning) # filter out warnings from transformer on classificatio head initialzation which is irrelevant

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
FRAME_NUM       = float(cfg["preprocess"]["frame_num"])
TIME            = float(cfg["preprocess"]["time"])
RPM_CLASS       = int(cfg["preprocess"]["rpm_class"])
AUG_BOOL        = cfg["preprocess"]["aug_bool"]
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

rank, world_size, local_rank = ddp_setup()
set_seed(SEED+rank)
print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

dataset_module = importlib.import_module(f"datasets.{DATASET}")
loss_module = importlib.import_module(f"losses.{LOSS}")
encoder_module = importlib.import_module(f"models.{ENCODER}")
flow_module = importlib.import_module(f"models.{FLOW}")
dataset_class = getattr(dataset_module, DATASET)

today = datetime.datetime.now().strftime("%m%d")
checkpoint = f"{CKPT_ROOT}{NAME}_{today}_{VER}.pth"
run_name = osp.basename(checkpoint).split(".")[0]

# LOAD DATA
video_paths = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT, NORM_SUBDIR, "*.json")))

sanity_check_alignment(video_paths, para_paths)

train_video_paths, val_video_paths = train_test_split(video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
train_para_paths, val_para_paths = train_test_split(para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)

train_ds = dataset_class(train_video_paths, train_para_paths, FRAME_NUM, TIME, AUG_BOOL)
val_ds = dataset_class(val_video_paths, val_para_paths, FRAME_NUM, TIME, aug_bool=False)

# train_ds = dataset_class(train_video_paths, train_para_paths, FRAME_NUM, TIME)
# val_ds = dataset_class(val_video_paths, val_para_paths, FRAME_NUM, TIME)

train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS)

# DEFINE MODEL
encoder_class = getattr(encoder_module, ENCODER)
flow_class = getattr(flow_module, FLOW)
criterion_class = getattr(loss_module, LOSS)
optim_class = getattr(optim, OPTIM_CLASS)
scheduler_class = getattr(optim.lr_scheduler, SCHEDULER_CLASS)
device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

# encoder = encoder_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, FLOW_BOOL, RPM_CLASS, EMBED_SIZE, WEIGHT).to(device)
encoder = encoder_class(DROP_RATE, OUTPUT_SIZE, FLOW_BOOL).to(device)
# state_dict = torch.load("src/weights/class10_rpm10_vivit_large_basetrain_meanpool_render20ddataset_norpmembed_augFALSE_0809_v0.pth")
# encoder.load_state_dict(state_dict, strict=False)
for param in encoder.parameters():
    param.requires_grad = True
encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
flow = flow_class(DIM, CON_DIM, HIDDEN_DIM, NUM_LAYERS).to(device) # this is also the flow model, but not utilized yet, not DDP wrapped
criterion = criterion_class(DESCALER, DATA_ROOT)

if FLOW_BOOL:
    optimizer = optim_class(list(encoder.parameters()) + list(flow.parameters()), lr=LR, weight_decay=W_DECAY)
else:
    optimizer = optim_class(encoder.parameters(), lr=LR, weight_decay=W_DECAY)
scheduler = scheduler_class(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

if rank == 0:
    wandb.init(project=PROJECT, name=run_name, reinit=True, resume="never", config= config)
    # wandb.watch(encoder, log="all", log_freq=10)

# TRAIN MODEL
best_val_loss = float("inf")
counter = 0
for epoch in range(NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    train_losses = []
    if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")
    encoder.train()
    for frames, parameters, hotvector, names, rpm_class in tqdm(train_dl):
        frames, parameters, hotvector, rpm_class = frames.to(device), parameters.to(device), hotvector.to(device), rpm_class.to(device)
        # print(rpm_class, hotvector)
        outputs = encoder(frames, rpm_class)

        if FLOW_BOOL:
            z, log_det_jacobian = flow(parameters, outputs)
            train_loss = criterion(z, log_det_jacobian)
            visc = flow.inverse(z, outputs)
            if rank == 0: MAPEflowcalculator(visc.detach(), parameters.detach(), DESCALER, "train", DATA_ROOT)
        else:
            train_loss = criterion(outputs, parameters, hotvector)
            # if rank == 0: MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "train", DATA_ROOT)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
            avg_train_loss = train_loss / world_size
            train_losses.append(avg_train_loss.item())

        if (len(train_losses)) % 2 == 0:
            mean_train_loss = mean(train_losses)
            train_losses.clear()
            if rank == 0: wandb.log({"train_loss": mean_train_loss})
    train_losses.clear()

    # VALIDATION
    encoder.eval()
    val_losses = []
    with torch.no_grad():
        if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
        for frames, parameters, hotvector, _, rpm_class in tqdm(val_dl):
            frames, parameters, hotvector, rpm_class = frames.to(device), parameters.to(device), hotvector.to(device), rpm_class.to(device)
            outputs = encoder(frames, rpm_class)

            if FLOW_BOOL:
                z, log_det_jacobian = flow(parameters, outputs)
                val_loss = criterion(z, log_det_jacobian)
                visc = flow.inverse(z, outputs)
                if rank == 0: MAPEflowcalculator(visc.detach(), parameters.detach(), DESCALER, "val", DATA_ROOT)
            else:
                val_loss = criterion(outputs, parameters, hotvector)
                # if rank == 0: MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "val", DATA_ROOT)
            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
            avg_val_loss = val_loss / world_size
            val_losses.append(avg_val_loss.item())

    mean_val_loss = mean(val_losses)
    val_losses.clear()
    if rank == 0: wandb.log({"val_loss": mean_val_loss})

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.7f}")
    val_losses.clear()

    # PATIENCE
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

if rank == 0:
    wandb.finish()
ddp_cleanup()



"""
# REAL WORLD calibration
encoder = encoder_class(DROP_RATE, OUTPUT_SIZE, FLOW_BOOL).to(device)
criterion = criterion_class(DESCALER, DATA_ROOT)
state_dict = torch.load("src/weights/classification_10_trans_total_0721_v0.pth")

# Remove the old FC layer weights
keys_to_remove = [k for k in state_dict if k.startswith("classifier.")]
for k in keys_to_remove:
    del state_dict[k]

# Load encoder weights only
encoder.load_state_dict(state_dict, strict=False)

for param in encoder.featureextractor.parameters():
    param.requires_grad = True

# Model Definition
encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
optimizer = torch.optim.Adam(encoder.parameters(), lr=REAL_LR, weight_decay=REAL_W_DECAY) 
scheduler = scheduler_class(optimizer, T_max=REAL_EPOCHS, eta_min=ETA_MIN)

if rank == 0:
    wandb.init(project=PROJECT, name=run_name, reinit=True, resume="never", config= config)
    wandb.watch(encoder, log="all", log_freq=10)

# Data Loader
real_video_paths = sorted(glob.glob(osp.join(REAL_ROOT, VIDEO_SUBDIR, "*.mp4")))
real_para_paths = sorted(glob.glob(osp.join(REAL_ROOT, NORM_SUBDIR, "*.json")))

real_train_video_paths, real_val_video_paths = train_test_split(real_video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
real_train_para_paths, real_val_para_paths = train_test_split(real_para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)

real_train_ds = dataset_class(real_train_video_paths, real_train_para_paths, FRAME_NUM, TIME)
real_val_ds = dataset_class(real_val_video_paths, real_val_para_paths, FRAME_NUM, TIME)

real_train_sampler = DistributedSampler(real_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
real_val_sampler = DistributedSampler(real_val_ds, num_replicas=world_size, rank=rank, shuffle=False)

real_train_dl = DataLoader(real_train_ds, batch_size=BATCH_SIZE, sampler=real_train_sampler, num_workers=NUM_WORKERS)
real_val_dl = DataLoader(real_val_ds, batch_size=BATCH_SIZE, sampler=real_val_sampler, num_workers=NUM_WORKERS)

# TRAINING
best_val_loss = float("inf")
counter = 0
for epoch in range(NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    train_losses = []
    if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")
    encoder.train()
    # for frames, parameters, names, rpm_class in tqdm(train_dl):
    #     frames, parameters, rpm_class = frames.to(device), parameters.to(device), rpm_class.to(device)
    for frames, parameters, hotvector, names, rpm_class in tqdm(train_dl):
        frames, parameters, hotvector, rpm_class = frames.to(device), parameters.to(device), hotvector.to(device), rpm_class.to(device)
        # print(hotvector)
        outputs = encoder(frames, rpm_class)

        if FLOW_BOOL:
            z, log_det_jacobian = flow(parameters, outputs)
            train_loss = criterion(z, log_det_jacobian)
            visc = flow.inverse(z, outputs)
            if rank == 0: MAPEflowcalculator(vissc.detach(), parameters.detach(), DESCALER, "train", DATA_ROOT)
        else:
            train_loss = criterion(outputs, parameters, hotvector)
            # train_loss = criterion(outputs, parameters)
            if rank == 0: MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "train", DATA_ROOT)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
            avg_train_loss = train_loss / world_size
            train_losses.append(avg_train_loss.item())

        if (len(train_losses)) % 50 == 0:
            mean_train_loss = mean(train_losses)
            if rank == 0: wandb.log({"train_loss": mean_train_loss})
    train_losses.clear()

    # VALIDATION
    encoder.eval()
    val_losses = []
    with torch.no_grad():
        if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
        # for frames, parameters, _, rpm_class in tqdm(val_dl):
        #     frames, parameters, rpm_class = frames.to(device), parameters.to(device), rpm_class.to(device)
        for frames, parameters, hotvector, _, rpm_class in tqdm(val_dl):
            frames, parameters, hotvector, rpm_class = frames.to(device), parameters.to(device), hotvector.to(device), rpm_class.to(device)
            outputs = encoder(frames, rpm_class)

            if FLOW_BOOL:
                z, log_det_jacobian = flow(parameters, outputs)
                val_loss = criterion(z, log_det_jacobian)
                visc = flow.inverse(z, outputs)
                if rank == 0: MAPEflowcalculator(visc.detach(), parameters.detach(), DESCALER, "val", DATA_ROOT)
            else:
                val_loss = criterion(outputs, parameters, hotvector)
                # val_loss = criterion(outputs, parameters)
                if rank == 0: MAPEcalculator(outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "val", DATA_ROOT)
            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
            avg_val_loss = val_loss / world_size
            val_losses.append(avg_val_loss.item())

    mean_val_loss = mean(val_losses)
    val_losses.clear()
    if rank == 0: wandb.log({"val_loss": mean_val_loss})

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
    if rank == 0: print(f"Epoch {epoch+1}/{NUM_EPOCHS} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.7f}")
    val_losses.clear()

if rank == 0:
    torch.save(encoder.module.state_dict(), checkpoint)
    print(f"Model saved to {checkpoint}")
    wandb.finish()

# Save the model
if rank == 0:
    real_checkpoint = "src/weights/curriculum.pth"
    torch.save(encoder.state_dict(), real_checkpoint)
    print(f"Model saved to {checkpoint}")
    wandb.finish()

ddp_cleanup()
"""