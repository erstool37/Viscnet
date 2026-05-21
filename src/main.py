import argparse
import datetime
import glob
import importlib
import json
import math
import os
import os.path as osp
import warnings
from statistics import mean

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from utils import (
    MAPEcalculator,
    MAPEGMMcalculator,
    calibrate_gmm,
    confusion_matrix,
    ddp_cleanup,
    ddp_setup,
    load_weights,
    plot_error_distribution,
    reliability_diagram,
    sanity_check_alignment,
    save_attention,
    set_seed,
    viz_attention,
    viz_gmm,
)

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*We recommend you.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*may impair performance.*", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)


def load_manifest_pairs(manifest_path):
    with open(manifest_path, "r") as file:
        payload = json.load(file)
    records = payload["samples"] if isinstance(payload, dict) else payload
    video_paths = [record["video_path"] for record in records]
    para_paths = [record["parameters_norm_path"] for record in records]
    return video_paths, para_paths


def names_to_list(names):
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return [str(names)]


def gather_object_records(local_records):
    bucket = [None] * dist.get_world_size()
    dist.all_gather_object(bucket, local_records)
    records = []
    for part in bucket:
        records.extend(part)
    return records


def dedupe_records_by_name(records):
    deduped = {}
    for record in records:
        deduped.setdefault(record["name"], record)
    return [deduped[name] for name in sorted(deduped)]


PROJECT = config["project"]
ENTITY = config.get("entity")
NAME = config["name"]
VER = config["version"]
# Basic settings
NUM_WORKERS = int(config["train_settings"]["num_workers"])
SEED = int(config["train_settings"]["seed"])
WATCH_BOOL = bool(config["train_settings"]["watch_bool"])
CLASS_BOOL = bool(config["train_settings"]["classification"])
GMM_BOOL = bool(config["train_settings"]["gmm_bool"])
TRAIN_BOOL = bool(config["train_settings"]["train_bool"])
TEST_BOOL = bool(config["train_settings"]["test_bool"])
ATTN_BOOL = bool(config["train_settings"]["attn_bool"])
SAN_BOOL = bool(config["train_settings"]["sanity_check_bool"])
# Dataset and Dataloader
SCALER = config["dataset"]["preprocess"]["scaler"]
DESCALER = config["dataset"]["preprocess"]["descaler"]
### For Train
DATA_ROOT_TRAIN = config["dataset"]["train"]["train_root"]
TRAIN_MANIFEST = config["dataset"]["train"].get("manifest")
FRAME_NUM = float(config["dataset"]["train"]["frame_num"])
TIME = float(config["dataset"]["train"]["time"])
RPM_CLASS = int(config["dataset"]["train"]["rpm_class"])
USE_ALL_TRAIN_SAMPLES = bool(config["dataset"]["train"].get("use_all_samples", False))
AUG_BOOL = bool(config["dataset"]["train"]["dataloader"]["aug_bool"])
BATCH_SIZE = int(config["dataset"]["train"]["dataloader"]["batch_size"])
TEST_SIZE = float(config["dataset"]["train"]["dataloader"]["test_size"])
RAND_STATE = int(config["dataset"]["train"]["dataloader"]["random_state"])
DATASET = config["dataset"]["train"]["dataloader"]["dataloader"]
### For Test
DATA_ROOT_TEST = config["dataset"]["test"]["test_root"]
TEST_MANIFEST = config["dataset"]["test"].get("manifest")
FRAME_NUM_TEST = float(config["dataset"]["test"]["frame_num"])
TIME_TEST = float(config["dataset"]["test"]["time"])
RPM_CLASS_TEST = int(config["dataset"]["test"]["rpm_class"])
AUG_BOOL_TEST = bool(config["dataset"]["test"]["dataloader"]["aug_bool"])
BATCH_SIZE_TEST = int(config["dataset"]["test"]["dataloader"]["batch_size"])
TEST_SIZE_TEST = float(config["dataset"]["test"]["dataloader"]["test_size"])
RAND_STATE_TEST = int(config["dataset"]["test"]["dataloader"]["random_state"])
DATASET_TEST = config["dataset"]["test"]["dataloader"]["dataloader"]
# Model Settings
TRANS_BOOL = config["model"]["transformer_bool"]
ENCODER = config["model"]["transformer"]["encoder"]
VISC_CLASS = config["model"]["transformer"]["class"]
TRANSFORMER_NUM_FRAMES = int(config["model"]["transformer"].get("num_frames", int(FRAME_NUM * TIME)))
CNN_TRAIN = bool(config["model"]["cnn"]["cnn_train"])
CNN = config["model"]["cnn"]["cnn"]
LSTM_SIZE = int(config["model"]["cnn"]["lstm_size"])
LSTM_LAYERS = int(config["model"]["cnn"]["lstm_layers"])
OUTPUT_SIZE = int(config["model"]["cnn"]["output_size"])
DROP_RATE = float(config["model"]["cnn"]["drop_rate"])
EMBED_SIZE = int(config["model"]["cnn"]["embedding_size"])
WEIGHT = float(config["model"]["cnn"]["embed_weight"])
GMM_NUM = int(config["model"]["gmm"]["gmm_num"])
RPM_BOOL = bool(config["model"]["embeddings"]["rpm_bool"])
PAT_BOOL = bool(config["model"]["embeddings"]["pat_bool"])

# Train Settings
VAL_TEST_BOOL = bool(config["train_settings"]["val_test_bool"])
CURR_BOOL = int(config["training"]["curr_bool"])
CURR_CKPT = config["training"]["curr_ckpt"]
CHECKPOINT_NAME = config["training"].get("checkpoint_name")
NUM_EPOCHS = int(config["training"]["num_epochs"])
LOSS = config["training"]["loss"]
SMTH_LABEL = float(config["training"]["label_smoothing"])
OPTIM_CLASS = config["training"]["optimizer"]["optim_class"]
SCHEDULER_CLASS = config["training"]["optimizer"]["scheduler_class"]
LR = float(config["training"]["optimizer"]["lr"])
ETA_MIN = float(config["training"]["optimizer"]["eta_min"])
W_DECAY = float(config["training"]["optimizer"]["weight_decay"])
PATIENCE = int(config["training"]["optimizer"]["patience"])
SCHEDULE_POLICY = config["training"]["optimizer"].get("schedule_policy", "cosine")
WARMUP_EPOCHS = float(config["training"]["optimizer"].get("warmup_epochs", 0.0))
WARMUP_START_FACTOR = float(config["training"]["optimizer"].get("warmup_start_factor", 0.1))
LR_HOLD_EPOCHS = float(config["training"]["optimizer"].get("lr_hold_epochs", 0.0))
UPDATE_DENSITY = config["training"].get("update_density", {})
OPTIMIZER_MICROBATCH_SIZE = UPDATE_DENSITY.get("optimizer_microbatch_size")
OPTIMIZER_MICROBATCH_SIZE = None if OPTIMIZER_MICROBATCH_SIZE is None else max(1, int(OPTIMIZER_MICROBATCH_SIZE))
# MISC Settings
CKPT_ROOT = config["misc_dir"]["ckpt_root"]
VIDEO_SUBDIR = config["misc_dir"]["video_subdir"]
PARA_SUBDIR = config["misc_dir"]["para_subdir"]
NORM_SUBDIR = config["misc_dir"]["norm_subdir"]
OUTPUT_ROOT = config["misc_dir"].get("output_root", "src/inference")

# DDP SETUP
rank, world_size, local_rank = ddp_setup()
set_seed(SEED + rank)
print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

# Set name
today = datetime.datetime.now().strftime("%m%d")
checkpoint = osp.join(CKPT_ROOT, CHECKPOINT_NAME) if CHECKPOINT_NAME else f"{CKPT_ROOT}{NAME}_{today}_{VER}.pth"
os.makedirs(osp.dirname(checkpoint), exist_ok=True)
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
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

# Model Definition
if TRANS_BOOL:
    encoder = encoder_class(
        DROP_RATE,
        OUTPUT_SIZE,
        CLASS_BOOL,
        VISC_CLASS,
        GMM_NUM,
        RPM_BOOL,
        PAT_BOOL,
        num_frames=TRANSFORMER_NUM_FRAMES,
    ).to(device)
else:
    encoder = encoder_class(
        LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, RPM_CLASS, EMBED_SIZE, WEIGHT, VISC_CLASS
    ).to(device)
if CURR_BOOL:
    load_weights(encoder, osp.join(CKPT_ROOT, CURR_CKPT))

encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
criterion = criterion_class(DESCALER, DATA_ROOT_TRAIN, SMTH_LABEL)
optimizer = optim_class(encoder.parameters(), lr=LR, weight_decay=W_DECAY)

# LOAD DATA
if TRAIN_MANIFEST:
    video_paths, para_paths = load_manifest_pairs(TRAIN_MANIFEST)
else:
    video_paths = sorted(glob.glob(osp.join(DATA_ROOT_TRAIN, VIDEO_SUBDIR, "*.mp4")))
    para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TRAIN, NORM_SUBDIR, "*.json")))

if rank == 0 and SAN_BOOL:
    sanity_check_alignment(video_paths, para_paths, FRAME_NUM, TIME)

if USE_ALL_TRAIN_SAMPLES:
    train_video_paths = video_paths
    train_para_paths = para_paths
    val_video_paths = video_paths[:1]
    val_para_paths = para_paths[:1]
else:
    train_video_paths, val_video_paths = train_test_split(video_paths, test_size=TEST_SIZE, random_state=RAND_STATE)
    train_para_paths, val_para_paths = train_test_split(para_paths, test_size=TEST_SIZE, random_state=RAND_STATE)

train_ds = dataset_class(train_video_paths, train_para_paths, FRAME_NUM, TIME, AUG_BOOL, VISC_CLASS)
val_ds = dataset_class(val_video_paths, val_para_paths, FRAME_NUM, TIME, aug_bool=False, visc_class=VISC_CLASS)

train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
SCHEDULER_STEP_PER_OPTIMIZER_STEP = OPTIMIZER_MICROBATCH_SIZE is not None and OPTIMIZER_MICROBATCH_SIZE < BATCH_SIZE
full_loader_batches, partial_loader_batch = divmod(len(train_sampler), BATCH_SIZE)
OPTIMIZER_STEPS_PER_EPOCH = full_loader_batches
if SCHEDULER_STEP_PER_OPTIMIZER_STEP:
    OPTIMIZER_STEPS_PER_EPOCH = full_loader_batches * math.ceil(BATCH_SIZE / OPTIMIZER_MICROBATCH_SIZE)
    if partial_loader_batch:
        OPTIMIZER_STEPS_PER_EPOCH += math.ceil(partial_loader_batch / OPTIMIZER_MICROBATCH_SIZE)
elif partial_loader_batch:
    OPTIMIZER_STEPS_PER_EPOCH += 1
SCHEDULER_T_MAX = NUM_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH if SCHEDULER_STEP_PER_OPTIMIZER_STEP else NUM_EPOCHS
SCHEDULER_UNITS_PER_EPOCH = OPTIMIZER_STEPS_PER_EPOCH if SCHEDULER_STEP_PER_OPTIMIZER_STEP else 1
if SCHEDULE_POLICY == "hold_then_cosine":
    hold_steps = int(round(LR_HOLD_EPOCHS * SCHEDULER_UNITS_PER_EPOCH))
    hold_steps = max(0, min(hold_steps, max(0, SCHEDULER_T_MAX - 1)))
    decay_steps = max(1, SCHEDULER_T_MAX - hold_steps)
    eta_factor = ETA_MIN / LR if LR > 0.0 else 0.0

    def hold_then_cosine_lambda(step):
        if step <= hold_steps:
            return 1.0
        progress = min(1.0, (step - hold_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_factor + (1.0 - eta_factor) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=hold_then_cosine_lambda)
elif SCHEDULE_POLICY == "warmup_hold_cosine":
    if not (0.0 < WARMUP_START_FACTOR <= 1.0):
        raise ValueError("warmup_start_factor must be in (0, 1].")
    warmup_steps = int(round(WARMUP_EPOCHS * SCHEDULER_UNITS_PER_EPOCH))
    warmup_steps = max(1, min(warmup_steps, max(1, SCHEDULER_T_MAX - 2)))
    max_hold_steps = max(0, SCHEDULER_T_MAX - warmup_steps - 1)
    hold_steps = int(round(LR_HOLD_EPOCHS * SCHEDULER_UNITS_PER_EPOCH))
    hold_steps = max(0, min(hold_steps, max_hold_steps))
    decay_steps = max(1, SCHEDULER_T_MAX - warmup_steps - hold_steps)
    eta_factor = ETA_MIN / LR if LR > 0.0 else 0.0

    def warmup_hold_cosine_lambda(step):
        if step <= warmup_steps:
            progress = step / warmup_steps
            return WARMUP_START_FACTOR + (1.0 - WARMUP_START_FACTOR) * progress
        hold_end = warmup_steps + hold_steps
        if step <= hold_end:
            return 1.0
        progress = min(1.0, (step - hold_end) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_factor + (1.0 - eta_factor) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_hold_cosine_lambda)
elif SCHEDULE_POLICY == "cosine":
    scheduler = scheduler_class(optimizer, T_max=SCHEDULER_T_MAX, eta_min=ETA_MIN)
else:
    raise ValueError(f"Unsupported schedule_policy: {SCHEDULE_POLICY}")
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKERS)

test_dataset_module = importlib.import_module(f"datasets.{DATASET_TEST}")
test_dataset_class = getattr(test_dataset_module, DATASET_TEST)
if TEST_MANIFEST:
    test_video_paths, test_para_paths = load_manifest_pairs(TEST_MANIFEST)
else:
    test_video_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, VIDEO_SUBDIR, "*.mp4")))
    test_para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, NORM_SUBDIR, "*.json")))
test_ds = test_dataset_class(
    test_video_paths, test_para_paths, FRAME_NUM_TEST, TIME_TEST, aug_bool=False, visc_class=VISC_CLASS
)
test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=NUM_WORKERS)

# WANDB INITIATE
if TRAIN_BOOL:
    if rank == 0:
        wandb.init(project=PROJECT, entity=ENTITY, name=run_name, reinit=True, resume="never", config=config)
        print(
            f"Optimizer steps per epoch: {OPTIMIZER_STEPS_PER_EPOCH}; "
            f"Scheduler T_max: {SCHEDULER_T_MAX}; scheduler_units_per_epoch: {SCHEDULER_UNITS_PER_EPOCH}; "
            f"schedule_policy: {SCHEDULE_POLICY}; "
            f"warmup_epochs: {WARMUP_EPOCHS}; lr_hold_epochs: {LR_HOLD_EPOCHS}"
        )
        if WATCH_BOOL:
            wandb.watch(encoder, log="all", log_freq=20)

    # TRAIN MODEL
    preds_train, tgts_train = [], []
    best_val_loss = float("inf")
    counter = 0
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        train_losses = []
        if rank == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training")
        encoder.train()
        epoch_train_losses = []
        optimizer.zero_grad(set_to_none=True)
        train_loader = tqdm(train_dl) if rank == 0 else train_dl
        for frames, parameters, hotvector, names, rpm_idx, pattern in train_loader:
            frames, parameters, hotvector, rpm_idx, pattern = (
                frames.to(device),
                parameters.to(device),
                hotvector.to(device),
                rpm_idx.to(device).long().squeeze(-1),
                pattern.to(device),
            )
            local_batch_size = frames.shape[0]
            microbatch_size = local_batch_size
            if OPTIMIZER_MICROBATCH_SIZE is not None and OPTIMIZER_MICROBATCH_SIZE < local_batch_size:
                microbatch_size = OPTIMIZER_MICROBATCH_SIZE

            for micro_start in range(0, local_batch_size, microbatch_size):
                micro_end = min(micro_start + microbatch_size, local_batch_size)
                frames_micro = frames[micro_start:micro_end]
                parameters_micro = parameters[micro_start:micro_end]
                hotvector_micro = hotvector[micro_start:micro_end]
                rpm_idx_micro = rpm_idx[micro_start:micro_end]
                pattern_micro = pattern[micro_start:micro_end]

                outputs = encoder(frames_micro, rpm_idx_micro, pattern_micro)
                if CLASS_BOOL:  # Classification
                    train_loss = criterion(outputs, hotvector_micro)
                else:  # Regression
                    train_loss = criterion(outputs, parameters_micro)
                    if GMM_BOOL:
                        if rank == 0:
                            MAPEGMMcalculator(
                                outputs, parameters_micro.detach().cpu(), DESCALER, "val", DATA_ROOT_TRAIN
                            )
                    else:
                        preds_train.append(outputs)
                        tgts_train.extend(parameters_micro.detach().cpu().numpy().tolist())
                        if rank == 0:
                            MAPEcalculator(
                                outputs.detach().cpu(),
                                parameters_micro.detach().cpu(),
                                DESCALER,
                                "train",
                                DATA_ROOT_TRAIN,
                            )
                train_loss.backward()
                optimizer.step()
                if SCHEDULER_STEP_PER_OPTIMIZER_STEP:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    avg_train_loss = train_loss.detach()
                    torch.distributed.all_reduce(avg_train_loss, op=torch.distributed.ReduceOp.SUM)
                    avg_train_loss = avg_train_loss / world_size
                    train_losses.append(avg_train_loss.item())
                    epoch_train_losses.append(avg_train_loss.item())

                if (len(train_losses)) % 10 == 0:
                    mean_train_loss = mean(train_losses)
                    train_losses.clear()
                    if rank == 0:
                        wandb.log({"train_loss": mean_train_loss})
        if train_losses:
            mean_train_loss = mean(train_losses)
            train_losses.clear()
            if rank == 0:
                wandb.log({"train_loss": mean_train_loss})
        if epoch_train_losses:
            mean_train_loss = mean(epoch_train_losses)
        train_losses.clear()
        if GMM_BOOL:
            viz_gmm(checkpoint, preds_train, tgts_train, DESCALER, DATA_ROOT_TEST)
            ### VALIDATION
        preds_val, tgts_val = [], []
        encoder.eval()
        val_losses = []
        with torch.no_grad():
            if rank == 0:
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Validation")
            val_loader = tqdm(val_dl) if rank == 0 else val_dl
            test_loader = tqdm(test_dl) if rank == 0 else test_dl
            for frames, parameters, hotvector, name, rpm_idx, pattern in test_loader if VAL_TEST_BOOL else val_loader:
                frames, parameters, hotvector, rpm_idx, pattern = (
                    frames.to(device),
                    parameters.to(device),
                    hotvector.to(device),
                    rpm_idx.to(device, dtype=torch.int).squeeze(-1),
                    pattern.to(device),
                )
                outputs = encoder(frames, rpm_idx, pattern)
                if CLASS_BOOL:  # Classification
                    val_loss = criterion(outputs, hotvector)
                else:  # Regression
                    val_loss = criterion(outputs, parameters)
                    if GMM_BOOL:
                        if rank == 0:
                            MAPEGMMcalculator(outputs, parameters.detach().cpu(), DESCALER, "val", DATA_ROOT_TRAIN)
                        preds_val.append(outputs)
                        tgts_val.extend(parameters.detach().cpu().numpy().tolist())
                    else:
                        if rank == 0:
                            MAPEcalculator(
                                outputs.detach().cpu(), parameters.detach().cpu(), DESCALER, "val", DATA_ROOT_TRAIN
                            )
                torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
                avg_val_loss = val_loss / world_size
                val_losses.append(avg_val_loss.item())
            if GMM_BOOL:
                if rank == 0:
                    viz_gmm(checkpoint, preds_val, tgts_val, DESCALER, DATA_ROOT_TEST)
                    calibrate_gmm(checkpoint)
        mean_val_loss = mean(val_losses)
        val_losses.clear()
        if rank == 0:
            wandb.log({"val_loss": mean_val_loss})
        if not SCHEDULER_STEP_PER_OPTIMIZER_STEP:
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.7f}"
            )
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
                print(f"Early stopping at epoch {epoch + 1}")
                break
    if rank == 0:
        print("Training complete.")

if TEST_BOOL:
    # Definition
    test_dataset_module = importlib.import_module(f"datasets.{DATASET_TEST}")
    test_dataset_class = getattr(test_dataset_module, DATASET_TEST)
    encoder.eval()
    if not TRAIN_BOOL:
        checkpoint = osp.join(CKPT_ROOT, CURR_CKPT)
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    encoder.module.load_state_dict(torch.load(checkpoint), strict=False)
    # DATASET LOAD
    if TEST_MANIFEST:
        test_video_paths, test_para_paths = load_manifest_pairs(TEST_MANIFEST)
    else:
        test_video_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, VIDEO_SUBDIR, "*.mp4")))
        test_para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, NORM_SUBDIR, "*.json")))
    test_ds = test_dataset_class(
        test_video_paths, test_para_paths, FRAME_NUM_TEST, TIME_TEST, aug_bool=False, visc_class=VISC_CLASS
    )
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_TEST,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    # TEST LOOP
    records_local = []
    preds_local, tgts_local = [], []
    with torch.no_grad():
        test_iter = tqdm(test_dl) if rank == 0 else test_dl
        for frames, parameters, hotvector, names, rpm_idx, pattern in test_iter:
            frames, parameters, hotvector, rpm_idx, pattern = (
                frames.to(device),
                parameters.to(device),
                hotvector.to(device),
                rpm_idx.to(device).long().squeeze(-1),
                pattern.to(device),
            )
            outputs = encoder(frames, rpm_idx, pattern)
            if ATTN_BOOL and rank == 0:
                save_attention(
                    encoder,
                    frames[:1],
                    rpm_idx[:1],
                    pattern[:1],
                    checkpoint,
                    names_to_list(names)[0],
                    outputs[:1],
                    hotvector[:1],
                )
            if CLASS_BOOL:  # Classification
                logits_cpu = outputs.detach().cpu().float()
                labels_cpu = hotvector.detach().cpu().view(-1).long()
                preds_cpu = logits_cpu.argmax(dim=1)
                for name, label, pred, logits_row in zip(
                    names_to_list(names), labels_cpu.tolist(), preds_cpu.tolist(), logits_cpu.tolist()
                ):
                    records_local.append(
                        {
                            "name": name,
                            "target": int(label),
                            "prediction": int(pred),
                            "logits": [float(value) for value in logits_row],
                        }
                    )
            else:  # Regression
                names_batch = names_to_list(names)
                params_cpu = parameters.detach().cpu().numpy().tolist()
                if GMM_BOOL:
                    outputs_cpu = outputs.detach().cpu().numpy().tolist()
                else:  # Simple Regression
                    outputs_cpu = outputs.detach().cpu().numpy().tolist()
                for name, pred, target in zip(names_batch, outputs_cpu, params_cpu):
                    records_local.append({"name": name, "prediction": pred, "target": target})
    records_all = dedupe_records_by_name(gather_object_records(records_local))
    if rank == 0:
        if CLASS_BOOL:
            logits = torch.tensor([record["logits"] for record in records_all], dtype=torch.float32)
            idx = torch.tensor([record["target"] for record in records_all], dtype=torch.long)
            onehot = F.one_hot(idx, num_classes=logits.shape[1]).float()
            confusion_matrix(run_name, logits.numpy(), idx.numpy(), save_dir=osp.join(OUTPUT_ROOT, "confusion_matrix"))
            reliability_diagram(logits, onehot, run_name, save_dir=osp.join(OUTPUT_ROOT, "reliability_plots"))
        else:
            preds_local = [record["prediction"] for record in records_all]
            tgts_local = [record["target"] for record in records_all]
            if GMM_BOOL:
                viz_gmm(checkpoint, preds_local, tgts_local, DESCALER, DATA_ROOT_TEST)
                calibrate_gmm(checkpoint)
            else:
                plot_error_distribution(run_name, preds_local, tgts_local, DESCALER, DATA_ROOT_TEST)
        if ATTN_BOOL:
            viz_attention(checkpoint)

if rank == 0:
    wandb.finish()
ddp_cleanup()
