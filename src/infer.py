import os
import re
import json
import csv
import glob
import datetime
import argparse
import importlib
import os.path as osp
from statistics import mean

import warnings

import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import sanity_check_alignment,set_seed, confusion_matrix

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

RESULT_DIR = "src/inference/predictions"

PROJECT = config["project"]
NAME = config["name"]
VER = config["version"]
NUM_WORKERS = int(config["train_settings"]["num_workers"])
SEED = int(config["train_settings"]["seed"])
WATCH_BOOL = bool(config["train_settings"]["watch_bool"])
CLASS_BOOL = bool(config["train_settings"]["classification"])
TEST_BOOL = bool(config["train_settings"]["test_bool"])

SCALER = config["dataset"]["preprocess"]["scaler"]
DESCALER = config["dataset"]["preprocess"]["descaler"]

DATA_ROOT_TRAIN = config["dataset"]["train"]["train_root"]
FRAME_NUM = float(config["dataset"]["train"]["frame_num"])
TIME = float(config["dataset"]["train"]["time"])
RPM_CLASS = int(config["dataset"]["train"]["rpm_class"])
AUG_BOOL = bool(config["dataset"]["train"]["dataloader"]["aug_bool"])
BATCH_SIZE = int(config["dataset"]["train"]["dataloader"]["batch_size"])
TEST_SIZE = float(config["dataset"]["train"]["dataloader"]["test_size"])
RAND_STATE = int(config["dataset"]["train"]["dataloader"]["random_state"])
DATASET = config["dataset"]["train"]["dataloader"]["dataloader"]

DATA_ROOT_TEST = config["dataset"]["test"]["test_root"]
FRAME_NUM_TEST = float(config["dataset"]["test"]["frame_num"])
TIME_TEST = float(config["dataset"]["test"]["time"])
RPM_CLASS_TEST = int(config["dataset"]["test"]["rpm_class"])
AUG_BOOL_TEST = bool(config["dataset"]["test"]["dataloader"]["aug_bool"])
BATCH_SIZE_TEST = int(config["dataset"]["test"]["dataloader"]["batch_size"])
TEST_SIZE_TEST = float(config["dataset"]["test"]["dataloader"]["test_size"])
RAND_STATE_TEST = int(config["dataset"]["test"]["dataloader"]["random_state"])
DATASET_TEST = config["dataset"]["test"]["dataloader"]["dataloader"]

TRANS_BOOL = config["model"]["transformer_bool"]
ENCODER = config["model"]["transformer"]["encoder"]
VISC_CLASS = config["model"]["transformer"]["class"]
CNN_TRAIN = bool(config["model"]["cnn"]["cnn_train"])
CNN = config["model"]["cnn"]["cnn"]
LSTM_SIZE = int(config["model"]["cnn"]["lstm_size"])
LSTM_LAYERS = int(config["model"]["cnn"]["lstm_layers"])
OUTPUT_SIZE = int(config["model"]["cnn"]["output_size"])
DROP_RATE = float(config["model"]["cnn"]["drop_rate"])
EMBED_SIZE = int(config["model"]["cnn"]["embedding_size"])
WEIGHT = float(config["model"]["cnn"]["embed_weight"])
GMM_NUM = int(config["model"]["gmm"]["gmm_num"])

CURR_CKPT = config["training"]["curr_ckpt"]

CKPT_ROOT = config["misc_dir"]["ckpt_root"]
VIDEO_SUBDIR = config["misc_dir"]["video_subdir"]
PARA_SUBDIR = config["misc_dir"]["para_subdir"]
NORM_SUBDIR = config["misc_dir"]["norm_subdir"]

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

today = datetime.datetime.now().strftime("%m%d")
checkpoint = "src/weights/" + CURR_CKPT
run_name = osp.basename(checkpoint).split(".")[0]

encoder_module = importlib.import_module(f"models.{ENCODER}")
encoder_class = getattr(encoder_module, ENCODER)

if TRANS_BOOL:
    encoder = encoder_class(DROP_RATE, OUTPUT_SIZE, CLASS_BOOL, VISC_CLASS, GMM_NUM).to(device)
else:
    encoder = encoder_class(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, RPM_CLASS, EMBED_SIZE, WEIGHT, VISC_CLASS).to(device)

state_dict = torch.load(osp.join(CKPT_ROOT, CURR_CKPT), map_location=device)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc")}
encoder.load_state_dict(state_dict, strict=False)

video_paths = sorted(glob.glob(osp.join(DATA_ROOT_TRAIN, VIDEO_SUBDIR, "*.mp4")))
para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TRAIN, NORM_SUBDIR, "*.json")))
sanity_check_alignment(video_paths, para_paths)

test_dataset_module = importlib.import_module(f"datasets.{DATASET_TEST}")
test_dataset_class = getattr(test_dataset_module, DATASET_TEST)

encoder.load_state_dict(torch.load(checkpoint, map_location=device))
encoder.eval()

test_video_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, VIDEO_SUBDIR, "*.mp4")))
test_para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, NORM_SUBDIR, "*.json")))
test_ds = test_dataset_class(test_video_paths, test_para_paths, FRAME_NUM, TIME, aug_bool=False, visc_class=VISC_CLASS)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=NUM_WORKERS, pin_memory=True)

os.makedirs(RESULT_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULT_DIR, f"{run_name}_summary.csv")

utils = importlib.import_module("utils")
descale = getattr(utils, DESCALER)

preds_local, tgts_local = [], []
summary_rows = []

with torch.no_grad():
    for frames, parameters, hotvector, names, rpm_idx, sample_info in tqdm(test_dl, desc="Testing"):
        frames = frames.to(device)
        parameters = parameters.to(device)
        hotvector = hotvector.to(device)
        rpm_idx = rpm_idx.to(device).long().squeeze(-1)

        outputs = encoder(frames, rpm_idx)

        base = sample_info["name"]
        render = sample_info.get("renderPattern") or ""
        rpm_from_name = sample_info.get("rpm_from_name")
        rpm_from_name = int(rpm_from_name) if rpm_from_name is not None else None

        if CLASS_BOOL:
            preds_local.extend(outputs.argmax(1).cpu().numpy().tolist())
            tgts_local.extend(hotvector.cpu().numpy().tolist())
        else:
            preds_local.extend(outputs.cpu().numpy().tolist())
            tgts_local.extend(parameters.cpu().numpy().tolist())

        kin_true = float(descale(parameters[0, 2].detach().cpu(), "kinematic_viscosity", DATA_ROOT_TEST).item())
        try:
            dyn_true = float(descale(parameters[0, 0].detach().cpu(), "dynamic_viscosity", DATA_ROOT_TEST).item())
        except Exception:
            dyn_true = None

        if CLASS_BOOL:
            true_visc_idx = int(sample_info.get("visc_index_true", int(hotvector.item())))
            pred_visc_idx = int(outputs.argmax(1).item())
            kin_pred = None
            dyn_pred = None
        else:
            true_visc_idx = None
            pred_visc_idx = None
            kin_pred = float(descale(outputs[0, 2].detach().cpu(), "kinematic_viscosity", DATA_ROOT_TEST).item())
            try:
                dyn_pred = float(descale(outputs[0, 0].detach().cpu(), "dynamic_viscosity", DATA_ROOT_TEST).item())
            except Exception:
                dyn_pred = None

        record = {
            "name": base,
            "renderPattern": render,
            "rpm": rpm_from_name,
            "rpm_idx": int(rpm_idx.item()),
            "true_visc_index": true_visc_idx,
            "pred_visc_index": pred_visc_idx,
            "true_kinematic_viscosity": kin_true,
            "pred_kinematic_viscosity": kin_pred,
            "true_dynamic_viscosity": dyn_true,
            "pred_dynamic_viscosity": dyn_pred,
        }

        json_path = os.path.join(RESULT_DIR, f"{base}.json")
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        summary_rows.append(record)

field_order = [
    "name",
    "renderPattern",
    "rpm",
    "rpm_idx",
    "true_visc_index",
    "pred_visc_index",
    "true_kinematic_viscosity",
    "pred_kinematic_viscosity",
    "true_dynamic_viscosity",
    "pred_dynamic_viscosity",
]
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=field_order)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Wrote {len(summary_rows)} JSON files to: {RESULT_DIR}")
print(f"Wrote summary CSV: {CSV_PATH}")

if CLASS_BOOL:
    confusion_matrix(run_name, preds_local, tgts_local)

try:
    import wandb
    if wandb.run is not None:
        wandb.finish()
except Exception:
    pass
