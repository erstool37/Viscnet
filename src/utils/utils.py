import torch
import os.path as osp
import json
from statistics import mean, stdev
import wandb
import importlib
import cv2

# log10 > 0 to 1
def loginterscaler(lst):
    log_lst = torch.log10(torch.tensor(lst, dtype=torch.float32))
    min_val = log_lst.min()
    max_val = log_lst.max()
    scaled = (log_lst - min_val) / (max_val - min_val)
    return scaled, max_val.item(), min_val.item()

def loginterdescaler(scaled_lst, property, path):
    root = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        max_val = torch.tensor(data[property]["max"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        min_val = torch.tensor(data[property]["min"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    log_val = scaled_lst * (max_val - min_val) + min_val
    return torch.pow(10, log_val)

# 0 to 1
def interscaler(lst): 
    lst = torch.tensor(lst, dtype=torch.float32)
    min_val = lst.min()
    max_val = lst.max()
    scaled = (lst - min_val) / (max_val - min_val)
    return scaled, max_val, min_val

def interdescaler(scaled_lst, property, path):
    root = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        max_val = torch.tensor(data[property]["max"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        min_val = torch.tensor( data[property]["min"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    return scaled_lst * (max_val - min_val) + min_val
    
# zscore mean to 0.5
def zscaler(lst):
    lst = torch.tensor(lst, dtype=torch.float32)
    mean_val = lst.mean()
    std_val = lst.std()
    scaled = [(x - mean_val) / (2 * std_val) + 0.5 for x in lst] 
    return scaled, mean_val, std_val

def zdescaler(scaled_lst, property, path):
    root = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        mean_val = torch.tensor(data[property]["mean"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        std_val = torch.tensor(data[property]["std"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    descaled = (scaled_lst - 0.5) * (2 * std_val) + mean_val
    return descaled

# log10 > zscore 0 to 1
def logzscaler(lst):
    log_lst = torch.log10(torch.tensor(lst, dtype=torch.float32))
    mean_val = log_lst.mean()
    std_val = log_lst.std()
    scaled = (log_lst - mean_val) / (2 * std_val) + 0.5
    return scaled, mean_val, std_val

def logzdescaler(scaled_lst, property, path):
    root = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, 'r') as file:
        data = json.load(file)
        mean_val = torch.tensor(data[property]["mean"], dtype=scaled_lst.dtype, device=scaled_lst.device)
        std_val = torch.tensor(data[property]["std"], dtype=scaled_lst.dtype, device=scaled_lst.device)
    descaled = (scaled_lst - 0.5) * (2 * std_val) + mean_val
    return torch.pow(10, descaled)

def noscaler(lst):
    lst = torch.tensor(lst, dtype=torch.float32)
    constant = torch.tensor(1.0, dtype=torch.float32)
    return lst, constant, constant 

def nodescaler(lst, property, path):
    return lst

# MEAN ABSOLUTE PERCENTAGE ERROR
def MAPEcalculator(pred, target, descaler, method, path):
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    
    pred_den = descaler(pred[:,0], "density", path).unsqueeze(-1)
    pred_surfT = descaler(pred[:,1], "surface_tension", path).unsqueeze(-1)
    pred_kinvisc = descaler(pred[:,2], "kinematic_viscosity", path).unsqueeze(-1)

    target_den = descaler(target[:,0], "density", path).unsqueeze(-1)
    target_surfT = descaler(target[:,1], "surface_tension", path).unsqueeze(-1)
    target_kinvisc = descaler(target[:,2], "kinematic_viscosity", path).unsqueeze(-1)

    loss_mape_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
    loss_mape_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)
    loss_mape_kinvisc = torch.mean((torch.abs(pred_kinvisc - target_kinvisc) / target_kinvisc)).unsqueeze(-1)

    # wandb.log({f"MAPE {method} den %" : loss_mape_den * 100})
    wandb.log({f"MAPE {method} kinvisc %" : loss_mape_kinvisc * 100})
    # wandb.log({f"MAPE {method} surfT %" : loss_mape_surfT * 100})
    
def MAPEGMMcalculator(pred, target, descaler, method, path):
    """
    Computes MAPE (%) for kinematic viscosity only.
    pred: dict containing key 'y_hat' -> Tensor [N,1]
    target: Tensor [N,1]
    """
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)

    # extract prediction tensor
    y_hat = pred["y_hat"].cpu()

    # descale
    pred_kinvisc = descaler(y_hat.squeeze(-1), "kinematic_viscosity", path).unsqueeze(-1)
    target_kinvisc = descaler(target.squeeze(-1), "kinematic_viscosity", path).unsqueeze(-1)

    # MAPE for kinematic viscosity
    loss_mape_kinvisc = torch.mean(
        torch.abs(pred_kinvisc - target_kinvisc) / (target_kinvisc + 1e-8)
    ).unsqueeze(-1)

    # log and return
    wandb.log({f"MAPE {method} kinvisc %": loss_mape_kinvisc * 100})

    return loss_mape_kinvisc.item() * 100


def sanity_check_alignment(video_paths, para_paths, expected_fps=10, expected_time=5):
    print("Initiating sanity check for video-parameter alignment...")
    expected_frames=expected_fps * expected_time
    if len(video_paths) != len(para_paths):
        raise ValueError(f"Length mismatch: {len(video_paths)} videos vs {len(para_paths)} params")

    name_mismatches = []
    fps_mismatches = []
    frame_mismatches = []

    for v_path, p_path in zip(video_paths, para_paths):
        v_name = osp.splitext(osp.basename(v_path))[0]
        p_name = osp.splitext(osp.basename(p_path))[0]

        # --- filename alignment ---
        if v_name != p_name:
            name_mismatches.append((v_name, p_name))
            continue

        # --- open video and check fps + frame count ---
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f" Could not open video: {v_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # --- load parameter file (optional cross-check) ---
        with open(p_path, "r") as jf:
            meta = json.load(jf)

        meta_fps = meta.get("fps", None)
        meta_frames = meta.get("frame_count", None)  # if stored, otherwise None

        # FPS check (both video metadata and stored meta)
        if abs(fps - expected_fps) > 0.5 or (meta_fps and abs(meta_fps - expected_fps) > 0.5):
            fps_mismatches.append((v_name, fps, meta_fps))

        # Frame count check (allow ±1 tolerance)
        if abs(frame_count - expected_frames) > 1 or (meta_frames and abs(meta_frames - expected_frames) > 1):
            frame_mismatches.append((v_name, frame_count, meta_frames))

    # --- Report ---
    if name_mismatches:
        print(f"Found {len(name_mismatches)} name mismatches:")
        for v, p in name_mismatches[:10]:
            print(f"  Video: {v}  ≠  Param: {p}")

    if fps_mismatches:
        print(f"❌ Found {len(fps_mismatches)} FPS mismatches:")
        for name, v_fps, m_fps in fps_mismatches[:5]:
            print(f"  {name}: video FPS={v_fps:.2f}, meta FPS={m_fps}")

    if frame_mismatches:
        print(f" Found {len(frame_mismatches)} frame-count mismatches:")
        for name, v_fr, m_fr in frame_mismatches[:5]:
            print(f"  {name}: video frames={v_fr}, meta frames={m_fr}")

    if not (name_mismatches or fps_mismatches or frame_mismatches):
        print(" Sanity check passed: all file names, FPS, and frame counts are aligned.")

def load_weights(model, ckpt_path: str):
    """
    1) Always loads feature extractor weights (strict on shape).
    2) Loads head (fc/gmm) weights only if shapes match (best-effort).
    - Handles checkpoints saved with/without 'module.' prefix.
    - Handles checkpoints saved under 'state_dict' or 'model' keys.
    """
    target = model.module if hasattr(model, "module") else model
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # If wrapped inside a dict, try common keys
    if isinstance(ckpt, dict) and all(not isinstance(v, torch.Tensor) for v in ckpt.values()):
        for key in ("state_dict", "model", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    # Strip DDP prefix if present
    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    msd = target.state_dict()

    # Partition keys
    feat_keys = [k for k in ckpt.keys() if k.startswith("featureextractor")]
    head_keys = [k for k in ckpt.keys() if k.startswith(("fc", "gmm"))]

    # 1) Feature extractor (strict: require matching key+shape)
    feat_load = {
        k: v for k, v in ckpt.items()
        if k in feat_keys and k in msd and msd[k].shape == v.shape
    }
    missing_feats = [k for k in feat_keys if not (k in feat_load)]
    if missing_feats:
        raise RuntimeError(f"Feature extractor mismatch/missing keys (first few): {missing_feats[:5]}")

    # 2) Head (best-effort: only load matching shapes)
    head_load = {
        k: v for k, v in ckpt.items()
        if k in head_keys and k in msd and msd[k].shape == v.shape
    }
    skipped = [k for k in head_keys if k not in head_load]
    if skipped:
        print(f"[load] Skipped head layers (shape mismatch): {skipped[:5]}...")

    # Combine and load
    load_dict = {**feat_load, **head_load}
    missing, unexpected = target.load_state_dict(load_dict, strict=False)
    print(f"[load] Features loaded: {len(feat_load)} | Head loaded: {len(head_load)} | Skipped: {len(skipped)}")
    if missing:
        print(f"[load] Missing keys (ignored): {missing[:5]}...")
    if unexpected:
        print(f"[load] Unexpected keys (ignored): {unexpected[:5]}...")