import json
import os.path as osp
import os
import glob
import numpy as np
import argparse
import yaml
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-m", "--TARGET", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

TARGET          = args.TARGET

DATA_ROOT_TRAIN = config["dataset"]["train"]["train_root"]
DATA_ROOT_TEST  = config["dataset"]["test"]["test_root"]

CKPT_ROOT       = config["misc_dir"]["ckpt_root"]
VIDEO_SUBDIR    = config["misc_dir"]["video_subdir"]
PARA_SUBDIR     = config["misc_dir"]["para_subdir"]
NORM_SUBDIR     = config["misc_dir"]["norm_subdir"]

NORMALIZE          = config["dataset"]["preprocess"]["scaler"]
UNNORMALIZE        = config["dataset"]["preprocess"]["descaler"]

if TARGET == "synthetic":
    para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TRAIN, PARA_SUBDIR, "*.json")))
    norm_path = osp.join(DATA_ROOT_TRAIN, NORM_SUBDIR)
    os.makedirs(norm_path, exist_ok=True)
    print(f"Normalizing {len(para_paths)} of synthetic data")
elif TARGET == "real":
    para_paths = sorted(glob.glob(osp.join(DATA_ROOT_TEST, PARA_SUBDIR, "*.json")))
    norm_path = osp.join(DATA_ROOT_TEST, NORM_SUBDIR)
    os.makedirs(norm_path, exist_ok=True)
    print("Normalizing real data")

utils = importlib.import_module("utils")
scaler = getattr(utils, NORMALIZE)
descaler = getattr(utils, UNNORMALIZE)

dynVisc = []
kinVisc = [] 
surfT = []
density = []
rpm = []

for path in para_paths:
    with open(path, 'r') as file:
        data = json.load(file)
        if TARGET == "synthetic":
            density.append(data["density"])
            rpm.append(data["RPM"])
            kinVisc.append(data["viscosity"])
            dynVisc.append(data["viscosity"] * data["density"])
            surfT.append(data["surface_tension"])
        else: # TARGET == "real"
            dynVisc.append(data["dynamic_viscosity"])
            surfT.append(data["surface_tension"])
            kinVisc.append(data["kinematic_viscosity"])
            density.append(data["density"])
            rpm.append(data["RPM"])

# Sanity check
parameters = [dynVisc, kinVisc, surfT, density, rpm]
for idx, lst in enumerate(parameters):
    if max(lst) == min(lst):
        eps = max(lst) * 1e-3
        noise = (np.random.rand(len(lst)) * eps).tolist()
        for j in range(len(lst)):
            lst[j] += noise[j]

# Normalize/store stats
dynViscnorm, con1dynVisc, con2dynVisc = scaler(dynVisc)
kinViscnorm, con1kinVisc, con2kinVisc = scaler(kinVisc)
surfTnorm, con1surfT, con2surfT = scaler(surfT)
densitynorm, con1density, con2density = scaler(density)
rpmnorm, con1rpm, con2rpm = scaler(rpm)

rpm_unique = sorted(set(rpm))
rpm_to_idx = {r: i for i, r in enumerate(rpm_unique)}

rounded_kinVisc = [round(v, 7) for v in kinVisc]
visc_unique = sorted(set(rounded_kinVisc))
visc_to_idx = {v: i for i, v in enumerate(visc_unique)}

if "z" in NORMALIZE:
    stats = {
        "dynamic_viscosity": {"mean": float(con1dynVisc), "std": float(con2dynVisc)},
        "kinematic_viscosity": {"mean": float(con1kinVisc), "std": float(con2kinVisc)},
        "surface_tension": {"mean": float(con1surfT), "std": float(con2surfT)},
        "density": {"mean": float(con1density), "std": float(con2density)},
        "rpm": {"mean": float(con1rpm), "std": float(con2rpm)}
    }
else:
    stats = {
        "dynamic_viscosity": {"max": float(con1dynVisc), "min": float(con2dynVisc)},
        "kinematic_viscosity": {"max": float(con1kinVisc), "min": float(con2kinVisc)},
        "surface_tension": {"max": float(con1surfT), "min": float(con2surfT)},
        "density": {"max": float(con1density), "min": float(con2density)},
        "rpm": {"max": float(con1rpm), "min": float(con2rpm)}
    }

# store normalized data
for idx in range(len(dynViscnorm)):
    data = {"dynamic_viscosity": float(dynViscnorm[idx]), "kinematic_viscosity": float(kinViscnorm[idx]), 
    "surface_tension": float(surfTnorm[idx]),  "density": float(densitynorm[idx]), "visc_index": visc_to_idx[round(kinVisc[idx], 7)], 
    "rpm": float(rpmnorm[idx]), "rpm_idx": int(rpm_to_idx[rpm[idx]])}
    original_name = osp.splitext(osp.basename(para_paths[idx]))[0]
    with open(osp.join(norm_path, f"{original_name}.json"), 'w') as file:
        json.dump(data, file, indent=4)

# store statistics data
with open(f'{norm_path}/../statistics.json', 'w') as file:
    json.dump(stats, file, indent=4)

print(f"Normalization complete")