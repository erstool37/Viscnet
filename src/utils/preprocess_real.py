import json
import os.path as osp
import os
import glob
import math
import numpy as np
import torch
import argparse
import yaml
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-m", "--method", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
cfg = config["regression"]

METHOD          = args.method
DATA_ROOT       = cfg["directories"]["data"]["data_root"]
TEST_ROOT       = cfg["directories"]["data"]["test_root"]
REAL_ROOT       = cfg["directories"]["data"]["real_root"]
PARA_SUBDIR     = cfg["directories"]["data"]["para_subdir"]
NORM_SUBDIR     = cfg["directories"]["data"]["norm_subdir"]
NORMALIZE       = cfg["preprocess"]["scaler"]
UNNORMALIZE     = cfg["preprocess"]["descaler"]

if METHOD == "test":
    para_paths = sorted(glob.glob(osp.join(TEST_ROOT, PARA_SUBDIR, "*.json")))
    norm_path = osp.join(TEST_ROOT, NORM_SUBDIR)
    os.makedirs(norm_path, exist_ok=True)
    print("Test mode: Normalizing test data")
elif METHOD == "real":
    para_paths = sorted(glob.glob(osp.join(REAL_ROOT, PARA_SUBDIR, "*.json")))
    norm_path = osp.join(REAL_ROOT, NORM_SUBDIR)
    os.makedirs(norm_path, exist_ok=True)
    print("Real mode: Normalizing real data")
else:
    para_paths = sorted(glob.glob(osp.join(DATA_ROOT, PARA_SUBDIR, "*.json")))
    norm_path = osp.join(DATA_ROOT, NORM_SUBDIR)
    os.makedirs(norm_path, exist_ok=True)
    print("Train mode: Normalizing training data")

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

        # for training
        # kinVisc.append(float(1.0))
        # density.append(data["density"])
        # rpm.append(data["rpm"])
        # dynVisc.append(data["xsph"])
        # surfT.append(data["surfaceTension"])

        # for real world inference
        dynVisc.append(data["dynamic_viscosity"])
        surfT.append(data["surface_tension"])
        kinVisc.append(data["kinematic_viscosity"])
        density.append(data["density"])
        rpm.append(data["RPM"])

# sanity check
parameters = [dynVisc, kinVisc, surfT, density, rpm]
for idx, lst in enumerate(parameters):
    if max(lst) == min(lst):
        eps = max(lst) * 1e-3
        noise = (np.random.rand(len(lst)) * eps).tolist()
        for j in range(len(lst)):
            lst[j] += noise[j]

# normalize/store stats
dynViscnorm, con1dynVisc, con2dynVisc = scaler(dynVisc)
kinViscnorm, con1kinVisc, con2kinVisc = scaler(kinVisc)
surfTnorm, con1surfT, con2surfT = scaler(surfT)
densitynorm, con1density, con2density = scaler(density)
rpmnorm, con1rpm, con2rpm = scaler(rpm)

rpm_unique = sorted(set(rpm))
rpm_to_idx = {r: i for i, r in enumerate(rpm_unique)}

visc_unique = sorted(set(dynVisc))
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
    "surface_tension": float(surfTnorm[idx]),  "density": float(densitynorm[idx]), "visc_index": visc_to_idx[dynVisc[idx]], 
    "rpm": float(rpmnorm[idx]), "rpm_idx": rpm_to_idx[rpm[idx]]}
    with open(f'{norm_path}/config_{(idx+1):04d}.json', 'w') as file:
        json.dump(data, file, indent=4)

# store statistics data
with open(f'{norm_path}/../statistics.json', 'w') as file:
    json.dump(stats, file, indent=4)

print(f"Normalization complete")
# print(f"Total unique visc classes: {len(visc_to_idx)}") 