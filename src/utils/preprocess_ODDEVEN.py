import os, os.path as osp, glob, json, argparse, yaml

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

STATS_ROOT      = "dataset/RealArchive/real_20rpm_increment"
DATA_ROOT_TRAIN = "dataset/RealArchive/real_20rpm_increment_odd_train"
DATA_ROOT_TEST  = "dataset/RealArchive/real_20rpm_increment_even_test"
PARA_SUBDIR     = "parameters"
NORM_SUBDIR     = "parametersNorm"

# load statistics.json
stat_path = osp.join(STATS_ROOT, "statistics.json")
with open(stat_path, "r") as f:
    stats = json.load(f)

def normalize(val, prop, mode):
    if mode == "interscaler":
        vmin = stats[prop]["min"]
        vmax = stats[prop]["max"]
        return (val - vmin) / (vmax - vmin)
    else:
        mean = stats[prop]["mean"]
        std  = stats[prop]["std"]
        return (val - mean) / (std)
        
for root in [DATA_ROOT_TRAIN, DATA_ROOT_TEST]:
    in_dir  = osp.join(root, PARA_SUBDIR)
    out_dir = osp.join(root, NORM_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(osp.join(in_dir, "*.json")))
    print(f"Normalizing {len(paths)} files from {in_dir} -> {out_dir} using {stat_path}")

    for p in paths:
        with open(p, "r") as f:
            d = json.load(f)

        dyn = float(d["dynamic_viscosity"])
        kin = float(d["kinematic_viscosity"])
        den = float(d["density"])
        st  = float(d["surface_tension"])
        rpm = float(d["RPM"])

        out = {
            "dynamic_viscosity": normalize(dyn, "dynamic_viscosity", "interscaler"),
            "kinematic_viscosity": normalize(kin, "kinematic_viscosity", "interscaler"),
            "density": normalize(den, "density", "interscaler"),
            "surface_tension": normalize(st,  "surface_tension", "interscaler"),
            "RPM": normalize(rpm, "rpm", "interscaler")
        }

        with open(osp.join(out_dir, osp.basename(p)), "w") as g:
            json.dump(out, g, indent=4)

    print("Normalization complete.")