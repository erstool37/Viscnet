# inference_real.py
import argparse, os.path as osp, glob, importlib, yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import set_seed, sanity_check_alignment  # from your repo

# -------- parse config --------
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="configs/config.yaml")
parser.add_argument("--ckpt", type=str, required=True, help="path/to/checkpoint.pth")
parser.add_argument("--split", type=str, default="test", choices=["train","val","test"])
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--outdir", type=str, default="inference_out")
parser.add_argument("--annotate_top", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# ---- pull cfg (use same keys as training) ----
SEED            = int(config["train_settings"]["seed"])
CLASS_BOOL      = bool(config["train_settings"]["classification"])
TRANS_BOOL      = bool(config["model"]["transformer_bool"])
ENCODER         = config["model"]["transformer"]["encoder"]
CNN             = config["model"]["cnn"]["cnn"]
DROP_RATE       = float(config["model"]["cnn"]["drop_rate"])
OUTPUT_SIZE     = int(config["model"]["cnn"]["output_size"])
LSTM_SIZE       = int(config["model"]["cnn"]["lstm_size"])
LSTM_LAYERS     = int(config["model"]["cnn"]["lstm_layers"])
CNN_TRAIN       = bool(config["model"]["cnn"]["cnn_train"])
RPM_CLASS       = int(config["dataset"][args.split]["rpm_class"])
EMBED_SIZE      = int(config["model"]["cnn"]["embedding_size"])
WEIGHT          = float(config["model"]["cnn"]["embed_weight"])
VISC_CLASS      = int(config["model"]["transformer"]["class"])
VIDEO_SUBDIR    = config["misc_dir"]["video_subdir"]
NORM_SUBDIR     = config["misc_dir"]["norm_subdir"]
DESCALER        = config["dataset"]["preprocess"]["descaler"]  # dotted path or empty
DATASET_NAME    = config["dataset"][args.split]["dataloader"]["dataloader"]
DATA_ROOT       = config["dataset"][args.split][f"{args.split}_root"]
FRAME_NUM       = float(config["dataset"][args.split]["frame_num"])
TIME            = float(config["dataset"][args.split]["time"])

# ---- setup ----
set_seed(SEED)
device = torch.device(args.device)
Path(args.outdir).mkdir(parents=True, exist_ok=True)

# ---- dataset ----
dataset_module = importlib.import_module(f"datasets.{DATASET_NAME}")
dataset_class = getattr(dataset_module, DATASET_NAME)

video_paths = sorted(glob.glob(osp.join(DATA_ROOT, VIDEO_SUBDIR, "*.mp4")))
para_paths  = sorted(glob.glob(osp.join(DATA_ROOT, NORM_SUBDIR, "*.json")))
sanity_check_alignment(video_paths, para_paths)

ds = dataset_class(video_paths, para_paths, FRAME_NUM, TIME, aug_bool=False, visc_class=VISC_CLASS)
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# ---- model ----
if TRANS_BOOL:
    enc_mod = importlib.import_module(f"models.{ENCODER}")
    Enc = getattr(enc_mod, ENCODER)
    model = Enc(DROP_RATE, OUTPUT_SIZE, CLASS_BOOL, VISC_CLASS).to(device)
else:
    enc_mod = importlib.import_module(f"models.{CNN}")
    Enc = getattr(enc_mod, CNN)
    model = Enc(LSTM_SIZE, LSTM_LAYERS, OUTPUT_SIZE, DROP_RATE, CNN, CNN_TRAIN, RPM_CLASS, EMBED_SIZE, WEIGHT, VISC_CLASS).to(device)

state = torch.load(args.ckpt, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

# ---- optional descaler (same convention as training cfg) ----
descale_fn = None
if isinstance(DESCALER, str) and DESCALER.strip():
    try:
        mod_name, fn_name = DESCALER.rsplit(".", 1)
        descale_fn = getattr(importlib.import_module(mod_name), fn_name)
    except Exception as e:
        print(f"[WARN] could not import DESCALER '{DESCALER}': {e}")

def _descale(x):
    if descale_fn is None:
        return x
    try:
        return descale_fn(x)
    except Exception as e:
        print(f"[WARN] descale failed ({e}); returning raw values")
        return x

# ---- inference loop (with tqdm) ----
rows = []
names_all, rpm_all, rho_all, sig_all, nu_all, err_all = [], [], [], [], [], []
running_err_sum, running_count = 0.0, 0

with torch.no_grad():
    pbar = tqdm(dl, total=len(dl), desc=f"Inference [{args.split}]", unit="batch")
    for batch in pbar:
        # dataset returns: frames, parameters, hotvector, names, rpm
        frames, parameters, hotvector, names, rpm_idx = batch
        frames = frames.to(device)
        rpm_idx = rpm_idx.to(device).long().squeeze(-1)

        out = model(frames, rpm_idx)

        if CLASS_BOOL:
            probs = F.softmax(out, dim=1)
            pred  = probs.argmax(1)
            tgt   = hotvector.to(device).long().view(-1)
            p_true = probs[torch.arange(probs.size(0)), tgt]
            err = (1.0 - p_true).detach().cpu().numpy()

            params_real = _descale(parameters.to(device)).detach().cpu().numpy()
            for i in range(params_real.shape[0]):
                nm = str(names[i])
                rho, sig, nu, rpm_i = params_real[i].tolist()
                names_all.append(nm); rpm_all.append(rpm_i); rho_all.append(rho); sig_all.append(sig); nu_all.append(nu); err_all.append(float(err[i]))
                rows.append({
                    "name": nm,
                    "density": rho, "surface_tension": sig, "kinematic_visc": nu, "rpm_idx": int(rpm_i),
                    "target_cluster": int(tgt[i].item()),
                    "pred_cluster": int(pred[i].item()),
                    "prob_true": float(1.0 - err[i]),
                    "error_score": float(err[i]),
                    "correct": bool(pred[i].item() == tgt[i].item()),
                })

            running_err_sum += float(err.sum())
            running_count   += len(err)
            pbar.set_postfix(avg_err=f"{(running_err_sum/max(running_count,1)):.4f}")

        else:
            pred_real = _descale(out)
            tgt_real  = _descale(parameters.to(device))
            pred_np = pred_real.detach().cpu().numpy()
            tgt_np  = tgt_real.detach().cpu().numpy()

            # use % relative error of ν for the bar metric
            batch_errs = []
            for i in range(pred_np.shape[0]):
                nm = str(names[i])
                rho_t, sig_t, nu_t, rpm_i = tgt_np[i].tolist()
                rho_p, sig_p, nu_p, _     = pred_np[i].tolist()
                ae_nu  = abs(nu_p - nu_t)
                re_nu  = float(ae_nu / (abs(nu_t) + 1e-9)) * 100.0  # %
                l1_all = float(np.mean(np.abs(np.array([rho_p-rho_t, sig_p-sig_t, nu_p-nu_t]))))

                names_all.append(nm); rpm_all.append(rpm_i); rho_all.append(rho_t); sig_all.append(sig_t); nu_all.append(nu_t); err_all.append(re_nu)
                rows.append({
                    "name": nm,
                    "density_true": rho_t, "surface_tension_true": sig_t, "kinematic_visc_true": nu_t, "rpm_idx": int(rpm_i),
                    "density_pred": rho_p, "surface_tension_pred": sig_p, "kinematic_visc_pred": nu_p,
                    "abs_err_nu": float(ae_nu),
                    "rel_err_nu_percent": re_nu,
                    "l1_error_params": l1_all,
                })
                batch_errs.append(re_nu)

            running_err_sum += float(np.sum(batch_errs))
            running_count   += len(batch_errs)
            pbar.set_postfix(avg_err=f"{(running_err_sum/max(running_count,1)):.4f}%")

# ---- save csv ----
import csv
csv_path = osp.join(args.outdir, f"results_{args.split}.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    if len(rows) > 0:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[INFO] saved: {csv_path}")

# ---- plots: dataset space w/ error overlays ----
def _scatter2d(xs, ys, errs, names, xlabel, ylabel, title, out_path, annotate_k):
    plt.figure(figsize=(8,6))
    sc = plt.scatter(xs, ys, c=errs, s=np.clip(np.array(errs)*200, 12, 300), alpha=0.85)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    cb = plt.colorbar(sc); cb.set_label("Error")
    if annotate_k > 0 and len(errs) > 0:
        idx = np.argsort(np.array(errs))[-annotate_k:][::-1]
        for i in idx:
            plt.annotate(str(names[i]), (xs[i], ys[i]), fontsize=8, xytext=(5,5), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[INFO] plot saved: {out_path}")

_sc1 = osp.join(args.outdir, f"scatter_rpm_vs_nu_{args.split}.png")
_sc2 = osp.join(args.outdir, f"scatter_rho_vs_sigma_{args.split}.png")
suffix = " (classification error=1-P(true))" if CLASS_BOOL else " (regression error=% rel ν)"

_scatter2d(np.array(rpm_all, dtype=float), np.array(nu_all, dtype=float), np.array(err_all, dtype=float),
           [str(n) for n in names_all], "RPM index", "Kinematic viscosity (ν)",
           f"{args.split.upper()} — RPM vs ν{suffix}", _sc1, args.annotate_top)

_scatter2d(np.array(rho_all, dtype=float), np.array(sig_all, dtype=float), np.array(err_all, dtype=float),
           [str(n) for n in names_all], "Density (ρ)", "Surface tension (σ)",
           f"{args.split.upper()} — ρ vs σ{suffix}", _sc2, args.annotate_top)

print("[INFO] inference complete.")
