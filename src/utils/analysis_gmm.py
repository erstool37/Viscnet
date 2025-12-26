import os
import csv
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import os.path as osp
import json

def viz_gmm(checkpoint, preds_list, targets_list, descaler, path):
    descaler = "zdescaler"
    """
    GMM inference visualization & data export (LOG space, after descaling)
    ---------------------------------------------------------------------
    Assumes:
      - model outputs GMM parameters in log-z space (normalized log10 ν)
      - `descaler` maps from normalized log space -> log10(ν) (descaled log space)

    Saves (in src/inference/GMM_plots/<run_name>/):
      ✓ y_true_log.npy        [N]            (true log-viscosity)
      ✓ mean_log.npy          [N]            (mixture mean in log space)
      ✓ std_log.npy           [N]            (mixture std in log space)
      ✓ gmm_mu_log.npy        [N, K]         (component means in log space)
      ✓ gmm_sigma_log.npy     [N, K]         (component stds in log space)
      ✓ gmm_weights.npy       [N, K]         (mixture weights)
      ✓ stats.csv             per-sample stats
      ✓ parity.png            parity plot (log space)
      ✓ ause_curve.png        sparsification / AUSE (log space)
    """

    # SETUP

    base = os.path.basename(checkpoint)
    run_name = os.path.splitext(base)[0]
    out_dir = os.path.join("src", "inference", "GMM_plots", run_name)
    os.makedirs(out_dir, exist_ok=True)

    parity_plot_path = os.path.join(out_dir, "parity.png")
    ause_plot_path   = os.path.join(out_dir, "ause_curve.png")
    stats_csv_path   = os.path.join(out_dir, "stats.csv")

    # load descaler function
    utils_mod = importlib.import_module("utils")
    descaler_fn = getattr(utils_mod, "zdescaler")

    # read normalization stats ONCE
    root      = osp.dirname(osp.abspath(__file__))
    stat_path = osp.join(root, "../..", path, "statistics.json")
    with open(stat_path, "r") as f:
        stats = json.load(f)
    sigma_log = stats["kinematic_viscosity"]["std"]  # std in log10 space

    target_index = 2  # viscosity index in target vector

    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # detect number of GMM components
    K = _np(preds_list[0]["pi"]).reshape(-1).shape[0]

    # STORAGE
    rows = []
    y_true_log_list = []
    mean_log_list   = []
    std_log_list    = []

    all_mu_log   = []
    all_sig_log  = []
    all_weights  = []

    err_abs_list = []   # |y - y_hat| in log space
    err_mse_list = []

    # MAIN LOO
    for i, (pred, tgt) in enumerate(zip(preds_list, targets_list), start=1):

        # ---- GMM params in normalized space ----
        mu_n  = _np(pred["mu"]).reshape(-1)     # [K]
        sig_n = _np(pred["sigma"]).reshape(-1)  # [K]
        w     = _np(pred["pi"]).reshape(-1)     # [K]
        w     = w / w.sum()

        # ---- TRUE target in descaled log space ----
        t_n = float(_np(tgt)[target_index])  # normalized log
        t_log = float(descaler_fn(torch.tensor([t_n]),
                                  "kinematic_viscosity", path))  # log10(ν)
        y_true_log_list.append(t_log)

        # ---- Component means in descaled LOG space ----
        mu_log = _np(descaler_fn(torch.tensor(mu_n),
                                 "kinematic_viscosity", path))   # [K], log10(ν)

        # ---- Component stds in descaled LOG space ----
        # normalized std -> log std
        sig_log = sigma_log * sig_n  # [K]
        sig_log = np.clip(sig_log, 1e-12, None)

        all_mu_log.append(mu_log)
        all_sig_log.append(sig_log)
        all_weights.append(w)

        # ---- Mixture mean & variance in LOG space ----
        # μ_mix = Σ w_k μ_k
        mu_mix_log = float(np.sum(w * mu_log))

        # Var_mix = Σ w_k (σ_k^2 + μ_k^2) - μ_mix^2
        comp = sig_log**2 + mu_log**2
        var_mix_log = float(np.sum(w * comp) - mu_mix_log**2)
        var_mix_log = max(var_mix_log, 1e-12)
        std_mix_log = float(np.sqrt(var_mix_log))

        mean_log_list.append(mu_mix_log)
        std_log_list.append(std_mix_log)

        # ---- Error metrics in LOG space ----
        err_abs = abs(mu_mix_log - t_log)
        err_mse = (mu_mix_log - t_log)**2
        err_abs_list.append(err_abs)
        err_mse_list.append(err_mse)

        rows.append({
            "idx": i,
            "true_log": t_log,
            "pred_log": mu_mix_log,
            "std_log": std_mix_log,
            "abs_err_log": err_abs,
        })

    # Convert to arrays
    y_true_log_arr = np.array(y_true_log_list)
    mean_log_arr   = np.array(mean_log_list)
    std_log_arr    = np.array(std_log_list)
    err_abs_arr    = np.array(err_abs_list)
    err_mse_arr    = np.array(err_mse_list) 

    all_mu_log  = np.array(all_mu_log)   # (N, K)
    all_sig_log = np.array(all_sig_log)  # (N, K)
    all_weights = np.array(all_weights)  # (N, K)

    N = len(y_true_log_arr)

    # ============================================================
    # SAVE NUMPY ARRAYS (for calibration, analysis)
    # ============================================================
    np.save(os.path.join(out_dir, "y_true_log.npy"),  y_true_log_arr)
    np.save(os.path.join(out_dir, "mean_log.npy"),    mean_log_arr)
    np.save(os.path.join(out_dir, "std_log.npy"),     std_log_arr)

    np.save(os.path.join(out_dir, "gmm_mu_log.npy"),      all_mu_log)
    np.save(os.path.join(out_dir, "gmm_sigma_log.npy"),   all_sig_log)
    np.save(os.path.join(out_dir, "gmm_weights.npy"),     all_weights)

    print("\n=== Saved GMM log-space arrays ===")
    print(" y_true_log.npy   ", y_true_log_arr.shape)
    print(" mean_log.npy     ", mean_log_arr.shape)
    print(" std_log.npy      ", std_log_arr.shape)
    print(" gmm_mu_log.npy   ", all_mu_log.shape)
    print(" gmm_sigma_log.npy", all_sig_log.shape)
    print(" gmm_weights.npy  ", all_weights.shape)

    # ============================================================
    # SAVE STATS CSV
    # ============================================================
    with open(stats_csv_path, "w", newline="") as fp:
        wcsv = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    # ============================================================
    # PARITY PLOT (LOG space)
    # ============================================================
    rmse_log = np.sqrt(np.mean((y_true_log_arr - mean_log_arr) ** 2))
    mae_log  = np.mean(np.abs(y_true_log_arr - mean_log_arr))

    plt.figure(figsize=(4, 4))
    plt.scatter(y_true_log_arr, mean_log_arr, s=10, alpha=0.5)
    min_val = min(y_true_log_arr.min(), mean_log_arr.min())
    max_val = max(y_true_log_arr.max(), mean_log_arr.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
    plt.xlabel("True log-viscosity")
    plt.ylabel("Predicted log-viscosity")
    plt.title(f"Parity (log space), RMSE={rmse_log:.4f}, MAE={mae_log:.4f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(parity_plot_path, dpi=300)
    plt.close()

    # ============================================================
    # AUSE Sparsification (LOG space)
    # ============================================================
    errors = err_abs_arr.copy()          # |y - ŷ| in log space
    mses   = err_mse_arr.copy()
    rmse   = mses**0.5
    rel_unc = std_log_arr.copy()         # σ in log space

    mean_err  = errors.mean()
    mean_rmse = rmse.mean()

    # Normalize errors -> mean = 1 (common convention in AUSE)
    errors /= (mean_err  + 1e-12)
    rmse   /= (mean_rmse + 1e-12)

    idx_or   = np.argsort(errors)        # oracle (smallest true error first)
    idx_unc  = np.argsort(rel_unc)       # model (smallest uncertainty first)
    idx_rand = np.random.permutation(N)  # random baseline
    print(len(idx_or), len(idx_rand), len(idx_unc))

    fractions = np.linspace(0, 1, 500)

    def curve(order):
        vals = []
        for f in fractions:
            k = max(1, int(f * N))
            vals.append(errors[order[:k]].mean())
        return fractions, np.array(vals)

    f_or,   curve_or   = curve(idx_or)
    f_mod,  curve_mod  = curve(idx_unc)
    f_rand, curve_rand = curve(idx_rand)

    ause = np.trapz(curve_mod - curve_or, f_or)
    aure = np.trapz(curve_rand - curve_or, f_or)

    print(f"AUSE={ause:.6f} | AURG={aure:.6f}")

    # ============================================================
    # SAVE CSV
    # ============================================================
    csv_path = os.path.join(out_dir, "ause_curves.csv")
    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["fraction_kept", "oracle", "model", "random"])
        for f, o, m, r in zip(f_or, curve_or, curve_mod, curve_rand):
            writer.writerow([f, o, m, r])   

    print(f"Saved sparsification curves to: {csv_path}")

    plt.figure(figsize=(5.5, 5))
    plt.plot(curve_or,  label="Oracle")
    plt.plot(curve_mod, label="Model")
    plt.plot(curve_rand, "--", label="Random")
    plt.title("Sparsification (log space)")
    plt.ylabel("Normalized MAE")
    plt.xlabel("Fraction kept")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ause_plot_path, dpi=300)
    plt.close()

    # ============================================================
    # SUMMARY LOG
    # ============================================================
    print("\n========== UQ SUMMARY (LOG space) ==========")
    print(f"RMSE_log = {rmse_log:.6f}")
    print(f"MAE_log = {mae_log:.6f}")
    print(f"AUSE      = {ause:.6f}")
    print(f"AURG      = {aure:.6f}")
    print("============================================\n")

def calibrate_gmm(checkpoint, levels=(0.5, 0.68, 0.95)):
    """
    Alpha-calibration for GMM predictive std (in LOG space).

    Loads:
      y_true_log.npy  [N]
      mean_log.npy    [N]
      std_log.npy     [N]

    Outputs:
      - previous (uncalibrated) coverage   (alpha = 1.0)
      - calibrated coverage                (alpha = best_alpha)
    """
    # -----------------------------------------------------
    # LOAD ARRAYS
    # -----------------------------------------------------
    base = os.path.basename(checkpoint)
    run_name = os.path.splitext(base)[0]
    out_dir = os.path.join("src", "inference", "GMM_plots", run_name)

    y_true = np.load(os.path.join(out_dir, "y_true_log.npy"))
    y_hat  = np.load(os.path.join(out_dir, "mean_log.npy"))
    std    = np.load(os.path.join(out_dir, "std_log.npy"))

    assert y_true.shape == y_hat.shape == std.shape
    N = len(y_true)

    # -----------------------------------------------------
    # Z-values per confidence level (two-sided)
    # -----------------------------------------------------
    z_table = {
        0.50: 0.67448975,
        0.68: 1.0,
        0.95: 1.95996398,
    }

    std = np.clip(std, 1e-12, None)

    # -----------------------------------------------------
    # Coverage computing helper
    # -----------------------------------------------------
    def coverage_for_alpha(alpha):
        cov = {}
        std_scaled = std * alpha
        for p in levels:
            z = z_table.get(p, 1.0)
            hit = np.abs(y_true - y_hat) <= z * std_scaled
            cov[p] = float(hit.mean())
        return cov

    # -----------------------------------------------------
    # FIRST: Uncalibrated coverage (α = 1.0)
    # -----------------------------------------------------
    raw_cov = coverage_for_alpha(1.0)
    raw_err = sum((raw_cov[p] - p) ** 2 for p in levels)

    # -----------------------------------------------------
    # SECOND: Grid-search for calibrated α
    # -----------------------------------------------------
    best_alpha, best_err = None, 1e9
    alphas = np.linspace(0.05, 3.0, 300)

    for a in alphas:
        cov = coverage_for_alpha(a)
        err = sum((cov[p] - p) ** 2 for p in levels)
        if err < best_err:
            best_err = err
            best_alpha = a

    best_cov = coverage_for_alpha(best_alpha)

    # -----------------------------------------------------
    # PRINT RESULT
    # -----------------------------------------------------
    print("\n====== Alpha Calibration (LOG space) ======")
    print(f">> Raw (uncalibrated, alpha = 1.0)")
    for p in levels:
        print(f" level {int(p*100):2d}%: empirical={raw_cov[p]:.4f}, nominal={p:.2f}")
    print(f" raw total squared error = {raw_err:.6e}\n")

    print(f">> Calibrated (alpha = {best_alpha:.4f})")
    for p in levels:
        print(f" level {int(p*100):2d}%: empirical={best_cov[p]:.4f}, nominal={p:.2f}")
    print(f" calibrated total squared error = {best_err:.6e}")
    print("===========================================\n")

    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------
    out_json = {
        "alpha": float(best_alpha),
        "raw_coverage":   {str(p): float(raw_cov[p])  for p in levels},
        "calib_coverage": {str(p): float(best_cov[p]) for p in levels},
        "raw_error":      float(raw_err),
        "calib_error":    float(best_err),
    }
    with open(os.path.join(out_dir, "alpha_calibration.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    return best_alpha, raw_cov, best_cov