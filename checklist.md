# Viscnet Reproduction Checker Checklist

Use this checklist to verify paper-reproduction classification runs produced by the
training subagent. This is a checker document, not a training recipe.

## Scope

Accept only reproduction runs for:

- Synthetic pretraining on `dataset/CFDArchive/sph_35000`.
- Real-only data-efficiency runs for `300, 400, 500, 600, 700, 800, 900, 993` samples.
- Synthetic-pretrained + real fine-tuning runs for the same sample counts.
- Imported pattern-generalization classification runs.

Reject or mark out of scope:

- Label smoothing above `0.0`.
- Hyperparameter sweeps or enhancement variants.
- Architecture changes.
- UQ/regression runs.
- Runs that train on copied or modified dataset folders instead of manifests or imported folders.

## Analyzer Figure Trigger Policy

After checker outputs exist for a completed classification run, the analyzer should
draw the pre-GMM paper-style Figure 3-6 diagnostics only for 993-sample runs:

- Eligible real-only runs: `realonly_993` and bounded 993 real-only retries such
  as microbatch or LR-hold variants.
- Eligible transfer runs: `transfer_993` and bounded 993 transfer retries such
  as microbatch, lower-start-LR, or LR-hold variants.
- Excluded runs: data-efficiency runs with `300, 400, 500, 600, 700, 800, 900`
  real samples, synthetic pretraining-only runs, pattern-generalization runs,
  and any GMM/regression/UQ run.

For eligible 993 runs, draw Figure 3-6 diagnostics from the pre-GMM classification
artifacts: confusion/per-class accuracy, encoder feature t-SNE, final-layer
attention binned by Re/Ca/We, and viscosity-gap `D_n` versus class accuracy.
These figures are analyzer diagnostics, not reproduction pass/fail criteria.

## Methodology Fit

For each run, classify methodology fit as one of:

- `exact reproduction`: dataset path, checkpoint chain, model, loss, optimizer, scheduler, seed, augmentation setting, and sample count match the W&B reference.
- `current-repo reproduction`: same intended scheme, but a known unavailable old asset is replaced by the current imported equivalent.
- `blocked/mismatched`: required dataset, checkpoint, code path, or config field is absent or materially changed.

Required checks:

- Config comes from `configs/rebuild/*.yaml`.
- W&B project is `viscnet-rebuild`.
- Model encoder is `VivitEmbed`.
- Embedding conditioning uses video plus RPM only: `rpm_bool: true`, `pat_bool: false`.
- Pattern images may still be loaded for dataloader interface compatibility, but they must not be embedded, subtracted, or used for classification features.
- Classification is `true`, GMM is `false`, loss is `CE`.
- `label_smoothing` is exactly `0.0`.
- Optimizer is `AdamW`; scheduler is `CosineAnnealingLR`.
- Learning rate is `1e-5`, weight decay is `1e-2`, patience is `10`.
- The active reproduction retry route is the optimizer-microbatch route: dataloader batch size is `10`, `training.update_density.optimizer_microbatch_size` is `1`, and `num_epochs` is `30`.
- In optimizer-microbatch mode, each local batch is split into optimizer microbatches and `CosineAnnealingLR` steps once per optimizer update, so the run gets roughly the same optimizer-update budget as a 300-epoch batch-10 run without requiring 300 dataset passes.
- A bounded transfer-learning microbatch retry may use `training.optimizer.schedule_policy: hold_then_cosine`, positive `training.optimizer.lr_hold_epochs`, and more than `30` but fewer than `300` epochs. This keeps the model in the useful LR region longer and then decays, without returning to the inefficient 300-epoch route.
- For the 993 transfer LR-hold retry, pass only if `training.acceptance.min_accuracy` is exceeded. The current threshold is `0.8501`, so the run must beat the previous 300-epoch real-only diagnostic accuracy of `0.8500`.
- A bounded lower-start-LR transfer retry may set `training.acceptance.allow_lr_override: true` with `0 < lr <= 1e-5`. This retry must keep the same optimizer, microbatch, data, label-smoothing, and artifact requirements, and must still beat the explicit `0.8501` threshold when that threshold is configured.
- A raw 993 LR-hold retry may use the same bounded schedule without synthetic pretraining: `curr_bool: false`, optimizer microbatch size `1`, `num_epochs: 60`, `schedule_policy: hold_then_cosine`, and `lr_hold_epochs: 12`.
- A user-requested long transfer LR-hold diagnostic may set `training.acceptance.allow_patience_override: true` with patience greater than `10`, while keeping the same optimizer, LR, microbatch, data, label-smoothing, and synthetic-pretrained initialization. This run is accepted only against its explicit `training.acceptance.min_accuracy`; the current long-transfer threshold is `0.9001`, meaning it must exceed `90%` test accuracy and beat the best real-only 993 LR-hold result of `0.8880`.
- Do not continue the 300-epoch route as the default reproduction retry route. Treat it only as diagnostic evidence unless this checker standard is explicitly updated before launch.
- Per-GPU training batch size is `10` on the 4x H200 pod, for an effective DDP training batch size of `40`.
- `num_workers` is `0` to avoid Kubernetes pod shared-memory exhaustion from large video tensors.
- Seed is `1205`; manifest generation seed is `37`.
- Checkpoint path is under `outputs/rebuild_reproduction/checkpoints`.
- Outputs are under `outputs/rebuild_reproduction/<run_name>/`.
- No generated outputs, checkpoints, W&B logs, or datasets are committed.

Known methodology caveats:

- Old synthetic W&B run used `dataset/CFDArchive/sph_realvisc_diffback_35000`; this repo currently has `dataset/CFDArchive/sph_35000`.
- Old W&B configs used per-GPU `batch_size: 1`; this rebuild intentionally uses per-GPU dataloader `batch_size: 10` because the active pod has four H200 GPUs with enough memory. Treat this as `current-repo reproduction`, not `exact reproduction`.
- The first 30-epoch real-only sweep is a failed diagnostic baseline because `batch_size: 10` also made the optimizer batch large and cut optimizer updates per epoch.
- The previous long-horizon `realonly_993` pass, `0.8500` observed accuracy versus the `0.7230` target, is diagnostic evidence that update density/training horizon can recover the 993 real-only result. It is not the new default route.
- The active retry returns to `num_epochs: 30` while restoring old-style update density with `optimizer_microbatch_size: 1`.
- The transfer LR-hold retry is a bounded scheduler retry, not a 300-epoch route. It must report `num_epochs`, `lr_hold_epochs`, optimizer updates per epoch, total planned optimizer updates, and the explicit `0.8501` pass threshold.
- The lower-start-LR transfer retry is an optimization-dynamics diagnostic for transfer overfitting. It must report the starting LR, LR hold, decay policy, stop reason, best epoch, and whether class `1`/`2` confusions improve without dropping below `0.8501`.
- The raw 993 LR-hold retry is the same bounded scheduler retry without synthetic initialization. It must report the same update-count fields and is compared against the real-only 993 target unless a stricter explicit threshold is set before launch.
- The long transfer LR-hold retry is a threshold diagnostic, not a new baseline curve. It must report `num_epochs`, `patience`, `lr_hold_epochs`, stop reason, best validation epoch, whether accuracy exceeds `0.9000`, and whether it improves class `1/2/3` confusions relative to `repro_transfer_993_microbatch_lrhold`.
- The matched long real-only LR-hold diagnostic uses the same optimizer, microbatching, `num_epochs`, patience, and LR-hold schedule as the successful `repro_transfer_993_microbatch_lrhold_ep90_pat25` run, but keeps `curr_bool: false`. It is compared against the standard real-only 993 target and separately analyzed against the transfer result `0.9020`; exceeding or failing to exceed `90%` is diagnostic evidence, not a retroactive reproduction pass/fail rule.
- The batch-8 warmup/hold/cosine diagnostics are non-microbatch runs requested to test whether fewer sequential optimizer steps can be faster while preserving useful LR exposure. They use `batch_size: 8`, no `training.update_density.optimizer_microbatch_size`, `num_epochs: 100`, `schedule_policy: warmup_hold_cosine`, positive `warmup_epochs`, positive `lr_hold_epochs`, and bounded override flags for batch size, epoch count, and patience. The real-only run remains judged against the standard real-only 993 target and is separately compared with `90%`; the transfer run may use an explicit `min_accuracy: 0.9001` if the question is whether this faster regime still exceeds `90%`.
- The short-warmup batch-8 retry fixes the scheduler unit bug before launch and uses the microbatch LR-hold shape without optimizer microbatching: `warmup_epochs: 1`, `warmup_start_factor: 0.5`, `lr_hold_epochs: 18`, `lr: 1e-5`, and `eta_min: 1e-10`.
- Before launching the 30-frame windowed real-only diagnostic, run the gated transfer LR-hold retry `repro_transfer_993_microbatch_lrhold_ep70_min87` and continue only if it exceeds `87%` accuracy.
- The 30-frame windowed real-only diagnostic expands only the 993 real training set into 21 fixed temporal windows per original video. It must keep `/Viscnet/dataset` untouched, write derived clips under `outputs/rebuild_reproduction/derived_datasets/`, set `model.transformer.num_frames: 30`, use `use_all_samples: true`, train for `50` epochs, and compare against the best real-only microbatch LR-hold baseline `0.8880`.
- Old W&B configs used DataLoader workers; this rebuild intentionally uses `num_workers: 0` because the active pod hit `/dev/shm` bus errors with worker multiprocessing.
- Current model code gates all pattern-feature paths behind `pat_bool`; reproduction configs keep `pat_bool: false`.
- Old pattern reference used unavailable `real_20rpm_increment_1back` / `real_20rpm_increment_4back` paths and unavailable `diffback_basetrain_256-8-1024_augTrue_0917_v0.pth`.
- Current pattern configs are therefore `current-repo reproduction`, not exact original reproduction.

## Reference Targets

Reference metrics are also stored in `configs/rebuild/reference_metrics.json`.

Real-only data-efficiency targets:

| Samples | W&B Run | Target Accuracy |
| ---: | --- | ---: |
| 300 | `dataefficiency/g5nheksd` | 0.3180 |
| 400 | `dataefficiency/883f5m6k` | 0.4210 |
| 500 | `dataefficiency/dp5p39bc` | 0.4860 |
| 600 | `dataefficiency/utgbwbeu` | 0.5210 |
| 700 | `dataefficiency/l4ibkqqg` | 0.6010 |
| 800 | `dataefficiency/hq4o5lfo` | 0.6460 |
| 900 | `dataefficiency/l7zrpyt7` | 0.6950 |
| 993 | `dataefficiency/i58ha0w0` | 0.7230 |

Synthetic-pretrained + real fine-tuning targets:

| Samples | W&B Run | Target Accuracy |
| ---: | --- | ---: |
| 300 | `dataefficiency/2fn4fl9b` | 0.6020 |
| 400 | `dataefficiency/ph6rwzbz` | 0.6930 |
| 500 | `dataefficiency/jvqqegb9` | 0.7160 |
| 600 | `dataefficiency/1gqccpob` | 0.7350 |
| 700 | `dataefficiency/3mn3dnfp` | 0.7720 |
| 800 | `dataefficiency/rznnlkan` | 0.8010 |
| 900 | `dataefficiency/zhc7d47c` | 0.7970 |
| 993 | `dataefficiency/6r1p7fet` | 0.8090 |

Pattern reference:

- `sph_test_run/p5iarxc7`, `5thpatVal_finetuning_0929_v0`, target accuracy `0.8421`.

Legacy synthetic pretraining reference:

- `sph_test_run/mffvc6vq`, `35000_weightmaking_smallerVivit_layer10_1024_v0`.
- Final train loss: `0.3351`.
- Final validation loss: `0.6146`.
- This legacy run used the missing synthetic dataset path `dataset/CFDArchive/sph_realvisc_diffback_35000`.
- Because the exact dataset is absent, these losses are not hard pass/fail criteria for the current `dataset/CFDArchive/sph_35000` pretrain. They are still the expected loss region: a current synthetic pretrain or long real-only retry should move at least somewhere near this regime before treating architecture or data changes as the primary fix.

## Update-Density Retry Hypotheses

The failed 30-epoch real-only baseline exposed an update-density problem: the real datasets are small while per-GPU batch size is large, so each epoch has few optimizer updates. Low gradient noise from a large effective batch does not compensate for too few update steps.

Allowed retry hypotheses after verifier evidence shows under-training:

- Large H200 VRAM should be used to stage video batches efficiently, not to make the optimizer batch statistically huge when the dataset is small.
- `training.update_density.optimizer_microbatch_size` is the focused retry control. It keeps the dataloader batch size large, then slices each local training batch along the batch dimension. Each slice runs its own forward pass, loss, backward pass, optimizer step, and zero-grad call.
- This is not gradient accumulation. Each optimizer microbatch must produce its own optimizer update.
- If `optimizer_microbatch_size` is absent, or greater than or equal to the current local dataloader batch size, training preserves the existing one optimizer step per dataloader batch behavior.
- This increases optimizer update density without repeating the whole loader and should preserve more staging throughput than lowering the dataloader batch size directly.
- Based on the long-horizon `realonly_993` diagnostic, validation loss reached the useful range while the 300-epoch cosine LR was still high. This supports the optimizer-microbatch route, but does not make the 300-epoch route the active default. The next retry should use `optimizer_microbatch_size: 1`, `num_epochs: 30`, LR `1e-5`, and `eta_min: 1e-10`. In optimizer-microbatch mode, the scheduler steps once per optimizer update, so the LR reaches its intended final state over the denser update sequence while the model still gets at least the intended update count.
- Adaptive optimizer variants beyond current AdamW, SAM, Lookahead, stronger augmentation, or label smoothing are enhancement/debug variants. They may be proposed by the analyzer, but they are blocked as valid reproduction until the checker standard is updated before launch.
- SAM and Lookahead are plausible stability/generalization retries for small-data regimes, but they add extra forward/backward or slow-weight dynamics and must be reported separately from deterministic reproduction.

Any update-density retry report must state:

- dataloader batch size
- optimizer microbatch size
- optimizer updates per epoch
- total planned optimizer updates
- why the variant is a reproduction retry rather than an enhancement

## Data-Efficiency Gate

Before launching or accepting the full real-only and transfer data-efficiency curves, run and check the 993-sample pair first:

- `993` real-only without synthetic pretraining, using the optimizer-microbatch route.
- `993` synthetic-pretrained transfer, using the optimizer-microbatch route.

The checker must apply the same pass/fail targets and artifact requirements to this gate:

- The real-only 993 run must satisfy the `0.7230` target.
- The transfer 993 run must satisfy the `0.8090` target.
- The transfer 993 accuracy must be equal to or greater than the real-only 993 accuracy.
- Both runs must have the required checkpoint, W&B config, confusion matrix, reliability plot, and JSON metrics.

For the transfer-learning LR-hold retry requested after the completed 993 gate:

- Use synthetic-pretrained `993` transfer with optimizer microbatching.
- Do not use 300 epochs.
- Keep LR high longer with `schedule_policy: hold_then_cosine`, then decay.
- Pass only if accuracy exceeds the previous 300-epoch real-only diagnostic accuracy, i.e. `accuracy >= 0.8501`.
- If this run fails, do not count the old `0.8190` transfer pass as satisfying the new user-requested threshold.

As of 2026-05-19, the user selected the completed 30-epoch microbatch 993 pair
as the working baseline for constructing the data-efficiency curve:

- `realonly_993`: accuracy `0.7190`, checker status `fail` against the legacy
  target `0.7230`.
- `transfer_993`: accuracy `0.8190`, checker status `pass` against the legacy
  target `0.8090`.

This authorizes running the remaining `300, 400, 500, 600, 700, 800, 900`
real-only and transfer configs with the same 30-epoch microbatch route. It does
not change the pass/fail target for any point, and it does not allow diagnostic
300-epoch or 60-epoch LR-hold artifacts to be substituted into the baseline
curve.

## Required Artifacts

For every completed run, verify:

- Checkpoint exists at the configured `training.checkpoint_name` under configured `misc_dir.ckpt_root`.
- W&B run exists and has the matching config.
- Training log contains completion or a clear early-stopping event.
- `outputs/rebuild_reproduction/<run_name>/confusion_matrix/<run_name>.png` exists for classification tests.
- `outputs/rebuild_reproduction/<run_name>/confusion_matrix/<run_name>_metrics.json` exists and contains `accuracy`.
- `outputs/rebuild_reproduction/<run_name>/reliability_plots/<run_name>.png` exists.
- `outputs/rebuild_reproduction/<run_name>/reliability_plots/<run_name>_metrics.json` exists and contains `ece` and `mce`.
- Per-class support and per-class accuracy are present in the confusion metrics JSON.

## Pass/Fail Rule

A run passes only if:

- Methodology fit is `exact reproduction` or `current-repo reproduction`.
- Accuracy is equal to or greater than the reference target for that run family and sample count.
- Required artifacts are present.
- Any current-repo substitution is explicitly documented.

A batch passes only if:

- Every required sample count has one real-only run and one transfer run.
- The transfer accuracy is equal to or greater than the real-only accuracy at the same sample count.
- The 993 transfer run reaches at least `0.8090`.
- The full data-efficiency curve was launched only after the 993 optimizer-microbatch gate was checked and supported by checker/analyzer evidence.
- Pattern runs are reported separately because imported pattern splits are not exact old W&B paths.

## Checker Output

Write checker results to:

- `outputs/rebuild_reproduction/checker_report.md`
- `outputs/rebuild_reproduction/metrics_table.json`

The report must include:

- Run name, config path, W&B run ID, checkpoint path.
- Methodology-fit label and deviation notes.
- Accuracy target, observed accuracy, pass/fail.
- Presence of confusion matrix, reliability plot, and JSON metrics.
- Final recommendation: accepted, retry required, or blocked by missing original data/checkpoint.
