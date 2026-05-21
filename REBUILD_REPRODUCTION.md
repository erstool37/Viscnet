# Viscnet Paper-Reproduction Runbook

This runbook is for rebuilding classification weights only. It deliberately excludes
enhancement tuning, label smoothing, architecture changes, and UQ/regression.

## Current State

- Rebuild configs: `configs/rebuild/*.yaml`
- Real-data subset manifests: `configs/rebuild/manifests/*.json`
- Checker checklist: `checklist.md`
- Reference metrics: `configs/rebuild/reference_metrics.json`
- Checker script: `scripts/check_rebuild_results.py`
- Sequential runner: `scripts/run_rebuild_reproduction.sh`
- Output root: `outputs/rebuild_reproduction/`

As of 2026-05-19 UTC, the long-horizon `realonly_993` diagnostic completed and
passed the checker target: observed accuracy `0.8500` against target `0.7230`.
That result is diagnostic evidence for the update-density hypothesis, not the
default route going forward.

The active route is now optimizer microbatching: dataloader batch size `10`,
`training.update_density.optimizer_microbatch_size: 1`, and `num_epochs: 30`.
Microbatch configs write distinct checkpoints/logs/outputs with a `_microbatch`
run suffix so old 300-epoch artifacts cannot be credited to new microbatch runs.

The previous detached 300-epoch retry session `rebuild_realonly_retry` was
stopped after the route change. The 993 microbatch gate then completed:

- `repro_realonly_993_microbatch`: checker `fail`, accuracy `0.7190` vs target `0.7230`.
- `repro_transfer_993_microbatch`: checker `pass`, accuracy `0.8190` vs target `0.8090`.

Checker/analyzer were regenerated after the gate. The 30-epoch microbatch 993
pair is now the user-selected baseline for the data-efficiency curve. This does
not retroactively make the real-only 993 point pass the legacy target; it means
the curve should use those 30-epoch microbatch artifacts as its baseline instead
of the diagnostic 300-epoch or 60-epoch LR-hold retries.

The transfer-learning LR-hold retry then completed:

- Config: `configs/rebuild/retries/transfer_993_microbatch_lrhold.yaml`
- Schedule: optimizer microbatch size `1`, `num_epochs: 60`, `schedule_policy: hold_then_cosine`, `lr_hold_epochs: 12`
- Pass threshold: `accuracy >= 0.8501`, so it must beat the previous 300-epoch real-only diagnostic accuracy `0.8500`
- Checker status: `pass`
- Accuracy: `0.8720`
- W&B run: `qykghfy9`
- Best validation loss: `0.3950` at epoch `44`; early stopping at epoch `54`

The raw 993 LR-hold retry then completed:

- Config: `configs/rebuild/retries/realonly_993_microbatch_lrhold.yaml`
- Schedule: optimizer microbatch size `1`, `num_epochs: 60`, `schedule_policy: hold_then_cosine`, `lr_hold_epochs: 12`
- Initialization: raw real-only, `curr_bool: false`
- Checker status: `pass`
- Accuracy: `0.8880` against the real-only 993 target `0.7230`
- W&B run: `gdzire3o`
- Best validation loss: `0.3174` at epoch `60`; training completed without early stopping

Completed 30-epoch microbatch data-efficiency curve:

- Completed: 2026-05-19 UTC
- tmux session: `rebuild_data_efficiency_30ep` completed and is no longer active
- completed real-only configs: `realonly_300`, `realonly_400`, `realonly_500`,
  `realonly_600`, `realonly_700`, `realonly_800`, `realonly_900`; the completed
  `realonly_993` microbatch gate is reused as the baseline endpoint
- completed transfer configs: `transfer_300`, `transfer_400`, `transfer_500`,
  `transfer_600`, `transfer_700`, `transfer_800`, `transfer_900`; the completed
  `transfer_993` microbatch gate is reused as the baseline endpoint
- checker final recommendation: `retry required`
- real-only passes: `300`, `600`, `700`, `800`, `900`; real-only fails:
  `400`, `500`, `993`
- transfer passes: `600`, `900`, `993`; transfer fails:
  `300`, `400`, `500`, `700`, `800`
- transfer accuracy is higher than real-only at every sample count, with mean
  gain `+0.161`
- graph/data outputs:
  `outputs/rebuild_reproduction/data_efficiency_curve.png`,
  `outputs/rebuild_reproduction/data_efficiency_metrics.csv`, and
  `outputs/rebuild_reproduction/data_efficiency_metrics.json`

No rebuild training process is active as of this update. Check process state
again before launching the next retry.

In future sessions, do not assume this snapshot is still live. Check before
starting or replacing training:

```bash
ps -eo pid,ppid,cmd | grep -E 'torchrun|src/main.py' | grep -v grep
tmux ls
```

## Hardware Assumption

The active pod has four H200 GPUs. Generated configs use:

- `NPROC_PER_NODE=4`
- per-GPU dataloader `batch_size: 10`
- optimizer microbatch size `1` for active retry configs, yielding old-style small optimizer batches while still staging larger video batches in H200 VRAM
- `num_workers: 0` to avoid pod shared-memory exhaustion from DataLoader worker multiprocessing
- `rpm_bool: true` and `pat_bool: false`; the model uses video plus RPM conditioning only
- default `num_epochs: 30` for active microbatch retry configs
- in optimizer-microbatch mode, `CosineAnnealingLR` steps once per optimizer update and reaches its final low-LR state over the denser update budget, yielding roughly the optimizer-update budget of a 300-epoch batch-10 run without requiring 300 dataset passes

This differs from old W&B configs that used per-GPU `batch_size: 1`, so the checker
marks rebuilt runs as `current-repo reproduction` rather than exact reproduction.

## Training Order

Run jobs sequentially in this order:

1. Preserve or produce `outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_sph35000.pth`.
2. Run the 993 microbatch gate:
   - `configs/rebuild/realonly_993.yaml`
   - `configs/rebuild/transfer_993.yaml`
3. Run checker plus analyzer and compare 993 real-only versus 993 transfer.
4. After the user accepts the 30-epoch microbatch 993 pair as the working
   baseline, run the remaining real-only and transfer data-efficiency curves.
5. Run imported pattern-generalization configs separately.

The transfer and pattern configs depend on:

```text
outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_sph35000.pth
```

Before full data-efficiency runs, inspect the 993 confusion matrices and analyzer output.
The first completed real-only sweep failed all reference points because the
dataloader `batch_size: 10` also became the optimizer batch, cutting optimizer
updates per epoch by about 10x. The long-horizon `realonly_993` diagnostic showed
good validation loss while the 300-epoch cosine scheduler was still near its
initial LR. Therefore the next retry should use optimizer microbatching while
preserving the optimizer-update budget without using 300 epochs:

1. Use dataloader `batch_size: 10`, `training.update_density.optimizer_microbatch_size: 1`, LR `1e-5`, `eta_min: 1e-10`, and `num_epochs: 30`.
2. Run `realonly_993` without synthetic and `transfer_993` with synthetic first.
3. If the user accepts the 993 pair as the working baseline after checker and
   analyzer review, rerun the full data-efficiency curve with the same
   microbatch schedule.
4. Do not launch the full data-efficiency curve before this 993 pair is checked.

## Launch Command

Only start this when the user explicitly wants training to run:

```bash
tmux new-session -d -s viscnet_rebuild \
  'cd /Viscnet; REBUILD_MIDRUN_ANALYSIS=1 NPROC_PER_NODE=4 MASTER_PORT=29513 bash scripts/run_rebuild_reproduction.sh'
```

The default runner now launches only the 993 microbatch gate:

```bash
bash scripts/run_rebuild_reproduction.sh
```

For the full microbatch data-efficiency curve after the 993 gate is analyzed,
reuse completed 993 baseline artifacts unless intentionally re-baselining:

```bash
REBUILD_CONFIGS="configs/rebuild/realonly_300.yaml configs/rebuild/realonly_400.yaml configs/rebuild/realonly_500.yaml configs/rebuild/realonly_600.yaml configs/rebuild/realonly_700.yaml configs/rebuild/realonly_800.yaml configs/rebuild/realonly_900.yaml configs/rebuild/transfer_300.yaml configs/rebuild/transfer_400.yaml configs/rebuild/transfer_500.yaml configs/rebuild/transfer_600.yaml configs/rebuild/transfer_700.yaml configs/rebuild/transfer_800.yaml configs/rebuild/transfer_900.yaml" \
REBUILD_MIDRUN_ANALYSIS=1 \
NPROC_PER_NODE=4 \
MASTER_PORT=29519 \
bash scripts/run_rebuild_reproduction.sh
```

The active graph can be refreshed manually with:

```bash
python scripts/plot_data_efficiency.py
```

To run checker plus analyzer after each config:

```bash
REBUILD_MIDRUN_ANALYSIS=1 bash scripts/run_rebuild_reproduction.sh
```

Monitor:

```bash
tmux capture-pane -t viscnet_rebuild -p | tail -120
tail -f outputs/rebuild_reproduction/logs/synthetic_pretrain.log
```

Stop:

```bash
tmux kill-session -t viscnet_rebuild
```

## Checker Workflow

Run the checker after any batch completes:

```bash
python scripts/check_rebuild_results.py
```

Outputs:

- `outputs/rebuild_reproduction/checker_report.md`
- `outputs/rebuild_reproduction/metrics_table.json`

Before training, all planned jobs should be `pending`. After training, a run passes
only if required artifacts exist and observed accuracy is equal to or better than
the reference target in `configs/rebuild/reference_metrics.json`. The checker also
parses the W&B run ID from each run log and, when `.env` provides `WANDB_API_KEY`,
confirms that the run exists in `jongwonsohn-seoul-national-university/viscnet-rebuild`.

## Analyzer Workflow

Run the analyzer only after the checker has written `checker_report.md` and
`metrics_table.json`:

```bash
python scripts/analyze_rebuild_results.py
```

Output:

- `outputs/rebuild_reproduction/analyzer_report.md`

The analyzer is downstream of `checklist.md`; it must not redefine pass/fail
validity. It reviews local run logs, confusion metrics, reliability metrics,
available W&B reference summaries, and optional attention/gradient evidence.
If attention maps or gradient traces are absent, it must report that absence and
recommend a separate diagnostic rerun instead of inferring those results.

Use the analyzer report to form debugging hypotheses about model quality,
training dynamics, calibration, per-class confusions, synthetic-real mismatch,
data-efficiency transfer gains, and follow-up diagnostic experiments. These
hypotheses are for later improvement work and do not change the reproduction
standard.

During training, the analyzer should explicitly check whether the current run is
under-trained for the chosen batch size. Warning signs include validation loss
still dropping sharply late in training, best validation loss at the final epochs,
and cosine LR already near zero. In that case the first retry is a longer horizon
or slower LR schedule, not an architecture change. During long single runs, inspect
the active log around epochs 75, 150, and 225 rather than waiting for the run to
finish.

## Known Reproduction Caveats

- Old W&B synthetic pretrain used `dataset/CFDArchive/sph_realvisc_diffback_35000`.
  This repo currently has `dataset/CFDArchive/sph_35000`.
- Old pattern reference used unavailable `real_20rpm_increment_1back`,
  `real_20rpm_increment_4back`, and `diffback_basetrain_256-8-1024_augTrue_0917_v0.pth`.
- Therefore, current configs are paper-methodology rebuilds using available repo data,
  not exact byte-for-byte old-run reproduction.
