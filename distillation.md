# Viscnet Final Handoff

Last updated: 2026-05-21 UTC.

This is the canonical handoff for a fresh agent. It merges the previous `REBUILD_REPRODUCTION.md` runbook with the current experiment summary, GitHub state, raw-video config work, and token-control rules. After the repo move, start from `/root/Viscnet`.

## Non-Negotiable Rules

- Read `AGENTS.md` first. If it still references `/Viscnet`, interpret the active root as `/root/Viscnet` after this move.
- Do not delete, reformat, replace, commit, or upload `dataset/`, `rawdataset/`, `outputs/`, checkpoints, videos, W&B logs, caches, or private data.
- `rawdataset/`, `dataset/RealArchive/`, `dataset/CFDArchive/`, and `outputs/` must stay ignored. The raw dataset must never be pushed.
- Do not use Codex as a training monitor. Launch training plus a detached watcher/finalizer, then stop. Only read bounded artifacts after completion.
- Do not repeatedly poll `tmux`, `nvidia-smi`, `ps`, `tail`, `grep`, `rg`, checker, or analyzer while training is active.

## Git And Workspace State

- GitHub repo: `https://github.com/erstool37/Viscnet.git`
- Branch: `main`
- Remote HEAD after message rewrite:

```text
8662716 edit: add rebuild inference and config tooling
779d9bb edit: move runtime secrets to env file
f1e1c0d init: refactor
```

- Pushed code excludes raw data, generated outputs, checkpoints, `.mov`, `.mp4`, `.pth`, and `.zip`.
- Local-only unstaged changes existed before this handoff and should not be assumed intentional for future commits:
  - `.env.example` deleted locally
  - `.gitignore` modified locally
  - `requirements.txt` modified locally
- `gh` is installed and authenticated in this pod. Credentials were saved in plain text because no secure credential store exists. Logout or revoke the token when GitHub work is finished.

## Current Code And Tooling

Multi-GPU/multi-batch inference and analysis are implemented.

- `src/main.py`: DDP final test inference uses all ranks, configurable test batch size, rank-0 gather/dedupe, and single writer behavior.
- `scripts/window21_test_inference.py`: supports `torchrun`, rank sharding, config defaults, multi-window batching, and rank-0 metrics/report writing.
- `scripts/analyze_993_attention.py`: supports `torchrun`, rank-sharded attention analysis, and configurable `--batch-size`.
- `scripts/build_raw_realvideo_configs.py`: builds metadata JSON configs for raw MOV archives without modifying videos.
- `pyproject.toml`: ruff config for repo linting.

Validation already run:

```bash
python -m ruff check .
python -m ruff format . --check
python -m compileall -q src scripts
torchrun --nproc_per_node=4 scripts/window21_test_inference.py --max-videos 4 --num-windows 2 --full-frames 31 --window-size 30 --window-batch-size 2 --output-dir outputs/rebuild_reproduction/tmp_window21_ddp4_smoke
torchrun --nproc_per_node=4 scripts/analyze_993_attention.py --max-samples 4 --batch-size 1 --output-root outputs/rebuild_reproduction/tmp_attention_ddp4_smoke_after_barrier_patch
```

## Rebuild Runbook

Rebuild scope is classification weights only. Excluded by default: label smoothing, architecture changes, uncertainty/regression variants, and unlabeled enhancement tuning.

Important files:

- Rebuild configs: `configs/rebuild/*.yaml`
- Retry configs: `configs/rebuild/retries/*.yaml`
- Real-data manifests: `configs/rebuild/manifests/*.json`
- Reference metrics: `configs/rebuild/reference_metrics.json`
- Checker: `scripts/check_rebuild_results.py`
- Analyzer: `scripts/analyze_rebuild_results.py`
- Runner: `scripts/run_rebuild_reproduction.sh`
- Output root: `outputs/rebuild_reproduction/`

Hardware assumptions:

- Current pod had four H200 GPUs.
- For 8 H200 pods, use `torchrun --nproc_per_node=8` for inference/analysis.
- Active microbatch route uses dataloader batch size `10`, optimizer microbatch size `1`, `num_workers: 0`, `rpm_bool: true`, and `pat_bool: false`.

Training order:

1. Preserve or produce `outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_sph35000.pth`.
2. Run the 993 real-only and transfer gate first.
3. Run checker and analyzer.
4. Only after the 993 pair is accepted, run the remaining real-only and transfer data-efficiency curve.
5. Run pattern-generalization configs separately.

Launch only when explicitly requested:

```bash
cd /root/Viscnet
REBUILD_MIDRUN_ANALYSIS=1 NPROC_PER_NODE=4 MASTER_PORT=29513 bash scripts/run_rebuild_reproduction.sh
```

Checker and analyzer:

```bash
python scripts/check_rebuild_results.py
python scripts/analyze_rebuild_results.py
```

The analyzer is downstream of `checklist.md`; it may propose hypotheses but must not redefine pass/fail validity.

## Current Experiment Summary

Key 993 results on the 1000-video test set:

| Run | Accuracy | Note |
| --- | ---: | --- |
| `repro_realonly_993_microbatch` | `0.719` | early 30-epoch microbatch baseline |
| `repro_transfer_993_microbatch` | `0.819` | transfer baseline |
| `repro_realonly_993_microbatch_lrhold` | `0.888` | strong real-only lr-hold result |
| `repro_transfer_993_microbatch_lrhold` | `0.872` | transfer lr-hold |
| `repro_transfer_993_microbatch_lrhold_ep70_min87` | `0.880` | transfer gate run |
| `repro_transfer_993_microbatch_lrhold_ep90_pat25` | `0.902` | best transfer result so far |
| `repro_realonly_993_batch8_shortwarm_lrhold_ep100` | `0.827` | no-microbatch batch-8 retry underperformed |
| `repro_transfer_993_batch8_shortwarm_lrhold_ep100` | `0.812` | no-microbatch transfer retry underperformed |
| `repro_realonly_993_window30x21_ep50` | `0.898` | trained on 21 fixed 30-frame windows per real video, first-30 test |
| `window21_test_inference` on that checkpoint | `0.940` | averages 21 contiguous 30-frame test logits per original video |

Main interpretation:

- LR-hold and sufficient optimizer updates matter more than blind epoch increases.
- Batch-8 no-microbatch retries did not match the strong microbatch lr-hold behavior.
- 30-frame window training plus 21-window averaged-logit inference is the strongest current real-only path.
- Real-only beating many transfer runs is evidence to investigate synthetic-real mismatch and schedule sensitivity, not proof that synthetic data is categorically harmful.

## 21-Window Inference

Final restored output:

```text
outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_test_inference/
```

Final metrics:

- Accuracy: `0.9400`
- Evaluated samples: `1000`
- Confusion-matrix total: `1000`
- Windows per video: `21`
- Window size: `30`
- Full frames read: `50`
- Window batch size: `16`

Config block lives in:

```text
configs/rebuild/retries/realonly_993_window30x21_ep50.yaml
```

For an 8 H200 pod, start with:

```bash
MASTER_PORT=29630 PYTHONPATH=src torchrun --nproc_per_node=8 --master_port=29630 scripts/window21_test_inference.py
MASTER_PORT=29631 PYTHONPATH=src torchrun --nproc_per_node=8 --master_port=29631 scripts/analyze_993_attention.py --batch-size 1
```

If memory is still low, increase `inference.temporal_window.window_batch_size` from `16` to `24` or `32` for inference only.

## Raw Video Configs

Raw MOV archives were not modified.

Generated config locations:

- `rawdataset/rawvideos/impeller_1000_originals/configs/` - 1000 configs
- `rawdataset/rawvideos/raw_real_20rpmincrement_1500/configs/` - 1500 configs
- `dataset/RealArchive/real_20rpm_increment_2500/parameters_rebuilt_from_raw/` - 2500 consolidated configs
- `dataset/RealArchive/real_20rpm_increment_2500/raw_config_manifest.json`
- `dataset/RealArchive/real_20rpm_increment_2500/raw_config_build_report.json`

Mapping logic:

- `impeller_1000_originals`: source files `0000.mov` to `999.mov`, renders `A-J`.
- `raw_real_20rpmincrement_1500`: source files `0001.mov` to `1500.mov`, renders `K-Y`.
- RPM cycles `270, 290, ..., 450`; render advances every 10 RPMs.
- Viscosity block advances by sorted source order.

Observed final-dataset issue and exclusion rule:

- Existing `dataset/RealArchive/real_20rpm_increment_2500/parameters/` has `2493` configs, not `2500`.
- It has mixed key schemas: `1494` configs include `dynamic_viscosity_str`, `RPM_index`, `RENDER`, `INDEX`; `999` configs omit those keys.
- The seven missing stems are connected to contaminated source videos with mixing and must not be used for training, validation, testing, or future dataset construction.
- `scripts/build_raw_realvideo_configs.py` marks these seven generated config records with `usable: false`, `use_in_training: false`, and `exclude_reason: contaminated_source_video_mixing`.
- The raw source videos to exclude are:
  - `rawdataset/rawvideos/impeller_1000_originals/204.mov` -> `decay_10fps_visc090.13457_rpm350_renderA`
  - `rawdataset/rawvideos/raw_real_20rpmincrement_1500/0574.mov` -> `decay_10fps_visc067.55088_rpm330_renderW`
  - `rawdataset/rawvideos/raw_real_20rpmincrement_1500/0761.mov` -> `decay_10fps_visc037.94112_rpm270_renderL`
  - `rawdataset/rawvideos/raw_real_20rpmincrement_1500/0901.mov` -> `decay_10fps_visc028.43477_rpm270_renderK`
  - `rawdataset/rawvideos/raw_real_20rpmincrement_1500/1122.mov` -> `decay_10fps_visc021.31029_rpm290_renderR`
  - `rawdataset/rawvideos/raw_real_20rpmincrement_1500/1304.mov` -> `decay_10fps_visc015.97088_rpm330_renderU`
  - `rawdataset/rawvideos/raw_real_20rpmincrement_1500/1461.mov` -> `decay_10fps_visc000.89274_rpm270_renderV`
- After excluding these contaminated records, `raw_config_build_report.json` reports no missing active rebuilt videos/configs.

## Known Caveats And Next Work

- Old W&B synthetic pretrain used `dataset/CFDArchive/sph_realvisc_diffback_35000`; this repo uses available `dataset/CFDArchive/sph_35000`.
- Old pattern references depend on unavailable old dataset/checkpoint paths.
- Treat the 30-frame temporal window method as an enhancement unless explicitly redefined as the new baseline.
- Run transfer-side window training only if the research question is whether synthetic pretraining helps under identical temporal-window augmentation.
- Before any future push, inspect staged scope and exclude `dataset/`, `rawdataset/`, `outputs/`, checkpoints, W&B data, videos, and caches.
