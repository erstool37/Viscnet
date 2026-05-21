# Viscnet Current Handoff

Last updated: 2026-05-21 UTC.

This is the concise starting document for a fresh agent in `/Viscnet`. Read `AGENTS.md` first. That file is authoritative for repo safety, dataset safety, and token-control rules.

## Hard Rules

- Do not delete, move, reformat, or commit `/Viscnet/dataset`, `/Viscnet/rawdataset`, checkpoints, generated outputs, `wandb/`, caches, or private data.
- Do not move `/Viscnet` or push to GitHub until the rawdataset transfer is verified complete.
- Rawdataset transfer status at this handoff: no `tar -xf - -C /Viscnet/rawdataset` process is active, but `outputs/rebuild_reproduction/notifications/rawdataset_transfer_complete.txt` is missing. Treat move/push as blocked until this is resolved.
- Do not monitor training from inside Codex with repeated polling. Launch training plus a detached watcher/finalizer, then stop. After completion, read only exact bounded artifacts.
- For active training, do not loop on `write_stdin`, `tmux capture-pane`, `nvidia-smi`, `ps`, `tail`, `grep`, `rg`, checker, or analyzer.

## Current Code State

Multi-GPU and multi-batch inference has been implemented and validated.

Important files:

- `src/main.py`: DDP final test inference uses all ranks, configurable test batch size, rank-0 gather/dedupe, and single writer behavior.
- `scripts/window21_test_inference.py`: supports `torchrun`, rank sharding, multi-window batching, config defaults, and rank-0 metric/report writing.
- `scripts/analyze_993_attention.py`: supports `torchrun`, rank-sharded attention analysis, and configurable `--batch-size`.
- `src/utils/ddp.py`: DDP gather helper cleanup.
- `configs/rebuild/retries/realonly_993_window30x21_ep50.yaml`: contains the visible inference block.
- `pyproject.toml`: ruff config for repo linting.

Validation already run:

```bash
python -m ruff check .
python -m ruff format . --check
python -m compileall -q src scripts
torchrun --nproc_per_node=4 scripts/window21_test_inference.py --max-videos 4 --num-windows 2 --full-frames 31 --window-size 30 --window-batch-size 2 --output-dir outputs/rebuild_reproduction/tmp_window21_ddp4_smoke
torchrun --nproc_per_node=4 scripts/analyze_993_attention.py --max-samples 4 --batch-size 1 --output-root outputs/rebuild_reproduction/tmp_attention_ddp4_smoke_after_barrier_patch
```

## Restored 21-Window Inference

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

The accidental 2-sample smoke metrics were moved out of the final path:

```text
outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_test_inference_smoke_bad_20260521/
```

The 4-GPU restore command used:

```bash
MASTER_PORT=29622 PYTHONPATH=src torchrun --nproc_per_node=4 --master_port=29622 scripts/window21_test_inference.py --output-dir outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_test_inference_full_restore_tmp
```

The temp output was verified for `1000/1000` and then moved into the final output path.

## Config-Visible Inference Settings

The active config is:

```text
configs/rebuild/retries/realonly_993_window30x21_ep50.yaml
```

It includes:

```yaml
inference:
  temporal_window:
    enabled: true
    average_logits: true
    checkpoint: outputs/rebuild_reproduction/checkpoints/repro_realonly_993_window30x21_ep50.pth
    test_root: dataset/RealArchive/test_1000_wo_pat2
    baseline_metrics: outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/confusion_matrix/repro_realonly_993_window30x21_ep50_metrics.json
    full_frames: 50
    window_size: 30
    num_windows: 21
    window_batch_size: 16
    output_dir: outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_test_inference
```

For an 8 H200 pod, start from `window_batch_size: 16`. If memory is still low, try `24` or `32` for inference only. Do not change training batch sizes just because inference has memory headroom.

## Current Experiment Summary

Key 993 results on the 1000-video test set:

| Run | Accuracy | Note |
| --- | ---: | --- |
| `repro_realonly_993_microbatch` | `0.719` | early 30-epoch style baseline |
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

- LR-hold and sufficient optimizer updates matter more than simply increasing epochs blindly.
- The no-microbatch batch-8 retry did not reproduce the strong microbatch lr-hold behavior, so update density and schedule shape remain important.
- The 30-frame window training plus 21-window test-time logit averaging is the strongest current real-only path. It strongly suggests temporal segment choice is a major source of variance.
- Real-only performance beating many transfer runs is not proof that synthetic data is harmful in general. It may reflect synthetic-real mismatch, transfer schedule sensitivity, and train/test similarity in the real archive.

## Next-Pod Instructions

1. Start in `/Viscnet`.
2. Read `AGENTS.md`, then this file.
3. Verify the rawdataset transfer marker before any move/push work:

```bash
test -f outputs/rebuild_reproduction/notifications/rawdataset_transfer_complete.txt
```

4. If the marker is still missing, do not move `/Viscnet` and do not push. Resolve transfer provenance first.
5. Confirm the restored inference metrics:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_test_inference/window21_test_inference_metrics.json")
m = json.loads(p.read_text())
print(m["accuracy"], m["evaluated_sample_count"], m["confusion_matrix_total"])
PY
```

Expected output values are `0.94`, `1000`, `1000`.

6. On an 8 H200 pod, run inference with 8 ranks:

```bash
MASTER_PORT=29630 PYTHONPATH=src torchrun --nproc_per_node=8 --master_port=29630 scripts/window21_test_inference.py
```

7. For attention analysis on 8 GPUs:

```bash
MASTER_PORT=29631 PYTHONPATH=src torchrun --nproc_per_node=8 --master_port=29631 scripts/analyze_993_attention.py --batch-size 1
```

8. For any new training, use a detached watcher/finalizer. Do not make Codex wait or poll.

## Open Work

- Decide whether the 30-frame temporal window method is an experimental enhancement or a new baseline. It should not be mixed into paper reproduction claims without labeling it.
- Run transfer-side window training only if the research question is whether synthetic pretraining helps under the same temporal-window augmentation.
- Cleanly resolve rawdataset transfer status before moving the repo or pushing to GitHub.
- If pushing, inspect staged scope carefully and exclude `dataset/`, `rawdataset/`, `outputs/`, checkpoints, W&B data, and caches.
