# Rebuild Artifact Index

Last updated: 2026-05-26 UTC

This index maps the local rebuild datasets, configs, scripts, outputs, weights,
and analysis artifacts by experiment goal. `results.md` is the metric summary;
this file is the file-system guide for where each artifact lives and what it is.

No-RPM policy update: from 2026-05-26 onward, new training configs should set
`model.embeddings.rpm_bool: false`. Existing RPM-enabled artifacts are retained
as historical comparators only.

## Top-Level Layout

| Path | Meaning |
| --- | --- |
| `configs/rebuild/` | Rebuild experiment configs and manifest definitions. |
| `configs/rebuild/retries/` | Recovery, reproduction, window, transfer, and ablation configs used after the first rebuild pass. |
| `configs/rebuild/raw_fps/` | FPS-aware raw-real benchmark configs. |
| `configs/rebuild/pattern_lopo/` | Leave-one-pattern-out configs. Goal 5 is currently omitted/aborted by request. |
| `configs/rebuild/manifests/` | Source-count manifests for 300-993 real training subsets. |
| `scripts/` | Reusable dataset builders, training queues, evaluation tools, post-run loggers, and analysis scripts. |
| `dataset/` | Local source datasets. Do not commit dataset contents. |
| `outputs/rebuild_reproduction/` | Main rebuild output root: metrics, plots, derived datasets, logs, queue summaries, and checkpoints. |
| `outputs/rebuild_reproduction/checkpoints/` | Model weights for rebuild and ablation runs. Do not commit. |
| `outputs/rebuild_reproduction/logs/` | Detached training/evaluation logs. |
| `outputs/rebuild_reproduction/session_markers/` | Queue completion/abort markers. |
| `wandb/` | Local W&B cache. Online rebuild runs should be in project `re-rebuild-viscnet`. |

## Shared Source Data

| Dataset | Used By | Meaning |
| --- | --- | --- |
| `dataset/CFDArchive/sph_35000` | Goal 0, transfer weights | Synthetic paper/pretraining source dataset. |
| `dataset/RealArchive/train_993_wo_pat2` | Goals 0, 1, 4, 6, no-RPM ablation | Original 993 real-video training split. |
| `dataset/RealArchive/test_1000_wo_pat2` | Goals 0, 1, 4, 6, no-RPM ablation | Original 1000 real-video held-out split. |
| `dataset/RealArchive/real_20rpm_increment_2500` | Goals 2, 3, 5 | Raw-real source pool and rebuilt raw metadata. |
| `dataset/RealArchive/dualpatterndataset_V2_450_train` | Dual-pattern diagnostic | Dual-pattern real-video training split; 224 videos and no `backgrounds/` directory. |
| `dataset/RealArchive/dualpatterndataset_V2_450_test` | Dual-pattern diagnostic | Dual-pattern real-video test split; 225 videos and no `backgrounds/` directory. |
| `outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1` | Goal 1 | 21 overlapping 30-frame clips per 50-frame real training source. |
| `outputs/rebuild_reproduction/derived_datasets/real_train_993_window10_stride5_phase5` | Goals 4, 6 | Five phase-offset stride-5 clips per 50-frame real source. |
| `outputs/rebuild_reproduction/derived_datasets/raw_fps_benchmark` | Goals 2, 3 | Raw-FPS benchmark manifests and optional clip roots. |
| `outputs/rebuild_reproduction/derived_datasets/pattern_lopo_1993` | Goal 5 | Full 1993-source four-pattern LOPO split data. Currently not queued. |

## Goal 0: Paper-Rebuild Provenance

Purpose: rebuild paper-related datasets, source splits, base weights, metrics,
and provenance.

| Artifact Type | Paths |
| --- | --- |
| Configs | `configs/rebuild/synthetic_pretrain.yaml`, `configs/rebuild/retries/synthetic_pretrain_window30_batch8_ep50.yaml` |
| Core checkpoints | `outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_sph35000.pth`, `outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_window30_batch8_ep50.pth`, `outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_window10_stride5_phase5_ep50.pth` |
| Reference logs | `outputs/rebuild_reproduction/wandb_reference_logs/` |
| Checker/analyzer | `outputs/rebuild_reproduction/checker_report.md`, `outputs/rebuild_reproduction/analyzer_report.md` |
| Goal 0 figures | `outputs/rebuild_reproduction/goal0_confusion_matrices.png`, `outputs/rebuild_reproduction/goal0_class123_analysis/` |
| Status | Base assets present. Goal 0 also tracks new paper-related ablations such as no-RPM. |

## Goal 1: 30-Frame Real-Only / Transfer Baselines

Purpose: reproduce and compare normal 30-frame real-only, transfer, and
21-window evaluation baselines.

| Run / Group | Config | Checkpoint | Outputs |
| --- | --- | --- | --- |
| `repro_realonly_993_batch8_normal_lr3e5_ep70` | `configs/rebuild/retries/realonly_993_batch8_normal_lr3e5_ep70.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_realonly_993_batch8_normal_lr3e5_ep70.pth` | `outputs/rebuild_reproduction/repro_realonly_993_batch8_normal_lr3e5_ep70/` |
| `repro_transfer_993_batch8_normal_lr3e5_ep70` | `configs/rebuild/retries/transfer_993_batch8_normal_lr3e5_ep70.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_transfer_993_batch8_normal_lr3e5_ep70.pth` | `outputs/rebuild_reproduction/repro_transfer_993_batch8_normal_lr3e5_ep70/` |
| `repro_transfer_993_batch8_normal_lr1e5_ep90_min93` | `configs/rebuild/retries/transfer_993_batch8_normal_lr1e5_ep90_min93.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_transfer_993_batch8_normal_lr1e5_ep90_min93.pth` | `outputs/rebuild_reproduction/repro_transfer_993_batch8_normal_lr1e5_ep90_min93/` |
| `repro_realonly_993_window30x21_ep50` | `configs/rebuild/retries/realonly_993_window30x21_ep50.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_realonly_993_window30x21_ep50.pth` | `outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/` |
| `repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93` | `configs/rebuild/retries/transfer_993_window30x21_from_synth30_lr1e5_ep45_min93.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93.pth` | `outputs/rebuild_reproduction/repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93/` |
| `repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70` | `configs/rebuild/retries/realonly_993_batch8_normal_no_rpm_lr3e5_ep70.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70.pth` | `outputs/rebuild_reproduction/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70/` |
| `repro_realonly_993_batch8_normal_no_rpm_lr1e5_ep90` | `configs/rebuild/retries/realonly_993_batch8_normal_no_rpm_lr1e5_ep90.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_realonly_993_batch8_normal_no_rpm_lr1e5_ep90.pth` | `outputs/rebuild_reproduction/repro_realonly_993_batch8_normal_no_rpm_lr1e5_ep90/` |
| `repro_synthetic_pretrain_sph35000_no_rpm_ep50` | `configs/rebuild/retries/synthetic_pretrain_sph35000_no_rpm_ep50.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_sph35000_no_rpm_ep50.pth` | `outputs/rebuild_reproduction/repro_synthetic_pretrain_sph35000_no_rpm_ep50/` |
| `repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70` | `configs/rebuild/retries/transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70.pth` | `outputs/rebuild_reproduction/repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70/` |

Goal 1 scripts and queues:

| Script | Meaning |
| --- | --- |
| `scripts/recover_remaining_batch8_normal.sh` | Recovery queue for normal batch-8 real-only/transfer runs. |
| `scripts/recover_window30x21_rebuild.sh` | Rebuilds 30-frame window configs/runs. |
| `scripts/run_transfer_min93_queue.sh` | Transfer recovery and 21-window min-93 pass. |
| `scripts/window21_test_inference.py` | 21-window source-averaged and clip-level inference. |
| `scripts/run_window21_clip_eval_queue.sh` | Re-logs completed 21-window clip evals to W&B and records queue summary. |
| `scripts/log_eval_metrics_to_wandb.py` | Generic post-eval W&B logger for standalone eval outputs. |
| `scripts/run_no_rpm_realonly_993.sh` | Current no-RPM real-only ablation launcher. |
| `scripts/run_no_rpm_transfer_policy_queue.sh` | No-RPM policy queue: synthetic pretrain, transfer, and dual-pattern synthetic-transfer. |
| `scripts/run_no_rpm_realonly_followup_queue.sh` | Follow-up no-RPM real-only recovery queue that waits behind the active no-RPM transfer policy queue. |
| `scripts/verify_no_rpm_policy.py` | Preflight checker used by no-RPM queues; fails before launch if a queued config has `model.embeddings.rpm_bool` not exactly `false`, or if a transfer config initializes from a checkpoint path that does not include `no_rpm`. |
| `scripts/summarize_no_rpm_policy_results.py` | Reads completed no-RPM metrics and writes target/pass summary artifacts. Use `--require-accepted` after completion to exit nonzero unless all metrics exist and the transfer target is met. |
| `scripts/watch_no_rpm_policy_completion.sh` | Detached watcher that waits for both no-RPM queues and runs the summarizer once. |
| `scripts/watch_no_rpm_policy_acceptance.sh` | Detached watcher that waits for both no-RPM queues and writes either an accepted or retry-required marker using the strict summary gate. |
| `tests/test_no_rpm_policy.py` | Unit tests for the no-RPM preflight checker accepting `rpm_bool: false`, rejecting `rpm_bool: true`, and rejecting transfer from RPM-trained checkpoints. |
| `tests/test_no_rpm_model_behavior.py` | Regression test proving `VivitEmbeddings` does not access the RPM embedding module when `rpm_bool` is false. |
| `tests/test_no_rpm_policy_summary.py` | Unit tests for the no-RPM completion report, including transfer target, transfer-minus-real-only, and W&B run ID parsing. |
| `tests/test_no_rpm_queue_scripts.py` | Regression tests proving no-RPM training queues call `scripts/verify_no_rpm_policy.py` before launch. |

Important Goal 1 output subfolders:

| Subfolder | Meaning |
| --- | --- |
| `confusion_matrix/` | Classification confusion matrix PNG and metrics JSON. |
| `reliability_plots/` | Calibration/reliability PNG and metrics JSON. |
| `window21_test_inference/` | Source-averaged 21-window metrics, predictions, and figure. |
| `window21_clip_test_inference/` | Unaveraged 21-window clip metrics, predictions, and figure. |
| `_distributed_records/` | File-backed DDP gather scratch records. Internal artifact, not a result. |

Current no-RPM ablation status:

| Item | Path / Value |
| --- | --- |
| Queue summary | `outputs/rebuild_reproduction/no_rpm_realonly_993/summary.json` |
| Log | `outputs/rebuild_reproduction/logs/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70.log` |
| W&B | `re-rebuild-viscnet/z3350p9p` |
| Status | Running as of this index. Fill final metric into `results.md` after completion. |

## Goal 2: Raw-FPS 224px Benchmark

Purpose: compare raw-real FPS policies at 224px.

| Run | Config | Checkpoint | Outputs |
| --- | --- | --- | --- |
| `rawfps_realonly_existing993_1000_legacy10fps_224px` | `configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_legacy10fps_224px.yaml` | `outputs/rebuild_reproduction/checkpoints/rawfps_realonly_existing993_1000_legacy10fps_224px.pth` | `outputs/rebuild_reproduction/rawfps_realonly_existing993_1000_legacy10fps_224px/` |
| `rawfps_realonly_existing993_1000_nativefps_224px` | `configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_nativefps_224px.yaml` | `outputs/rebuild_reproduction/checkpoints/rawfps_realonly_existing993_1000_nativefps_224px.pth` | `outputs/rebuild_reproduction/rawfps_realonly_existing993_1000_nativefps_224px/` |
| `rawfps_realonly_ratio1500_500_legacy10fps_224px` | `configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_legacy10fps_224px.yaml` | `outputs/rebuild_reproduction/checkpoints/rawfps_realonly_ratio1500_500_legacy10fps_224px.pth` | `outputs/rebuild_reproduction/rawfps_realonly_ratio1500_500_legacy10fps_224px/` |
| `rawfps_realonly_ratio1500_500_nativefps_224px` | `configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_nativefps_224px.yaml` | `outputs/rebuild_reproduction/checkpoints/rawfps_realonly_ratio1500_500_nativefps_224px.pth` | `outputs/rebuild_reproduction/rawfps_realonly_ratio1500_500_nativefps_224px/` |

Supporting artifacts:

| Path | Meaning |
| --- | --- |
| `outputs/rebuild_reproduction/derived_datasets/raw_fps_benchmark/build_report_image224.json` | 224px manifest/source-count audit. |
| `scripts/build_raw_fps_window_dataset.py` | Builds FPS-aware raw-real manifests/configs. |
| `scripts/build_raw_realvideo_configs.py` | Raw-video config discovery/rebuild helper. |
| `outputs/rebuild_reproduction/logs/rawfps_realonly_*_224px.log` | Training logs. |
| `outputs/rebuild_reproduction/logs/eval_rawfps_realonly_*_224px.log` | Evaluation logs. |

## Goal 3: 336px Raw-FPS Probe

Purpose: test 336px feasibility after 224px baseline.

| Run | Config | Checkpoint | Outputs / Status |
| --- | --- | --- | --- |
| `rawfps_realonly_existing993_1000_legacy10fps_336px` | `configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_legacy10fps_336px.yaml` | `outputs/rebuild_reproduction/checkpoints/rawfps_realonly_existing993_1000_legacy10fps_336px.pth` | `outputs/rebuild_reproduction/rawfps_realonly_existing993_1000_legacy10fps_336px/`; finished before abort. |
| `rawfps_realonly_existing993_1000_nativefps_336px` | `configs/rebuild/raw_fps/rawfps_realonly_existing993_1000_nativefps_336px.yaml` | `outputs/rebuild_reproduction/checkpoints/rawfps_realonly_existing993_1000_nativefps_336px.pth` | Aborted by request; checkpoint may be partial/incomplete. |
| `rawfps_realonly_ratio1500_500_legacy10fps_336px` | `configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_legacy10fps_336px.yaml` |  | Not started; aborted by request. |
| `rawfps_realonly_ratio1500_500_nativefps_336px` | `configs/rebuild/raw_fps/rawfps_realonly_ratio1500_500_nativefps_336px.yaml` |  | Not started; aborted by request. |

Supporting artifacts:

| Path | Meaning |
| --- | --- |
| `outputs/rebuild_reproduction/rawfps336_requeue/summary.json` | 336px queue summary and abort reason. |
| `outputs/rebuild_reproduction/derived_datasets/raw_fps_benchmark/build_report_image336.json` | 336px manifest/source-count audit. |
| `outputs/rebuild_reproduction/session_markers/rawfps336_requeue.aborted` | Durable abort marker. |

## Goal 4: Phase-Offset Training / Validation

Purpose: train/evaluate phase-offset stride-5 clips from the original 50-frame
real videos.

| Run | Config Source | Checkpoint | Accepted Output |
| --- | --- | --- | --- |
| `repro_realonly_993_window10_stride5_phase5_ep50` | `outputs/rebuild_reproduction/all_pending_training_queue/configs/repro_realonly_993_window10_stride5_phase5_ep50.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_realonly_993_window10_stride5_phase5_ep50.pth` | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_realonly_993_window10_stride5_phase5_ep50/summary.json` |
| `repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45` | `outputs/rebuild_reproduction/all_pending_training_queue/configs/repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45.pth` | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45/summary.json` |
| `repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55` | `outputs/rebuild_reproduction/all_pending_training_queue/configs/repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55.yaml` | `outputs/rebuild_reproduction/checkpoints/repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55.pth` | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55/summary.json` |

Supporting artifacts:

| Path | Meaning |
| --- | --- |
| `scripts/run_mod_frame_test_queue.sh` | Goal 4/6 evaluation queue. |
| `scripts/mod_frame_test_inference.py` | Correct stride-5 phase-offset real-test evaluator. |
| `outputs/rebuild_reproduction/mod_frame_test_validation/results.md` | Accepted corrected Goal 4/6 result summary. |
| `outputs/rebuild_reproduction/phase_offset_window10_analysis.md` | Analysis notes for phase-offset/window behavior. |
| `outputs/rebuild_reproduction/repro_*window41_test_inference/` | Historical phase/window evals. Not accepted as corrected Goal 4/6. |

## Goal 5: Full Leave-One-Pattern-Out Validation

Purpose: train on three real patterns and test on the held-out pattern.

Status: omitted/aborted by current user priority. Keep existing partial outputs
as historical evidence, but do not run more LOPO unless explicitly requested.

| Artifact Type | Paths |
| --- | --- |
| Configs | `configs/rebuild/pattern_lopo/*.yaml` |
| Builder | `scripts/build_pattern_lopo_configs.py` |
| Queue | `scripts/run_pattern_lopo_queue.sh` |
| Derived data | `outputs/rebuild_reproduction/derived_datasets/pattern_lopo_1993` |
| Queue summary | `outputs/rebuild_reproduction/pattern_lopo_queue/summary.json` |
| Abort marker | `outputs/rebuild_reproduction/session_markers/pattern_lopo_queue.aborted` |
| Partial run outputs | `outputs/rebuild_reproduction/pattern_lopo_realonly_train_not1_test1/`, `outputs/rebuild_reproduction/pattern_lopo_transfer_synth30_train_not1_test1/`, `outputs/rebuild_reproduction/pattern_lopo_realonly_train_not2_test2/`, `outputs/rebuild_reproduction/pattern_lopo_transfer_synth30_train_not2_test2/`, `outputs/rebuild_reproduction/pattern_lopo_realonly_train_not3_test3/` |
| Partial checkpoints | `outputs/rebuild_reproduction/checkpoints/pattern_lopo_*.pth` |

## Dual-Pattern Diagnostic

Purpose: train the available dual-pattern real split and produce paper-style
confusion matrices for raw counts and row-normalized accuracy. This is not the
full four-pattern LOPO protocol.

| Artifact Type | Paths |
| --- | --- |
| Dataset | `dataset/RealArchive/dualpatterndataset_V2_450_train`, `dataset/RealArchive/dualpatterndataset_V2_450_test` |
| Config | `configs/rebuild/retries/dual_pattern_realonly_v2_450_ep70.yaml` |
| Dataset adapter | `src/datasets/VideoDatasetRealZeroPattern.py` |
| Queue | `scripts/run_dual_pattern_queue.sh` |
| Confusion plotting | `scripts/plot_dual_pattern_confusion.py` |
| Queue summary | `outputs/rebuild_reproduction/dual_pattern_queue/summary.json` |
| Log | `outputs/rebuild_reproduction/logs/dual_pattern_realonly_v2_450_ep70.log` |
| Checkpoint | `outputs/rebuild_reproduction/checkpoints/dual_pattern_realonly_v2_450_ep70.pth` |
| Metrics | `outputs/rebuild_reproduction/dual_pattern_realonly_v2_450_ep70/confusion_matrix/dual_pattern_realonly_v2_450_ep70_metrics.json` |
| Paper-style figures | `outputs/rebuild_reproduction/dual_pattern_realonly_v2_450_ep70/paper_confusion/dual_pattern_confusion_counts_full.png`, `outputs/rebuild_reproduction/dual_pattern_realonly_v2_450_ep70/paper_confusion/dual_pattern_confusion_normalized_full.png`, `outputs/rebuild_reproduction/dual_pattern_realonly_v2_450_ep70/paper_confusion/dual_pattern_confusion_counts_active.png`, `outputs/rebuild_reproduction/dual_pattern_realonly_v2_450_ep70/paper_confusion/dual_pattern_confusion_normalized_active.png` |
| No-RPM config | `configs/rebuild/retries/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70.yaml` |
| No-RPM outputs | `outputs/rebuild_reproduction/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70/` |
| Status | RPM-enabled `dual_pattern_realonly_v2_450_ep70` finished as historical evidence; no-RPM synthetic-transfer replacement is queued in `no_rpm_transfer_policy_queue`. The active no-RPM queue is currently on synthetic pretraining. |

## Goal 6: Modulo-Frame Real-Test Reconstruction

Purpose: evaluate the 1000 held-out real videos using stride-5 phase clips and
aggregate predictions back to source-level metrics.

Goal 6 shares the accepted outputs with Goal 4:

| Artifact | Path |
| --- | --- |
| Source/clip summary | `outputs/rebuild_reproduction/mod_frame_test_validation/summary.json` |
| Per-run summaries | `outputs/rebuild_reproduction/mod_frame_test_validation/<run_name>/summary.json` |
| Evaluator | `scripts/mod_frame_test_inference.py` |
| Queue | `scripts/run_mod_frame_test_queue.sh` |

## Cross-Pattern Validation Artifacts

These are diagnostic artifacts related to pattern generalization, separate from
the aborted full LOPO queue.

| Path | Meaning |
| --- | --- |
| `scripts/cross_pattern_validation.py` | Cross-pattern validation evaluator. |
| `scripts/run_cross_pattern_validation.sh` | Cross-pattern validation runner. |
| `scripts/run_cross_pattern_cpu_prefill.sh` | CPU-side prefill/helper for cross-pattern validation. |
| `outputs/rebuild_reproduction/cross_pattern_validation/summary.json` | Cross-pattern validation summary. |
| `outputs/rebuild_reproduction/cross_pattern_validation/summary.md` | Human-readable cross-pattern summary. |
| `outputs/rebuild_reproduction/cross_pattern_validation_smoke/` | Smoke-test outputs. |

## Queue And Status Artifacts

| Path | Meaning |
| --- | --- |
| `outputs/rebuild_reproduction/all_pending_training_queue/summary.json` | Historical all-pending queue summary. |
| `outputs/rebuild_reproduction/transfer_min93_queue/summary.json` | Transfer min-93 recovery queue summary. |
| `outputs/rebuild_reproduction/window21_clip_eval_queue/summary.json` | Clip-level 21-window eval queue summary and W&B run IDs. |
| `outputs/rebuild_reproduction/no_rpm_realonly_993/summary.json` | Current no-RPM ablation queue summary. |
| `outputs/rebuild_reproduction/dual_pattern_queue/summary.json` | Dual-pattern training queue summary and paper-confusion artifact paths. |
| `outputs/rebuild_reproduction/no_rpm_transfer_policy_queue/summary.json` | No-RPM synthetic/transfer/dual-pattern queue summary. |
| `outputs/rebuild_reproduction/no_rpm_realonly_followup_queue/summary.json` | Follow-up no-RPM real-only recovery queue summary. |
| `outputs/rebuild_reproduction/no_rpm_policy_report/summary.md` | Final no-RPM policy acceptance report after both queues complete. |
| `outputs/rebuild_reproduction/no_rpm_policy_report/summary.json` | Machine-readable no-RPM policy acceptance report. |
| `outputs/rebuild_reproduction/session_markers/no_rpm_policy.accepted` | Written only if all no-RPM metrics exist and transfer accuracy is at least `0.9001`. |
| `outputs/rebuild_reproduction/session_markers/no_rpm_policy.retry_required` | Written if the no-RPM suite finishes but metrics are missing or transfer target is not met. |
| `outputs/rebuild_reproduction/notifications/` | Short post-run notification summaries. |
| `outputs/rebuild_reproduction/session_markers/*.done` | Completed queue markers. |
| `outputs/rebuild_reproduction/session_markers/*.aborted` | Aborted queue markers. |

## Script Index By Role

| Role | Scripts |
| --- | --- |
| Dataset/config builders | `scripts/generate_rebuild_reproduction.py`, `scripts/build_real_window_dataset.py`, `scripts/build_raw_fps_window_dataset.py`, `scripts/build_raw_realvideo_configs.py`, `scripts/build_pattern_lopo_configs.py` |
| Main training launchers | `scripts/dev.sh`, `scripts/run_rebuild_reproduction.sh`, `scripts/run_all_pending_training_queue.sh`, `scripts/recover_remaining_batch8_normal.sh`, `scripts/recover_top5_rerebuild.sh`, `scripts/run_transfer_min93_queue.sh`, `scripts/run_no_rpm_realonly_993.sh`, `scripts/run_dual_pattern_queue.sh`, `scripts/run_no_rpm_transfer_policy_queue.sh`, `scripts/run_no_rpm_realonly_followup_queue.sh` |
| Evaluation | `scripts/window21_test_inference.py`, `scripts/mod_frame_test_inference.py`, `scripts/cross_pattern_validation.py`, `scripts/run_window21_clip_eval_queue.sh`, `scripts/run_mod_frame_test_queue.sh`, `scripts/run_cross_pattern_validation.sh`, `scripts/summarize_no_rpm_policy_results.py` |
| Post-run/checking | `scripts/check_rebuild_results.py`, `scripts/analyze_rebuild_results.py`, `scripts/post_rebuild_training.py`, `scripts/require_rebuild_accuracy.py`, `scripts/log_eval_metrics_to_wandb.py`, `scripts/watch_tmux_completion.py`, `scripts/verify_no_rpm_policy.py`, `scripts/watch_no_rpm_policy_completion.sh`, `scripts/watch_no_rpm_policy_acceptance.sh` |
| Plotting/diagnostics | `scripts/plot_data_efficiency.py`, `scripts/plot_993_confusion_attention.py`, `scripts/analyze_993_attention.py`, `scripts/plot_dual_pattern_confusion.py` |

## Weight Index By Goal

| Goal | Checkpoint Pattern |
| --- | --- |
| Goal 0 synthetic/base | `outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain*.pth` |
| Goal 1 normal/window/no-RPM | `outputs/rebuild_reproduction/checkpoints/repro_realonly_993*.pth`, `outputs/rebuild_reproduction/checkpoints/repro_transfer_993*.pth` |
| Goal 2 raw-FPS 224px | `outputs/rebuild_reproduction/checkpoints/rawfps_realonly_*_224px.pth` |
| Goal 3 raw-FPS 336px | `outputs/rebuild_reproduction/checkpoints/rawfps_realonly_*_336px.pth` |
| Goal 4/6 phase-offset | `outputs/rebuild_reproduction/checkpoints/*window10_stride5_phase5*.pth` |
| Goal 5 LOPO | `outputs/rebuild_reproduction/checkpoints/pattern_lopo_*.pth` |
| Dual-pattern diagnostic | `outputs/rebuild_reproduction/checkpoints/dual_pattern_realonly_v2_450_ep70.pth`, `outputs/rebuild_reproduction/checkpoints/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70.pth` |

Checkpoint caveats:

- `*.wrong_project_partial.pth` and `*.batch16_partial.pth` are historical/partial
  artifacts, not accepted final weights.
- Active or aborted queues may leave checkpoints before final metrics exist.
  Use `results.md` plus the queue summary before treating a checkpoint as a
  completed result.

## Result-Reading Rules

- Use `results.md` for accepted metrics and current status.
- Use this index to locate artifacts by goal.
- Treat `confusion_matrix/*_metrics.json` as classification metric truth.
- Treat `reliability_plots/*_metrics.json` as calibration metric truth.
- Treat `window21_test_inference/*metrics.json` as source-averaged 21-window
  metric truth.
- Treat `window21_clip_test_inference/*metrics.json` as unaveraged clip-level
  21-window metric truth.
- Treat `mod_frame_test_validation/*/summary.json` as corrected Goal 4/6 metric
  truth.
- Do not claim W&B-backed completion unless the run is in
  `jongwonsohn-seoul-national-university/re-rebuild-viscnet` or was explicitly
  re-logged there.
