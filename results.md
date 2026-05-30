# Results

Last updated: 2026-05-26 UTC

## Goal 0: Rebuild Paper Data And Weights

Question: can we rebuild the data splits, derived clips, checkpoints, metrics, and W&B provenance needed to reproduce the paper-related training path?

This provenance goal tracks the core paper assets and W&B records. Finished training runs below have local W&B metadata under `wandb/`. The active training processes were checked and have `WANDB_API_KEY` and `WANDB_PROJECT` set from their launch environment.

| Asset / Run | Role | Local Status | W&B Run ID | Result / Note |
| --- | --- | --- | --- | --- |
| `dataset/CFDArchive/sph_35000` | synthetic paper dataset | present |  | source dataset available |
| `dataset/RealArchive/train_993_wo_pat2` | real training split | present |  | 993 real videos |
| `dataset/RealArchive/test_1000_wo_pat2` | real held-out split | present |  | 1000 real videos |
| `outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1` | 30-frame window training data | present |  | required for 30-frame baseline |
| `outputs/rebuild_reproduction/derived_datasets/real_train_993_window10_stride5_phase5` | phase-offset training data | present |  | corrected validation finished |
| `outputs/rebuild_reproduction/derived_datasets/raw_fps_benchmark` | raw-FPS derived data | present |  | 224px finished; 336px partly finished then aborted by request |
| `outputs/rebuild_reproduction/derived_datasets/pattern_lopo_1993` | four-pattern LOPO splits | present |  | Goal 5 aborted by request after partial results |
| `repro_synthetic_pretrain_sph35000` | original synthetic pretrain weight | checkpoint present | `o80yid35` | W&B metadata present |
| `repro_synthetic_pretrain_window30_batch8_ep50` | 30-frame synthetic transfer weight | checkpoint present | `gy33rxno` | required by transfer and LOPO transfer |
| `repro_synthetic_pretrain_window10_stride5_phase5_ep50` | 10-frame phase synthetic pretrain | checkpoint present | `e36h1mbn` | required by phase transfer runs |

## Analysis Until Now

The strongest finished baseline is still `repro_realonly_993_window30x21_ep50` at `0.966`. The best finished raw-FPS result is the 224px 1500/500 split at `0.978`. One 336px run finished before we stopped the queue: `rawfps_realonly_existing993_1000_legacy10fps_336px` reached `0.970` with ECE `0.0084`; the rest of the 336px queue was aborted by request so Goal 4/6 and Goal 5 can run first.

Goal 4/6 corrected modulo-frame validation finished. Goal 5 was aborted by request after partial LOPO results. The intermediate 30-frame window baseline evaluation also finished: both models were evaluated without 21-window logit averaging, producing `1000 * 21 = 21000` clip-level predictions per model and syncing those metrics to W&B under the `re-rebuild-viscnet` project.

No-RPM policy update: from 2026-05-26 onward, new training configs should keep `model.embeddings.rpm_bool: false`. Earlier RPM-enabled results remain historical comparators, not policy-compliant final candidates.

No-RPM queue guard: `scripts/verify_no_rpm_policy.py` is wired into the no-RPM queues and fails before launch if any queued config has `model.embeddings.rpm_bool` not exactly `false`.

Allnew no-RPM augmentation queue: `scripts/run_allnew_no_rpm_aug_queue.sh` is the launch wrapper for W&B project `allnewViscnet`. It preflights no-RPM policy, trains the `augv1` and `augv2` synthetic checkpoints from scratch, runs frozen held-out real-video eval for both, writes class-distribution summaries, and selects a transfer candidate without launching transfer.

Allnew loss-curve adjustment: on 2026-05-27, `allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80` had best observed validation loss at epoch 59 (`train_loss=0.5912`, `val_loss=0.6220`) while LR had decayed to `3.7e-6`. The next `augv2` run keeps the existing active queue path but now uses `num_epochs: 60`, `lr_hold_epochs: 50`, and `eta_min: 5e-6` so training spends less time in the low-LR tail.

No-RPM completion report: `scripts/watch_no_rpm_policy_completion.sh` waits for the no-RPM queues to finish and then writes `outputs/rebuild_reproduction/no_rpm_policy_report/summary.md` and `summary.json` once.

No-RPM acceptance marker: `scripts/watch_no_rpm_policy_acceptance.sh` waits for the no-RPM queues and writes either `outputs/rebuild_reproduction/session_markers/no_rpm_policy.accepted` or `outputs/rebuild_reproduction/session_markers/no_rpm_policy.retry_required`.

## Frame-Sampling Accounting

This table counts frames relative to the original raw capture whenever the source can be traced. The old renamed datasets are not changed by the MP4 FPS metadata during model loading; the important value is the original raw-frame index gap used when the clips were built.

| Dataset / Eval | Clip construction | Model input per clip | Raw-frame index gap | Raw frames covered by one clip | Notes |
| --- | --- | ---: | --- | ---: | --- |
| Original renamed real videos from `videorename.py` / `video1500rename.py` | first 50 raw frames copied, then encoded as 10 FPS MP4 | 50 for old normal configs, 30 for window-30 configs, 10 for phase configs | 1 | 50 available | This is the user-built baseline: generated frame `i` equals raw frame `i`, so raw gap is 1. |
| Goal 1 normal 993/1000 runs | load generated 50-frame MP4 directly | 50 | 1 | first 50 raw frames | Configs with `frame_num: 10`, `time: 5`. |
| Goal 1 30-frame `window30x21` train/eval | 21 overlapping windows from each generated 50-frame MP4 | 30 | 1 | 30 per clip, all 50 across 21 windows | Window starts 0..20; clip `w00` uses 0..29, `w20` uses 20..49. |
| Goal 2/3 `legacy10fps` raw-FPS clips | raw MOV spans of 30, 36, 42, 50 frames, then sampled to 30 frames | 30 | span 30: all 1; span 36/42/50: mix of 1 and 2 | 30, 36, 42, or 50 raw frames | Not just FPS metadata. For span 50, the 30 selected indices have 20 gaps of 2 and 9 gaps of 1. |
| Goal 2/3 `nativefps` raw-FPS clips | physical-duration spans 1.25, 1.50, 1.75, 2.08s from raw MOV, then sampled to 30 frames | 30 | mostly 1 or 2; 30fps 2.08s span can include gap 3 | depends on source FPS | At ~24fps spans are about 30, 36, 42, 50 raw frames. At ~30fps spans are about 37, 45, 52, 62 raw frames. |
| Goal 4/6 modulo-frame phase clips | phase offsets 0..4 from generated 50-frame MP4 with stride 5 | 10 | 5 within one phase clip | 10 per clip, all 50 across 5 phases | Phase 0 uses 0,5,...,45; phase 1 uses 1,6,...,46; together phases 0..4 cover all 50 original raw frames. |

Raw-FPS manifest audit: usable raw sources are 2493 total, with 1120 in the `~24fps` bin, 1358 in the `~30fps` bin, and 15 outlier/unknown. Raw-FPS generated manifests use `model_num_frames: 30`; no raw-FPS training row feeds 50 frames to the model.

Exact 30-frame linspace examples from original raw indices:

| Source span | Selected original indices relative to window start | Gap counts |
| ---: | --- | --- |
| 30 | `0..29` | 29 gaps of 1 |
| 36 | `0,1,2,4,5,6,7,8,10,...,35` | 23 gaps of 1, 6 gaps of 2 |
| 42 | `0,1,3,4,6,7,8,10,11,...,41` | 17 gaps of 1, 12 gaps of 2 |
| 50 | `0,2,3,5,7,8,10,12,14,...,49` | 9 gaps of 1, 20 gaps of 2 |
| 62 | `0,2,4,6,8,11,13,15,17,...,61` | 26 gaps of 2, 3 gaps of 3 |

## Goal 1: 30-Frame Real-Only / Transfer Baseline

| Run | Setup | Accuracy | W&B Run ID | Artifact |
| --- | --- | ---: | --- | --- |
| `repro_realonly_993_batch8_normal_lr3e5_ep70` | real-only 993/1000, normal 30-frame eval | 0.899 | `yvq161vk` | `outputs/rebuild_reproduction/repro_realonly_993_batch8_normal_lr3e5_ep70/confusion_matrix/repro_realonly_993_batch8_normal_lr3e5_ep70_metrics.json` |
| `repro_transfer_993_batch8_normal_lr3e5_ep70` | transfer 993/1000, normal 30-frame eval | 0.893 | `mg0p76us` | `outputs/rebuild_reproduction/repro_transfer_993_batch8_normal_lr3e5_ep70/confusion_matrix/repro_transfer_993_batch8_normal_lr3e5_ep70_metrics.json` |
| `repro_transfer_993_batch8_normal_lr1e5_ep90_min93` | transfer recovery attempt, normal eval | 0.805 | `db3e6yv3` | `outputs/rebuild_reproduction/repro_transfer_993_batch8_normal_lr1e5_ep90_min93/confusion_matrix/repro_transfer_993_batch8_normal_lr1e5_ep90_min93_metrics.json` |
| `repro_realonly_993_window30x21_ep50` | real-only 993-source baseline, 21-window test eval | 0.966 | `aewy3fyv` | `outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_test_inference/window21_test_inference_metrics.json` |
| `repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93` | synthetic-30 transfer, 21-window test eval | 0.948 | `mhxwelmj` | `outputs/rebuild_reproduction/repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93/window21_test_inference/window21_test_inference_metrics.json` |
| `repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70` | real-only 993/1000 ablation, RPM embedding disabled | 0.880 | `z3350p9p` | `outputs/rebuild_reproduction/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70/confusion_matrix/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70_metrics.json` |

### No-RPM Transfer Policy Suite

Objective: never pass RPM conditioning to the model, rebuild synthetic weights once with previous hyperparameters, then evaluate no-RPM transfer against the existing no-RPM real-only result. Target transfer accuracy is over `0.900`; transfer better than real-only is optional.

| Run | Role | RPM Embedding | Status | Accuracy | W&B Run ID | Artifact |
| --- | --- | --- | --- | ---: | --- | --- |
| `repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70` | no-RPM real-only comparator | false | done | 0.880 | `z3350p9p` | `outputs/rebuild_reproduction/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70/confusion_matrix/repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70_metrics.json` |
| `repro_realonly_993_batch8_normal_no_rpm_lr1e5_ep90` | no-RPM real-only recovery attempt | false | queued after active no-RPM transfer policy queue |  |  | `outputs/rebuild_reproduction/repro_realonly_993_batch8_normal_no_rpm_lr1e5_ep90/confusion_matrix/repro_realonly_993_batch8_normal_no_rpm_lr1e5_ep90_metrics.json` |
| `repro_synthetic_pretrain_sph35000_no_rpm_ep50` | no-RPM synthetic pretrain weight | false | running |  |  | `outputs/rebuild_reproduction/checkpoints/repro_synthetic_pretrain_sph35000_no_rpm_ep50.pth` |
| `repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70` | no-RPM synthetic-to-real transfer | false | queued after synthetic no-RPM |  |  | `outputs/rebuild_reproduction/repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70/confusion_matrix/repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70_metrics.json` |

### 21-Window Clip-Level Eval

These rows use the same 1000 test source videos and same 21 windows, but do not average logits back to one source prediction. Each row should produce 21000 clip-level predictions.

| Run | Setup | Status | Clip Predictions | Clip Accuracy | W&B Run | Artifact |
| --- | --- | --- | ---: | ---: | --- | --- |
| `repro_realonly_993_window30x21_ep50_window21_clip` | real-only baseline, 21 unaveraged windows per source | W&B logged | 21000 | 0.9502 | `ajf2pjou` | `outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_clip_test_inference/window21_clip_test_inference_metrics.json` |
| `repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93_window21_clip` | synthetic-30 transfer baseline, 21 unaveraged windows per source | W&B logged | 21000 | 0.9166 | `vpxv6bex` | `outputs/rebuild_reproduction/repro_transfer_993_window30x21_from_synth30_lr1e5_ep45_min93/window21_clip_test_inference/window21_clip_test_inference_metrics.json` |

## Goal 2: Raw-FPS 224px

| Run | Split | FPS Policy | Accuracy | ECE | W&B Run ID | Artifact |
| --- | --- | --- | ---: | ---: | --- | --- |
| `rawfps_realonly_existing993_1000_legacy10fps_224px` | 993/1000 | legacy 10 FPS | 0.975 | 0.0231 | `31ruw8fo` | `outputs/rebuild_reproduction/rawfps_realonly_existing993_1000_legacy10fps_224px/confusion_matrix/rawfps_realonly_existing993_1000_legacy10fps_224px_metrics.json` |
| `rawfps_realonly_existing993_1000_nativefps_224px` | 993/1000 | native FPS | 0.971 | 0.0406 | `16nc6la9` | `outputs/rebuild_reproduction/rawfps_realonly_existing993_1000_nativefps_224px/confusion_matrix/rawfps_realonly_existing993_1000_nativefps_224px_metrics.json` |
| `rawfps_realonly_ratio1500_500_legacy10fps_224px` | 1500/500 | legacy 10 FPS | 0.978 | 0.0280 | `gaaeyuir` | `outputs/rebuild_reproduction/rawfps_realonly_ratio1500_500_legacy10fps_224px/confusion_matrix/rawfps_realonly_ratio1500_500_legacy10fps_224px_metrics.json` |
| `rawfps_realonly_ratio1500_500_nativefps_224px` | 1500/500 | native FPS | 0.978 | 0.0267 | `a1eg8vqr` | `outputs/rebuild_reproduction/rawfps_realonly_ratio1500_500_nativefps_224px/confusion_matrix/rawfps_realonly_ratio1500_500_nativefps_224px_metrics.json` |

## Goal 3: 336px Raw-FPS

The remaining 336px runs were aborted by request on 2026-05-25 so Goal 4/6 and Goal 5 can run first.

| Run | Split | FPS Policy | Status | Accuracy | ECE | Artifact |
| --- | --- | --- | --- | ---: | ---: | --- |
| `rawfps_realonly_existing993_1000_legacy10fps_336px` | 993/1000 | legacy 10 FPS | finished before abort | 0.970 | 0.0084 | `outputs/rebuild_reproduction/rawfps_realonly_existing993_1000_legacy10fps_336px/confusion_matrix/rawfps_realonly_existing993_1000_legacy10fps_336px_metrics.json` |
| `rawfps_realonly_existing993_1000_nativefps_336px` | 993/1000 | native FPS | aborted by request |  |  |  |
| `rawfps_realonly_ratio1500_500_legacy10fps_336px` | 1500/500 | legacy 10 FPS | not started, aborted by request |  |  |  |
| `rawfps_realonly_ratio1500_500_nativefps_336px` | 1500/500 | native FPS | not started, aborted by request |  |  |  |

## Goal 4: Phase-Offset Training / Validation

The corrected post queue finished. These are the accepted modulo-frame validation results over 1000 held-out real videos and 5000 reconstructed phase clips.

| Run | Purpose | Status | Accuracy | Artifact |
| --- | --- | --- | ---: | --- |
| `repro_realonly_993_window10_stride5_phase5_ep50` | corrected stride-5 phases 0-4 real-test validation | done | 0.707 source / 0.6816 clip | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_realonly_993_window10_stride5_phase5_ep50/summary.json` |
| `repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45` | corrected stride-5 phases 0-4 real-test validation | done | 0.312 source / 0.2970 clip | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45/summary.json` |
| `repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55` | corrected stride-5 phases 0-4 real-test validation | done | 0.404 source / 0.3724 clip | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55/summary.json` |

Historical phase-offset transfer rows below have metrics, but are not accepted as corrected Goal 4/6 results.

| Run | Eval | Accuracy | W&B Run ID | Artifact |
| --- | --- | ---: | --- | --- |
| `repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45` | window41 historical eval | 0.154 | `y6movp2a` | `outputs/rebuild_reproduction/repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45/window41_test_inference/window21_test_inference_metrics.json` |
| `repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55` | window41 historical eval | 0.247 | `xlzvb6ga` | `outputs/rebuild_reproduction/repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55/window41_test_inference/window21_test_inference_metrics.json` |

## Goal 5: Full Leave-One-Pattern-Out Validation

This was aborted by request on 2026-05-26 so the 21-window clip-level baseline evaluation can run first. It uses all 1993 real videos: pattern 1 = 499, pattern 2 = 498, pattern 3 = 499, pattern 4 = 497.

| Run | Train Patterns | Held-Out Pattern | Train Videos | Test Videos | Status | Accuracy | Artifact |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| `pattern_lopo_realonly_train_not1_test1` | 2,3,4 | 1 | 1494 | 499 | done batch 8 | 0.1523 | `outputs/rebuild_reproduction/pattern_lopo_realonly_train_not1_test1/confusion_matrix/pattern_lopo_realonly_train_not1_test1_metrics.json` |
| `pattern_lopo_transfer_synth30_train_not1_test1` | 2,3,4 | 1 | 1494 | 499 | done batch 8 | 0.1603 | `outputs/rebuild_reproduction/pattern_lopo_transfer_synth30_train_not1_test1/confusion_matrix/pattern_lopo_transfer_synth30_train_not1_test1_metrics.json` |
| `pattern_lopo_realonly_train_not2_test2` | 1,3,4 | 2 | 1495 | 498 | done batch 8 | 0.2450 | `outputs/rebuild_reproduction/pattern_lopo_realonly_train_not2_test2/confusion_matrix/pattern_lopo_realonly_train_not2_test2_metrics.json` |
| `pattern_lopo_transfer_synth30_train_not2_test2` | 1,3,4 | 2 | 1495 | 498 | done batch 8 | 0.4197 | `outputs/rebuild_reproduction/pattern_lopo_transfer_synth30_train_not2_test2/confusion_matrix/pattern_lopo_transfer_synth30_train_not2_test2_metrics.json` |
| `pattern_lopo_realonly_train_not3_test3` | 1,2,4 | 3 | 1494 | 499 | done batch 8 | 0.2705 | `outputs/rebuild_reproduction/pattern_lopo_realonly_train_not3_test3/confusion_matrix/pattern_lopo_realonly_train_not3_test3_metrics.json` |
| `pattern_lopo_transfer_synth30_train_not3_test3` | 1,2,4 | 3 | 1494 | 499 | aborted by request |  |  |
| `pattern_lopo_realonly_train_not4_test4` | 1,2,3 | 4 | 1496 | 497 | not started, aborted by request |  |  |
| `pattern_lopo_transfer_synth30_train_not4_test4` | 1,2,3 | 4 | 1496 | 497 | not started, aborted by request |  |  |

## Dual-Pattern Diagnostic

Purpose: train the available dual-pattern real split and export paper-style confusion matrices for both raw counts and row-normalized accuracy. This is separate from full four-pattern LOPO, which remains omitted by request.

| Run | Train Videos | Test Videos | Status | Accuracy | W&B Run ID | Artifact |
| --- | ---: | ---: | --- | ---: | --- | --- |
| `dual_pattern_realonly_v2_450_ep70` | 224 | 225 | historical RPM-enabled result; not no-RPM policy compliant | 0.920 | `l3xh11z3` | `outputs/rebuild_reproduction/dual_pattern_realonly_v2_450_ep70/confusion_matrix/dual_pattern_realonly_v2_450_ep70_metrics.json` |
| `dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70` | 224 | 225 | queued after no-RPM transfer |  |  | `outputs/rebuild_reproduction/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70/confusion_matrix/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70_metrics.json` |

Expected paper-style figures after completion:

| Figure | Path |
| --- | --- |
| Count confusion matrix, full 10-class head | `outputs/rebuild_reproduction/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70/paper_confusion/dual_pattern_confusion_counts_full.png` |
| Normalized confusion matrix, full 10-class head | `outputs/rebuild_reproduction/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70/paper_confusion/dual_pattern_confusion_normalized_full.png` |
| Count confusion matrix, active classes only | `outputs/rebuild_reproduction/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70/paper_confusion/dual_pattern_confusion_counts_active.png` |
| Normalized confusion matrix, active classes only | `outputs/rebuild_reproduction/dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70/paper_confusion/dual_pattern_confusion_normalized_active.png` |

## Goal 6: Modulo-Frame Real-Test Reconstruction

This is the source-averaged version of the corrected Goal 4 validation. It evaluates the 1000 held-out real test videos with stride 5 and phases 0-4, then aggregates predictions back to source videos.

| Run | Test Root | Reconstruction | Status | Source Accuracy | Clip Accuracy | Artifact |
| --- | --- | --- | --- | ---: | ---: | --- |
| `repro_realonly_993_window10_stride5_phase5_ep50` | `dataset/RealArchive/test_1000_wo_pat2` | stride 5, phases 0-4 | done | 0.707 | 0.6816 | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_realonly_993_window10_stride5_phase5_ep50/summary.json` |
| `repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45` | `dataset/RealArchive/test_1000_wo_pat2` | stride 5, phases 0-4 | done | 0.312 | 0.2970 | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_transfer_993_window10_stride5_phase5_from_synth10_lr1e5_ep45/summary.json` |
| `repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55` | `dataset/RealArchive/test_1000_wo_pat2` | stride 5, phases 0-4 | done | 0.404 | 0.3724 | `outputs/rebuild_reproduction/mod_frame_test_validation/repro_transfer_993_window10_stride5_phase5_from_synth10_lr5e6_ep55/summary.json` |

## Active Queue

| Queue Item | Purpose | Status |
| --- | --- | --- |
| `rawfps336_requeue` | 336px raw-FPS queue | aborted after first completed 336px result |
| `post_rawfps336_eval_queue` | Goal 4/6 first, then Goal 5 | failed at Goal 5 metric gather; Goal 4/6 completed |
| `pattern_lopo_queue` | Goal 5 full LOPO after file-backed gather fix | aborted by request |
| `window21_clip_eval_queue` | 21000 clip-level predictions for real-only and transfer window baselines | finished; both evals W&B logged |
| `no_rpm_realonly_993` | real-only 993/1000 ablation with RPM embedding disabled | finished |
| `dual_pattern_queue` | dual-pattern real-only training plus paper-style count/normalized confusion matrices | finished; historical RPM-enabled |
| `no_rpm_transfer_policy_queue` | no-RPM synthetic pretrain, no-RPM transfer, no-RPM synthetic-transfer dual-pattern | running; active run `repro_synthetic_pretrain_sph35000_no_rpm_ep50` |
| `no_rpm_realonly_followup_queue` | one bounded no-RPM real-only recovery attempt at lower LR/longer horizon | queued after `no_rpm_transfer_policy_queue` |
| `no_rpm_policy_report_watcher` | waits for both no-RPM queues, then writes final no-RPM acceptance report | running |
| `no_rpm_policy_acceptance_watcher` | waits for both no-RPM queues, then writes accepted/retry-required marker | running |
