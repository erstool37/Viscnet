# Important Result: Real-Only No-RPM 30-Frame Variable-Window Evaluation

Date recorded: 2026-05-28

This is currently the most important result for the no-RPM real-only direction and may be used as a paper-candidate result. Treat the paths below as the primary provenance record.

## Run Identity

- Training run: `repro_realonly_993_window30x21_no_rpm_ep50`
- Evaluation rerun: `repro_realonly_993_window30x21_no_rpm_ep50_realtest_1000x21_windows_rerun`
- Config: `configs/rebuild/retries/realonly_993_window30x21_no_rpm_ep50.yaml`
- Checkpoint: `outputs/rebuild_reproduction/checkpoints/repro_realonly_993_window30x21_no_rpm_ep50.pth`
- Summary JSON: `outputs/rebuild_reproduction/repro_realonly_993_window30x21_no_rpm_ep50_1000x21_inference_rerun/repro_realonly_993_window30x21_no_rpm_ep50_realtest_1000x21_windows_rerun/summary.json`
- Training W&B project: `re-rebuild-viscnet`
- Training W&B run ID: `168147oi`

## Method

- Model input uses video frames only plus normal visual pattern tensor loading, with RPM disabled.
- `model.embeddings.rpm_bool: false`
- Runtime diagnostic confirmed RPM passed to the model was zeroed:
  - `runtime_zeroed_rpm_confirmed: true`
  - `rpm_actual_nonzero_count: 19005`
  - `rpm_model_nonzero_count: 0`
- Training data: real-only `993` train videos expanded into `993 * 21 = 20,853` fixed 30-frame clips.
- Training windows: all contiguous 30-frame windows from source frame starts `0..20`.
- Real held-out evaluation: `1000` real test videos, evaluated over `21` windows each.
- Evaluation size: `1000 * 21 = 21,000` windows.
- Evaluation windows: contiguous 30-frame windows from starts `0..20` within the first 50 frames.

## Key Metrics

### All Windows Counted Per Window

- Accuracy: `0.9441904761904761`
- Test loss per window: `0.20828659610293107`
- Predicted classes used: `10/10`
- Max predicted-class share: `0.10938095238095238`
- Zero-predicted classes: `[]`

Predicted class counts across `21,000` windows:

| Class | Count | Share |
|---:|---:|---:|
| 0 | 2171 | 0.10338095238095238 |
| 1 | 2087 | 0.09938095238095238 |
| 2 | 1951 | 0.0929047619047619 |
| 3 | 2162 | 0.10295238095238095 |
| 4 | 1916 | 0.09123809523809524 |
| 5 | 2177 | 0.10366666666666667 |
| 6 | 2025 | 0.09642857142857143 |
| 7 | 2026 | 0.09647619047619048 |
| 8 | 2188 | 0.1041904761904762 |
| 9 | 2297 | 0.10938095238095238 |

Confusion matrix:

`outputs/rebuild_reproduction/repro_realonly_993_window30x21_no_rpm_ep50_1000x21_inference_rerun/repro_realonly_993_window30x21_no_rpm_ep50_realtest_1000x21_windows_rerun/confusion_matrix/variable_window_all_starts_per_window.png`

Metrics JSON:

`outputs/rebuild_reproduction/repro_realonly_993_window30x21_no_rpm_ep50_1000x21_inference_rerun/repro_realonly_993_window30x21_no_rpm_ep50_realtest_1000x21_windows_rerun/confusion_matrix/variable_window_all_starts_per_window_metrics.json`

### Mean Logits Across 21 Windows Per Video

- Accuracy: `0.96`
- Predicted classes used: `10/10`
- Max predicted-class share: `0.109`
- Zero-predicted classes: `[]`

Predicted class counts across `1000` videos:

| Class | Count | Share |
|---:|---:|---:|
| 0 | 102 | 0.102 |
| 1 | 102 | 0.102 |
| 2 | 95 | 0.095 |
| 3 | 100 | 0.1 |
| 4 | 91 | 0.091 |
| 5 | 103 | 0.103 |
| 6 | 97 | 0.097 |
| 7 | 96 | 0.096 |
| 8 | 105 | 0.105 |
| 9 | 109 | 0.109 |

Confusion matrix:

`outputs/rebuild_reproduction/repro_realonly_993_window30x21_no_rpm_ep50_1000x21_inference_rerun/repro_realonly_993_window30x21_no_rpm_ep50_realtest_1000x21_windows_rerun/confusion_matrix/variable_window_mean_logits_per_video.png`

Metrics JSON:

`outputs/rebuild_reproduction/repro_realonly_993_window30x21_no_rpm_ep50_1000x21_inference_rerun/repro_realonly_993_window30x21_no_rpm_ep50_realtest_1000x21_windows_rerun/confusion_matrix/variable_window_mean_logits_per_video_metrics.json`

### Fixed First-30 Reference

- Accuracy: `0.923`
- Predicted classes used: `10/10`
- Max predicted-class share: `0.111`
- Zero-predicted classes: `[]`

Variable-window mean-logits evaluation improved accuracy from fixed first-30 `0.923` to `0.96`, while preserving a balanced prediction distribution.

## Interpretation

Observed:

- The no-RPM real-only model does not collapse on the held-out real test videos.
- All classes receive predictions under both per-window and per-video mean-logits evaluation.
- The predicted distribution is close to uniform, with max predicted-class share around `0.109`.
- Runtime checks confirm RPM values existed in the data but the tensor given to the model was zeroed.

Paper-candidate claim, if supported by later review:

- Real-only training with 30-frame temporal window expansion and no RPM input can produce a well-distributed, high-accuracy held-out real-video classifier when inference averages or aggregates over all 30-frame windows from the first 50 frames.

Do not claim from this artifact alone:

- That the model is invariant to RPM.
- That this result transfers to synthetic-to-real training.
- That attention maps prove causal viscosity reasoning.
- That variable-window inference is equivalent to fixed-window inference.

## Entropy-Attention Regime-Flip Diagnostic

Date recorded: 2026-05-29

This section links the entropy-change analysis to the saved attention maps. It is
diagnostic evidence for model behavior, not a replacement for the primary
real-only variable-window result above.

Attention provenance:

- Attention run:
  `allnew_synth_no_rpm_augv1_weight_real_attention_synth_val`
- Attention split: real test
- Attention summary:
  `outputs/rebuild_reproduction/allnew_no_rpm_diagnostics/allnew_synth_no_rpm_augv1_weight_real_attention_synth_val/real_test_attention/summary.json`
- Attention panel:
  `outputs/rebuild_reproduction/allnew_no_rpm_diagnostics/allnew_synth_no_rpm_augv1_weight_real_attention_synth_val/real_test_attention/panels/attention_by_true_viscosity_class.png`
- Attention volumes:
  `outputs/rebuild_reproduction/allnew_no_rpm_diagnostics/allnew_synth_no_rpm_augv1_weight_real_attention_synth_val/real_test_attention/volumes`

Important caveat:

- This attention run is a frozen synthetic-to-real diagnostic with real-test
  accuracy `0.139`. It is useful for explaining a collapsed shortcut-like model,
  but it should not be described as the attention behavior of the high-accuracy
  real-only no-RPM model.

Temporal entropy-attention figures:

- Side-by-side entropy/attention heatmaps:
  `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/entropy_attention_correlation/temporal_entropy_attention_side_by_side_heatmaps.png`
- Regime-flip explanation graph:
  `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/entropy_attention_correlation/temporal_entropy_attention_regime_flip_explanation.png`
- Per-class temporal correlation table:
  `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/entropy_attention_correlation/entropy_attention_profile_correlation_by_class.csv`

Key temporal finding:

- Classes `0-3`: strong anti-correlation between temporal entropy profile and
  temporal attention profile. Mean Pearson is about `-0.97`. The main entropy
  peak occurs around tubelets `8-9`, but attention peaks very late, around
  tubelet `24`.
- Classes `4-9`: strong positive correlation between temporal entropy profile
  and temporal attention profile. Mean Pearson is about `0.96`. Attention peaks
  near the entropy peak, usually around tubelets `7-10`.

Interpretation:

- The attention/entropy relationship flips across viscosity regimes.
- Low-viscosity classes have high temporal entropy, but the collapsed
  synthetic-to-real model does not attend to the main entropy-change window. It
  shifts attention toward the end of the clip, which suggests that its attention
  is not stably tracking the physically meaningful deformation interval.
- Higher-viscosity classes have lower temporal entropy and show much better
  alignment between attention and entropy. In those classes, attention peaks near
  the same temporal region where the video has the strongest entropy-change
  signal.
- This supports the shortcut hypothesis for the collapsed synthetic-to-real
  model: the model is sensitive to viscosity-regime structure, but its attention
  is not a regime-invariant detector of fluid-induced deformation. It behaves
  differently in low- and high-viscosity regimes.

How to read the temporal graphs:

- In `temporal_entropy_attention_side_by_side_heatmaps.png`, the left panel is
  row-normalized temporal entropy by class, the middle panel is row-normalized
  attention by class, and the right panel is the z-score difference between
  attention shape and entropy shape.
- In `temporal_entropy_attention_regime_flip_explanation.png`, the top-left
  panel gives the per-class profile correlation, the top-right panel compares
  entropy peak tubelet against attention peak tubelet, and the bottom panel
  overlays representative class curves.

Spatial entropy-attention figures:

- Spatial side-by-side maps by class:
  `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/spatial_entropy_attention_correlation/spatial_change_attention_side_by_side_by_class.png`
- Spatial significance by class:
  `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/spatial_entropy_attention_correlation/spatial_attention_patch_correlation_significance_by_class.png`
- Spatial overall summary:
  `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/spatial_entropy_attention_correlation/spatial_attention_overall_summary.png`
- Spatial per-sample correlation table:
  `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/spatial_entropy_attention_correlation/spatial_attention_per_sample_patch_correlations.csv`
- Spatial significance table:
  `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/spatial_entropy_attention_correlation/spatial_attention_patch_correlation_significance.csv`

Key spatial finding:

- The temporal dimension was collapsed first. Attention volumes `(14, 14, 25)`
  were summed over time into `14x14` spatial attention maps. Video-derived maps
  were collapsed over frames `0..49` into matching `14x14` patch maps.
- Overall spatial alignment with attention is statistically significant for all
  tested proxies, but the strongest effects are from frame-difference maps:
  - Frame-difference magnitude: mean patch correlation `0.174`, p `1.33e-60`.
  - Frame-difference patch entropy: mean patch correlation `0.165`, p
    `1.89e-56`.
  - Residual patch entropy: mean patch correlation `0.074`, p `2.35e-30`.
  - Residual magnitude: mean patch correlation `0.065`, p `1.03e-28`.
- The meaningful spatial alignment is concentrated in high-viscosity classes
  `6-9`. Low classes `0-3` are weak, negative, or inconsistent.

Spatial class-mean examples:

| Class | Frame-diff entropy corr. | Frame-diff magnitude corr. |
|---:|---:|---:|
| 0 | `-0.1109` | `0.0090` |
| 1 | `-0.0998` | `-0.0733` |
| 2 | `-0.1483` | `-0.0947` |
| 3 | `-0.0852` | `-0.1518` |
| 6 | `0.6966` | `0.6661` |
| 7 | `0.7381` | `0.7582` |
| 8 | `0.7975` | `0.8076` |
| 9 | `0.8457` | `0.8755` |

Spatial interpretation:

- The spatial result matches the temporal result. The collapsed model aligns
  with spatial deformation/entropy structure mainly for higher-viscosity
  classes.
- For low-viscosity classes, attention does not reliably sit on the regions with
  the strongest frame-difference entropy or magnitude.
- This makes the shortcut concern stronger: the model does not appear to use one
  stable spatiotemporal deformation rule across all viscosity classes.

## Analysis Operations Log

Date recorded: 2026-05-29

The following analyses were run in the side conversation and produced local
artifacts under `outputs/rebuild_reproduction/entropy_probe/`.

1. Viscosity adjacency, entropy, and accuracy link.
   - Built logarithmic viscosity-adjacency graphs linking class accuracy and
     temporal entropy change.
   - Main graph:
     `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/paper_adjacency_entropy_link/viscosity_adjacency_entropy_accuracy_log.png`
   - Compact overlay:
     `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/paper_adjacency_entropy_link/viscosity_adjacency_accuracy_entropy_overlay_log.png`
   - Key result: `log10_nearest_viscosity_gap` vs accuracy had Pearson
     `0.7942` and Spearman `0.8268`; `log10_nearest_viscosity_gap` vs temporal
     entropy had Pearson `-0.8997` and Spearman `-0.9119`.

2. RPM versus viscosity correlation with accuracy.
   - Compared sample-level and grouped correlations against correctness.
   - Key result: raw sample-level correlations were weak for both RPM and
     viscosity; class-level viscosity adjacency was the stronger explanatory
     axis than raw RPM or raw viscosity.

3. Baseline clean-pattern entropy versus pattern accuracy.
   - Built a 4-pattern graph of clean background entropy against pattern-level
     accuracy.
   - Graph:
     `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/pattern_entropy_accuracy/pattern_entropy_vs_accuracy.png`
   - Key result: clean pattern entropy alone was not a stable explanation
     (`n=4`, Pearson `-0.5674`, Spearman `-0.2000`).

4. Temporal frame-difference entropy over the first 50 frames.
   - Computed frame-to-frame entropy for frames `0..49` across the `1000` real
     test videos, matching the `1000*21` variable-window context.
   - Main graph:
     `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/frame_temporal_entropy/temporal_diff_entropy_over_first50_frames.png`
   - Compact graph:
     `outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/frame_temporal_entropy/temporal_diff_entropy_over_first50_frames_compact.png`
   - Key result: mean temporal entropy was `5.3964` overall, `5.3867` for
     correct videos, and `5.6279` for wrong videos.

5. Temporal entropy profile versus temporal attention profile.
   - Collapsed attention volumes into 25 temporal tubelets and aligned entropy
     transitions from the first 50 frames to those tubelets.
   - Figures and tables are listed in the Entropy-Attention Regime-Flip
     Diagnostic section above.
   - Key result: classes `0-3` anti-align; classes `4-9` align.

6. Spatial video-change maps versus spatial attention maps.
   - Collapsed attention over temporal tubelets into `14x14` spatial maps.
   - Computed `14x14` video-derived maps for frame-difference magnitude,
     frame-difference patch entropy, residual magnitude, and residual patch
     entropy after collapsing over frames `0..49`.
   - Figures and tables are listed in the spatial part of the
     Entropy-Attention Regime-Flip Diagnostic section above.
   - Key result: spatial alignment is significant overall but concentrated in
     high-viscosity classes `6-9`.

7. Artifact safety note.
   - These operations used local CPU/OpenCV/Pandas/Matplotlib analysis only.
   - No training was launched from this side conversation.
   - No checkpoint, dataset, W&B run, or source code path was modified by these
     analyses, except for this documentation update to `important.md`.
