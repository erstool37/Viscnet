# Viscnet Dataset-Agnostic Paper Rebuild Pipeline

This document is the future new-dataset protocol for paper-rebuild planning,
execution, QA, and reviewer passes. It is intentionally dataset-agnostic: new
simulation and real-video datasets enter through manifests and dataset configs,
not through hardcoded local paths.

Use this file only after the user provides or explicitly scopes work to a new
dataset. For current rebuild runs, `checklist.md` remains the active acceptance
contract unless the user explicitly accepts a migration before launch.

For new-dataset work, Planner, QA, reviewer, and implementation subagents must
treat this file as the method contract unless the user explicitly accepts a
change before launching the affected run.

## 1. Clarified Protocol Decisions

The following decisions came from the planning conversation and should not be
re-litigated during cue generation unless the user explicitly asks.

- The scope is a strict paper-rebuild protocol, not a general benchmark suite.
- This protocol is future/new-dataset scoped. It is not a retroactive acceptance
  contract for existing runs.
- Datasets plug in through a standard manifest. Dataset-specific adapters are
  allowed only if they emit the standard manifest before any training stage.
- Targets are both viscosity class and continuous `log10(nu)` viscosity.
- Mandatory RGB encoder families are ViViT, VideoMAE-family video transformer,
  and 3D ResNet.
- Optical-flow models are optional after RGB baselines complete.
- Every mandatory RGB encoder runs synthetic classification, real transfer
  classification, and smaller-LR real regression.
- Grid search is 3 learning rates by 3 epoch budgets per stage.
- Grid defaults are based on prior successful and failed local training logs,
  not arbitrary broad sweeps.
- Checkpoint selection is loss-based only. Use a named selection split. If a
  legacy config calls that stream `test_loss`, the report must label it as the
  selection split and reserve separate held-out evidence for paper claims when
  the new dataset provides it.
- Accuracy, MAE, confusion distribution, calibration, and attention diagnostics
  are post-hoc evidence, never checkpoint-selection metrics.
- Plateau assessment uses the loss tail and patience behavior together.
- Non-plateau synthetic weights may feed downstream stages, but must be marked
  caveated and cannot be promoted as clean paper evidence.
- Frozen synthetic-to-real confusion distribution is required as a diagnostic,
  but poor distribution must not block transfer learning.
- Labeled real manifests are required for paper evidence.
- Fine-tuning defaults to unfreezing the full model.
- Regression trains with MSE on normalized `log10(nu)` and reports MAE in
  descaled `log10(nu)` space.
- Regression LR is smaller than the classification LR region and follows the
  evidence-based regression grid in this document.
- Re, Ca, and We values come from dataset config or metadata; if absent, compute
  documented proxies and label them as proxies.
- Re/Ca/We grouping is dataset-config-defined. Do not invent automatic bins when
  the dataset provides meaningful groups or values.
- The full analysis suite is run on the best ViViT classification-to-regression
  chain after the baseline comparison.
- GMM ablation starts from the best ViViT regression checkpoint.
- GMM component counts are `K = 1, 3, 5, 7`.
- Calibration error (CE) is the mean absolute nominal-vs-empirical coverage gap.
- "Bayesian inference" is not a required method label. Epistemic evidence is
  produced by a 3-seed final ensemble.
- `pipeline.md` must support future Planner and QA subagents through run cards,
  QA gates, and cue-generation rules.
- All W&B-backed training, evaluation logging, queue summaries, and post-eval
  provenance must use exactly one project: `REALVISCNET`.

## 2. Dataset Contract

Every dataset must be represented by one or more immutable manifest files. A
manifest row represents one train/eval sample. For windowed video protocols, a
row may represent a window; include the source video ID so per-video aggregation
can be reconstructed.

Required manifest fields:

- `sample_id`: stable unique row ID.
- `domain`: `simulation` or `real`.
- `split`: `train`, `val`, `test`, or an explicit named split such as
  `real_frozen_eval`.
- `video_path`: path to the video or pre-extracted clip/window.
- `viscosity_class`: integer class label for classification.
- `log_viscosity`: continuous target in `log10(nu)` space for regression.
- `rpm`: rotation speed or equivalent driving-rate metadata.
- `pattern_id`: background/render/pattern identifier.

Required for physics analysis:

- Dataset-provided `Re`, `Ca`, and `We`, or enough raw fields to compute
  documented proxies.
- If proxies are computed, the report must name them `Re_proxy`, `Ca_proxy`, and
  `We_proxy`, define the formula, and avoid presenting them as exact values.

Preferred manifest fields:

- `parameters_path`, `parameters_norm_path`, `background_path`.
- `density`, `kinematic_viscosity`, `dynamic_viscosity`, `surface_tension`.
- `source_video_id`, `window_start`, `window_end`, `frame_count`, `fps`.
- Dataset-config group labels for Re/Ca/We regimes when the dataset owner
  provides meaningful non-binned groupings.

Dataset rules:

- No active run may depend on a private folder convention that is not captured
  in the manifest or config.
- Dataset adapters must be deterministic and must write a manifest snapshot
  before training.
- Dataset-specific manifest adapters are intentionally deferred until the new
  dataset is provided. Do not treat the absence of those adapters before dataset
  delivery as a methodology blocker.
- The manifest snapshot used for a run is a required QA artifact.
- Labeled real test/eval manifests are mandatory for paper evidence. Unlabeled
  deployment diagnostics are out of scope for acceptance claims.
- Windowed datasets must split by `source_video_id` or a stricter group key
  before window generation. Overlapping windows from the same source video must
  not cross train, selection, or held-out evidence splits.

## 3. Global Execution And Logging Rules

All stages use the same provenance rules.

- W&B project must be exactly `REALVISCNET`.
- Launchers must source the intended environment and reject a run if
  `WANDB_PROJECT` resolves to anything else, including historical variants such
  as `allnewViscnet`, `allnewviscnet`, `viscnet`, `viscnet-rebuild`, or
  `re-rebuild-viscnet`.
- Required W&B scalars are `train_loss`, `val_loss`, and `test_loss` when those
  values exist for the stage.
- Rich diagnostics such as confusion matrices, distributions, reliability plots,
  attention maps, Grad-CAM maps, t-SNE coordinates, sparsification curves, and
  calibration tables are written as local artifacts and may be linked or logged
  as files/images.
- Every run must record config path, manifest path, run name, seed, git commit,
  dirty flag, launch command, W&B run ID/URL, checkpoint path, output root, and
  log path.
- Long training must run through detached queues/watchers. The watcher writes a
  completion marker and summary after the process exits; Codex must not monitor
  active training in a loop.
- Generated datasets, checkpoints, outputs, W&B logs, and large binaries are not
  committed unless explicitly requested.

## 4. Run Card Schema For Planner Cues

Planner subagents should emit one run card per concrete run. QA subagents should
read the same card to decide whether evidence is complete.

Required run-card fields:

- `stage`: one of the stages in Section 5.
- `dataset_id`: dataset/config identifier, not a raw folder name.
- `encoder`: `vivit`, `videomae`, `3d_resnet`, or optional extension name.
- `task`: `classification`, `frozen_eval`, `regression`, `gmm`, or `analysis`.
- `input_manifest`: manifest snapshot path.
- `train_manifest`, `val_manifest`, `test_manifest`: explicit split manifests.
- `init_checkpoint`: required for transfer, regression, and GMM; empty for
  from-scratch synthetic pretraining.
- `output_checkpoint`: checkpoint path to write.
- `loss`: `CE`, `MSE`, or `GMM_NLL`.
- `lr`, `max_epochs`, `patience`, `seed`, `batch_size`, `num_workers`.
- `checkpoint_selection_metric`: always `val_loss` or another loss stream
  explicitly named as the selection split before launch.
- `selection_split`: named split used for checkpoint selection.
- `heldout_evidence_split`: final split reserved for paper metrics when the
  dataset provides enough labeled data.
- `wandb_project`: must be `REALVISCNET`.
- `required_artifacts`: list from Section 8.
- `depends_on`: prior run cards or artifacts.
- `qa_status`: `pending`, `accepted`, `retry required`, or `blocked`.

Recommended run-name template:

```text
{dataset_id}_{stage}_{encoder}_{task}_lr{lr}_ep{max_epochs}_seed{seed}
```

For GMM:

```text
{dataset_id}_gmm_vivit_k{K}_lr{lr}_ep{max_epochs}_seed{seed}
```

## 5. Ordered Training Pipeline

### Stage 0: Preflight

Purpose: prove the dataset, config, and provenance setup are ready before any
expensive run.

Required checks:

- Manifests exist and parse.
- All required fields in Section 2 are present.
- Class counts, split sizes, pattern counts, and Re/Ca/We group/value coverage
  are summarized.
- Video paths and metadata paths resolve.
- Target scaling/descaling for `log10(nu)` is defined and tested on a small
  sample.
- W&B project resolves to `REALVISCNET`.
- Checkpoint and output roots are writable and ignored by Git.
- The launch plan identifies GPU count, batch size, workers, seed, and whether
  the run is classification, regression, GMM, frozen eval, or analysis.

Stage output:

- `preflight_summary.json`
- `preflight_summary.md`
- Manifest snapshots used by later run cards.

### Stage 1: Synthetic Classification Pretraining

Purpose: train each mandatory encoder on the simulation dataset and build
classification weights.

Mandatory encoders:

- ViViT.
- VideoMAE-family video transformer.
- 3D ResNet.

Task setup:

- Train from scratch or from accepted public pretrained initialization when the
  model family convention requires it.
- Use CE loss on `viscosity_class`.
- Use loss-based checkpoint selection only.
- Run the full 3x3 LR/epoch grid unless a dataset-specific launch plan approved
  before launch narrows the grid.

Default grid:

- LR: `5e-6`, `1e-5`, `2e-5`.
- Max epochs: `50`, `80`, `110`.

Context behind the defaults:

- Prior local synthetic runs around `1e-5` improved for long schedules; one
  completed no-RPM synthetic run improved through epoch 80.
- Short or interrupted synthetic runs did not prove plateau.
- The grid therefore keeps `1e-5` as the center and includes lower/higher
  options without a broad sweep.

Required stage evidence:

- Train/validation/test loss curves.
- Best checkpoint and best epoch.
- Plateau assessment.
- Classification accuracy and confusion matrix on simulation validation/test.
- W&B run ID/URL in `REALVISCNET`.

### Stage 2: Frozen Synthetic-to-Real Inference

Purpose: test how synthetic classification weights behave on labeled real video
before transfer learning.

Task setup:

- Use each Stage 1 checkpoint with no training.
- Run inference on the labeled real frozen-eval manifest.
- Record confusion and class-distribution diagnostics.

Required diagnostics:

- Accuracy and confusion matrix.
- Predicted class counts and shares.
- Number of predicted classes used.
- Maximum predicted-class share.
- Zero-predicted classes.
- Per-class accuracy.

Protocol rule:

- Poor frozen real distribution does not block transfer learning.
- The QA status should note whether distribution is healthy, collapsed, or
  ambiguous, but this is report-only.

### Stage 3: Real Classification Transfer

Purpose: adapt each synthetic-pretrained encoder to labeled real videos and
build real classification weights.

Task setup:

- Initialize from the corresponding best or caveated Stage 1 checkpoint.
- Fine-tune all model parameters by default.
- Use CE loss on `viscosity_class`.
- Use loss-based checkpoint selection only.
- Run the 3x3 LR/epoch grid.

Default grid:

- LR: `5e-6`, `1e-5`, `2e-5`.
- Max epochs: `25`, `45`, `70`.

Context behind the defaults:

- Prior real transfer runs often reached best validation loss early and then
  overfit within roughly 10-20 epochs at `1e-5`.
- The epoch grid includes a short budget for overfit-prone transfers, a middle
  budget matching prior transfer configs, and a longer budget for datasets that
  improve more slowly.

Required stage evidence:

- Train/validation/test loss curves.
- Best checkpoint and best epoch.
- Plateau assessment.
- Real classification accuracy, confusion matrix, per-class accuracy, and
  reliability/calibration diagnostic if available.
- Direct comparison against the corresponding frozen synthetic-to-real result.

### Stage 4: Real Regression Fine-Tuning

Purpose: convert the real classification chain into a continuous viscosity
regressor and build regression weights.

Task setup:

- Initialize from the corresponding Stage 3 real classification checkpoint.
- Fine-tune all model parameters by default.
- Replace/enable the regression head.
- Train with MSE on normalized `log10(nu)`.
- Report MAE in descaled `log10(nu)` space.
- Use loss-based checkpoint selection only.
- Run the 3x3 LR/epoch grid.

Default grid:

- LR: `2e-6`, `5e-6`, `1e-5`.
- Max epochs: `20`, `35`, `50`.

Context behind the defaults:

- Prior local regression runs stabilized with LRs in the `5e-6` to `1e-5`
  region and often early-stopped around 17-25 epochs.
- The grid is intentionally smaller than or equal to the classification LR
  region because regression fine-tunes from a trained classifier.

Required stage evidence:

- Train/validation/test loss curves.
- Best checkpoint and best epoch.
- Plateau assessment.
- MAE in descaled `log10(nu)`.
- Error distribution and parity plot.
- Per-pattern, per-viscosity-class, and Re/Ca/We-group error summaries when
  group metadata is available.

### Stage 5: Baseline Comparison And Winner Selection

Purpose: decide which encoder chain supports headline analysis and which models
serve as baselines.

Comparison table must include:

- Encoder family.
- Synthetic classification accuracy and plateau status.
- Frozen synthetic-to-real accuracy and distribution status.
- Real transfer classification accuracy and plateau status.
- Regression MAE in `log10(nu)` and plateau status.
- W&B run IDs and checkpoint paths.
- Caveats for non-plateau or incomplete runs.

Winner rule:

- For classification, choose the highest real transfer accuracy after plateau
  or caveat annotation. Loss is the tie-breaker.
- For regression, choose the lowest descaled `log10(nu)` MAE after plateau or
  caveat annotation. Loss is the tie-breaker.
- The full analysis suite in Stage 6 uses the best ViViT chain, even if another
  encoder has a better headline metric. The other encoders remain baselines.

Acceptance-threshold rule:

- Dataset-specific thresholds for classification accuracy, regression MAE,
  AUSE, CE, non-inferiority, or confidence intervals are intentionally deferred
  until the new dataset is provided.
- Before any new-dataset launch, the Planner must record the accepted thresholds
  or state that the run is diagnostic only.
- A reviewer should not mark the methodology weak merely because thresholds are
  absent before dataset delivery; the methodology is robust when it requires
  threshold binding before claims are made.

## 6. Plateau And Checkpoint Policy

Checkpoint selection:

- Allowed: `val_loss` or another loss stream explicitly named as the selection
  split before launch.
- If a legacy config reports the selection stream as `test_loss`, the paper
  report must call it the selection split, not the final held-out test.
- When a new dataset provides enough labeled data, reserve a separate held-out
  evidence split for final accuracy, MAE, AUSE, CE, calibration, and uncertainty
  claims.
- Blocked: real-test distribution score, class-count balance, confusion shape,
  accuracy, MAE, AUSE, calibration, attention, Grad-CAM, or entropy metrics.

Plateau status:

- `plateau`: early stopping occurred by patience after the best loss, or the
  final patience window improved by no more than 1 percent relative to the best
  loss and no new best occurred in the final patience window.
- `still improving`: the best loss occurs inside the final patience window, or
  the final patience window improves by more than 1 percent.
- `overfit after best`: validation/test loss worsens after the best epoch while
  training loss continues downward.
- `insufficient`: loss history is missing or too short to classify.

Evidence handling:

- A `still improving` or `insufficient` run may be used for downstream probing
  only with an explicit caveat.
- A caveated upstream checkpoint must propagate its caveat to transfer,
  regression, analysis, and GMM reports.
- No run can be accepted as clean paper evidence without loss-curve inspection.

## 7. Stage 6: ViViT-Best Analysis Suite

Run this suite only after Stage 5 identifies the best ViViT classification and
linked regression checkpoints.

Required classification analyses:

- Confusion matrix and per-class accuracy.
- Final-layer attention maps grouped by dataset-provided Re/Ca/We values or
  groups.
- Final-layer attention maps grouped by `pattern_id` and `viscosity_class`.
- Correct-vs-wrong attention summaries when enough samples exist.
- t-SNE or comparable embedding projection colored by RPM, viscosity class,
  continuous `log10(nu)`, and pattern ID.

Required regression analyses:

- MAE and error distribution in `log10(nu)`.
- Error summaries grouped by pattern ID, viscosity class, RPM, and dataset
  Re/Ca/We groups.
- Relationship between pattern entropy and regression absolute error.
- Relationship between pattern entropy and classification correctness/accuracy.

Required Grad-CAM analyses:

- Grad-CAM overlays on original frames or background images.
- Aggregated Grad-CAM by pattern ID.
- Aggregated Grad-CAM by pattern ID and viscosity class.
- Grad-CAM summaries by Re/Ca/We groups when the dataset provides those groups.

Required entropy analyses:

- Static pattern entropy.
- Video/frame entropy summaries.
- Residual or temporal entropy summaries when available.
- Correlations between entropy features and classification accuracy.
- Correlations between entropy features and regression absolute error.

Analysis rule:

- Attention, Grad-CAM, entropy, and t-SNE are interpretability evidence, not
  checkpoint-selection criteria.
- Claims must distinguish observed correlations from causal explanations.

## 8. Stage 7: GMM And Uncertainty Protocol

Run GMM only after the best ViViT regression checkpoint exists.

### GMM Ablation

Component counts:

- `K = 1`
- `K = 3`
- `K = 5`
- `K = 7`

Task setup:

- Initialize from the best ViViT regression checkpoint.
- Train a GMM regression head using negative log likelihood in normalized
  `log10(nu)` space.
- Report point prediction as the mixture mean in descaled `log10(nu)` space.
- Run the 3x3 LR/epoch grid for each `K`.

Default grid:

- LR: `1e-6`, `2e-6`, `5e-6`.
- Max epochs: `20`, `35`, `50`.

Required metrics:

- MAE in descaled `log10(nu)`.
- GMM predictive mean and variance in `log10(nu)`.
- Sparsification plot with absolute `log10(nu)` error on the y-axis.
- AUSE for error-uncertainty alignment.
- Randomized-uncertainty AUSE baseline.
- Calibration coverage before and after post-hoc calibration.
- CE before and after post-hoc calibration.

AUSE definition:

- Error is absolute error in descaled `log10(nu)`.
- Normalize total error so the maximum curve value becomes 1 before AUSE
  comparison.
- Compute model sparsification by removing or retaining samples according to
  predicted uncertainty.
- Compute oracle sparsification by sorting according to true error.
- Compute randomized baseline by random uncertainty assignments and report the
  mean over repeated randomizations when practical.
- Report AUSE relative to the randomized baseline.

CE definition:

- Use nominal coverage levels `0.50`, `0.68`, and `0.95`.
- CE is the mean absolute difference between nominal and empirical coverage.
- Report raw CE and post-hoc calibrated CE.
- Report the calibration transform or scale factor used.

### Epistemic Evidence

The accepted epistemic-evidence method is a final 3-seed ensemble for the
selected ViViT-GMM setting.

Required setup:

- Train or fine-tune the selected GMM setting with three distinct seeds.
- Keep dataset split, manifests, architecture, loss, LR, epoch budget, and
  checkpoint-selection policy fixed across seeds.
- Record all three W&B run IDs in `REALVISCNET`.

Uncertainty decomposition:

- Aleatoric uncertainty is the per-seed GMM predictive variance in `log10(nu)`,
  including mixture component variance and mixture spread.
- Epistemic uncertainty is the between-seed variance of predictive means.
- Total uncertainty is aleatoric plus epistemic under the ensemble summary.

Claim rule:

- The paper may claim aleatoric uncertainty is dominant only if aleatoric
  contribution is consistently larger than epistemic contribution on the full
  real test split and in grouped analyses by pattern, viscosity class, and
  Re/Ca/We groups.
- If the dominance is mixed or group-specific, report the mixed result rather
  than claiming global dominance.

## 9. Required Artifacts

Each training run must produce:

- Config snapshot.
- Manifest snapshot.
- W&B run ID/URL in `REALVISCNET`.
- Local log path.
- Checkpoint path.
- Train/validation/test loss history.
- Plateau report.
- Metrics JSON.
- Completion summary.

Classification runs must also produce:

- Confusion matrix image and JSON.
- Per-class accuracy.
- Prediction records with sample IDs, targets, predictions, and logits.
- Reliability/calibration diagnostics when available.

Regression runs must also produce:

- Prediction records with sample IDs, targets, predictions, and errors.
- MAE in descaled `log10(nu)`.
- Parity plot.
- Error distribution plot.
- Grouped error summaries for pattern, viscosity class, RPM, and Re/Ca/We
  groups when available.

Analysis runs must also produce:

- Attention figures and metrics.
- Grad-CAM overlays and aggregate maps.
- Entropy summary CSV/JSON.
- Entropy-vs-accuracy and entropy-vs-error plots.
- t-SNE coordinate files and plots.

GMM runs must also produce:

- GMM component parameters.
- Predictive mean and variance arrays.
- Sparsification curves.
- AUSE metrics with randomized baseline.
- Calibration coverage and CE before/after post-hoc calibration.
- Ensemble uncertainty decomposition for the selected final setting.

## 10. QA Gates

QA subagents must classify every run and stage as one of:

- `accepted`: method, artifacts, W&B provenance, metrics, and loss-curve
  evidence satisfy this protocol.
- `retry required`: run exists but is incomplete, caveated beyond accepted
  evidence, missing non-blocking artifacts, or failed a required grid cell.
- `blocked`: required inputs, labels, checkpoints, W&B provenance, or method
  constraints are absent or wrong.

Blocking conditions:

- W&B project is not exactly `REALVISCNET`.
- Real evidence manifest is unlabeled.
- Manifest snapshot is missing.
- Checkpoint is missing.
- Loss history is missing.
- Checkpoint was selected by any non-loss diagnostic.
- Dataset-specific assumptions are not captured in manifest/config.
- Required prior-stage checkpoint is absent.
- GMM or uncertainty claims are made without required GMM/ensemble artifacts.

Retry-required conditions:

- Plateau is `still improving`, `overfit after best`, or `insufficient` and the
  run is being treated as clean paper evidence.
- A required grid cell failed and affects comparison.
- Frozen real confusion/distribution artifacts are absent.
- Regression MAE artifacts are absent.
- Any `K` in the GMM ablation is missing.
- AUSE, randomized AUSE baseline, calibration coverage, or CE is missing.
- Aleatoric dominance is claimed without the 3-seed decomposition.

Accepted-stage requirements:

- Stage 1 accepted only after all mandatory encoders have synthetic
  classification weights or explicitly caveated retry-required records.
- Stage 2 accepted only after frozen real inference artifacts exist for each
  Stage 1 checkpoint.
- Stage 3 accepted only after all mandatory encoders have real transfer
  classification evidence.
- Stage 4 accepted only after all mandatory encoders have real regression
  evidence.
- Stage 5 accepted only after a comparison table selects the best ViViT chain
  and records baseline outcomes.
- Stage 6 accepted only after all ViViT-best analysis artifacts exist.
- Stage 7 accepted only after GMM K ablation, calibration, sparsification, and
  3-seed uncertainty decomposition exist.

## 11. Reviewer Evaluation Checklist

A reviewer subagent should evaluate the methodology as a paper-review process,
not as a current-repo implementation readiness review. The reviewer should ask
whether the proposed future-dataset paper method is robust, well described, and
grounded under the stated assumptions. If the method is robust, the reviewer
should state that clearly and then list any remaining refinements.

Do not count these intentionally deferred items as methodology blockers before
the new dataset is provided:

- Dataset-specific manifest adapter implementation.
- Dataset-specific accuracy, MAE, AUSE, CE, or non-inferiority thresholds.
- Dataset-specific Re/Ca/We grouping definitions.
- Dataset-specific split sizes and class supports.

Review against these points:

- Dataset agnosticism: every dataset-specific assumption is represented in a
  manifest, config, or adapter output.
- Reproducibility: every run has config, manifest, seed, checkpoint, W&B
  provenance, log, and output root.
- Fair baseline comparison: ViViT, VideoMAE-family, and 3D ResNet all pass
  through the same stage structure before analysis is restricted to ViViT-best.
- Selection hygiene: checkpoints are selected only by loss, while headline
  metrics remain post-hoc evidence.
- Plateau evidence: loss curves, patience behavior, best epoch, and tail slope
  are reported before accepting weights.
- Domain transfer transparency: frozen synthetic-to-real confusion is reported
  even when poor, and poor distribution is not hidden or used as a blocker.
- Regression correctness: training target and reported MAE both use the intended
  `log10(nu)` space with documented scaling/descaling.
- Interpretability grounding: attention, Grad-CAM, entropy, and t-SNE claims are
  tied to concrete artifacts and not used as proof of causality.
- UQ grounding: MAE, AUSE, randomized AUSE baseline, CE, and 3-seed uncertainty
  decomposition are all present before uncertainty claims.
- Claim discipline: aleatoric dominance is reported only when supported across
  aggregate and grouped evidence.

Reviewer verdict guidance:

- `robust`: the methodology is scientifically coherent, evidence-producing, and
  has explicit gates for future dataset-specific adapters and thresholds.
- `robust with refinements`: the method is sound, but specific documentation or
  implementation details should be tightened before launch.
- `not robust yet`: the method would allow unsupported paper claims even after a
  dataset is provided.

## 12. Planner Cue Generation Rules

Planner subagents should generate cues in this order:

1. Preflight manifest and W&B checks.
2. Stage 1 synthetic classification grid for each mandatory encoder.
3. Stage 2 frozen synthetic-to-real inference for each completed Stage 1
   checkpoint.
4. Stage 3 real transfer classification grid for each mandatory encoder.
5. Stage 4 real regression grid for each mandatory encoder.
6. Stage 5 comparison and ViViT-best chain selection.
7. Stage 6 ViViT-best analysis suite.
8. Stage 7 ViViT-GMM K ablation.
9. Stage 7 selected-GMM 3-seed ensemble and uncertainty decomposition.
10. Final QA and reviewer methodology report.

Cue-generation constraints:

- Do not launch downstream runs until the required upstream checkpoint exists.
- Downstream diagnostic runs may proceed from caveated weights, but the caveat
  must be copied into the run card.
- Do not broaden the grid after seeing results unless the user explicitly
  accepts the new grid before launch.
- Do not add new acceptance gates after a run launches.
- Do not silently substitute a dataset, split, checkpoint, or W&B project.

## 13. Methodology Status Standard

Use these labels consistently in reports:

- `clean evidence`: accepted run with complete artifacts, correct W&B project,
  loss-based checkpointing, and plateau evidence.
- `caveated evidence`: run can inform downstream work but has non-plateau,
  incomplete, or method caveats.
- `diagnostic only`: useful for interpretation but not acceptance or headline
  claims.
- `retry required`: method is valid but artifacts or run quality are incomplete.
- `blocked`: required inputs, provenance, or method constraints are missing.
