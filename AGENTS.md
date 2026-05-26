# Viscnet Agent Instructions

These are local instructions for `/root/Viscnet`. They specialize the global
Codex operating laws with concrete repository structure, commands, and
experiment workflow.

## Instruction Hierarchy

Follow instructions in this order:

1. User request and current conversation context.
2. This local `AGENTS.md` and nested project instructions.
3. Rules under `rules/`.
4. Active skills and workflow references.
5. General conventions.

Never override truth, provenance, or user-safety requirements from higher-priority
instructions. If evidence is incomplete, say what is known, what is inferred, and
what remains unverified.

## Repository Boundary

- Work inside `/root/Viscnet` unless the user explicitly changes scope.
- Use local Git as the operational record. Do not use GitHub, push, publish, or
  upload artifacts unless the user explicitly asks.
- Keep real datasets, generated outputs, model weights, checkpoints, W&B logs,
  caches, virtualenvs, and secrets out of Git.
- Do not create ad hoc operational journals when local Git status, committed
  docs, checker reports, analyzer reports, and completion markers already carry
  the evidence.

## Repository Purpose

Viscnet is a computer-vision viscosity estimation project. The active codebase is
a PyTorch video training and evaluation stack for synthetic CFD data, real-video
classification, transfer learning, pattern-generalization runs, and rebuild
reproduction analysis.

## Repository Map

- `configs/config.yaml`: default experiment config.
- `configs/rebuild/*.yaml`: rebuild reproduction configs and bounded retry
  variants.
- `configs/rebuild/reference_metrics.json`: reference targets for reproduction
  checks.
- `src/main.py`: primary config-driven training entrypoint.
- `src/datasets/`: dataset and dataloader classes.
- `src/models/`: model architectures and heads.
- `src/losses/`: selectable loss implementations.
- `src/inference/`: inference and diagnostic helpers.
- `src/utils/`: preprocessing, metrics, distributed setup, analysis, and IO
  helpers.
- `scripts/`: operator scripts for setup, training batches, post-training
  analysis, checker reports, plotting, and dataset derivation.
- `checklist.md`: current reproduction validity standard. Treat it as a checker
  contract, not loose notes.
- `distillation.md`: canonical rebuild handoff and current experiment state.
- `REBUILD_REPRODUCTION.md`: compatibility pointer to `distillation.md`.
- `ARCHIVE/`: old experimental code. Do not make archived code part of the active
  execution path unless the user explicitly asks.
- `dataset/`, `outputs/`, `wandb/`, checkpoints, and derived datasets are local
  artifacts and should not be committed unless explicitly requested.

## Active Source Of Truth

- `checklist.md` is the durable reproduction and acceptance contract.
- `distillation.md` is the canonical handoff for current rebuild state.
- Durable workflow rules live in `rules/`.
- Subagent or role-specific conclusions must be reflected in durable files when
  they change gates, acceptance criteria, required artifacts, or launch policy.

## Standard Commands

Use these entrypoints where applicable:

- Setup: `bash scripts/setup.sh`
- Default training command: `bash scripts/dev.sh`
- Rebuild suite: `bash scripts/run_rebuild_reproduction.sh`
- Transfer LR-hold then window diagnostic: `bash scripts/run_transfer_lrhold_then_window30.sh`
- Rebuild checker: `python3 scripts/check_rebuild_results.py`
- Rebuild analysis: `python3 scripts/analyze_rebuild_results.py`
- Post-training artifacts: `python3 scripts/post_rebuild_training.py`
- Ruff lint: `python3 -m ruff check .`

Before running expensive training, verify the requested config, dataset path,
checkpoint path, output path, GPU assumptions, and W&B behavior. W&B-backed
rebuild runs require `WANDB_API_KEY`; if it is absent, do not claim acceptance
for completed runs until W&B provenance is restored or the user explicitly
accepts a non-W&B diagnostic.

## W&B Provenance And Troubleshooting

Training through `src/main.py` logs to W&B because it calls `wandb.init` and
`wandb.log`. Standalone inference/evaluation scripts do not count as W&B-backed
unless they explicitly initialize W&B or call a post-eval logger such as
`scripts/log_eval_metrics_to_wandb.py`.

When the user reports that W&B is not updating:

- First determine whether the active command is a training path or standalone
  eval path. Do not assume eval scripts log to W&B just because training does.
- Check the active process environment without printing secrets. Verify only
  presence/absence of `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, and
  `WANDB_MODE`.
- Check whether the launcher sources `.env`. Detached queue scripts that need
  W&B must load `.env` before launching Python or post-eval loggers.
- The canonical rebuild project is `re-rebuild-viscnet`. Rebuild queues and
  post-eval loggers must set or pass this project explicitly; do not let `.env`
  silently redirect rebuild evidence to a generic project such as `viscnet`.
- `src/main.py` forces configs launched from `configs/rebuild/` to use
  `re-rebuild-viscnet` even if an older YAML still has `project:
  viscnet-rebuild`.
- Verify the actual W&B project and entity from the environment or run URL before
  deciding that a run is missing. If a rebuild run lands outside
  `re-rebuild-viscnet`, re-log or rerun the provenance step into
  `re-rebuild-viscnet` and record the corrected run ID.
- Check local `wandb/run-*` directories and the exact queue log for a new W&B run
  ID, URL, offline mode, authentication error, or network failure.
- When local logs claim sync but the UI does not show the run, verify through the
  W&B API using only project/entity/run IDs, never printing the API key.
- If metrics already exist locally but W&B has no run, log the completed metrics
  with `scripts/log_eval_metrics_to_wandb.py` and record the returned W&B run ID
  and URL in the queue summary.
- If a queue has both compute and post-eval W&B upload, treat W&B upload failure
  as an incomplete provenance state, not as accepted completion. Record the local
  metric path and the missing/failed W&B state explicitly.
- Never write API keys, tokens, or `.env` contents into docs, logs, summaries, or
  final responses.

## Machine Learning Workflow

For machine learning training, refactor, debugging, or result-analysis work, read:

- `rules/subagents/machine-learning.md`
- `rules/structures/machine-learning-codebase-structure.md`

Use `checklist.md` as the active standard for reproduction validity. Do not
retroactively change acceptance criteria to make a run pass. Enhancement variants,
hyperparameter sweeps, architecture changes, label smoothing, UQ/regression work,
or alternative datasets are out of scope for reproduction unless the user accepts
that change before launch.

For substantial ML tasks:

- Distill the requested method and expected artifacts before launch.
- Do not use `training.update_density.optimizer_microbatch_size` or per-sample
  optimizer stepping for rebuild training unless the user explicitly asks for it.
  Normal batch training is the default for new recovery runs. If an old config
  contains microbatch optimizer updates, fork a normal-batch config with a new
  run/checkpoint name rather than silently launching the slow path. Existing
  microbatch artifacts may be analyzed as historical evidence, but do not promote
  them as the active launch policy without explicit user approval.
- Use Superpowers-style brainstorming and plan writing for substantial,
  ambiguous, high-risk, or multi-run requests. Refine the user's rough goal into
  a concrete method, constraints, run list, expected artifacts, and acceptance
  checks before editing configs or launching jobs.
- Do not start training, destructive operations, dataset generation, expensive
  inference, broad sweeps, or long analysis jobs if material details are unclear.
  Material details include target configs, dataset readiness, W&B project/entity,
  checkpoint chain, GPU count, expected outputs, pass/fail thresholds, and
  whether the run is reproduction, diagnostic, or enhancement.
- If clarification is required, ask concise blocking questions first. In Plan
  mode, use the user-input question tool when available so the user can answer
  in the structured popup. Outside Plan mode, ask directly and wait.
- Treat "make a plan first", "think through options", "not sure", and conflicting
  run instructions as a stop signal for launch. Produce the plan and wait for
  explicit approval before starting.
- Keep experiment wiring config-driven where practical.
- Preserve `src/main.py` as the discoverable training path.
- Record or verify run names, checkpoints, metrics JSON, confusion matrices,
  reliability plots, and W&B run IDs when judging completion.
- Separate observed metrics from inferred causes.
- Compare only against matching split, sample count, config family, and evaluation
  scope.

## Analyzer And Checker Workflow

- Use an analyzer role for artifact-grounded debugging, hypothesis generation,
  and next-method recommendations.
- Use a checker role for pass/fail readiness judgments against `checklist.md`.
  Repo-local `checker` means the `Checklist` role defined in
  `rules/subagents/machine-learning.md`.
- Use Superpowers systematic-debugging discipline for analysis and debugging
  runs: observe first, isolate the failure mode, form evidence-tied hypotheses,
  probe the smallest discriminating question, then verify the fix.
- Analyzer work should collect the relevant configs, metrics files, checkpoint
  traces, prediction artifacts, logs, loss traces, confusion matrices, reports,
  and model-interface diagnostics when available.
- Every result-analysis pass must include loss-graph or loss-curve analysis when
  train/validation loss history exists. This is mandatory even when confusion
  matrices or accuracy already seem explanatory. Inspect W&B curves when
  available, otherwise reconstruct curves from local logs. Record train-loss and
  validation-loss shape, divergence/overfit timing, checkpoint-save epochs,
  early-stopping behavior, scheduler phase, and whether the loss curves support
  or contradict the accuracy/confusion interpretation.
- Loss-graph analysis must be performed by a `Result Analyzer` subagent when
  the runtime supports subagents and the user has not prohibited delegation. If subagents are not
  available, state that explicitly and perform the same loss-curve analysis
  locally before making result-analysis claims.
- Compare current results against the matching baseline and previous best result
  on the same split and evaluation scope.
- Inspect loss behavior and checkpoint traces when available, including training
  loss, validation loss/metrics, checkpoint selection, validation-to-test drift,
  and any logged optimization diagnostics.
- Inspect error structure with the project-appropriate confusion matrices,
  per-class metrics, calibration/reliability artifacts, and failure cases.
- Tie every hypothesis to evidence. State what it predicts and what artifact or
  run would falsify it.
- Prefer focused probing code over broad guessing. Good probes include LR-curve
  reconstruction, scheduler-step accounting, epoch/horizon comparisons,
  dataloader batch-size and optimizer-update-density calculations, gradient norm
  tracking, train/validation loss drift checks, checkpoint-selection traces,
  attention/attribution exports, calibration/reliability checks, confusion-pair
  audits, and small max-sample smoke runs.
- Probe creatively but keep probes bounded and reversible. Put reusable probes in
  `scripts/` when they may be repeated; keep one-off exploratory output under
  `outputs/rebuild_reproduction/` and out of Git.
- Do not promote a probe result into a new method or acceptance gate until it is
  mapped to evidence and explicitly accepted before launch.
- Analyzer reports must include explicit allowed-next-action and
  blocked-next-action guidance.
- Do not ask for or issue a completion/pass judgment until claims are mapped to
  concrete artifacts and remaining blockers are stated.
- Update `checklist.md` only when a new durable gate or changed acceptance
  criterion is explicitly accepted before the relevant run launches.

## Long-Running Training

- Do not check training results repeatedly while a run is active; it wastes tokens
  and does not improve the experiment.
- Do not use Codex as a waiting loop.
- Do not repeatedly call `write_stdin`, `tmux capture-pane`, `nvidia-smi`, `ps`,
  `tail`, `grep`, `rg`, checker scripts, analyzer scripts, or W&B queries while
  training is still running.
- Do not calculate an ETA and then check later inside Codex. That still creates
  repeated monitoring turns.
- Launch long-running training in detached `tmux` sessions or an equivalent
  detached runner.
- Attach a detached watcher outside Codex. The watcher should wait for process,
  tmux session, or signal completion without reading logs while training runs.
- The watcher must run required post-training commands once after completion when
  analysis is part of the launch plan.
- The watcher must write a completion marker, short final status summary,
  checker/analyzer output paths, and exact log path.
- After launching training and watcher, end the Codex turn immediately.
- If training is already running, attach a detached watcher to the existing PID,
  signal, log, or tmux session, then stop.
- If training is unfinished and no bounded completion artifact exists, report
  `unfinished; watcher should handle completion` and stop.
- Only after completion should Codex read bounded outputs: exact config, exact
  metrics JSON, exact checker/analyzer report, completion summary, and at most
  `tail -n 80` of the exact log.
- Run checker/analyzer once after completion. Do not rerun them in loops.
- If multiple runs are queued, prefer one scripted queue with per-run logs and a
  completion alert over manual babysitting.

## Data And Artifact Safety

- Do not commit local datasets, derived clips, model weights, W&B logs, generated
  outputs, or large binary artifacts unless explicitly requested.
- Keep derived rebuild datasets under `outputs/rebuild_reproduction/derived_datasets/`
  unless a config already specifies another accepted location.
- Keep checkpoint outputs under the configured checkpoint root; for rebuild work
  this is normally `outputs/rebuild_reproduction/checkpoints`.
- If a referenced dataset, checkpoint, W&B run, or output artifact is missing,
  state that explicitly rather than assuming it exists.

## Editing Rules

- Prefer existing repository patterns over new abstractions.
- Keep changes scoped to the requested behavior.
- Avoid hidden one-off scripts when a config option, existing utility, or existing
  script can express the workflow.
- Do not rewrite archived code, datasets, generated outputs, or unrelated files as
  part of routine fixes.
- Respect dirty worktrees. Existing user changes, including deleted data artifacts,
  must not be reverted unless the user asks.

## Validation

- After meaningful code, config, or data-layout changes, run the relevant
  project gates when feasible.
- Prefer focused validation first: YAML parsing for config changes, `bash -n` for
  shell wrappers, `python3 -m py_compile` for changed Python scripts, and ruff or
  tests when the blast radius justifies it.
- Report any validation that was not run and why.

## Reporting

When reporting results, lead with the actionable status: accepted, retry required,
blocked, implemented, or not run. Include exact file paths, commands, configs,
metrics, and missing artifacts when they matter. Keep reproduction judgments
separate from improvement proposals.
