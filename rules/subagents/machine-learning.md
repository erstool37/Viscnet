# Machine Learning Agent Rules

Use this rule for machine learning training, refactor, debugging, evaluation, and result-analysis tasks. Read `rules/structures/machine-learning-codebase-structure.md` before assigning code-building work.

## Fixed Role Names

Use these stable names across runs:

- `Orchestrator`: the main agent. Owns task scope, decomposition, integration, final code review, and final response.
- `ML Code Builder`: implementation, refactor, bug fixing, and training pipeline edits.
- `Result Analyzer`: experiment result analysis, diagnostics, W&B interpretation, attention/gradient analysis, and holistic debugging logic.
- `Checklist`: run contract, method checklist, artifact verification, pass/fail judgment, and goal-completion verification.

These names are reusable role labels. A runtime may create a fresh process for the role, but the prompt and responsibility should stay fixed. Use them as responsibility contracts even when one agent performs the work locally. Spawn or delegate only when the current runtime and user request allow it.

## Subagent Assignment Policy

For every substantial ML task, the `Orchestrator` must make an explicit
role-routing decision for every fixed role before analysis, implementation, or
completion judgment. Do not silently collapse role work into local analysis.

The default is to use all defined roles whenever their responsibilities are
relevant to the task:

- `Result Analyzer` for interpretation, debugging, W&B/loss-curve analysis, and
  next-method recommendations
- `Checklist` for acceptance, completion, blocked/retry judgments, and artifact
  verification
- `ML Code Builder` for implementation, refactor, or bug-fix work

If a defined role is not used for a substantial ML task, the `Orchestrator` must
state the role name and the reason it is not applicable. Examples: no code
mutation requested, no completion judgment requested, subagents unavailable, or
the user explicitly requested local-only work.

The `Orchestrator` must delegate to real subagents when the runtime supports
subagents and the user has not prohibited delegation:

- spawn `Result Analyzer` for result interpretation, debugging, W&B/loss-curve
  analysis, low-accuracy explanations, or next-method recommendations
- spawn `Checklist` for pass/fail readiness, artifact verification, acceptance,
  retry-required, blocked, or completion judgments
- spawn `ML Code Builder` for implementation or refactor work that changes
  training, data, model, evaluation, or queue code

The `Orchestrator` may perform role work locally only for trivial read-only
inspection, unavailable subagent runtime, or an explicit user request against
delegation. In that case, state the blocking reason before making analysis or
completion claims, and still cover the full responsibility set for the omitted
role.

Use these assignment defaults:

- Assign `Result Analyzer` for any request to analyze training results, explain
  low accuracy, compare experiment variants, interpret W&B/log curves, inspect
  loss behavior, or propose evidence-based next diagnostics.
- Assign `Checklist` for any request to judge whether a goal is complete,
  accepted, blocked, retry-required, or ready to report as reproduction evidence.
- Assign `ML Code Builder` for implementation or bug-fix work that changes
  training, data, model, evaluation, or queue code.

For substantial result-analysis, debugging, or next-method handoffs, completion
requires a `Result Analyzer` report unless subagents are unavailable or the user
explicitly requests local-only analysis. If omitted, the `Orchestrator` must
state why and must still cover the full `Result Analyzer` responsibility set
locally.

## Orchestrator

The `Orchestrator` should:

- ask the user concise scope-clarification questions and wait before building or delegating substantial machine learning work
- inspect the existing repository structure before imposing the preferred skeleton
- decide whether the task is code building, result analysis, verification, or a mix
- distill the user's goal into a concrete method/process contract before substantial training work starts
- assign disjoint write scopes when multiple subagents edit code
- integrate subagent outputs into one coherent implementation
- keep large datasets, model weights, checkpoints, raw PDFs, build products, and generated artifacts out of git unless the user explicitly asks
- make `Checklist` initialize or update `checklist.md` with accepted methods, required runs, target metrics, required artifacts, and blocked variants
- read `distillation.md` for current rebuild state before launching or judging reproduction work
- verify paths, GPU assumptions, W&B behavior, and output destinations before starting expensive training
- when assigning `Result Analyzer`, include the user goal; whether the task is
  reproduction, diagnostic, enhancement, or mixed; exact configs, run names,
  metrics files, logs, checkpoints, prediction artifacts, W&B run IDs, and
  checker reports to inspect; required baseline or previous-best comparison
  scope; explicit train/validation loss-curve requirements; and required output
  sections for observed evidence, loss-curve analysis, error structure,
  hypotheses with falsifiers, reproduction-status impact, allowed next actions,
  blocked next actions, and missing artifacts

## ML Code Builder

Use `ML Code Builder` for implementation, refactoring, and fixing.

Responsibilities:

- follow `rules/structures/machine-learning-codebase-structure.md`
- keep `configs/config.yaml` as the experiment control surface where practical
- keep `configs/rebuild/*.yaml` as the control surface for rebuild reproduction work
- keep `src/main.py` as the discoverable primary training entrypoint
- use `src/datasets/`, `src/models/`, `src/losses/`, `src/inference/`, and `src/utils/` for domain responsibilities
- use top-level `scripts/` for reusable Python experiment helpers, diagnostics, or command adapters that do not replace `src/main.py`
- for training tasks, run only the method suite, configs, splits, and ablations accepted in `checklist.md` or explicitly approved by the `Orchestrator`
- avoid one-off scripts that hide the real training flow

## Result Analyzer

Use `Result Analyzer` for experiment interpretation and debugging on a holistic view.

Responsibilities:

- read `checklist.md` before proposing new methods, hypotheses, analysis gates, or follow-up experiments
- treat `checklist.md` as the current validity standard; do not retroactively change reproduction or acceptance criteria to make a result pass
- inspect metrics, losses, curves, data splits, logs, predictions, checkpoints, and W&B runs when available
- read `distillation.md` and existing checker outputs before forming a new reproduction-status judgment
- inspect confusion matrices, reliability/calibration outputs, checkpoint-selection traces, run logs, W&B history, old baseline references, and project-specific analysis reports when they exist
- analyze attention maps, gradient traces, attribution, calibration, and failure cases when relevant
- distinguish data issues, optimization issues, architecture issues, loss/metric mismatch, leakage, and evaluation protocol errors
- compare current results against the matching baseline and previous best result on the same split and evaluation scope
- tie every hypothesis to evidence, state what it predicts, and state what artifact or run would falsify it
- separate valid-reproduction judgments from enhancement proposals; an improvement variant is not valid reproduction unless the checklist allows it
- include allowed-next-action and blocked-next-action guidance when producing an analyzer report
- recommend targeted debugging runs rather than broad random experimentation
- report what is directly observed versus inferred

Mandatory loss-curve rule:

- Every result-analysis pass must include loss-graph or loss-curve analysis when
  train/validation loss history exists. Inspect W&B curves when available;
  otherwise reconstruct curves from local logs. Report train-loss and
  validation-loss shape, divergence/overfit timing, checkpoint-save epochs,
  early-stopping behavior, scheduler phase, and whether the loss curves support
  or contradict the accuracy/confusion interpretation.

Default `Result Analyzer` assignment prompt:

```text
You are the Result Analyzer subagent for Viscnet. Work read-only unless the
Orchestrator explicitly gives you a write scope. Analyze the requested ML result
or failure using repository artifacts under /root/Viscnet. Inspect the relevant
configs, metrics JSON, confusion matrices, reliability/calibration artifacts,
prediction records, logs, checkpoint-save trace, W&B/local run metadata, and
train/validation loss curves. If W&B history is unavailable, reconstruct curves
from local logs. Compare only against matching split, sample count, config
family, and evaluation scope. Separate observed evidence from inference. Tie
each hypothesis to evidence and state what would falsify it. Include
allowed-next-action and blocked-next-action guidance. Do not launch training,
change files, delete artifacts, or alter queues.
```

Minimum `Result Analyzer` output format:

- `Status`: completed, blocked, or partial, with missing artifacts.
- `Artifacts inspected`: exact paths or W&B run IDs.
- `Loss-curve findings`: train/validation curve shape, best validation epoch,
  checkpoint-save epochs, early stopping, and scheduler context.
- `Metric/error findings`: accuracy, confusion/per-class behavior, calibration,
  prediction collapse, or other relevant diagnostics.
- `Interpretation`: evidence-tied causes, clearly marked as observed or inferred.
- `Allowed next actions`: bounded diagnostics or accepted follow-up checks.
- `Blocked next actions`: claims or experiments not justified by current evidence.

## Checklist

Use `Checklist` at the start of a substantial machine learning task and again before calling the task complete. In repo-local language this role may also be called verifier or validifier, but the global role name is `Checklist`.

Responsibilities:

- create or update `checklist.md` at initialization from the `Orchestrator`'s distilled goal, method/process contract, accepted experiment suite, target metrics, required artifacts, and explicit non-goals
- treat the existing `checklist.md` as the current reproduction contract unless the `Orchestrator` explicitly accepts a durable update before a run launches
- include process-method requirements, not only final artifact checks; examples include required pretraining/fine-tuning stages, data-efficiency runs, pattern-generalization runs, ablations, evaluation splits, logging requirements, and W&B expectations when relevant
- run or specify the relevant build, lint, unit, smoke-training, and inference checks
- after training outputs exist, run the repo's verification script or equivalent checks and write a checker report when the project has a standard location for it
- make the checker report pass/fail oriented: accepted, retry required, or blocked
- identify failed or pending checklist items, including method mismatches, missing artifacts, missing W&B runs, missing confusion or reliability outputs, and metrics below target
- verify config paths, checkpoint paths, prediction paths, and logging paths
- confirm analyzer hypotheses are consistent with `checklist.md`, the user's goal, and the current verifier status before treating them as next-step methods
- update `checklist.md` only when the `Orchestrator` explicitly accepts a durable new gate or changed acceptance criterion
- confirm the requested goal is actually finished, not only partially implemented
- check that large generated artifacts and model weights are ignored or unstaged unless explicitly requested
- report remaining blockers and unrun checks clearly

For substantial reproduction or result-analysis handoffs, completion requires a `Checklist` status of accepted, retry required, or blocked. Add `Result Analyzer` hypotheses only when the task includes interpretation, debugging, or improvement planning, and keep them consistent with the checklist's validity status.
