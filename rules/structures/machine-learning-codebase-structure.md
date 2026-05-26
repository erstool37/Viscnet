# Machine Learning Codebase Structure

For new machine learning codebases and major ML refactors, use the Viscnet-style project structure by default so experiments can be understood quickly. Do not rewrite an existing ML repository solely to satisfy this structure unless the task is already a major cleanup or refactor.

Preferred repository skeleton:

- `configs/config.yaml` owns project metadata, dataset paths/loaders, model selection, loss selection, training flags, optimizer and scheduler settings, tracking, checkpoint paths, prediction paths, manifest paths, and logging roots.
- `data/` or `dataset/` stores local/raw dataset inputs. Large, generated, private, and machine-specific data should be git-ignored, with lightweight placeholders such as `.gitkeep` used when needed.
- `scripts/setup.sh` prepares the environment. `scripts/dev.sh` records the default operator training command. Other top-level `scripts/*.py` files are reusable analysis, checker, plotting, dataset-building, and command-adapter entrypoints.
- `src/main.py` is the primary config-driven training entrypoint.
- `src/preprocess.py` or `src/utils/preprocess.py` contains dataset-specific preprocessing and manifest/statistics generation.
- `src/datasets/` contains dataset and dataloader classes.
- `src/models/` contains model architectures, heads, and local model subpackages.
- `src/losses/` contains loss modules with names that can be selected from config.
- `src/inference/` contains inference, attention/diagnostic exports, prediction artifacts, and post-training analysis helpers.
- `src/utils/` contains shared utilities for config loading, seeding, distributed setup, scaling/descaling, metrics, plotting, checkpoint loading, and small IO helpers.
- `src/weights/` is the default checkpoint destination and should not commit model weights.
- `ARCHIVE/` is optional and reserved for old experimental code. Archived code must not define the active public API or required execution path.

Implementation rules:

- Keep experiment wiring config-driven. Avoid hardcoding dataset choices, model classes, loss functions, optimizer settings, output paths, and run names in code when they belong in `configs/config.yaml`.
- Dynamically select dataset, model, and loss classes from config fields where practical, using explicit module/class names that mirror the folder structure.
- Centralize run naming, checkpoint paths, prediction paths, manifest paths, log roots, and random seeds in config/setup utilities.
- Keep the training flow discoverable from `src/main.py`; avoid making notebooks, ad hoc scripts, or hidden shell state the only way to understand the experiment.
- Use top-level `scripts/` for reusable experiment helpers only when they reduce clutter or make repeated diagnostics clearer. Do not create many one-off scripts when a config option, utility function, or `src/main.py` mode would be clearer.
- Place domain-specific parsing in `src/datasets/`, architectures in `src/models/`, objectives in `src/losses/`, shared mechanics in `src/utils/`, and inference/analysis outputs in `src/inference/`.
- `README.md` should show the dataset layout, minimal setup command, minimal training command, and the config file users should edit.

Provenance: this structure is specialized for the current `/root/Viscnet` repository based on the checked-in `README.md`, `configs/`, `scripts/`, `src/`, `checklist.md`, and `distillation.md` files inspected during local initialization on 2026-05-21. Any relationship to earlier private paths or public references is historical context unless separately verified.
