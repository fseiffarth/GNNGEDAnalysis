# Repository Guidelines

## Project Structure & Module Organization
This repository is a lightweight experiment wrapper for GNN-based GED analysis.

- `train_models.py`: CLI entry point for preprocessing, training, and validation evaluation.
- `evaluate_models.py`: CLI entry point for loading trained models and evaluating splits.
- `configs/`: YAML experiment and network definitions (`main_config.yml`, `network_*.yml`, `paths_config.yml`).
- `Data/`: local datasets and split artifacts (gitignored for large/generated files).
- `Results/`: model outputs and evaluation artifacts (gitignored).
- `utils/`: helper scripts and placeholders for GED-specific utilities.

Keep new code in focused modules under `utils/` (or a new package directory) instead of adding large logic blocks to top-level scripts.

## Build, Test, and Development Commands
No build system is configured; use Python CLI execution.

- `python train_models.py --num_threads 4`: run preprocessing + training workflow.
- `python evaluate_models.py --num_threads 4`: evaluate trained models on configured splits.
- `python -m py_compile train_models.py evaluate_models.py utils/*.py`: quick syntax check.

Run commands from repository root so relative config paths resolve correctly.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and descriptive YAML names like `network_GATv2.yml`.
- Keep scripts CLI-friendly with `click` options and small `main()` wrappers.
- Prefer explicit imports and short, targeted comments only where behavior is non-obvious.

## Testing Guidelines
Automated tests are currently minimal. For contributions:

- Add focused `pytest` tests under a new `tests/` directory when introducing logic-heavy utilities.
- Name tests as `test_<module>.py` and test functions as `test_<behavior>()`.
- At minimum, run syntax checks and one end-to-end smoke run (`train_models.py` or `evaluate_models.py`) with a small dataset/config.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit messages (for example, `Update .gitignore ...`, `Initial commit`). Continue that style:

- Commit format: `Verb + scope` (for example, `Add GED split loader`).
- Keep commits atomic and tied to one change intent.
- PRs should include: purpose, changed files, config/data assumptions, and a short validation note (command run + result).
- Link related issues and include result snapshots only when outputs materially change.
