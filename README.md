# GNNGEDAnalysis

GNNGEDAnalysis is a lightweight experiment wrapper for studying how graph neural
networks behave along graph edit distance (GED) paths. It trains graph
classification models with the `simplegnn` framework, evaluates trained models
on original graph splits and generated GED path graphs, merges evaluation
artifacts, and runs post-hoc analyses for decision changes along edit paths.

## Features

- Train GNN classifiers for configured datasets and architectures.
- Evaluate trained models on train, validation, and GED path graph variants.
- Copy generated GED graph path datasets into the local data layout.
- Merge per-fold evaluation outputs into `results/all_results.csv`.
- Enrich merged results with GED metadata.
- Run experiment summaries and plots for edit-operation instability.

## Repository Layout

```text
.
|-- train_models.py                 # Training and validation evaluation CLI
|-- evaluate_models.py              # Model and GED path evaluation CLI
|-- analyze_evaluated_results.py    # Merge evaluation artifacts into CSV
|-- experiments.py                  # Main post-hoc experiment suite
|-- experiments_additional.py       # Additional post-hoc experiment suite
|-- copy_ged_graphs.py              # Copy GED path graphs from GNNGED output
|-- configs/                        # Per-dataset experiment configs
|-- splits/                         # Split artifacts for selected datasets
|-- tests/                          # Pytest coverage for CLIs and utilities
|-- utils/                          # Shared utilities
|-- data/                           # Local datasets and generated graph data
`-- results/                        # Training, evaluation, and analysis output
```

Large or generated data is expected under `data/`, `tmp/`, `results/`, and
`results_old/`.

## Requirements

- Python 3.9 or newer.
- [`simplegnn`](https://github.com/fseiffarth/SimpleGNN), either installed as a
  package or available at `../SimpleGNN/repo/src` relative to this repository.
- Python packages used by the scripts: `click`, `joblib`, `torch`, `polars`,
  `pytest`, and optionally `matplotlib` for plots.

This repository does not currently include a `requirements.txt` or packaging
configuration. Install dependencies in the environment used for `simplegnn`.

## Setup

```bash
git clone <repository-url>
cd GNNGEDAnalysis

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install click joblib torch polars pytest matplotlib
```

If `simplegnn` is not installed, clone the
[SimpleGNN repository](https://github.com/fseiffarth/SimpleGNN) next to this
repository so that `../SimpleGNN/repo/src` exists, or install it into the active
environment.

## Configuration

Experiment behavior is driven by YAML files under `configs/<dataset>/`:

- `main_config.yml`: original dataset, model config, hyperparameters, split
  file, and results directory.
- `paths_config.yml`: GED path graph datasets for evaluation.
- `parameters.yml`: optimizer, loss, batch size, learning rate, epochs,
  feature handling, and model-saving options.
- `models_<algorithm>.yml`: model architecture definitions for algorithms such
  as `GIN`, `GCN`, `GAT`, `GATv2`, and `GraphSAGE`.

Run commands from the repository root so relative paths in these configs resolve
correctly.

## Usage

### Train Models

Train the default dataset (`MUTAG`):

```bash
python train_models.py --num_threads 4
```

Train one or more configured datasets:

```bash
python train_models.py --db MUTAG --db PTC_MR --num_threads 4
```

Train every dataset with `configs/<dataset>/main_config.yml`:

```bash
python train_models.py --all --num_threads 4
```

### Evaluate Models on GED Paths

Evaluate the default dataset, path strategy, and algorithms:

```bash
python evaluate_models.py --num_threads 4
```

Evaluate selected datasets, strategies, and algorithms:

```bash
python evaluate_models.py \
  --db MUTAG \
  --path-strategy d-E_d-IsoN \
  --path-strategy i-E_d-IsoN \
  --gnn-algorithm GIN \
  --gnn-algorithm GCN \
  --num_threads 4
```

Before evaluation, `evaluate_models.py` copies GED graph datasets from
`../GNNGED/Results` into `data/GEDGraphs` by default. Override those paths when
needed:

```bash
python evaluate_models.py \
  --copy-source-root /path/to/GNNGED/Results \
  --copy-dest-root data/GEDGraphs
```

### Merge Evaluation Results

Merge per-fold train, validation, and path outputs into a single CSV:

```bash
python analyze_evaluated_results.py --results-path results
```

The default output is:

```text
results/all_results.csv
```

### Enrich Merged Results

Add GED graph metadata to merged results:

```bash
python utils/enrich_all_results.py \
  --input-csv results/all_results.csv \
  --output-csv results/all_results_enriched.csv \
  --ged-root data/GEDGraphs \
  --overwrite
```

### Run Post-hoc Experiments

Run the main experiment suite:

```bash
python experiments.py \
  --data-path results/all_results.csv \
  --output-dir results/Experiments
```

Run the additional experiment suite:

```bash
python experiments_additional.py \
  --data-path results/all_results.csv \
  --output-dir results/ExperimentsAdditional
```

Both experiment CLIs support filters such as `--dataset`, `--gnn`,
`--path-strategy`, `--val-id`, `--path-split`, and `--correct-filter`.

## Outputs

Training outputs are written under the `results` path configured in each
dataset's YAML files. Evaluation writes artifacts to the selected
`--evaluation-folder` under each algorithm results directory, for example:

```text
results/GIN/path_evaluation/
|-- train_results_config0_val0_MUTAG.pt
|-- validation_results_config0_val0_MUTAG.pt
|-- path_results_config0_val0_MUTAG_d-E_d-IsoN.pt
`-- path_results_config0_val0_MUTAG_d-E_d-IsoN.txt
```

Post-hoc experiment outputs are written to `results/Experiments` and
`results/ExperimentsAdditional` by default.

## Development

Run the test suite:

```bash
python -m pytest
```

Run a quick syntax check:

```bash
python -m py_compile \
  train_models.py \
  evaluate_models.py \
  analyze_evaluated_results.py \
  experiments.py \
  experiments_additional.py \
  copy_ged_graphs.py \
  utils/*.py
```

## Notes

- The training and evaluation scripts validate requested dataset names against
  `configs/*/main_config.yml`.
- `--all_dbs_classical` in `evaluate_models.py` targets the classical paper
  datasets listed in the script.
- GED path evaluation expects path graph data with `processed/`, `raw/`, and
  `<dataset>_edit_paths_data.txt` present for each copied dataset.
