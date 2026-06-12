# GNNGEDAnalysis

GNNGEDAnalysis is a lightweight experiment wrapper for studying how graph neural
networks behave along graph edit distance (GED) paths. It trains graph
classification models with the `simplegnn` framework, evaluates trained models
on original graph splits and generated GED path graphs, merges evaluation
artifacts, and runs post-hoc analyses for decision changes along edit paths.

The GED path graphs themselves are generated upstream by the
[gedpaths (GNNGED) repository](#generating-the-pipeline-input-with-gedpaths)
and consumed here as PyTorch Geometric datasets.

## Features

- Train GNN classifiers for configured datasets and architectures.
- Evaluate trained models on train, validation, and GED path graph variants.
- Copy generated GED graph path datasets into the local data layout.
- Merge per-fold evaluation outputs into `results/all_results.csv`.
- Enrich merged results with GED metadata.
- Run experiment summaries and plots for edit-operation instability.

## Pipeline Overview

The full workflow consists of an upstream data-generation stage (in the
`gedpaths`/GNNGED repository) followed by six stages in this repository. Each
stage reads the output of the previous one:

```text
gedpaths (GNNGED repo)                      GNNGEDAnalysis (this repo)
======================                      ==========================

experiment.sh -db <DB>
  |- CreateMappings   (GED mappings)
  |- CreatePaths      (edit paths, 4 strategies)
  |- AnalyzePaths     (path statistics)
  `- bgf_to_pt.py     (PyTorch Geometric conversion)
        |
        v
../GNNGED/Results/Paths_<STRATEGY>/<METHOD>/<DB>/
        |
        |  (1) copy_ged_graphs.py  (auto-run by evaluate_models.py)
        v
data/GEDGraphs/<METHOD>/<DB>_<STRATEGY>/
        |
        |  (2) train_models.py              -> trained models under results/<GNN>/
        |  (3) evaluate_models.py           -> results/<GNN>/path_evaluation/*.pt|*.txt
        |  (4) analyze_evaluated_results.py -> results/all_results.csv
        |  (5) utils/enrich_all_results.py  -> results/all_results_enriched.csv
        v
        (6) experiments.py / experiments_additional.py
            -> results/Experiments/, results/ExperimentsAdditional/
```

### Stage 1 — Copy GED path graphs (`copy_ged_graphs.py`)

Copies the generated path-graph datasets from the gedpaths results tree into
the local data layout. For every `Paths_<STRATEGY>` directory under the source
root it scans the mapping-method subdirectories (only `F2` and `Precomputed`
are accepted) and copies each dataset folder that contains all three required
items:

- `processed/` — the PyTorch Geometric dataset (`data.pt`),
- `raw/` — the raw-data folder created during conversion,
- `<DB>_edit_paths_data.txt` — the per-step edit-operation metadata.

The copy target is `data/GEDGraphs/<METHOD>/<DB>_<STRATEGY>/`, e.g.
`../GNNGED/Results/Paths_d-E_d-IsoN/F2/MUTAG/` becomes
`data/GEDGraphs/F2/MUTAG_d-E_d-IsoN/`. Incomplete source folders are skipped
and reported. This stage runs automatically at the start of
`evaluate_models.py` (with `--copy-skip-existing` enabled by default), but can
also be run standalone.

The `<DB>_edit_paths_data.txt` file contains one edit operation per line in the
format

```text
<source_graph_id> <step_id> <target_graph_id> <ELEMENT_TYPE> <ELEMENT> <OPERATION>
```

for example `0 1 2 EDGE 0--5 DELETE` (parsed by `utils/load_path_info.py` and
by the evaluation/merge stages to attribute each intermediate path graph to an
edit operation such as `EDGE INSERT`, `NODE DELETE`, or `NODE RELABEL`).

### Stage 2 — Train models (`train_models.py`)

For each selected dataset, loads `configs/<DB>/main_config.yml` into the
`simplegnn` `FrameworkMain`, preprocesses the original dataset (e.g. TUDataset,
downloaded automatically into `data/TUDatasets/`), trains every configured
architecture (GIN, GCN, GAT, GATv2, GraphSAGE — one config entry per
`models_<GNN>.yml`) across all cross-validation folds defined in the split
file, and evaluates validation performance. Trained model checkpoints and
training logs are written under the per-architecture `results/<GNN>/` directory
configured in the YAML.

### Stage 3 — Evaluate models on GED paths (`evaluate_models.py`)

For every combination of dataset, path strategy, and GNN architecture:

1. Preprocesses both the original dataset (`main_config.yml`) and the path
   graph dataset `<DB>_<STRATEGY>` (`paths_config.yml`, loaded from
   `data/GEDGraphs/<METHOD>/`), aligning path-graph node features with the
   original feature space.
2. Loads the trained model checkpoint for each configuration and validation
   fold.
3. Runs inference on the train split, the validation split, and **all** path
   graphs (source graphs, intermediate edit-step graphs, and target graphs).
4. Saves per-fold tensors and a human-readable text dump to
   `results/<GNN>/path_evaluation/` (configurable via `--evaluation-folder`):
   - `train_results_config<C>_val<V>_<DB>.pt`
   - `validation_results_config<C>_val<V>_<DB>.pt`
   - `path_results_config<C>_val<V>_<DB>_<STRATEGY>.pt`
   - `path_results_config<C>_val<V>_<DB>_<STRATEGY>.txt`

   Each path-result row records the source graph id, edit step id, target
   graph id, the model output logits, and the edit operation applied at that
   step.

### Stage 4 — Merge evaluation results (`analyze_evaluated_results.py`)

Walks the evaluation folders for all selected datasets, strategies, and
architectures, joins the per-fold train/validation/path outputs, and derives
per-row analysis columns such as `predicted_label`, `true_label`,
`is_flipping` (does the prediction change at this edit step), `is_correct`,
`is_source`/`is_target`, train/validation membership of path endpoints, and
endpoint-label agreement. The merged table is written to
`results/all_results.csv` (one row per graph and edit step, per fold, per
architecture, per strategy).

### Stage 5 — Enrich merged results (`utils/enrich_all_results.py`)

Joins `results/all_results.csv` with structural metadata recomputed from the
copied GED graph datasets under `data/GEDGraphs/` (e.g. number of connected
components and cycle counts of length 3–6 per path graph) and writes
`results/all_results_enriched.csv`.

### Stage 6 — Post-hoc experiments (`experiments.py`, `experiments_additional.py`)

Consume the merged CSV and produce summary tables and plots, e.g. prediction
flips per edit-operation type (absolute, relative, combined), flip statistics,
and class-change heatmaps along edit paths. Outputs go to
`results/Experiments/` and `results/ExperimentsAdditional/` by default. Both
CLIs support filtering by dataset, GNN, path strategy, validation fold, path
split (`train`/`validation`), and correctness of path endpoints.

## Generating the Pipeline Input with gedpaths

The input to this pipeline — the GED path graph datasets — is produced by the
`gedpaths` library (the GNNGED repository, expected as a sibling checkout at
`../GNNGED`). gedpaths builds on `libGraph` and GEDLIB to compute graph edit
distance mappings between graph pairs and to materialize the intermediate
graphs along the corresponding edit paths.

### Running the gedpaths pipeline

The simplest way is the bundled experiment script (see the GNNGED README for
build prerequisites; the exact solvers such as `F2` require GUROBI):

```bash
cd ../GNNGED
chmod u+x experiment.sh
./experiment.sh -db <DB>            # e.g. ./experiment.sh -db MUTAG,NCI1
```

For each dataset this runs, in order:

1. **Download** the TU dataset (`python_src/data_loader.py`) into
   `Data/Graphs/<DB>/`.
2. **Compute GED mappings** (`build/CreateMappings`, default method `F2`,
   5000 graph pairs) into `Results/Mappings/<METHOD>/<DB>/`
   (`<DB>_ged_mapping.bin`, `<DB>_ged_mapping.csv`, `graph_ids.txt`).
3. **Validate mappings** (`build/AnalyzeMappings`).
4. **Build edit paths** (`build/CreatePaths`) for the four path strategies
   into `Results/Paths_<STRATEGY>/<METHOD>/<DB>/`. This produces the edit-path
   graphs (`<DB>_edit_paths.bgf`) and the edit-operation metadata file
   `<DB>_edit_paths_data.txt` required by this repository.
5. **Compute path statistics** (`build/AnalyzePaths`).
6. **Convert to PyTorch Geometric** (`python_src/converter/bgf_to_pt.py`),
   which creates `processed/data.pt` (and the `raw/` folder) inside each
   `Results/Paths_<STRATEGY>/<METHOD>/<DB>/` directory.
7. Optional plotting and Weisfeiler–Leman analysis stages.

The four path strategies are:

| Strategy id  | Meaning                                        |
|--------------|------------------------------------------------|
| `Rnd`        | Random edit-operation order                    |
| `Rnd_d-IsoN` | Random order, delete isolated nodes            |
| `i-E_d-IsoN` | Insert edges first, delete isolated nodes      |
| `d-E_d-IsoN` | Delete edges first, delete isolated nodes      |

### Expected input layout

After a successful gedpaths run, each dataset folder must contain the three
items checked by `copy_ged_graphs.py`:

```text
../GNNGED/Results/Paths_<STRATEGY>/<METHOD>/<DB>/
|-- processed/                  # PyTorch Geometric dataset (data.pt, ...)
|-- raw/                        # raw-data folder from the conversion
`-- <DB>_edit_paths_data.txt    # one edit operation per line
```

Only mapping methods `F2` and `Precomputed` are picked up by the copy step;
other mapping directories are ignored.

### Continuing in GNNGEDAnalysis after a new gedpaths run

Suppose you ran `./experiment.sh -db <DB>` in the gedpaths repository for a new
dataset `<DB>`. To analyze it here:

1. **Create the dataset config folder** `configs/<DB>/` (easiest: copy an
   existing one such as `configs/MUTAG/` and replace the dataset name):
   - `main_config.yml` — one entry per architecture pointing at
     `data/TUDatasets/`, `results/<GNN>/`, `models_<GNN>.yml`,
     `parameters.yml`, and the split file (for TU datasets:
     `../SimpleGNN/repo/src/simplegnn/datasets/splits/tu_splits/<DB>_splits.json`;
     custom split files live under `splits/`).
   - `paths_config.yml` — one entry per path strategy with
     `name: "<DB>_<STRATEGY>"`, `source: "path"`, and
     `data: "data/GEDGraphs/<METHOD>/"` (matching the copy destination).
   - `parameters.yml` and `models_<GNN>.yml` — hyperparameters and
     architecture definitions (usually reusable as-is).

   > **Note:** The train/validation/test split file is the one input that is
   > *not* produced by gedpaths. For the classical TU datasets, ready-made
   > splits ship with SimpleGNN under
   > `../SimpleGNN/repo/src/simplegnn/datasets/splits/tu_splits/<DB>_splits.json`.
   > For any other dataset you must provide a split JSON yourself (place it
   > under `splits/` in this repository and reference it from
   > `main_config.yml` and `paths_config.yml`; see the existing files in
   > `splits/` for the expected format). Training, evaluation, and the
   > train/validation columns in the merged results all depend on this file.

2. **Train** the models on the original dataset:

   ```bash
   python train_models.py --db <DB> --num_threads 4
   ```

3. **Evaluate** on the GED paths (this first copies
   `../GNNGED/Results/Paths_*` into `data/GEDGraphs/` automatically):

   ```bash
   python evaluate_models.py --db <DB> \
     --path-strategy d-E_d-IsoN --path-strategy i-E_d-IsoN \
     --num_threads 4
   ```

4. **Merge** the evaluation artifacts:

   ```bash
   python analyze_evaluated_results.py --db <DB> --results-path results
   ```

5. **Enrich** (optional) and **run the experiments**:

   ```bash
   python utils/enrich_all_results.py \
     --input-csv results/all_results.csv \
     --output-csv results/all_results_enriched.csv \
     --ged-root data/GEDGraphs --overwrite

   python experiments.py --data-path results/all_results.csv \
     --output-dir results/Experiments --dataset <DB>
   ```

## Repository Layout

```text
.
|-- train_models.py                 # Training and validation evaluation CLI
|-- evaluate_models.py              # Model and GED path evaluation CLI
|-- analyze_evaluated_results.py    # Merge evaluation artifacts into CSV
|-- experiments.py                  # Main post-hoc experiment suite
|-- experiments_additional.py       # Additional post-hoc experiment suite
|-- copy_ged_graphs.py              # Copy GED path graphs from gedpaths output
|-- configs/                        # Per-dataset experiment configs
|-- splits/                         # Split artifacts for selected datasets
|-- tests/                          # Pytest coverage for CLIs and utilities
|-- utils/                          # Shared utilities (path info loader, enrichment, plotting)
|-- data/                           # Local datasets and copied GED graph data
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
- For generating new input data: a sibling checkout of the gedpaths/GNNGED
  repository at `../GNNGED` with its C++ tools built (see its
  `INSTALLATION.md`; exact GED solvers require GUROBI).

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
- `paths_config.yml`: GED path graph datasets for evaluation (one entry per
  path strategy, with `source: "path"` and data rooted at
  `data/GEDGraphs/<METHOD>/`).
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

The copy step can also be run standalone:

```bash
python copy_ged_graphs.py --source-root ../GNNGED/Results --dest-root data/GEDGraphs
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
