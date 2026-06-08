import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click


REQUIRED_COLUMNS = [
    "row_idx",
    "path_idx_int",
    "source_id",
    "step_id",
    "target_id",
    "operation",
    "operation_str",
    "class_0",
    "class_1",
    "predicted_label",
    "true_label",
    "is_flipping",
    "is_correct",
    "is_source",
    "is_target",
    "is_train",
    "is_validation",
    "is_path",
    "is_correct_path_endpoint",
    "is_correct_path",
    "same_endpoint_labels_predicted",
    "same_endpoint_labels_true",
    "is_train_path",
    "is_validation_path",
    "val_id",
    "config_id",
    "gnn_algorithm",
    "dataset",
    "path_strategy",
]


ALL_EXPERIMENTS = (
    "flips_per_operation",
    "flips_per_operation_relative",
    "flips_per_operation_relative_combined",
    "flips_statistics",
    "class_change_heatmaps",
    "class_change_heatmaps_flipping_only",
)
DEFAULT_EXPERIMENTS = ALL_EXPERIMENTS

ALLOWED_PATH_SPLITS = ("train", "validation")
ALLOWED_CORRECT_FILTERS = ("correct", "all", "incorrect")
FLIPS_PER_OPERATION_NAMES = (
    "EDGE INSERT",
    "EDGE DELETE",
    "NODE INSERT",
    "NODE DELETE",
    "EDGE RELABEL",
    "NODE RELABEL",
)
FLIPS_PER_OPERATION_SPLITS = ("all", "train", "validation")
FLIPS_STATISTICS_SPLITS = ("all", "train", "validation")


def _extract_class_columns(fieldnames: Iterable[str]) -> List[str]:
    return sorted(
        [name for name in fieldnames if name.startswith("class_")],
        key=lambda item: _to_int(item.replace("class_", ""), default=0),
    )


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slugify(parts: Iterable[str]) -> str:
    clean = []
    for part in parts:
        text = str(part)
        out = []
        for ch in text:
            if ch.isalnum() or ch in ("-", "_"):
                out.append(ch)
            else:
                out.append("_")
        clean.append("".join(out))
    return "__".join(clean)


def _display_gnn_label(gnn: str) -> str:
    if gnn == "GraphSAGE":
        return "GSAGE"
    return gnn


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean()
        var = max(self.total_sq / self.count - mean * mean, 0.0)
        return math.sqrt(var)


def discover_metadata(data_path: Path) -> Dict[str, object]:
    with data_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {data_path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        class_columns = _extract_class_columns(reader.fieldnames)
        if not class_columns:
            raise ValueError("CSV must include at least one class_* column.")

        datasets = set()
        gnns = set()
        strategies = set()
        vals = set()
        max_step = -1
        row_count = 0

        for row in reader:
            row_count += 1
            datasets.add(row["dataset"])
            gnns.add(row["gnn_algorithm"])
            strategies.add(row["path_strategy"])
            vals.add(row["val_id"])
            step_id = _to_int(row["step_id"], default=-1)
            if step_id > max_step:
                max_step = step_id

    return {
        "row_count": row_count,
        "datasets": sorted(datasets),
        "gnns": sorted(gnns),
        "strategies": sorted(strategies),
        "vals": sorted(vals),
        "max_step": max_step,
        "class_columns": class_columns,
    }


def _step_group_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (row["dataset"], row["gnn_algorithm"], row["path_strategy"], row["val_id"])


def _path_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str]:
    return (
        row["dataset"],
        row["gnn_algorithm"],
        row["path_strategy"],
        row["config_id"],
        row["val_id"],
        row["path_idx_int"],
    )


def _path_passes_correct_filter(is_correct_path: int, correct_filter: str) -> bool:
    if correct_filter == "all":
        return True
    if correct_filter == "correct":
        return is_correct_path == 1
    if correct_filter == "incorrect":
        return is_correct_path == 0
    raise ValueError(f"Unsupported correct_filter: {correct_filter}")


class ExperimentRunner:
    def __init__(
        self,
        data_path: Path,
        output_dir: Path,
        selected_datasets: List[str],
        selected_gnns: List[str],
        selected_strategies: List[str],
        selected_vals: List[str],
        class_columns: List[str],
        selected_path_splits: List[str],
        correct_filter: str,
        min_step: int,
        max_step: int,
        max_sampled_paths: int,
        seed: int,
        experiments: List[str],
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.selected_datasets = set(selected_datasets)
        self.selected_gnns = set(selected_gnns)
        self.selected_strategies = set(selected_strategies)
        self.selected_vals = set(selected_vals)
        self.class_columns = list(class_columns)
        self.selected_path_splits = {split.lower() for split in selected_path_splits}
        self.correct_filter = correct_filter.lower()
        self.min_step = min_step
        self.max_step = max_step
        self.max_sampled_paths = max_sampled_paths
        self.seed = seed
        self.experiments = set(experiments)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.flips_per_operation_counts = defaultdict(int)
        self.flips_per_operation_total_counts = defaultdict(int)
        self.flips_per_operation_present = set()
        self.flips_statistics_counts = defaultdict(int)
        self.flips_statistics_present = set()
        self.class_change_total = defaultdict(float)
        self.class_change_count = defaultdict(int)
        self.class_change_flipping_total = defaultdict(float)
        self.class_change_flipping_count = defaultdict(int)
        self.class_change_present = set()

    def _group_in_scope(self, group_key: Tuple[str, str, str, str]) -> bool:
        dataset, gnn, strategy, val_id = group_key
        return (
            dataset in self.selected_datasets
            and gnn in self.selected_gnns
            and strategy in self.selected_strategies
            and val_id in self.selected_vals
        )

    def _finalize_path(
        self,
        path_key: Tuple[str, str, str, str, str, str],
        rows: List[Dict[str, object]],
        endpoint_is_correct_path: int,
        endpoint_is_train_path: int,
        endpoint_is_validation_path: int,
    ) -> None:
        dataset, gnn, strategy, _, val_id, _ = path_key
        group_key = (dataset, gnn, strategy, val_id)
        if not self._group_in_scope(group_key):
            return

        if not _path_passes_correct_filter(endpoint_is_correct_path, self.correct_filter):
            return

        split_names = ["all"]
        if endpoint_is_train_path == 1:
            split_names.append("train")
        if endpoint_is_validation_path == 1:
            split_names.append("validation")
        num_decision_changes = sum(int(item["is_flipping"]) for item in rows)

        if (
            "flips_per_operation" in self.experiments
            or "flips_per_operation_relative" in self.experiments
            or "flips_per_operation_relative_combined" in self.experiments
        ):
            for split_name in split_names:
                self.flips_per_operation_present.add((dataset, gnn, strategy, val_id, split_name))

            for item in rows:
                operation_str = str(item["operation_str"])
                is_flipping = int(item["is_flipping"])
                if operation_str not in FLIPS_PER_OPERATION_NAMES:
                    continue
                for split_name in split_names:
                    base_key = (dataset, gnn, strategy, val_id, split_name, operation_str)
                    self.flips_per_operation_total_counts[base_key] += 1
                    if is_flipping == 1:
                        self.flips_per_operation_counts[base_key] += 1

        if "flips_statistics" in self.experiments:
            for split_name in split_names:
                self.flips_statistics_present.add((dataset, gnn, strategy, val_id, split_name))
                self.flips_statistics_counts[
                    (dataset, gnn, strategy, val_id, split_name, num_decision_changes)
                ] += 1

        if "class_change_heatmaps" in self.experiments or "class_change_heatmaps_flipping_only" in self.experiments:
            self.class_change_present.add((dataset, gnn, strategy, val_id))
            previous_scores = None
            for item in rows:
                operation_str = str(item["operation_str"])
                is_flipping = int(item["is_flipping"])
                class_scores = item["class_scores"]
                if not isinstance(class_scores, dict):
                    continue
                if previous_scores is None:
                    previous_scores = class_scores
                    continue
                if operation_str not in FLIPS_PER_OPERATION_NAMES:
                    previous_scores = class_scores
                    continue
                for class_name in self.class_columns:
                    current_value = _to_float(class_scores.get(class_name, 0.0), default=0.0)
                    previous_value = _to_float(previous_scores.get(class_name, 0.0), default=0.0)
                    abs_delta = abs(current_value - previous_value)
                    base_key = (dataset, gnn, strategy, val_id, operation_str, class_name)
                    self.class_change_total[base_key] += abs_delta
                    self.class_change_count[base_key] += 1
                    if is_flipping == 1:
                        self.class_change_flipping_total[base_key] += abs_delta
                        self.class_change_flipping_count[base_key] += 1
                previous_scores = class_scores

    def run(self) -> None:
        path_states: Dict[Tuple[str, str, str, str, str, str], Dict[str, object]] = {}

        with self.data_path.open("r", newline="", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                path_key = _path_key(row)
                group_key = _step_group_key(row)
                if not self._group_in_scope(group_key):
                    continue

                state = path_states.setdefault(
                    path_key,
                    {
                        "rows": [],
                        "endpoint_is_correct_path": 0,
                        "endpoint_is_train_path": 0,
                        "endpoint_is_validation_path": 0,
                    },
                )

                step_id = _to_int(row["step_id"], default=-1)
                is_path = _to_int(row["is_path"], default=0)
                is_flipping = _to_int(row["is_flipping"], default=0)
                operation_str = row["operation_str"]

                if is_path == 1 and step_id >= 0:
                    class_scores = {class_name: row.get(class_name, "0.0") for class_name in self.class_columns}
                    state["rows"].append(
                        {
                            "operation_str": operation_str,
                            "is_flipping": is_flipping,
                            "class_scores": class_scores,
                        }
                    )

                if _to_int(row["is_target"], default=0) == 1:
                    state["endpoint_is_correct_path"] = _to_int(row["is_correct_path"], default=0)
                    state["endpoint_is_train_path"] = _to_int(row["is_train_path"], default=0)
                    state["endpoint_is_validation_path"] = _to_int(row["is_validation_path"], default=0)

                    self._finalize_path(
                        path_key=path_key,
                        rows=state["rows"],
                        endpoint_is_correct_path=int(state["endpoint_is_correct_path"]),
                        endpoint_is_train_path=int(state["endpoint_is_train_path"]),
                        endpoint_is_validation_path=int(state["endpoint_is_validation_path"]),
                    )
                    del path_states[path_key]

        if path_states:
            warning_path = self.output_dir / "warnings.txt"
            with warning_path.open("w", encoding="utf-8") as file_obj:
                file_obj.write(
                    f"{len(path_states)} paths were not finalized because no endpoint row was encountered.\n"
                )

        self._write_outputs()
        self._write_run_filters()

    def _write_run_filters(self) -> None:
        out_path = self.output_dir / "run_filters.txt"
        with out_path.open("w", encoding="utf-8") as file_obj:
            file_obj.write("Applied path-level filters\n")
            file_obj.write("==========================\n")
            file_obj.write(f"path_splits: {','.join(sorted(self.selected_path_splits))}\n")
            file_obj.write(f"correct_filter: {self.correct_filter}\n")
            file_obj.write(f"datasets: {','.join(sorted(self.selected_datasets))}\n")
            file_obj.write(f"gnns: {','.join(sorted(self.selected_gnns))}\n")
            file_obj.write(f"path_strategies: {','.join(sorted(self.selected_strategies))}\n")
            file_obj.write(f"val_ids: {','.join(sorted(self.selected_vals))}\n")

    def _write_outputs(self) -> None:
        if "flips_per_operation" in self.experiments:
            self._write_flips_per_operation_outputs()
        if "flips_per_operation_relative" in self.experiments:
            self._write_flips_per_operation_relative_outputs()
        if "flips_per_operation_relative_combined" in self.experiments:
            self._write_flips_per_operation_relative_combined_outputs()
        if "flips_statistics" in self.experiments:
            self._write_flips_statistics_outputs()
        if "class_change_heatmaps" in self.experiments:
            self._write_class_change_heatmap_outputs(flipping_only=False)
        if "class_change_heatmaps_flipping_only" in self.experiments:
            self._write_class_change_heatmap_outputs(flipping_only=True)

    def _write_flips_per_operation_outputs(self) -> None:
        base_dir = self.output_dir / "experiments_flips_per_operation"
        base_dir.mkdir(parents=True, exist_ok=True)

        def _sort_val_ids(values: Iterable[str]) -> List[str]:
            return sorted(values, key=lambda item: _to_int(item, default=0))

        selected_vals = _sort_val_ids(self.selected_vals)
        present_triples = {
            (dataset, gnn, strategy)
            for dataset, gnn, strategy, _, _ in self.flips_per_operation_present
        }

        fold_rows: List[Dict[str, object]] = []
        for dataset, gnn, strategy in sorted(present_triples):
            for split_name in FLIPS_PER_OPERATION_SPLITS:
                for val_id in selected_vals:
                    if (dataset, gnn, strategy, val_id, split_name) not in self.flips_per_operation_present:
                        continue
                    for operation_name in FLIPS_PER_OPERATION_NAMES:
                        fold_rows.append(
                            {
                                "dataset": dataset,
                                "gnn_algorithm": gnn,
                                "path_strategy": strategy,
                                "split": split_name,
                                "val_id": val_id,
                                "operation_str": operation_name,
                                "decision_change_count": self.flips_per_operation_counts.get(
                                    (dataset, gnn, strategy, val_id, split_name, operation_name), 0
                                ),
                            }
                        )

        fold_path = base_dir / "flips_per_operation_fold_counts.csv"
        with fold_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "val_id",
                    "operation_str",
                    "decision_change_count",
                ],
            )
            writer.writeheader()
            for row in fold_rows:
                writer.writerow(row)

        summary_path = base_dir / "flips_per_operation_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "operation_str",
                    "num_folds",
                    "mean_decision_change_count",
                    "std_decision_change_count",
                    "min_decision_change_count",
                    "max_decision_change_count",
                ],
            )
            writer.writeheader()

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                plt = None

            grouped_rows = defaultdict(list)
            for row in fold_rows:
                grouped_rows[
                    (row["dataset"], row["gnn_algorithm"], row["path_strategy"], row["split"])
                ].append(row)

            for key in sorted(grouped_rows):
                dataset, gnn, strategy, split_name = key
                op_values = {op: [] for op in FLIPS_PER_OPERATION_NAMES}
                for row in grouped_rows[key]:
                    op_values[row["operation_str"]].append(_to_float(row["decision_change_count"], default=0.0))

                data_for_plot = []
                for operation_name in FLIPS_PER_OPERATION_NAMES:
                    values = op_values[operation_name]
                    if not values:
                        values = [0.0]
                    data_for_plot.append(values)
                    stats = RunningStats()
                    for value in values:
                        stats.add(value)
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "gnn_algorithm": gnn,
                            "path_strategy": strategy,
                            "split": split_name,
                            "operation_str": operation_name,
                            "num_folds": len(values),
                            "mean_decision_change_count": stats.mean(),
                            "std_decision_change_count": stats.std(),
                            "min_decision_change_count": stats.min_value if stats.count else 0.0,
                            "max_decision_change_count": stats.max_value if stats.count else 0.0,
                        }
                    )

                if plt is not None:
                    figure = plt.figure(figsize=(11, 6))
                    ax = figure.add_subplot(111)
                    ax.boxplot(data_for_plot, patch_artist=True)
                    ax.set_xticks(range(1, len(FLIPS_PER_OPERATION_NAMES) + 1))
                    ax.set_xticklabels(FLIPS_PER_OPERATION_NAMES, rotation=25, ha="right")
                    ax.set_xlabel("Graph Edit Operation")
                    ax.set_ylabel("Number of Decision Changes")
                    ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
                    figure.tight_layout()
                    out_png = base_dir / f"flips_per_operation__{_slugify((dataset, gnn, strategy, split_name))}.png"
                    figure.savefig(out_png, dpi=150)
                    plt.close(figure)

    def _write_flips_per_operation_relative_outputs(self) -> None:
        base_dir = self.output_dir / "experiments_flips_per_operation_relative"
        base_dir.mkdir(parents=True, exist_ok=True)

        def _sort_val_ids(values: Iterable[str]) -> List[str]:
            return sorted(values, key=lambda item: _to_int(item, default=0))

        selected_vals = _sort_val_ids(self.selected_vals)
        present_triples = {
            (dataset, gnn, strategy)
            for dataset, gnn, strategy, _, _ in self.flips_per_operation_present
        }

        fold_rows: List[Dict[str, object]] = []
        for dataset, gnn, strategy in sorted(present_triples):
            for split_name in FLIPS_PER_OPERATION_SPLITS:
                for val_id in selected_vals:
                    if (dataset, gnn, strategy, val_id, split_name) not in self.flips_per_operation_present:
                        continue
                    for operation_name in FLIPS_PER_OPERATION_NAMES:
                        decision_changes = self.flips_per_operation_counts.get(
                            (dataset, gnn, strategy, val_id, split_name, operation_name), 0
                        )
                        operation_count = self.flips_per_operation_total_counts.get(
                            (dataset, gnn, strategy, val_id, split_name, operation_name), 0
                        )
                        relative_count = (decision_changes / operation_count) if operation_count > 0 else 0.0
                        fold_rows.append(
                            {
                                "dataset": dataset,
                                "gnn_algorithm": gnn,
                                "path_strategy": strategy,
                                "split": split_name,
                                "val_id": val_id,
                                "operation_str": operation_name,
                                "decision_change_count": decision_changes,
                                "operation_count": operation_count,
                                "relative_decision_change_count": relative_count,
                            }
                        )

        fold_path = base_dir / "flips_per_operation_relative_fold_counts.csv"
        with fold_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "val_id",
                    "operation_str",
                    "decision_change_count",
                    "operation_count",
                    "relative_decision_change_count",
                ],
            )
            writer.writeheader()
            for row in fold_rows:
                writer.writerow(row)

        summary_path = base_dir / "flips_per_operation_relative_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "operation_str",
                    "num_folds",
                    "mean_relative_decision_change_count",
                    "std_relative_decision_change_count",
                    "min_relative_decision_change_count",
                    "max_relative_decision_change_count",
                ],
            )
            writer.writeheader()

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                plt = None

            grouped_rows = defaultdict(list)
            for row in fold_rows:
                grouped_rows[
                    (row["dataset"], row["gnn_algorithm"], row["path_strategy"], row["split"])
                ].append(row)

            for key in sorted(grouped_rows):
                dataset, gnn, strategy, split_name = key
                op_values = {op: [] for op in FLIPS_PER_OPERATION_NAMES}
                for row in grouped_rows[key]:
                    op_values[row["operation_str"]].append(_to_float(row["relative_decision_change_count"], default=0.0))

                data_for_plot = []
                for operation_name in FLIPS_PER_OPERATION_NAMES:
                    values = op_values[operation_name]
                    if not values:
                        values = [0.0]
                    data_for_plot.append(values)
                    stats = RunningStats()
                    for value in values:
                        stats.add(value)
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "gnn_algorithm": gnn,
                            "path_strategy": strategy,
                            "split": split_name,
                            "operation_str": operation_name,
                            "num_folds": len(values),
                            "mean_relative_decision_change_count": stats.mean(),
                            "std_relative_decision_change_count": stats.std(),
                            "min_relative_decision_change_count": stats.min_value if stats.count else 0.0,
                            "max_relative_decision_change_count": stats.max_value if stats.count else 0.0,
                        }
                    )

                if plt is not None:
                    figure = plt.figure(figsize=(11, 6))
                    ax = figure.add_subplot(111)
                    ax.boxplot(data_for_plot, patch_artist=True)
                    ax.set_xticks(range(1, len(FLIPS_PER_OPERATION_NAMES) + 1))
                    ax.set_xticklabels(FLIPS_PER_OPERATION_NAMES, rotation=25, ha="right")
                    ax.set_xlabel("Graph Edit Operation")
                    ax.set_ylabel("Relative Number of Decision Changes")
                    ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
                    figure.tight_layout()
                    out_png = base_dir / f"flips_per_operation_relative__{_slugify((dataset, gnn, strategy, split_name))}.png"
                    figure.savefig(out_png, dpi=150)
                    plt.close(figure)

    def _write_flips_per_operation_relative_combined_outputs(self) -> None:
        base_dir = self.output_dir / "experiments_flips_per_operation_relative_combined"
        base_dir.mkdir(parents=True, exist_ok=True)
        preferred_gnn_order = ["GCN", "GATv2", "GraphSAGE", "GIN"]

        def _sort_val_ids(values: Iterable[str]) -> List[str]:
            return sorted(values, key=lambda item: _to_int(item, default=0))

        def _order_gnns(gnns: Iterable[str]) -> List[str]:
            gnn_list = list(gnns)
            known = [gnn for gnn in preferred_gnn_order if gnn in gnn_list]
            extras = sorted(gnn for gnn in gnn_list if gnn not in preferred_gnn_order)
            return known + extras

        selected_vals = _sort_val_ids(self.selected_vals)
        present_groups = {
            (dataset, strategy, split_name)
            for dataset, _, strategy, _, split_name in self.flips_per_operation_present
        }

        grouped_values: Dict[Tuple[str, str, str], Dict[Tuple[str, str], List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        y_limits_by_dataset_split: Dict[Tuple[str, str], Tuple[float, float]] = {}

        for dataset, strategy, split_name in sorted(present_groups):
            for gnn in _order_gnns(self.selected_gnns):
                for val_id in selected_vals:
                    if (dataset, gnn, strategy, val_id, split_name) not in self.flips_per_operation_present:
                        continue
                    for operation_name in FLIPS_PER_OPERATION_NAMES:
                        decision_changes = self.flips_per_operation_counts.get(
                            (dataset, gnn, strategy, val_id, split_name, operation_name), 0
                        )
                        operation_count = self.flips_per_operation_total_counts.get(
                            (dataset, gnn, strategy, val_id, split_name, operation_name), 0
                        )
                        relative_count = (decision_changes / operation_count) if operation_count > 0 else 0.0
                        grouped_values[(dataset, strategy, split_name)][(gnn, operation_name)].append(relative_count)

        for dataset, split_name in sorted({(dataset, split_name) for dataset, _, split_name in present_groups}):
            values = []
            for strategy in self.selected_strategies:
                key = (dataset, strategy, split_name)
                if key not in grouped_values:
                    continue
                for gnn, operation_name in grouped_values[key]:
                    values.extend(grouped_values[key][(gnn, operation_name)])
            if not values:
                y_limits_by_dataset_split[(dataset, split_name)] = (0.0, 1.0)
                continue
            ymin = min(values)
            ymax = max(values)
            if math.isclose(ymin, ymax):
                if math.isclose(ymax, 0.0):
                    ymax = 1.0
                else:
                    ymin = min(0.0, ymin)
            y_limits_by_dataset_split[(dataset, split_name)] = (ymin, ymax)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        for key in sorted(grouped_values):
            dataset, strategy, split_name = key
            gnns_for_group = _order_gnns(
                {
                    gnn
                    for gnn in self.selected_gnns
                    if any(
                        (dataset, gnn, strategy, val_id, split_name) in self.flips_per_operation_present
                        for val_id in selected_vals
                    )
                }
            )
            if not gnns_for_group:
                continue

            include_operations = list(FLIPS_PER_OPERATION_NAMES)
            edge_relabel_name = "EDGE RELABEL"
            edge_relabel_total = sum(
                sum(grouped_values[key].get((gnn, edge_relabel_name), [])) for gnn in gnns_for_group
            )
            if edge_relabel_total == 0.0:
                include_operations = [name for name in include_operations if name != edge_relabel_name]

            if not include_operations:
                continue

            figure = plt.figure()
            ax = figure.add_subplot(111)
            data_groups = []
            positions = []
            gap = 1
            width = 0.6
            num_gnns = len(gnns_for_group)
            for operation_idx, operation_name in enumerate(include_operations):
                for gnn_idx, gnn in enumerate(gnns_for_group):
                    values = grouped_values[key].get((gnn, operation_name), [])
                    data_groups.append(values or [0.0])
                    positions.append(operation_idx * (num_gnns + gap) + gnn_idx)

            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
            algorithm_colors = {
                "GCN": color_cycle[0 % len(color_cycle)],
                "GATv2": color_cycle[1 % len(color_cycle)],
                "GraphSAGE": color_cycle[2 % len(color_cycle)],
                "GIN": color_cycle[3 % len(color_cycle)],
            }
            fallback_colors = {
                gnn: color_cycle[idx % len(color_cycle)] for idx, gnn in enumerate(gnns_for_group)
            }

            boxplot = ax.boxplot(data_groups, positions=positions, widths=width, patch_artist=True)
            for idx, box in enumerate(boxplot["boxes"]):
                gnn = gnns_for_group[idx % num_gnns]
                color = algorithm_colors.get(gnn, fallback_colors[gnn])
                box.set_facecolor(color)
                box.set_edgecolor("black")
            for whisker in boxplot["whiskers"]:
                whisker.set_color("black")
            for cap in boxplot["caps"]:
                cap.set_color("black")
            for median in boxplot["medians"]:
                median.set_color("white")
                median.set_linewidth(1.5)
            for flier in boxplot.get("fliers", []):
                flier.set_markeredgecolor("black")

            xtick_positions = [
                operation_idx * (num_gnns + gap) + (num_gnns - 1) / 2
                for operation_idx in range(len(include_operations))
            ]
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(include_operations, rotation=15, ha="right")
            ax.set_ylabel("Percentage of Decision Changes")
            ax.set_ylim(*y_limits_by_dataset_split[(dataset, split_name)])

            figure.tight_layout()
            out_png = base_dir / (
                f"flips_per_operation_relative_combined__{_slugify((dataset, strategy, split_name))}.png"
            )
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

    def _write_flips_statistics_outputs(self) -> None:
        base_dir = self.output_dir / "experiments_flips_statistics"
        base_dir.mkdir(parents=True, exist_ok=True)

        selected_vals = sorted(self.selected_vals, key=lambda item: _to_int(item, default=0))
        present_triples = {
            (dataset, gnn, strategy)
            for dataset, gnn, strategy, _, _ in self.flips_statistics_present
        }

        fold_rows: List[Dict[str, object]] = []
        for dataset, gnn, strategy in sorted(present_triples):
            for split_name in FLIPS_STATISTICS_SPLITS:
                max_decision_changes_for_group = 0
                for key in self.flips_statistics_counts:
                    key_dataset, key_gnn, key_strategy, _, key_split_name, key_decision_change_count = key
                    if (
                        key_dataset == dataset
                        and key_gnn == gnn
                        and key_strategy == strategy
                        and key_split_name == split_name
                    ):
                        max_decision_changes_for_group = max(max_decision_changes_for_group, key_decision_change_count)

                for val_id in selected_vals:
                    if (dataset, gnn, strategy, val_id, split_name) not in self.flips_statistics_present:
                        continue
                    for decision_change_count in range(max_decision_changes_for_group + 1):
                        fold_rows.append(
                            {
                                "dataset": dataset,
                                "gnn_algorithm": gnn,
                                "path_strategy": strategy,
                                "split": split_name,
                                "val_id": val_id,
                                "decision_change_count": decision_change_count,
                                "path_count": self.flips_statistics_counts.get(
                                    (dataset, gnn, strategy, val_id, split_name, decision_change_count), 0
                                ),
                            }
                        )

        fold_path = base_dir / "flips_statistics_fold_counts.csv"
        with fold_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "val_id",
                    "decision_change_count",
                    "path_count",
                ],
            )
            writer.writeheader()
            for row in fold_rows:
                writer.writerow(row)

        summary_path = base_dir / "flips_statistics_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "decision_change_count",
                    "num_folds",
                    "mean_path_count",
                    "std_path_count",
                    "min_path_count",
                    "max_path_count",
                ],
            )
            writer.writeheader()

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                plt = None

            grouped_rows = defaultdict(list)
            for row in fold_rows:
                grouped_rows[
                    (row["dataset"], row["gnn_algorithm"], row["path_strategy"], row["split"])
                ].append(row)

            for key in sorted(grouped_rows):
                dataset, gnn, strategy, split_name = key
                decision_change_values = defaultdict(list)
                for row in grouped_rows[key]:
                    decision_change_values[_to_int(str(row["decision_change_count"]), default=0)].append(
                        _to_float(str(row["path_count"]), default=0.0)
                    )

                ordered_decision_change_counts = sorted(decision_change_values.keys())
                means_for_plot: List[float] = []
                stds_for_plot: List[float] = []
                for decision_change_count in ordered_decision_change_counts:
                    values = decision_change_values[decision_change_count] or [0.0]
                    stats = RunningStats()
                    for value in values:
                        stats.add(value)
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "gnn_algorithm": gnn,
                            "path_strategy": strategy,
                            "split": split_name,
                            "decision_change_count": decision_change_count,
                            "num_folds": len(values),
                            "mean_path_count": stats.mean(),
                            "std_path_count": stats.std(),
                            "min_path_count": stats.min_value if stats.count else 0.0,
                            "max_path_count": stats.max_value if stats.count else 0.0,
                        }
                    )
                    means_for_plot.append(stats.mean())
                    stds_for_plot.append(stats.std())

                if plt is not None and ordered_decision_change_counts:
                    figure = plt.figure(figsize=(11, 6))
                    ax = figure.add_subplot(111)
                    positions = list(range(len(ordered_decision_change_counts)))
                    ax.bar(positions, means_for_plot, yerr=stds_for_plot, capsize=4)
                    ax.set_xticks(positions)
                    ax.set_xticklabels([str(item) for item in ordered_decision_change_counts])
                    ax.set_xlabel("Number of Decision Changes")
                    ax.set_ylabel("Number of Paths")
                    ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
                    figure.tight_layout()
                    out_png = base_dir / f"flips_statistics__{_slugify((dataset, gnn, strategy, split_name))}.png"
                    figure.savefig(out_png, dpi=150)
                    plt.close(figure)

    def _write_class_change_heatmap_outputs(self, flipping_only: bool) -> None:
        heatmap_tick_fontsize = 20
        heatmap_annotation_fontsize = 24

        def _collect_mode_summary_stats(
            totals: Dict[Tuple[str, str, str, str, str, str], float],
            counts: Dict[Tuple[str, str, str, str, str, str], int],
        ) -> Dict[Tuple[str, str, str, str, str], Tuple[float, float]]:
            mode_summary_stats: Dict[Tuple[str, str, str, str, str], Tuple[float, float]] = {}
            for dataset, gnn, strategy in sorted(present_triples):
                for class_name in self.class_columns:
                    for operation_name in FLIPS_PER_OPERATION_NAMES:
                        stats = RunningStats()
                        for val_id in selected_vals:
                            if (dataset, gnn, strategy, val_id) not in self.class_change_present:
                                continue
                            key = (dataset, gnn, strategy, val_id, operation_name, class_name)
                            total = totals.get(key, 0.0)
                            count = counts.get(key, 0)
                            stats.add((total / count) if count > 0 else 0.0)
                        mode_summary_stats[(dataset, strategy, class_name, gnn, operation_name)] = (
                            stats.mean(),
                            stats.std(),
                        )
            return mode_summary_stats

        def _shared_color_limits(
            dataset: str,
            strategy: str,
            class_name: str,
            primary_stats: Dict[Tuple[str, str, str, str, str], Tuple[float, float]],
            secondary_stats: Dict[Tuple[str, str, str, str, str], Tuple[float, float]],
            gnns_sorted: List[str],
        ) -> Tuple[float, float]:
            values = []
            for stats_map in (primary_stats, secondary_stats):
                for gnn in gnns_sorted:
                    for operation_name in FLIPS_PER_OPERATION_NAMES:
                        mean_value, _ = stats_map.get((dataset, strategy, class_name, gnn, operation_name), (0.0, 0.0))
                        values.append(mean_value)
            if not values:
                return (0.0, 1.0)
            vmin = min(values)
            vmax = max(values)
            if math.isclose(vmin, vmax):
                if math.isclose(vmin, 0.0):
                    return (0.0, 1.0)
                if vmin > 0.0:
                    return (0.0, vmin)
                return (vmin, 0.0)
            return (vmin, vmax)

        if flipping_only:
            base_dir = self.output_dir / "experiments_class_change_heatmaps_flipping_only"
            fold_filename = "class_change_heatmaps_flipping_only_fold_values.csv"
            summary_filename = "class_change_heatmaps_flipping_only_summary.csv"
        else:
            base_dir = self.output_dir / "experiments_class_change_heatmaps"
            fold_filename = "class_change_heatmaps_fold_values.csv"
            summary_filename = "class_change_heatmaps_summary.csv"
        base_dir.mkdir(parents=True, exist_ok=True)

        selected_vals = sorted(self.selected_vals, key=lambda item: _to_int(item, default=0))
        present_triples = {(dataset, gnn, strategy) for dataset, gnn, strategy, _ in self.class_change_present}

        if flipping_only:
            total_map = self.class_change_flipping_total
            count_map = self.class_change_flipping_count
        else:
            total_map = self.class_change_total
            count_map = self.class_change_count

        fold_rows: List[Dict[str, object]] = []
        for dataset, gnn, strategy in sorted(present_triples):
            for val_id in selected_vals:
                if (dataset, gnn, strategy, val_id) not in self.class_change_present:
                    continue
                for class_name in self.class_columns:
                    for operation_name in FLIPS_PER_OPERATION_NAMES:
                        key = (dataset, gnn, strategy, val_id, operation_name, class_name)
                        total = total_map.get(key, 0.0)
                        count = count_map.get(key, 0)
                        mean_change = (total / count) if count > 0 else 0.0
                        fold_rows.append(
                            {
                                "dataset": dataset,
                                "gnn_algorithm": gnn,
                                "path_strategy": strategy,
                                "val_id": val_id,
                                "class_column": class_name,
                                "operation_str": operation_name,
                                "mean_class_change": mean_change,
                            }
                        )

        fold_path = base_dir / fold_filename
        with fold_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "val_id",
                    "class_column",
                    "operation_str",
                    "mean_class_change",
                ],
            )
            writer.writeheader()
            for row in fold_rows:
                writer.writerow(row)

        summary_path = base_dir / summary_filename
        with summary_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "class_column",
                    "operation_str",
                    "num_folds",
                    "mean_class_change",
                    "std_class_change",
                    "min_class_change",
                    "max_class_change",
                ],
            )
            writer.writeheader()

            grouped_rows = defaultdict(list)
            for row in fold_rows:
                grouped_rows[(row["dataset"], row["gnn_algorithm"], row["path_strategy"], row["class_column"])].append(row)

            summary_stats = {}
            for key in sorted(grouped_rows):
                dataset, gnn, strategy, class_name = key
                op_values = {operation: [] for operation in FLIPS_PER_OPERATION_NAMES}
                for row in grouped_rows[key]:
                    op_values[row["operation_str"]].append(_to_float(row["mean_class_change"], default=0.0))
                for operation_name in FLIPS_PER_OPERATION_NAMES:
                    stats = RunningStats()
                    values = op_values[operation_name] or [0.0]
                    for value in values:
                        stats.add(value)
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "gnn_algorithm": gnn,
                            "path_strategy": strategy,
                            "class_column": class_name,
                            "operation_str": operation_name,
                            "num_folds": len(values),
                            "mean_class_change": stats.mean(),
                            "std_class_change": stats.std(),
                            "min_class_change": stats.min_value if stats.count else 0.0,
                            "max_class_change": stats.max_value if stats.count else 0.0,
                        }
                    )
                    summary_stats[(dataset, strategy, class_name, gnn, operation_name)] = (stats.mean(), stats.std())

            alternate_summary_stats = _collect_mode_summary_stats(
                self.class_change_flipping_total if not flipping_only else self.class_change_total,
                self.class_change_flipping_count if not flipping_only else self.class_change_count,
            )

            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.cm as cm
                import matplotlib.colors as colors
                import matplotlib.pyplot as plt
                import numpy as np
            except ImportError:
                return

            present_trios_for_plot = {(row["dataset"], row["path_strategy"], row["class_column"]) for row in fold_rows}
            gnns_sorted = sorted(self.selected_gnns)
            for dataset, strategy, class_name in sorted(present_trios_for_plot):
                data = np.zeros((len(gnns_sorted), len(FLIPS_PER_OPERATION_NAMES)))
                mean_annotations = [["" for _ in FLIPS_PER_OPERATION_NAMES] for _ in gnns_sorted]
                mean_std_annotations = [["" for _ in FLIPS_PER_OPERATION_NAMES] for _ in gnns_sorted]
                vmin, vmax = _shared_color_limits(
                    dataset,
                    strategy,
                    class_name,
                    summary_stats,
                    alternate_summary_stats,
                    gnns_sorted,
                )
                for row_idx, gnn in enumerate(gnns_sorted):
                    for col_idx, operation_name in enumerate(FLIPS_PER_OPERATION_NAMES):
                        mean_value, std_value = summary_stats.get(
                            (dataset, strategy, class_name, gnn, operation_name), (0.0, 0.0)
                        )
                        data[row_idx, col_idx] = mean_value
                        mean_annotations[row_idx][col_idx] = f"{mean_value:.2f}"
                        mean_std_annotations[row_idx][col_idx] = f"{mean_value:.2f}\n+- {std_value:.2f}"

                mode_suffix = "flipping_only" if flipping_only else "all_operations"
                output_slug = _slugify((dataset, strategy, class_name, mode_suffix))

                def _save_heatmap_variant(
                    filename_suffix: str,
                    annotations: Optional[List[List[str]]],
                ) -> None:
                    figure = plt.figure(figsize=(12, 6))
                    ax = figure.add_subplot(111)
                    ax.imshow(data, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
                    ax.set_xticks(range(len(FLIPS_PER_OPERATION_NAMES)))
                    ax.set_xticklabels(
                        FLIPS_PER_OPERATION_NAMES,
                        rotation=25,
                        ha="right",
                        fontsize=heatmap_tick_fontsize,
                    )
                    ax.set_yticks(range(len(gnns_sorted)))
                    ax.set_yticklabels(
                        [_display_gnn_label(gnn) for gnn in gnns_sorted],
                        fontsize=heatmap_tick_fontsize,
                    )
                    if annotations is not None:
                        for annotation_row_idx in range(len(gnns_sorted)):
                            for annotation_col_idx in range(len(FLIPS_PER_OPERATION_NAMES)):
                                ax.text(
                                    annotation_col_idx,
                                    annotation_row_idx,
                                    annotations[annotation_row_idx][annotation_col_idx],
                                    ha="center",
                                    va="center",
                                    color="white",
                                    fontsize=heatmap_annotation_fontsize,
                                )
                    figure.tight_layout()
                    out_png = base_dir / f"class_change_heatmap__{output_slug}{filename_suffix}.png"
                    figure.savefig(out_png, dpi=150)
                    plt.close(figure)

                _save_heatmap_variant("__no_numbers", None)
                _save_heatmap_variant("", mean_annotations)
                _save_heatmap_variant("__mean_std", mean_std_annotations)

                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                scalar_mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
                scalar_mappable.set_array([])
                legend_ticks = [vmin, (vmin + vmax) / 2, vmax]
                legend_tick_labels = [f"{value:.2f}" for value in legend_ticks]

                legend_figure = plt.figure(figsize=(2.6, 4.2))
                legend_ax = legend_figure.add_axes([0.18, 0.12, 0.10, 0.76])
                colorbar = legend_figure.colorbar(scalar_mappable, cax=legend_ax)
                colorbar.set_ticks(legend_ticks)
                colorbar.set_ticklabels(legend_tick_labels)
                colorbar.ax.tick_params(labelsize=20)
                colorbar.set_label("Mean Class Change", fontsize=22, labelpad=14)
                legend_png = base_dir / f"class_change_heatmap_legend__{output_slug}.png"
                legend_figure.savefig(legend_png, dpi=150, bbox_inches="tight", pad_inches=0.12)
                plt.close(legend_figure)

                horizontal_legend_figure = plt.figure(figsize=(4.8, 1.4))
                horizontal_legend_ax = horizontal_legend_figure.add_axes([0.12, 0.48, 0.76, 0.18])
                horizontal_colorbar = horizontal_legend_figure.colorbar(
                    scalar_mappable,
                    cax=horizontal_legend_ax,
                    orientation="horizontal",
                )
                horizontal_colorbar.set_ticks(legend_ticks)
                horizontal_colorbar.set_ticklabels(legend_tick_labels)
                horizontal_colorbar.ax.tick_params(labelsize=20)
                horizontal_colorbar.set_label("Mean Class Change", fontsize=22)
                horizontal_legend_png = base_dir / f"class_change_heatmap_legend_horizontal__{output_slug}.png"
                horizontal_legend_figure.savefig(horizontal_legend_png, dpi=150, bbox_inches="tight", pad_inches=0.12)
                plt.close(horizontal_legend_figure)


@click.command()
@click.option("--data-path", type=click.Path(path_type=Path), default=Path("results/all_results.csv"), show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("results/Experiments"), show_default=True)
@click.option("--dataset", "datasets", multiple=True, help="Datasets to include. Defaults to all.")
@click.option("--gnn", "gnns", multiple=True, help="GNNs to include. Defaults to all.")
@click.option("--path-strategy", "path_strategies", multiple=True, help="Path strategies to include. Defaults to all.")
@click.option("--val-id", "val_ids", multiple=True, help="Validation fold ids to include. Defaults to all.")
@click.option(
    "--path-split",
    "path_splits",
    multiple=True,
    type=click.Choice(ALLOWED_PATH_SPLITS, case_sensitive=False),
    default=("validation",),
    show_default=True,
    help="Path split scope (repeatable). A path is included if it matches any selected split.",
)
@click.option(
    "--correct-filter",
    type=click.Choice(ALLOWED_CORRECT_FILTERS, case_sensitive=False),
    default="correct",
    show_default=True,
    help="Path correctness filter.",
)
@click.option("--min-step", type=int, default=0, show_default=True)
@click.option("--max-step", type=int, default=None)
@click.option("--max-sampled-paths", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--experiments",
    default=",".join(DEFAULT_EXPERIMENTS),
    show_default=True,
    help="Comma-separated subset of experiments.",
)
def main(
    data_path: Path,
    output_dir: Path,
    datasets: Tuple[str, ...],
    gnns: Tuple[str, ...],
    path_strategies: Tuple[str, ...],
    val_ids: Tuple[str, ...],
    path_splits: Tuple[str, ...],
    correct_filter: str,
    min_step: int,
    max_step: Optional[int],
    max_sampled_paths: int,
    seed: int,
    experiments: str,
):
    if not data_path.exists():
        raise click.ClickException(f"Results file not found: {data_path}")

    metadata = discover_metadata(data_path)
    chosen_datasets = list(datasets) if datasets else metadata["datasets"]
    chosen_gnns = list(gnns) if gnns else metadata["gnns"]
    chosen_strategies = list(path_strategies) if path_strategies else metadata["strategies"]
    chosen_vals = list(val_ids) if val_ids else metadata["vals"]
    chosen_path_splits = [item.lower() for item in path_splits]
    chosen_correct_filter = correct_filter.lower()
    resolved_max_step = metadata["max_step"] if max_step is None else max_step
    selected_experiments = [item.strip() for item in experiments.split(",") if item.strip()]

    unknown = sorted(set(selected_experiments) - set(ALL_EXPERIMENTS))
    if unknown:
        raise click.ClickException(f"Unknown experiments requested: {unknown}")

    runner = ExperimentRunner(
        data_path=data_path,
        output_dir=output_dir,
        selected_datasets=chosen_datasets,
        selected_gnns=chosen_gnns,
        selected_strategies=chosen_strategies,
        selected_vals=chosen_vals,
        class_columns=metadata["class_columns"],
        selected_path_splits=chosen_path_splits,
        correct_filter=chosen_correct_filter,
        min_step=min_step,
        max_step=resolved_max_step,
        max_sampled_paths=max_sampled_paths,
        seed=seed,
        experiments=selected_experiments,
    )
    runner.run()

    click.echo("Experiment run complete.")
    click.echo(f"Rows scanned: {metadata['row_count']}")
    click.echo(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
