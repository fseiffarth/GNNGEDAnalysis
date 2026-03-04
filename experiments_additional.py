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
    "flip_position_sensitivity",
    "flip_streaks_and_recovery",
    "margin_to_flip_analysis",
    "operation_transition_instability",
    "path_consistency_score",
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
POSITION_BINS = (
    (0.0, 1.0 / 3.0, "early"),
    (1.0 / 3.0, 2.0 / 3.0, "mid"),
    (2.0 / 3.0, 1.0 + 1e-12, "late"),
)
POSITION_BIN_ORDER = ("early", "mid", "late")


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


def _position_bin(rel_pos: float) -> str:
    for start, end, name in POSITION_BINS:
        if start <= rel_pos < end:
            return name
    return "late"


def _parse_margin_bins(value: str) -> List[float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) < 2:
        raise click.ClickException("margin-bins must contain at least two numeric edges.")
    edges = []
    for part in parts:
        try:
            edges.append(float(part))
        except ValueError as exc:
            raise click.ClickException(f"Invalid margin bin edge: {part}") from exc
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise click.ClickException("margin-bins must be strictly increasing.")
    return edges


def _margin_bin(value: float, edges: List[float]) -> str:
    if value < edges[0]:
        return f"(-inf,{edges[0]:.6g})"
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        is_last = idx == len(edges) - 2
        if (left <= value < right) or (is_last and left <= value <= right):
            close = "]" if is_last else ")"
            return f"[{left:.6g},{right:.6g}{close}"
    return f"({edges[-1]:.6g},inf)"


def _margin_bin_left_edge(label: str) -> float:
    if label.startswith("(-inf"):
        return float("-inf")
    if label.startswith("("):
        body = label[1:]
    elif label.startswith("["):
        body = label[1:]
    else:
        return float("inf")
    left_part = body.split(",", 1)[0].strip()
    try:
        return float(left_part)
    except ValueError:
        return float("inf")


def _sorted_margin_bins(labels: Iterable[str]) -> List[str]:
    return sorted(labels, key=_margin_bin_left_edge)


def _sort_position_bins(labels: Iterable[str]) -> List[str]:
    index_map = {name: idx for idx, name in enumerate(POSITION_BIN_ORDER)}
    return sorted(labels, key=lambda item: index_map.get(item, len(index_map)))


def _path_passes_correct_filter(is_correct_path: int, correct_filter: str) -> bool:
    if correct_filter == "all":
        return True
    if correct_filter == "correct":
        return is_correct_path == 1
    if correct_filter == "incorrect":
        return is_correct_path == 0
    raise ValueError(f"Unsupported correct_filter: {correct_filter}")


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


class AdditionalExperimentRunner:
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
        margin_bins: List[float],
        min_transition_count: int,
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
        self.margin_bins = margin_bins
        self.min_transition_count = max(min_transition_count, 1)
        self.experiments = set(experiments)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.position_counts = defaultdict(int)

        self.streak_path_rows: List[Dict[str, object]] = []
        self.streak_hist = defaultdict(int)

        self.margin_counts = defaultdict(int)
        self.margin_distribution = defaultdict(lambda: {"flip": RunningStats(), "non_flip": RunningStats()})

        self.transition_op_totals = defaultdict(int)
        self.transition_op_flips = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.bigram_flips = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        self.trigram_flips = defaultdict(int)

        self.consistency_path_rows: List[Dict[str, object]] = []

    def _group_in_scope(self, group_key: Tuple[str, str, str, str]) -> bool:
        dataset, gnn, strategy, val_id = group_key
        return (
            dataset in self.selected_datasets
            and gnn in self.selected_gnns
            and strategy in self.selected_strategies
            and val_id in self.selected_vals
        )

    def _path_split_names(self, endpoint_is_train_path: int, endpoint_is_validation_path: int) -> List[str]:
        available = []
        if endpoint_is_train_path == 1:
            available.append("train")
        if endpoint_is_validation_path == 1:
            available.append("validation")
        if not available:
            return []
        if self.selected_path_splits and not (set(available) & self.selected_path_splits):
            return []
        split_names = ["all"]
        split_names.extend(available)
        return split_names

    def _finalize_path(
        self,
        path_key: Tuple[str, str, str, str, str, str],
        rows: List[Dict[str, object]],
        endpoint_is_correct_path: int,
        endpoint_is_train_path: int,
        endpoint_is_validation_path: int,
    ) -> None:
        dataset, gnn, strategy, _, val_id, path_idx = path_key
        group_key = (dataset, gnn, strategy, val_id)

        if not self._group_in_scope(group_key):
            return
        if not _path_passes_correct_filter(endpoint_is_correct_path, self.correct_filter):
            return
        split_names = self._path_split_names(endpoint_is_train_path, endpoint_is_validation_path)
        if not split_names:
            return

        steps = sorted(rows, key=lambda item: int(item["step_id"]))
        if not steps:
            return

        if "flip_position_sensitivity" in self.experiments:
            denom = max(len(steps) - 1, 1)
            for idx, item in enumerate(steps):
                if item["operation_str"] not in FLIPS_PER_OPERATION_NAMES:
                    continue
                rel_pos = idx / denom
                pos_bin = _position_bin(rel_pos)
                is_flipping = int(item["is_flipping"])
                for split_name in split_names:
                    base_key = (
                        dataset,
                        gnn,
                        strategy,
                        val_id,
                        split_name,
                        pos_bin,
                        item["operation_str"],
                        "total",
                    )
                    self.position_counts[base_key] += 1
                    if is_flipping == 1:
                        flip_key = (
                            dataset,
                            gnn,
                            strategy,
                            val_id,
                            split_name,
                            pos_bin,
                            item["operation_str"],
                            "flip",
                        )
                        self.position_counts[flip_key] += 1

        if "flip_streaks_and_recovery" in self.experiments:
            flips = [int(item["is_flipping"]) for item in steps]
            pred_labels = [str(item["predicted_label"]) for item in steps]
            streak_lengths = []
            current = 0
            for flip in flips:
                if flip == 1:
                    current += 1
                elif current > 0:
                    streak_lengths.append(current)
                    current = 0
            if current > 0:
                streak_lengths.append(current)

            num_flips = sum(flips)
            num_streaks = len(streak_lengths)
            max_streak = max(streak_lengths) if streak_lengths else 0
            mean_streak = (sum(streak_lengths) / num_streaks) if num_streaks else 0.0

            initial_label = pred_labels[0]
            final_label = pred_labels[-1]
            recovered = int(num_flips > 0 and final_label == initial_label)

            steps_to_recovery = ""
            if num_flips > 0:
                first_flip_idx = next((i for i, value in enumerate(flips) if value == 1), None)
                if first_flip_idx is not None:
                    for idx in range(first_flip_idx + 1, len(pred_labels)):
                        if pred_labels[idx] == initial_label:
                            steps_to_recovery = str(idx - first_flip_idx)
                            break

            for split_name in split_names:
                self.streak_path_rows.append(
                    {
                        "dataset": dataset,
                        "gnn_algorithm": gnn,
                        "path_strategy": strategy,
                        "val_id": val_id,
                        "split": split_name,
                        "path_idx_int": path_idx,
                        "num_steps": len(steps),
                        "num_flips": num_flips,
                        "num_flip_streaks": num_streaks,
                        "max_flip_streak": max_streak,
                        "mean_flip_streak": mean_streak,
                        "initial_label": initial_label,
                        "final_label": final_label,
                        "recovered_to_initial_label": recovered,
                        "steps_to_recovery": steps_to_recovery,
                    }
                )
                for streak_len in streak_lengths:
                    self.streak_hist[(dataset, gnn, strategy, val_id, split_name, streak_len)] += 1

        if "margin_to_flip_analysis" in self.experiments:
            for item in steps:
                if item["operation_str"] not in FLIPS_PER_OPERATION_NAMES:
                    continue
                margin = float(item["margin"])
                margin_bin = _margin_bin(margin, self.margin_bins)
                is_flip = int(item["is_flipping"])
                for split_name in split_names:
                    total_key = (dataset, gnn, strategy, val_id, split_name, margin_bin, item["operation_str"], "total")
                    self.margin_counts[total_key] += 1
                    if is_flip == 1:
                        flip_key = (dataset, gnn, strategy, val_id, split_name, margin_bin, item["operation_str"], "flip")
                        self.margin_counts[flip_key] += 1
                    dist_key = (dataset, gnn, strategy, val_id, split_name)
                    bucket = "flip" if is_flip == 1 else "non_flip"
                    self.margin_distribution[dist_key][bucket].add(margin)

        if "operation_transition_instability" in self.experiments:
            op_steps = [item for item in steps if item["operation_str"] in FLIPS_PER_OPERATION_NAMES]
            for split_name in split_names:
                for item in op_steps:
                    op_key = (dataset, gnn, strategy, val_id, split_name, item["operation_str"])
                    self.transition_op_totals[op_key] += 1
                    if int(item["is_flipping"]) == 1:
                        self.transition_op_flips[op_key] += 1

                for idx in range(1, len(op_steps)):
                    prev_item = op_steps[idx - 1]
                    curr_item = op_steps[idx]
                    key = (
                        dataset,
                        gnn,
                        strategy,
                        val_id,
                        split_name,
                        prev_item["operation_str"],
                        curr_item["operation_str"],
                    )
                    self.bigram_counts[key] += 1
                    if int(curr_item["is_flipping"]) == 1:
                        self.bigram_flips[key] += 1

                for idx in range(2, len(op_steps)):
                    first = op_steps[idx - 2]
                    second = op_steps[idx - 1]
                    third = op_steps[idx]
                    key = (
                        dataset,
                        gnn,
                        strategy,
                        val_id,
                        split_name,
                        first["operation_str"],
                        second["operation_str"],
                        third["operation_str"],
                    )
                    self.trigram_counts[key] += 1
                    if int(third["is_flipping"]) == 1:
                        self.trigram_flips[key] += 1

        if "path_consistency_score" in self.experiments:
            labels = [str(item["predicted_label"]) for item in steps]
            confidences = [float(item["top1_confidence"]) for item in steps]
            if len(labels) > 1:
                switch_count = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
                switch_rate = switch_count / (len(labels) - 1)
            else:
                switch_count = 0
                switch_rate = 0.0

            endpoint_disagreement = 1 if labels[-1] != labels[0] else 0
            conf_stats = RunningStats()
            for conf in confidences:
                conf_stats.add(conf)
            confidence_volatility = conf_stats.std()
            normalized_volatility = min(confidence_volatility / 0.5, 1.0)

            score = 1.0 - (0.5 * switch_rate + 0.3 * endpoint_disagreement + 0.2 * normalized_volatility)
            score = max(0.0, min(1.0, score))

            for split_name in split_names:
                self.consistency_path_rows.append(
                    {
                        "dataset": dataset,
                        "gnn_algorithm": gnn,
                        "path_strategy": strategy,
                        "val_id": val_id,
                        "split": split_name,
                        "path_idx_int": path_idx,
                        "num_steps": len(steps),
                        "label_switch_count": switch_count,
                        "switch_rate": switch_rate,
                        "endpoint_disagreement": endpoint_disagreement,
                        "confidence_volatility": confidence_volatility,
                        "normalized_volatility": normalized_volatility,
                        "consistency_score": score,
                    }
                )

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
                if is_path == 1 and self.min_step <= step_id <= self.max_step:
                    class_values = [_to_float(row.get(name, "0.0"), default=0.0) for name in self.class_columns]
                    ordered = sorted(class_values, reverse=True)
                    top1 = ordered[0] if ordered else 0.0
                    top2 = ordered[1] if len(ordered) >= 2 else 0.0
                    margin = top1 - top2
                    state["rows"].append(
                        {
                            "step_id": step_id,
                            "operation_str": row["operation_str"],
                            "is_flipping": _to_int(row["is_flipping"], default=0),
                            "predicted_label": row.get("predicted_label", ""),
                            "top1_confidence": top1,
                            "margin": margin,
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
            file_obj.write(f"min_step: {self.min_step}\n")
            file_obj.write(f"max_step: {self.max_step}\n")

    def _write_outputs(self) -> None:
        if "flip_position_sensitivity" in self.experiments:
            self._write_flip_position_sensitivity()
        if "flip_streaks_and_recovery" in self.experiments:
            self._write_flip_streaks_and_recovery()
        if "margin_to_flip_analysis" in self.experiments:
            self._write_margin_to_flip_analysis()
        if "operation_transition_instability" in self.experiments:
            self._write_operation_transition_instability()
        if "path_consistency_score" in self.experiments:
            self._write_path_consistency_score()

    @staticmethod
    def _load_plotting(need_numpy: bool = False):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if need_numpy:
                import numpy as np

                return plt, np
            return plt, None
        except ImportError:
            return None, None

    def _write_flip_position_sensitivity(self) -> None:
        base_dir = self.output_dir / "experiments_flip_position_sensitivity"
        base_dir.mkdir(parents=True, exist_ok=True)

        fold_rows = []
        key_prefixes = {
            key[:-1]
            for key in self.position_counts.keys()
            if key[-1] == "total"
        }
        for key in sorted(key_prefixes):
            total = self.position_counts.get((*key, "total"), 0)
            flips = self.position_counts.get((*key, "flip"), 0)
            rate = (flips / total) if total else 0.0
            dataset, gnn, strategy, val_id, split_name, pos_bin, operation = key
            fold_rows.append(
                {
                    "dataset": dataset,
                    "gnn_algorithm": gnn,
                    "path_strategy": strategy,
                    "val_id": val_id,
                    "split": split_name,
                    "position_bin": pos_bin,
                    "operation_str": operation,
                    "operation_count": total,
                    "decision_change_count": flips,
                    "flip_rate": rate,
                }
            )

        with (base_dir / "flip_position_sensitivity_fold_counts.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "val_id",
                    "split",
                    "position_bin",
                    "operation_str",
                    "operation_count",
                    "decision_change_count",
                    "flip_rate",
                ],
            )
            writer.writeheader()
            for row in fold_rows:
                writer.writerow(row)

        summary_groups = defaultdict(list)
        for row in fold_rows:
            summary_groups[
                (
                    row["dataset"],
                    row["gnn_algorithm"],
                    row["path_strategy"],
                    row["split"],
                    row["position_bin"],
                    row["operation_str"],
                )
            ].append(float(row["flip_rate"]))

        with (base_dir / "flip_position_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "position_bin",
                    "operation_str",
                    "num_folds",
                    "mean_flip_rate",
                    "std_flip_rate",
                    "min_flip_rate",
                    "max_flip_rate",
                ],
            )
            writer.writeheader()
            for key in sorted(summary_groups):
                stats = RunningStats()
                for value in summary_groups[key]:
                    stats.add(value)
                dataset, gnn, strategy, split_name, pos_bin, operation = key
                writer.writerow(
                    {
                        "dataset": dataset,
                        "gnn_algorithm": gnn,
                        "path_strategy": strategy,
                        "split": split_name,
                        "position_bin": pos_bin,
                        "operation_str": operation,
                        "num_folds": stats.count,
                        "mean_flip_rate": stats.mean(),
                        "std_flip_rate": stats.std(),
                        "min_flip_rate": stats.min_value if stats.count else 0.0,
                        "max_flip_rate": stats.max_value if stats.count else 0.0,
                    }
                )

        plt, np = self._load_plotting(need_numpy=True)
        if plt is None or np is None:
            return

        grouped_values: Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], float]] = defaultdict(dict)
        grouped_bins: Dict[Tuple[str, str, str, str], set] = defaultdict(set)
        for key in summary_groups:
            dataset, gnn, strategy, split_name, pos_bin, operation = key
            stats = RunningStats()
            for value in summary_groups[key]:
                stats.add(value)
            group_key = (dataset, gnn, strategy, split_name)
            grouped_values[group_key][(pos_bin, operation)] = stats.mean()
            grouped_bins[group_key].add(pos_bin)

        for group_key in sorted(grouped_values.keys()):
            dataset, gnn, strategy, split_name = group_key
            pos_bins = _sort_position_bins(grouped_bins[group_key])
            if not pos_bins:
                continue

            matrix = np.zeros((len(pos_bins), len(FLIPS_PER_OPERATION_NAMES)))
            for row_idx, pos_bin in enumerate(pos_bins):
                for col_idx, operation in enumerate(FLIPS_PER_OPERATION_NAMES):
                    matrix[row_idx, col_idx] = grouped_values[group_key].get((pos_bin, operation), 0.0)

            figure = plt.figure(figsize=(12, 6))
            ax = figure.add_subplot(111)
            x = np.arange(len(FLIPS_PER_OPERATION_NAMES))
            width = 0.8 / max(len(pos_bins), 1)
            for row_idx, pos_bin in enumerate(pos_bins):
                ax.bar(
                    x + (row_idx - (len(pos_bins) - 1) / 2.0) * width,
                    matrix[row_idx, :],
                    width=width,
                    label=pos_bin,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(FLIPS_PER_OPERATION_NAMES, rotation=25, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Graph Edit Operation")
            ax.set_ylabel("Mean Flip Rate")
            ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
            ax.legend(title="Position Bin")
            figure.tight_layout()
            out_png = base_dir / f"flip_position_sensitivity_summary__{_slugify(group_key)}.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

            figure = plt.figure(figsize=(10, 4.5))
            ax = figure.add_subplot(111)
            image = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(FLIPS_PER_OPERATION_NAMES)))
            ax.set_xticklabels(FLIPS_PER_OPERATION_NAMES, rotation=25, ha="right")
            ax.set_yticks(range(len(pos_bins)))
            ax.set_yticklabels(pos_bins)
            ax.set_xlabel("Graph Edit Operation")
            ax.set_ylabel("Position Bin")
            ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
            figure.colorbar(image, ax=ax, label="Mean Flip Rate")
            figure.tight_layout()
            out_png = base_dir / f"flip_position_sensitivity_heatmap__{_slugify(group_key)}.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

    def _write_flip_streaks_and_recovery(self) -> None:
        base_dir = self.output_dir / "experiments_flip_streaks_and_recovery"
        base_dir.mkdir(parents=True, exist_ok=True)

        with (base_dir / "flip_streaks_and_recovery_path_metrics.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "val_id",
                    "split",
                    "path_idx_int",
                    "num_steps",
                    "num_flips",
                    "num_flip_streaks",
                    "max_flip_streak",
                    "mean_flip_streak",
                    "initial_label",
                    "final_label",
                    "recovered_to_initial_label",
                    "steps_to_recovery",
                ],
            )
            writer.writeheader()
            for row in self.streak_path_rows:
                writer.writerow(row)

        hist_rows = []
        for key, count in sorted(self.streak_hist.items()):
            dataset, gnn, strategy, val_id, split_name, streak_len = key
            hist_rows.append(
                {
                    "dataset": dataset,
                    "gnn_algorithm": gnn,
                    "path_strategy": strategy,
                    "val_id": val_id,
                    "split": split_name,
                    "streak_length": streak_len,
                    "streak_count": count,
                }
            )
        with (base_dir / "flip_streak_length_histogram_fold.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=["dataset", "gnn_algorithm", "path_strategy", "val_id", "split", "streak_length", "streak_count"],
            )
            writer.writeheader()
            for row in hist_rows:
                writer.writerow(row)

        fold_groups = defaultdict(list)
        for row in self.streak_path_rows:
            fold_groups[
                (
                    row["dataset"],
                    row["gnn_algorithm"],
                    row["path_strategy"],
                    row["val_id"],
                    row["split"],
                )
            ].append(row)

        summary_groups = defaultdict(list)
        for key, rows in fold_groups.items():
            dataset, gnn, strategy, val_id, split_name = key
            num_paths = len(rows)
            mean_num_flips = sum(int(row["num_flips"]) for row in rows) / num_paths if num_paths else 0.0
            mean_max_streak = sum(int(row["max_flip_streak"]) for row in rows) / num_paths if num_paths else 0.0
            recovery_rate = sum(int(row["recovered_to_initial_label"]) for row in rows) / num_paths if num_paths else 0.0
            recovery_steps = [int(row["steps_to_recovery"]) for row in rows if str(row["steps_to_recovery"]).strip() != ""]
            mean_steps_to_recovery = sum(recovery_steps) / len(recovery_steps) if recovery_steps else 0.0
            summary_groups[(dataset, gnn, strategy, split_name)].append(
                {
                    "val_id": val_id,
                    "num_paths": num_paths,
                    "mean_num_flips": mean_num_flips,
                    "mean_max_flip_streak": mean_max_streak,
                    "recovery_rate": recovery_rate,
                    "mean_steps_to_recovery": mean_steps_to_recovery,
                }
            )

        with (base_dir / "flip_streaks_and_recovery_summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "num_folds",
                    "mean_num_paths",
                    "mean_num_flips",
                    "mean_max_flip_streak",
                    "mean_recovery_rate",
                    "mean_steps_to_recovery",
                ],
            )
            writer.writeheader()
            for key in sorted(summary_groups):
                rows = summary_groups[key]
                stats_paths = RunningStats()
                stats_flips = RunningStats()
                stats_max_streak = RunningStats()
                stats_recovery = RunningStats()
                stats_steps = RunningStats()
                for row in rows:
                    stats_paths.add(float(row["num_paths"]))
                    stats_flips.add(float(row["mean_num_flips"]))
                    stats_max_streak.add(float(row["mean_max_flip_streak"]))
                    stats_recovery.add(float(row["recovery_rate"]))
                    stats_steps.add(float(row["mean_steps_to_recovery"]))
                dataset, gnn, strategy, split_name = key
                writer.writerow(
                    {
                        "dataset": dataset,
                        "gnn_algorithm": gnn,
                        "path_strategy": strategy,
                        "split": split_name,
                        "num_folds": len(rows),
                        "mean_num_paths": stats_paths.mean(),
                        "mean_num_flips": stats_flips.mean(),
                        "mean_max_flip_streak": stats_max_streak.mean(),
                        "mean_recovery_rate": stats_recovery.mean(),
                        "mean_steps_to_recovery": stats_steps.mean(),
                    }
                )

        plt, _ = self._load_plotting(need_numpy=False)
        if plt is None:
            return

        group_keys = sorted(summary_groups.keys())
        for group_key in group_keys:
            dataset, gnn, strategy, split_name = group_key

            streak_counts = defaultdict(int)
            for key, count in self.streak_hist.items():
                dset, model, strat, _, split_hist, streak_len = key
                if (dset, model, strat, split_hist) == group_key:
                    streak_counts[int(streak_len)] += int(count)
            if streak_counts:
                x_vals = sorted(streak_counts.keys())
                y_vals = [streak_counts[item] for item in x_vals]
                figure = plt.figure(figsize=(10, 5))
                ax = figure.add_subplot(111)
                ax.bar(x_vals, y_vals)
                ax.set_xlabel("Flip Streak Length")
                ax.set_ylabel("Streak Count")
                ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
                figure.tight_layout()
                out_png = base_dir / f"flip_streak_distribution__{_slugify(group_key)}.png"
                figure.savefig(out_png, dpi=150)
                plt.close(figure)

            path_rows = [
                row
                for row in self.streak_path_rows
                if (
                    row["dataset"],
                    row["gnn_algorithm"],
                    row["path_strategy"],
                    row["split"],
                )
                == group_key
            ]
            if not path_rows:
                continue

            recovered_by_flips = defaultdict(lambda: {"total": 0, "recovered": 0, "steps_total": 0, "steps_count": 0})
            for row in path_rows:
                num_flips = int(row["num_flips"])
                recovered = int(row["recovered_to_initial_label"])
                bucket = recovered_by_flips[num_flips]
                bucket["total"] += 1
                bucket["recovered"] += recovered
                steps_value = str(row["steps_to_recovery"]).strip()
                if steps_value:
                    bucket["steps_total"] += int(steps_value)
                    bucket["steps_count"] += 1

            x_vals = sorted(recovered_by_flips.keys())
            if not x_vals:
                continue
            recovery_rates = [
                (recovered_by_flips[item]["recovered"] / recovered_by_flips[item]["total"])
                if recovered_by_flips[item]["total"]
                else 0.0
                for item in x_vals
            ]
            mean_recovery_steps = [
                (recovered_by_flips[item]["steps_total"] / recovered_by_flips[item]["steps_count"])
                if recovered_by_flips[item]["steps_count"]
                else 0.0
                for item in x_vals
            ]

            figure = plt.figure(figsize=(10, 5))
            ax1 = figure.add_subplot(111)
            ax1.plot(x_vals, recovery_rates, marker="o", label="Recovery Rate")
            ax1.set_xlabel("Number of Flips per Path")
            ax1.set_ylabel("Recovery Rate")
            ax1.set_ylim(0.0, 1.0)
            ax2 = ax1.twinx()
            ax2.plot(x_vals, mean_recovery_steps, marker="s", color="tab:orange", label="Mean Steps to Recovery")
            ax2.set_ylabel("Steps to Recovery")
            ax1.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
            figure.tight_layout()
            out_png = base_dir / f"flip_streak_recovery__{_slugify(group_key)}.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

    def _write_margin_to_flip_analysis(self) -> None:
        base_dir = self.output_dir / "experiments_margin_to_flip_analysis"
        base_dir.mkdir(parents=True, exist_ok=True)

        fold_rows = []
        key_prefixes = {
            key[:-1]
            for key in self.margin_counts.keys()
            if key[-1] == "total"
        }
        for key in sorted(key_prefixes):
            total = self.margin_counts.get((*key, "total"), 0)
            flips = self.margin_counts.get((*key, "flip"), 0)
            rate = flips / total if total else 0.0
            dataset, gnn, strategy, val_id, split_name, margin_bin, operation = key
            fold_rows.append(
                {
                    "dataset": dataset,
                    "gnn_algorithm": gnn,
                    "path_strategy": strategy,
                    "val_id": val_id,
                    "split": split_name,
                    "margin_bin": margin_bin,
                    "operation_str": operation,
                    "step_count": total,
                    "flip_count": flips,
                    "flip_rate": rate,
                }
            )

        with (base_dir / "margin_to_flip_calibration_fold.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "val_id",
                    "split",
                    "margin_bin",
                    "operation_str",
                    "step_count",
                    "flip_count",
                    "flip_rate",
                ],
            )
            writer.writeheader()
            for row in fold_rows:
                writer.writerow(row)

        summary_groups = defaultdict(list)
        for row in fold_rows:
            summary_groups[
                (
                    row["dataset"],
                    row["gnn_algorithm"],
                    row["path_strategy"],
                    row["split"],
                    row["margin_bin"],
                    row["operation_str"],
                )
            ].append(float(row["flip_rate"]))

        with (base_dir / "margin_to_flip_calibration_summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "margin_bin",
                    "operation_str",
                    "num_folds",
                    "mean_flip_rate",
                    "std_flip_rate",
                    "min_flip_rate",
                    "max_flip_rate",
                ],
            )
            writer.writeheader()
            for key in sorted(summary_groups):
                stats = RunningStats()
                for value in summary_groups[key]:
                    stats.add(value)
                dataset, gnn, strategy, split_name, margin_bin, operation = key
                writer.writerow(
                    {
                        "dataset": dataset,
                        "gnn_algorithm": gnn,
                        "path_strategy": strategy,
                        "split": split_name,
                        "margin_bin": margin_bin,
                        "operation_str": operation,
                        "num_folds": stats.count,
                        "mean_flip_rate": stats.mean(),
                        "std_flip_rate": stats.std(),
                        "min_flip_rate": stats.min_value if stats.count else 0.0,
                        "max_flip_rate": stats.max_value if stats.count else 0.0,
                    }
                )

        with (base_dir / "margin_to_flip_distribution_summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "val_id",
                    "split",
                    "mean_margin_flipping",
                    "std_margin_flipping",
                    "mean_margin_non_flipping",
                    "std_margin_non_flipping",
                ],
            )
            writer.writeheader()
            for key in sorted(self.margin_distribution.keys()):
                dataset, gnn, strategy, val_id, split_name = key
                stats_flip = self.margin_distribution[key]["flip"]
                stats_non_flip = self.margin_distribution[key]["non_flip"]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "gnn_algorithm": gnn,
                        "path_strategy": strategy,
                        "val_id": val_id,
                        "split": split_name,
                        "mean_margin_flipping": stats_flip.mean(),
                        "std_margin_flipping": stats_flip.std(),
                        "mean_margin_non_flipping": stats_non_flip.mean(),
                        "std_margin_non_flipping": stats_non_flip.std(),
                    }
                )

        plt, _ = self._load_plotting(need_numpy=False)
        if plt is None:
            return

        grouped_rates: Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], float]] = defaultdict(dict)
        grouped_margin_bins: Dict[Tuple[str, str, str, str], set] = defaultdict(set)
        for key in summary_groups:
            dataset, gnn, strategy, split_name, margin_bin, operation = key
            stats = RunningStats()
            for value in summary_groups[key]:
                stats.add(value)
            group_key = (dataset, gnn, strategy, split_name)
            grouped_rates[group_key][(margin_bin, operation)] = stats.mean()
            grouped_margin_bins[group_key].add(margin_bin)

        for group_key in sorted(grouped_rates.keys()):
            dataset, gnn, strategy, split_name = group_key
            margin_bins = _sorted_margin_bins(grouped_margin_bins[group_key])
            if not margin_bins:
                continue

            figure = plt.figure(figsize=(12, 6))
            ax = figure.add_subplot(111)
            for operation in FLIPS_PER_OPERATION_NAMES:
                y_vals = [grouped_rates[group_key].get((margin_bin, operation), 0.0) for margin_bin in margin_bins]
                ax.plot(margin_bins, y_vals, marker="o", label=operation)
            ax.set_xlabel("Margin Bin")
            ax.set_ylabel("Mean Flip Rate")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
            ax.tick_params(axis="x", rotation=35)
            ax.legend(loc="best", fontsize=8)
            figure.tight_layout()
            out_png = base_dir / f"margin_to_flip_calibration__{_slugify(group_key)}.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

            stats_flip = RunningStats()
            stats_non_flip = RunningStats()
            for dist_key in sorted(self.margin_distribution.keys()):
                dset, model, strat, _, split_dist = dist_key
                if (dset, model, strat, split_dist) != group_key:
                    continue
                fold_flip = self.margin_distribution[dist_key]["flip"]
                fold_non_flip = self.margin_distribution[dist_key]["non_flip"]
                if fold_flip.count:
                    stats_flip.add(fold_flip.mean())
                if fold_non_flip.count:
                    stats_non_flip.add(fold_non_flip.mean())

            if stats_flip.count == 0 and stats_non_flip.count == 0:
                continue
            figure = plt.figure(figsize=(7, 5))
            ax = figure.add_subplot(111)
            ax.bar(
                ["Flipping Steps", "Non-Flipping Steps"],
                [stats_flip.mean() if stats_flip.count else 0.0, stats_non_flip.mean() if stats_non_flip.count else 0.0],
            )
            ax.set_ylabel("Mean Margin")
            ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
            figure.tight_layout()
            out_png = base_dir / f"margin_to_flip_distribution__{_slugify(group_key)}.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

    def _write_operation_transition_instability(self) -> None:
        base_dir = self.output_dir / "experiments_operation_transition_instability"
        base_dir.mkdir(parents=True, exist_ok=True)

        bigram_fold_rows = []
        for key, seq_count in sorted(self.bigram_counts.items()):
            if seq_count < self.min_transition_count:
                continue
            seq_flip_count = self.bigram_flips.get(key, 0)
            seq_rate = seq_flip_count / seq_count if seq_count else 0.0
            dataset, gnn, strategy, val_id, split_name, op_prev, op_curr = key
            baseline_total = self.transition_op_totals.get((dataset, gnn, strategy, val_id, split_name, op_curr), 0)
            baseline_flips = self.transition_op_flips.get((dataset, gnn, strategy, val_id, split_name, op_curr), 0)
            baseline_rate = baseline_flips / baseline_total if baseline_total else 0.0
            enrichment = (seq_rate / baseline_rate) if baseline_rate > 0 else 0.0
            bigram_fold_rows.append(
                {
                    "dataset": dataset,
                    "gnn_algorithm": gnn,
                    "path_strategy": strategy,
                    "val_id": val_id,
                    "split": split_name,
                    "operation_prev": op_prev,
                    "operation_curr": op_curr,
                    "sequence_count": seq_count,
                    "sequence_flip_count": seq_flip_count,
                    "sequence_flip_rate": seq_rate,
                    "baseline_flip_rate_curr_operation": baseline_rate,
                    "enrichment_ratio": enrichment,
                }
            )

        with (base_dir / "operation_transition_bigrams_fold.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "val_id",
                    "split",
                    "operation_prev",
                    "operation_curr",
                    "sequence_count",
                    "sequence_flip_count",
                    "sequence_flip_rate",
                    "baseline_flip_rate_curr_operation",
                    "enrichment_ratio",
                ],
            )
            writer.writeheader()
            for row in bigram_fold_rows:
                writer.writerow(row)

        trigram_fold_rows = []
        for key, seq_count in sorted(self.trigram_counts.items()):
            if seq_count < self.min_transition_count:
                continue
            seq_flip_count = self.trigram_flips.get(key, 0)
            seq_rate = seq_flip_count / seq_count if seq_count else 0.0
            dataset, gnn, strategy, val_id, split_name, op_prev2, op_prev1, op_curr = key
            baseline_total = self.transition_op_totals.get((dataset, gnn, strategy, val_id, split_name, op_curr), 0)
            baseline_flips = self.transition_op_flips.get((dataset, gnn, strategy, val_id, split_name, op_curr), 0)
            baseline_rate = baseline_flips / baseline_total if baseline_total else 0.0
            enrichment = (seq_rate / baseline_rate) if baseline_rate > 0 else 0.0
            trigram_fold_rows.append(
                {
                    "dataset": dataset,
                    "gnn_algorithm": gnn,
                    "path_strategy": strategy,
                    "val_id": val_id,
                    "split": split_name,
                    "operation_prev2": op_prev2,
                    "operation_prev1": op_prev1,
                    "operation_curr": op_curr,
                    "sequence_count": seq_count,
                    "sequence_flip_count": seq_flip_count,
                    "sequence_flip_rate": seq_rate,
                    "baseline_flip_rate_curr_operation": baseline_rate,
                    "enrichment_ratio": enrichment,
                }
            )

        with (base_dir / "operation_transition_trigrams_fold.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "val_id",
                    "split",
                    "operation_prev2",
                    "operation_prev1",
                    "operation_curr",
                    "sequence_count",
                    "sequence_flip_count",
                    "sequence_flip_rate",
                    "baseline_flip_rate_curr_operation",
                    "enrichment_ratio",
                ],
            )
            writer.writeheader()
            for row in trigram_fold_rows:
                writer.writerow(row)

        self._write_transition_summary(
            base_dir / "operation_transition_bigrams_summary.csv",
            bigram_fold_rows,
            ["operation_prev", "operation_curr"],
        )
        self._write_transition_summary(
            base_dir / "operation_transition_trigrams_summary.csv",
            trigram_fold_rows,
            ["operation_prev2", "operation_prev1", "operation_curr"],
        )

        plt, np = self._load_plotting(need_numpy=True)
        if plt is None or np is None:
            return

        grouped_bigrams = defaultdict(list)
        for row in bigram_fold_rows:
            grouped_bigrams[
                (
                    row["dataset"],
                    row["gnn_algorithm"],
                    row["path_strategy"],
                    row["split"],
                )
            ].append(row)

        for group_key in sorted(grouped_bigrams.keys()):
            dataset, gnn, strategy, split_name = group_key
            rate_stats = defaultdict(RunningStats)
            enrich_stats = defaultdict(RunningStats)
            for row in grouped_bigrams[group_key]:
                pair = (row["operation_prev"], row["operation_curr"])
                rate_stats[pair].add(float(row["sequence_flip_rate"]))
                enrich_stats[pair].add(float(row["enrichment_ratio"]))

            rate_matrix = np.zeros((len(FLIPS_PER_OPERATION_NAMES), len(FLIPS_PER_OPERATION_NAMES)))
            enrich_matrix = np.zeros((len(FLIPS_PER_OPERATION_NAMES), len(FLIPS_PER_OPERATION_NAMES)))
            for row_idx, op_prev in enumerate(FLIPS_PER_OPERATION_NAMES):
                for col_idx, op_curr in enumerate(FLIPS_PER_OPERATION_NAMES):
                    if (op_prev, op_curr) in rate_stats:
                        rate_matrix[row_idx, col_idx] = rate_stats[(op_prev, op_curr)].mean()
                    if (op_prev, op_curr) in enrich_stats:
                        enrich_matrix[row_idx, col_idx] = enrich_stats[(op_prev, op_curr)].mean()

            figure = plt.figure(figsize=(10, 7))
            ax = figure.add_subplot(111)
            image = ax.imshow(rate_matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(FLIPS_PER_OPERATION_NAMES)))
            ax.set_xticklabels(FLIPS_PER_OPERATION_NAMES, rotation=25, ha="right")
            ax.set_yticks(range(len(FLIPS_PER_OPERATION_NAMES)))
            ax.set_yticklabels(FLIPS_PER_OPERATION_NAMES)
            ax.set_xlabel("Current Operation")
            ax.set_ylabel("Previous Operation")
            ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
            figure.colorbar(image, ax=ax, label="Mean Sequence Flip Rate")
            figure.tight_layout()
            out_png = base_dir / f"operation_transition_bigrams_rate__{_slugify(group_key)}.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

            figure = plt.figure(figsize=(10, 7))
            ax = figure.add_subplot(111)
            image = ax.imshow(enrich_matrix, cmap="plasma", aspect="auto")
            ax.set_xticks(range(len(FLIPS_PER_OPERATION_NAMES)))
            ax.set_xticklabels(FLIPS_PER_OPERATION_NAMES, rotation=25, ha="right")
            ax.set_yticks(range(len(FLIPS_PER_OPERATION_NAMES)))
            ax.set_yticklabels(FLIPS_PER_OPERATION_NAMES)
            ax.set_xlabel("Current Operation")
            ax.set_ylabel("Previous Operation")
            ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
            figure.colorbar(image, ax=ax, label="Mean Enrichment Ratio")
            figure.tight_layout()
            out_png = base_dir / f"operation_transition_bigrams_enrichment__{_slugify(group_key)}.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

    def _write_transition_summary(self, out_path: Path, rows: List[Dict[str, object]], op_cols: List[str]) -> None:
        summary_groups = defaultdict(list)
        for row in rows:
            key = [row["dataset"], row["gnn_algorithm"], row["path_strategy"], row["split"]]
            key.extend(row[col] for col in op_cols)
            summary_groups[tuple(key)].append(row)

        fieldnames = ["dataset", "gnn_algorithm", "path_strategy", "split"]
        fieldnames.extend(op_cols)
        fieldnames.extend(["num_folds", "mean_sequence_flip_rate", "std_sequence_flip_rate", "mean_enrichment_ratio"])

        with out_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(summary_groups):
                stats_rate = RunningStats()
                stats_enrich = RunningStats()
                for row in summary_groups[key]:
                    stats_rate.add(float(row["sequence_flip_rate"]))
                    stats_enrich.add(float(row["enrichment_ratio"]))
                out = {
                    "dataset": key[0],
                    "gnn_algorithm": key[1],
                    "path_strategy": key[2],
                    "split": key[3],
                    "num_folds": len(summary_groups[key]),
                    "mean_sequence_flip_rate": stats_rate.mean(),
                    "std_sequence_flip_rate": stats_rate.std(),
                    "mean_enrichment_ratio": stats_enrich.mean(),
                }
                for idx, col in enumerate(op_cols):
                    out[col] = key[4 + idx]
                writer.writerow(out)

    def _write_path_consistency_score(self) -> None:
        base_dir = self.output_dir / "experiments_path_consistency_score"
        base_dir.mkdir(parents=True, exist_ok=True)

        with (base_dir / "path_consistency_path_metrics.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "val_id",
                    "split",
                    "path_idx_int",
                    "num_steps",
                    "label_switch_count",
                    "switch_rate",
                    "endpoint_disagreement",
                    "confidence_volatility",
                    "normalized_volatility",
                    "consistency_score",
                ],
            )
            writer.writeheader()
            for row in self.consistency_path_rows:
                writer.writerow(row)

        fold_groups = defaultdict(list)
        for row in self.consistency_path_rows:
            fold_groups[
                (
                    row["dataset"],
                    row["gnn_algorithm"],
                    row["path_strategy"],
                    row["val_id"],
                    row["split"],
                )
            ].append(float(row["consistency_score"]))

        summary_rows = []
        summary_groups = defaultdict(list)
        for key, scores in fold_groups.items():
            fold_stats = RunningStats()
            for score in scores:
                fold_stats.add(score)
            dataset, gnn, strategy, val_id, split_name = key
            summary_groups[(dataset, gnn, strategy, split_name)].append(fold_stats.mean())

        for key, fold_means in summary_groups.items():
            stats = RunningStats()
            for value in fold_means:
                stats.add(value)
            dataset, gnn, strategy, split_name = key
            summary_rows.append(
                {
                    "dataset": dataset,
                    "gnn_algorithm": gnn,
                    "path_strategy": strategy,
                    "split": split_name,
                    "num_folds": len(fold_means),
                    "mean_consistency_score": stats.mean(),
                    "std_consistency_score": stats.std(),
                    "min_consistency_score": stats.min_value if stats.count else 0.0,
                    "max_consistency_score": stats.max_value if stats.count else 0.0,
                }
            )

        summary_rows_sorted = sorted(
            summary_rows,
            key=lambda row: (
                row["dataset"],
                row["gnn_algorithm"],
                row["path_strategy"],
                row["split"],
            ),
        )

        with (base_dir / "path_consistency_summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "num_folds",
                    "mean_consistency_score",
                    "std_consistency_score",
                    "min_consistency_score",
                    "max_consistency_score",
                ],
            )
            writer.writeheader()
            for row in summary_rows_sorted:
                writer.writerow(row)

        ranking_rows = sorted(
            summary_rows,
            key=lambda row: row["mean_consistency_score"],
            reverse=True,
        )
        for idx, row in enumerate(ranking_rows, start=1):
            row["rank"] = idx

        with (base_dir / "path_consistency_configuration_ranking.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "rank",
                    "dataset",
                    "gnn_algorithm",
                    "path_strategy",
                    "split",
                    "mean_consistency_score",
                    "num_folds",
                ],
            )
            writer.writeheader()
            for row in ranking_rows:
                writer.writerow(
                    {
                        "rank": row["rank"],
                        "dataset": row["dataset"],
                        "gnn_algorithm": row["gnn_algorithm"],
                        "path_strategy": row["path_strategy"],
                        "split": row["split"],
                        "mean_consistency_score": row["mean_consistency_score"],
                        "num_folds": row["num_folds"],
                    }
                )

        plt, _ = self._load_plotting(need_numpy=False)
        if plt is None:
            return

        grouped_rows = defaultdict(list)
        for row in self.consistency_path_rows:
            grouped_rows[
                (
                    row["dataset"],
                    row["gnn_algorithm"],
                    row["path_strategy"],
                    row["split"],
                )
            ].append(row)

        for group_key in sorted(grouped_rows.keys()):
            dataset, gnn, strategy, split_name = group_key
            by_fold = defaultdict(list)
            for row in grouped_rows[group_key]:
                by_fold[str(row["val_id"])].append(float(row["consistency_score"]))
            ordered_folds = sorted(by_fold.keys(), key=lambda item: _to_int(item, default=0))
            if not ordered_folds:
                continue
            data = [by_fold[val_id] if by_fold[val_id] else [0.0] for val_id in ordered_folds]
            figure = plt.figure(figsize=(10, 5))
            ax = figure.add_subplot(111)
            ax.boxplot(data, patch_artist=True)
            ax.set_xticks(range(1, len(ordered_folds) + 1))
            ax.set_xticklabels(ordered_folds)
            ax.set_xlabel("Validation Fold")
            ax.set_ylabel("Consistency Score")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{dataset} | {gnn} | {strategy} | {split_name}")
            figure.tight_layout()
            out_png = base_dir / f"path_consistency_scores__{_slugify(group_key)}.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)

        if ranking_rows:
            labels = [
                _slugify((row["dataset"], row["gnn_algorithm"], row["path_strategy"], row["split"]))
                for row in ranking_rows
            ]
            values = [float(row["mean_consistency_score"]) for row in ranking_rows]
            positions = list(range(len(ranking_rows)))
            figure = plt.figure(figsize=(max(10, len(ranking_rows) * 0.5), 5))
            ax = figure.add_subplot(111)
            ax.bar(positions, values)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
            ax.set_ylabel("Mean Consistency Score")
            ax.set_title("Path Consistency Configuration Ranking")
            figure.tight_layout()
            out_png = base_dir / "path_consistency_ranking.png"
            figure.savefig(out_png, dpi=150)
            plt.close(figure)


@click.command()
@click.option("--data-path", type=click.Path(path_type=Path), default=Path("results/all_results.csv"), show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("results/ExperimentsAdditional"),
    show_default=True,
)
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
@click.option(
    "--experiments",
    default=",".join(DEFAULT_EXPERIMENTS),
    show_default=True,
    help="Comma-separated subset of experiments.",
)
@click.option(
    "--margin-bins",
    type=str,
    default="0,0.05,0.1,0.2,0.4,1.0",
    show_default=True,
    help="Comma-separated margin bin edges.",
)
@click.option("--min-transition-count", type=int, default=5, show_default=True)
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
    experiments: str,
    margin_bins: str,
    min_transition_count: int,
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
    parsed_margin_bins = _parse_margin_bins(margin_bins)

    unknown = sorted(set(selected_experiments) - set(ALL_EXPERIMENTS))
    if unknown:
        raise click.ClickException(f"Unknown experiments requested: {unknown}")

    runner = AdditionalExperimentRunner(
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
        margin_bins=parsed_margin_bins,
        min_transition_count=min_transition_count,
        experiments=selected_experiments,
    )
    runner.run()

    click.echo("Additional experiment run complete.")
    click.echo(f"Rows scanned: {metadata['row_count']}")
    click.echo(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
