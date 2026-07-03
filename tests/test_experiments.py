import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from click.testing import CliRunner

from experiments import ALL_EXPERIMENTS, DEFAULT_EXPERIMENTS, main


def _has_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


HEADER = [
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


def _base_row(**overrides):
    row = {
        "row_idx": "0",
        "path_idx_int": "0",
        "source_id": "0",
        "step_id": "0",
        "target_id": "1",
        "operation": "3",
        "operation_str": "EDGE INSERT",
        "class_0": "0.0",
        "class_1": "1.0",
        "predicted_label": "1",
        "true_label": "-1",
        "is_flipping": "0",
        "is_correct": "-1",
        "is_source": "0",
        "is_target": "0",
        "is_train": "0",
        "is_validation": "0",
        "is_path": "1",
        "is_correct_path_endpoint": "-1",
        "is_correct_path": "1",
        "same_endpoint_labels_predicted": "1",
        "same_endpoint_labels_true": "1",
        "is_train_path": "0",
        "is_validation_path": "1",
        "val_id": "0",
        "config_id": "0",
        "gnn_algorithm": "GIN",
        "dataset": "MUTAG",
        "path_strategy": "Rnd",
    }
    row.update(overrides)
    return row


def _write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class _FakeAxes:
    def __init__(self):
        self.imshow_calls = []
        self.boxplot_calls = []
        self.text_calls = []
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.xtick_kwargs = None
        self.ytick_kwargs = None
        self.ylim = None

    def imshow(self, data, **kwargs):
        self.imshow_calls.append({"data": data, "kwargs": kwargs})
        return object()

    def boxplot(self, data, **kwargs):
        self.boxplot_calls.append({"data": data, "kwargs": kwargs})
        count = len(data)
        return {
            "boxes": [_FakeArtist() for _ in range(count)],
            "whiskers": [_FakeArtist() for _ in range(count * 2)],
            "caps": [_FakeArtist() for _ in range(count * 2)],
            "medians": [_FakeArtist() for _ in range(count)],
            "fliers": [_FakeArtist() for _ in range(count)],
        }

    def set_xticks(self, ticks):
        self.xticks = list(ticks)

    def set_xticklabels(self, labels, **kwargs):
        self.xticklabels = list(labels)
        self.xtick_kwargs = kwargs

    def set_yticks(self, ticks):
        self.yticks = list(ticks)

    def set_yticklabels(self, labels, **kwargs):
        self.yticklabels = list(labels)
        self.ytick_kwargs = kwargs

    def set_xlabel(self, text):
        self.xlabel = text

    def set_ylabel(self, text):
        self.ylabel = text

    def set_title(self, text):
        self.title = text

    def set_ylim(self, ymin, ymax):
        self.ylim = (ymin, ymax)

    def text(self, *args, **kwargs):
        self.text_calls.append({"args": args, "kwargs": kwargs})
        return None


class _FakeFigure:
    def __init__(self, figure_kwargs=None):
        self.axes = []
        self.add_axes_calls = []
        self.colorbar_calls = 0
        self.colorbar_call_kwargs = []
        self.saved_path = None
        self.saved_dpi = None
        self.savefig_kwargs = None
        self.figure_kwargs = figure_kwargs or {}
        self.legend_calls = []
        self.colorbars = []
        self.tight_layout_called = False

    def add_subplot(self, *args, **kwargs):
        axis = _FakeAxes()
        self.axes.append(axis)
        return axis

    def add_axes(self, rect):
        axis = _FakeAxes()
        axis.rect = list(rect)
        self.axes.append(axis)
        self.add_axes_calls.append(list(rect))
        return axis

    def colorbar(self, *args, **kwargs):
        self.colorbar_calls += 1
        self.colorbar_call_kwargs.append(kwargs)
        colorbar = _FakeColorbar()
        self.colorbars.append(colorbar)
        return colorbar

    def tight_layout(self):
        self.tight_layout_called = True
        return None

    def savefig(self, path, dpi=None, **kwargs):
        self.saved_path = Path(path)
        self.saved_dpi = dpi
        self.savefig_kwargs = kwargs


class _FakeArray:
    def __init__(self, shape):
        rows, cols = shape
        self.values = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, key):
        row_idx, col_idx = key
        return self.values[row_idx][col_idx]

    def __setitem__(self, key, value):
        row_idx, col_idx = key
        self.values[row_idx][col_idx] = value


class _FakeArtist:
    def __init__(self):
        self.facecolor = None
        self.edgecolor = None
        self.color = None
        self.linewidth = None
        self.markeredgecolor = None

    def set_facecolor(self, color):
        self.facecolor = color

    def set_edgecolor(self, color):
        self.edgecolor = color

    def set_color(self, color):
        self.color = color

    def set_linewidth(self, linewidth):
        self.linewidth = linewidth

    def set_markeredgecolor(self, color):
        self.markeredgecolor = color


class _FakeColorbar:
    def __init__(self):
        self.label = None
        self.label_kwargs = None
        self.ticks = None
        self.ticklabels = None
        self.ax = _FakeColorbarAxes()

    def set_label(self, label, **kwargs):
        self.label = label
        self.label_kwargs = kwargs

    def set_ticks(self, ticks):
        self.ticks = list(ticks)

    def set_ticklabels(self, ticklabels):
        self.ticklabels = list(ticklabels)


class _FakeColorbarAxes:
    def __init__(self):
        self.tick_params_kwargs = None

    def tick_params(self, **kwargs):
        self.tick_params_kwargs = kwargs


class _FakePropCycle:
    def __init__(self, colors):
        self.colors = colors

    def by_key(self):
        return {"color": self.colors}


class _FakePatch:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeNormalize:
    def __init__(self, vmin=None, vmax=None):
        self.vmin = vmin
        self.vmax = vmax


class _FakeScalarMappable:
    def __init__(self, norm=None, cmap=None):
        self.norm = norm
        self.cmap = cmap
        self.array = None

    def set_array(self, values):
        self.array = values


def _build_fake_matplotlib_patches_module():
    patches_module = ModuleType("matplotlib.patches")
    patches_module.Patch = _FakePatch
    return patches_module


def _build_fake_matplotlib_modules(figures):
    matplotlib_module = ModuleType("matplotlib")
    pyplot_module = ModuleType("matplotlib.pyplot")
    cm_module = ModuleType("matplotlib.cm")
    colors_module = ModuleType("matplotlib.colors")
    matplotlib_module.__path__ = []
    cm_module.ScalarMappable = _FakeScalarMappable
    colors_module.Normalize = _FakeNormalize

    def _figure(*args, **kwargs):
        figure = _FakeFigure(figure_kwargs=kwargs)
        figures.append(figure)
        pyplot_module._current_figure = figure
        return figure

    pyplot_module.figure = _figure
    pyplot_module.close = lambda figure: None
    pyplot_module._current_figure = None
    pyplot_module.rcParams = {
        "axes.prop_cycle": _FakePropCycle(["C0", "C1", "C2", "C3"])
    }
    pyplot_module.legend = lambda *args, **kwargs: pyplot_module._current_figure.legend_calls.append(
        {"args": args, "kwargs": kwargs}
    )
    matplotlib_module.use = lambda backend: None
    matplotlib_module.pyplot = pyplot_module
    matplotlib_module.cm = cm_module
    matplotlib_module.colors = colors_module
    return matplotlib_module, pyplot_module


def _build_fake_numpy_module():
    numpy_module = ModuleType("numpy")
    numpy_module.zeros = lambda shape: _FakeArray(shape)
    return numpy_module


class ExperimentsCliTests(unittest.TestCase):
    def test_default_experiments_match_all_experiments(self):
        self.assertEqual(set(DEFAULT_EXPERIMENTS), set(ALL_EXPERIMENTS))

    def test_unknown_experiment_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = [
                _base_row(row_idx="0", path_idx_int="10", step_id="0", operation_str="EDGE INSERT", is_flipping="1"),
                _base_row(row_idx="1", path_idx_int="10", step_id="0", is_target="1", is_path="0", operation_str="NONE", operation="-1"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--data-path",
                    str(data_path),
                    "--output-dir",
                    str(output_dir),
                    "--experiments",
                    "overview",
                ],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Unknown experiments requested", result.output)

    def test_default_outputs_include_all_experiment_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = [
                _base_row(row_idx="0", path_idx_int="11", step_id="0", operation_str="EDGE INSERT", is_flipping="1", is_validation_path="1", is_train_path="0"),
                _base_row(row_idx="1", path_idx_int="11", step_id="1", operation_str="EDGE DELETE", is_flipping="0", is_validation_path="1", is_train_path="0"),
                _base_row(row_idx="2", path_idx_int="11", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="1", is_train_path="0"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, ["--data-path", str(data_path), "--output-dir", str(output_dir)])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            self.assertTrue((output_dir / "run_filters.txt").exists())
            self.assertTrue((output_dir / "experiments_flips_per_operation").exists())
            self.assertTrue((output_dir / "experiments_flips_per_operation_relative").exists())
            self.assertTrue((output_dir / "experiments_flips_statistics").exists())
            self.assertTrue((output_dir / "experiments_class_change_heatmaps").exists())
            self.assertTrue((output_dir / "experiments_class_change_heatmaps_flipping_only").exists())

            self.assertFalse((output_dir / "overview_groups.csv").exists())
            self.assertFalse((output_dir / "strategy_delta_pairs.csv").exists())
            self.assertFalse((output_dir / "operation_effects_groups.csv").exists())

    def test_flips_per_operation_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = []

            rows.append(_base_row(row_idx="0", val_id="0", path_idx_int="40", step_id="0", operation_str="EDGE INSERT", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="1", val_id="0", path_idx_int="40", step_id="1", operation_str="EDGE DELETE", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="2", val_id="0", path_idx_int="40", step_id="2", operation_str="EDGE RELABEL", is_flipping="0", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="3", val_id="0", path_idx_int="40", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="1", is_train_path="0"))

            rows.append(_base_row(row_idx="4", val_id="0", path_idx_int="41", step_id="0", operation_str="NODE INSERT", is_flipping="1", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="5", val_id="0", path_idx_int="41", step_id="1", operation_str="NODE DELETE", is_flipping="0", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="6", val_id="0", path_idx_int="41", step_id="2", operation_str="NODE RELABEL", is_flipping="1", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="7", val_id="0", path_idx_int="41", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="0", is_train_path="1"))

            rows.append(_base_row(row_idx="8", val_id="1", path_idx_int="42", step_id="0", operation_str="EDGE INSERT", is_flipping="0", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="9", val_id="1", path_idx_int="42", step_id="1", operation_str="EDGE DELETE", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="10", val_id="1", path_idx_int="42", step_id="2", operation_str="EDGE RELABEL", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="11", val_id="1", path_idx_int="42", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="1", is_train_path="0"))

            rows.append(_base_row(row_idx="12", val_id="1", path_idx_int="43", step_id="0", operation_str="NODE INSERT", is_flipping="0", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="13", val_id="1", path_idx_int="43", step_id="1", operation_str="NODE DELETE", is_flipping="1", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="14", val_id="1", path_idx_int="43", step_id="2", operation_str="NODE RELABEL", is_flipping="0", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="15", val_id="1", path_idx_int="43", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="0", is_train_path="1"))

            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, ["--data-path", str(data_path), "--output-dir", str(output_dir)])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            flips_dir = output_dir / "experiments_flips_per_operation"
            self.assertTrue(flips_dir.exists())

            fold_counts = flips_dir / "flips_per_operation_fold_counts.csv"
            self.assertTrue(fold_counts.exists())
            with fold_counts.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertGreater(len(rows), 0)
            self.assertEqual({row["split"] for row in rows}, {"all", "train", "validation"})
            expected_operation_order = [
                "EDGE INSERT",
                "EDGE DELETE",
                "NODE INSERT",
                "NODE DELETE",
                "EDGE RELABEL",
                "NODE RELABEL",
            ]

            by_group = {}
            for row in rows:
                key = (
                    row["dataset"],
                    row["gnn_algorithm"],
                    row["path_strategy"],
                    row["split"],
                    row["val_id"],
                )
                by_group.setdefault(key, []).append(row["operation_str"])
            for operations in by_group.values():
                self.assertEqual(operations, expected_operation_order)

            summary = flips_dir / "flips_per_operation_summary.csv"
            self.assertTrue(summary.exists())
            with summary.open("r", newline="", encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertGreater(len(summary_rows), 0)
            summary_group_rows = [
                row
                for row in summary_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["split"] == "validation"
            ]
            self.assertEqual(
                [row["operation_str"] for row in summary_group_rows],
                expected_operation_order,
            )

    def test_flips_per_operation_relative_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = []
            rows.append(_base_row(row_idx="0", val_id="0", path_idx_int="50", step_id="0", operation_str="EDGE INSERT", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="1", val_id="0", path_idx_int="50", step_id="1", operation_str="EDGE INSERT", is_flipping="0", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="2", val_id="0", path_idx_int="50", step_id="2", operation_str="EDGE DELETE", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="3", val_id="0", path_idx_int="50", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="1", is_train_path="0"))

            rows.append(_base_row(row_idx="4", val_id="1", path_idx_int="51", step_id="0", operation_str="NODE DELETE", is_flipping="1", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="5", val_id="1", path_idx_int="51", step_id="1", operation_str="NODE DELETE", is_flipping="0", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="6", val_id="1", path_idx_int="51", step_id="2", operation_str="NODE INSERT", is_flipping="1", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="7", val_id="1", path_idx_int="51", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="0", is_train_path="1"))

            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, ["--data-path", str(data_path), "--output-dir", str(output_dir)])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            rel_dir = output_dir / "experiments_flips_per_operation_relative"
            self.assertTrue(rel_dir.exists())

            fold_counts = rel_dir / "flips_per_operation_relative_fold_counts.csv"
            self.assertTrue(fold_counts.exists())
            with fold_counts.open("r", newline="", encoding="utf-8") as f:
                fold_rows = list(csv.DictReader(f))
            self.assertGreater(len(fold_rows), 0)
            self.assertEqual({row["split"] for row in fold_rows}, {"all", "train", "validation"})
            expected_operation_order = [
                "EDGE INSERT",
                "EDGE DELETE",
                "NODE INSERT",
                "NODE DELETE",
                "EDGE RELABEL",
                "NODE RELABEL",
            ]
            fold_group_rows = [
                row
                for row in fold_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["split"] == "validation"
                and row["val_id"] == "0"
            ]
            self.assertEqual(
                [row["operation_str"] for row in fold_group_rows],
                expected_operation_order,
            )

            target = next(
                row
                for row in fold_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["split"] == "validation"
                and row["val_id"] == "0"
                and row["operation_str"] == "EDGE INSERT"
            )
            self.assertEqual(target["decision_change_count"], "1")
            self.assertEqual(target["operation_count"], "2")
            self.assertAlmostEqual(float(target["relative_decision_change_count"]), 0.5)

            summary = rel_dir / "flips_per_operation_relative_summary.csv"
            self.assertTrue(summary.exists())
            with summary.open("r", newline="", encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertGreater(len(summary_rows), 0)
            summary_group_rows = [
                row
                for row in summary_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["split"] == "validation"
            ]
            self.assertEqual(
                [row["operation_str"] for row in summary_group_rows],
                expected_operation_order,
            )

    def test_flips_per_operation_relative_combined_plot_matches_legacy_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = [
                _base_row(
                    row_idx="0",
                    val_id="0",
                    path_idx_int="100",
                    step_id="0",
                    operation_str="EDGE INSERT",
                    is_flipping="1",
                    gnn_algorithm="GCN",
                ),
                _base_row(
                    row_idx="1",
                    val_id="0",
                    path_idx_int="100",
                    step_id="1",
                    operation_str="EDGE INSERT",
                    is_flipping="0",
                    gnn_algorithm="GCN",
                ),
                _base_row(
                    row_idx="2",
                    val_id="0",
                    path_idx_int="100",
                    step_id="1",
                    operation_str="NONE",
                    operation="-1",
                    is_target="1",
                    is_path="0",
                    gnn_algorithm="GCN",
                ),
                _base_row(
                    row_idx="3",
                    val_id="1",
                    path_idx_int="101",
                    step_id="0",
                    operation_str="NODE DELETE",
                    is_flipping="1",
                    gnn_algorithm="GIN",
                ),
                _base_row(
                    row_idx="4",
                    val_id="1",
                    path_idx_int="101",
                    step_id="1",
                    operation_str="NODE DELETE",
                    is_flipping="0",
                    gnn_algorithm="GIN",
                ),
                _base_row(
                    row_idx="5",
                    val_id="1",
                    path_idx_int="101",
                    step_id="1",
                    operation_str="NONE",
                    operation="-1",
                    is_target="1",
                    is_path="0",
                    gnn_algorithm="GIN",
                ),
            ]
            _write_rows(data_path, rows)

            figures = []
            fake_matplotlib, fake_pyplot = _build_fake_matplotlib_modules(figures)
            runner = CliRunner()
            with mock.patch.dict(
                sys.modules,
                {
                    "matplotlib": fake_matplotlib,
                    "matplotlib.pyplot": fake_pyplot,
                },
            ):
                result = runner.invoke(
                    main,
                    [
                        "--data-path",
                        str(data_path),
                        "--output-dir",
                        str(output_dir),
                        "--experiments",
                        "flips_per_operation_relative_combined",
                    ],
                )

            self.assertEqual(result.exit_code, 0, msg=result.output)

            plot_figure = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name.startswith("flips_per_operation_relative_combined__MUTAG__Rnd__")
            )
            self.assertEqual(plot_figure.figure_kwargs, {})
            self.assertTrue(plot_figure.tight_layout_called)
            self.assertEqual(plot_figure.saved_dpi, 150)

            axis = plot_figure.axes[0]
            self.assertEqual(axis.xtick_kwargs["rotation"], 15)
            self.assertEqual(axis.xtick_kwargs["ha"], "right")
            self.assertEqual(axis.ylabel, "Percentage of Decision Changes")
            self.assertEqual(axis.ylim, (0.0, 0.5))
            self.assertEqual(plot_figure.legend_calls, [])

    def test_flips_statistics_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = []
            rows.append(_base_row(row_idx="0", val_id="0", path_idx_int="80", step_id="0", operation_str="EDGE INSERT", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="1", val_id="0", path_idx_int="80", step_id="1", operation_str="EDGE DELETE", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="2", val_id="0", path_idx_int="80", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="1", is_train_path="0"))

            rows.append(_base_row(row_idx="3", val_id="0", path_idx_int="81", step_id="0", operation_str="NODE INSERT", is_flipping="1", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="4", val_id="0", path_idx_int="81", step_id="1", operation_str="NODE DELETE", is_flipping="1", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="5", val_id="0", path_idx_int="81", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="0", is_train_path="1"))

            rows.append(_base_row(row_idx="6", val_id="1", path_idx_int="82", step_id="0", operation_str="EDGE INSERT", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="7", val_id="1", path_idx_int="82", step_id="1", operation_str="EDGE DELETE", is_flipping="1", is_validation_path="1", is_train_path="0"))
            rows.append(_base_row(row_idx="8", val_id="1", path_idx_int="82", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="1", is_train_path="0"))

            rows.append(_base_row(row_idx="9", val_id="1", path_idx_int="83", step_id="0", operation_str="NODE INSERT", is_flipping="1", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="10", val_id="1", path_idx_int="83", step_id="1", operation_str="NODE DELETE", is_flipping="0", is_validation_path="0", is_train_path="1"))
            rows.append(_base_row(row_idx="11", val_id="1", path_idx_int="83", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1", is_validation_path="0", is_train_path="1"))

            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, ["--data-path", str(data_path), "--output-dir", str(output_dir)])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            stats_dir = output_dir / "experiments_flips_statistics"
            self.assertTrue(stats_dir.exists())

            fold_counts = stats_dir / "flips_statistics_fold_counts.csv"
            self.assertTrue(fold_counts.exists())
            with fold_counts.open("r", newline="", encoding="utf-8") as f:
                fold_rows = list(csv.DictReader(f))
            self.assertGreater(len(fold_rows), 0)
            self.assertEqual({row["split"] for row in fold_rows}, {"all", "train", "validation"})

            def _find_fold_row(split, val_id, decision_change_count):
                return next(
                    row
                    for row in fold_rows
                    if row["dataset"] == "MUTAG"
                    and row["gnn_algorithm"] == "GIN"
                    and row["path_strategy"] == "Rnd"
                    and row["split"] == split
                    and row["val_id"] == val_id
                    and row["decision_change_count"] == str(decision_change_count)
                )

            self.assertEqual(_find_fold_row("validation", "0", 2)["path_count"], "1")
            self.assertEqual(_find_fold_row("validation", "0", 1)["path_count"], "0")
            self.assertEqual(_find_fold_row("train", "1", 1)["path_count"], "1")
            self.assertEqual(_find_fold_row("train", "1", 2)["path_count"], "0")
            self.assertEqual(_find_fold_row("all", "0", 2)["path_count"], "2")
            self.assertEqual(_find_fold_row("all", "1", 1)["path_count"], "1")
            self.assertEqual(_find_fold_row("all", "1", 0)["path_count"], "0")

            summary = stats_dir / "flips_statistics_summary.csv"
            self.assertTrue(summary.exists())
            with summary.open("r", newline="", encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertGreater(len(summary_rows), 0)

            all_split_k2 = next(
                row
                for row in summary_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["split"] == "all"
                and row["decision_change_count"] == "2"
            )
            self.assertEqual(all_split_k2["num_folds"], "2")
            self.assertAlmostEqual(float(all_split_k2["mean_path_count"]), 1.5)
            self.assertAlmostEqual(float(all_split_k2["std_path_count"]), 0.5)

            if _has_matplotlib():
                png_path = stats_dir / "flips_statistics__MUTAG__GIN__Rnd__all.png"
                self.assertTrue(png_path.exists())

    def test_class_change_heatmaps_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = []
            rows.append(_base_row(row_idx="0", val_id="0", path_idx_int="60", step_id="0", operation_str="EDGE INSERT", is_flipping="0", class_0="0.1", class_1="0.9"))
            rows.append(_base_row(row_idx="1", val_id="0", path_idx_int="60", step_id="1", operation_str="EDGE INSERT", is_flipping="1", class_0="0.3", class_1="0.7"))
            rows.append(_base_row(row_idx="2", val_id="0", path_idx_int="60", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1"))

            rows.append(_base_row(row_idx="3", val_id="1", path_idx_int="61", step_id="0", operation_str="EDGE INSERT", is_flipping="0", class_0="0.2", class_1="0.8"))
            rows.append(_base_row(row_idx="4", val_id="1", path_idx_int="61", step_id="1", operation_str="EDGE INSERT", is_flipping="1", class_0="0.6", class_1="0.4"))
            rows.append(_base_row(row_idx="5", val_id="1", path_idx_int="61", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1"))

            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--data-path",
                    str(data_path),
                    "--output-dir",
                    str(output_dir),
                    "--experiments",
                    "class_change_heatmaps",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)

            heatmap_dir = output_dir / "experiments_class_change_heatmaps"
            self.assertTrue(heatmap_dir.exists())

            fold_file = heatmap_dir / "class_change_heatmaps_fold_values.csv"
            self.assertTrue(fold_file.exists())
            with fold_file.open("r", newline="", encoding="utf-8") as f:
                fold_rows = list(csv.DictReader(f))
            self.assertEqual(len(fold_rows), 24)

            target_fold = next(
                row
                for row in fold_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["val_id"] == "0"
                and row["class_column"] == "class_0"
                and row["operation_str"] == "EDGE INSERT"
            )
            self.assertAlmostEqual(float(target_fold["mean_class_change"]), 0.2)

            summary_file = heatmap_dir / "class_change_heatmaps_summary.csv"
            self.assertTrue(summary_file.exists())
            with summary_file.open("r", newline="", encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))

            target_summary = next(
                row
                for row in summary_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["class_column"] == "class_0"
                and row["operation_str"] == "EDGE INSERT"
            )
            self.assertEqual(target_summary["num_folds"], "2")
            self.assertAlmostEqual(float(target_summary["mean_class_change"]), 0.3)
            self.assertAlmostEqual(float(target_summary["std_class_change"]), 0.1)

            if _has_matplotlib():
                png_path = heatmap_dir / "class_change_heatmap__MUTAG__Rnd__class_0__all_operations.png"
                no_numbers_path = heatmap_dir / "class_change_heatmap__MUTAG__Rnd__class_0__all_operations__no_numbers.png"
                mean_std_path = heatmap_dir / "class_change_heatmap__MUTAG__Rnd__class_0__all_operations__mean_std.png"
                legend_path = heatmap_dir / "class_change_heatmap_legend__MUTAG__Rnd__class_0__all_operations.png"
                horizontal_legend_path = (
                    heatmap_dir / "class_change_heatmap_legend_horizontal__MUTAG__Rnd__class_0__all_operations.png"
                )
                self.assertTrue(png_path.exists())
                self.assertTrue(no_numbers_path.exists())
                self.assertTrue(mean_std_path.exists())
                self.assertTrue(legend_path.exists())
                self.assertTrue(horizontal_legend_path.exists())

    def test_class_change_heatmaps_flipping_only_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = []
            rows.append(_base_row(row_idx="0", val_id="0", path_idx_int="70", step_id="0", operation_str="EDGE INSERT", is_flipping="0", class_0="0.1", class_1="0.9"))
            rows.append(_base_row(row_idx="1", val_id="0", path_idx_int="70", step_id="1", operation_str="EDGE INSERT", is_flipping="1", class_0="0.5", class_1="0.5"))
            rows.append(_base_row(row_idx="2", val_id="0", path_idx_int="70", step_id="2", operation_str="EDGE INSERT", is_flipping="0", class_0="0.9", class_1="0.1"))
            rows.append(_base_row(row_idx="3", val_id="0", path_idx_int="70", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1"))

            rows.append(_base_row(row_idx="4", val_id="1", path_idx_int="71", step_id="0", operation_str="EDGE INSERT", is_flipping="0", class_0="0.2", class_1="0.8"))
            rows.append(_base_row(row_idx="5", val_id="1", path_idx_int="71", step_id="1", operation_str="EDGE INSERT", is_flipping="0", class_0="0.6", class_1="0.4"))
            rows.append(_base_row(row_idx="6", val_id="1", path_idx_int="71", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1"))

            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--data-path",
                    str(data_path),
                    "--output-dir",
                    str(output_dir),
                    "--experiments",
                    "class_change_heatmaps_flipping_only",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)

            heatmap_dir = output_dir / "experiments_class_change_heatmaps_flipping_only"
            self.assertTrue(heatmap_dir.exists())

            fold_file = heatmap_dir / "class_change_heatmaps_flipping_only_fold_values.csv"
            self.assertTrue(fold_file.exists())
            with fold_file.open("r", newline="", encoding="utf-8") as f:
                fold_rows = list(csv.DictReader(f))
            self.assertEqual(len(fold_rows), 24)

            fold_val0 = next(
                row
                for row in fold_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["val_id"] == "0"
                and row["class_column"] == "class_0"
                and row["operation_str"] == "EDGE INSERT"
            )
            fold_val1 = next(
                row
                for row in fold_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["val_id"] == "1"
                and row["class_column"] == "class_0"
                and row["operation_str"] == "EDGE INSERT"
            )
            self.assertAlmostEqual(float(fold_val0["mean_class_change"]), 0.4)
            self.assertAlmostEqual(float(fold_val1["mean_class_change"]), 0.0)

            summary_file = heatmap_dir / "class_change_heatmaps_flipping_only_summary.csv"
            self.assertTrue(summary_file.exists())
            with summary_file.open("r", newline="", encoding="utf-8") as f:
                summary_rows = list(csv.DictReader(f))

            target_summary = next(
                row
                for row in summary_rows
                if row["dataset"] == "MUTAG"
                and row["gnn_algorithm"] == "GIN"
                and row["path_strategy"] == "Rnd"
                and row["class_column"] == "class_0"
                and row["operation_str"] == "EDGE INSERT"
            )
            self.assertEqual(target_summary["num_folds"], "2")
            self.assertAlmostEqual(float(target_summary["mean_class_change"]), 0.2)
            self.assertAlmostEqual(float(target_summary["std_class_change"]), 0.2)

            if _has_matplotlib():
                png_path = heatmap_dir / "class_change_heatmap__MUTAG__Rnd__class_0__flipping_only.png"
                no_numbers_path = heatmap_dir / "class_change_heatmap__MUTAG__Rnd__class_0__flipping_only__no_numbers.png"
                mean_std_path = heatmap_dir / "class_change_heatmap__MUTAG__Rnd__class_0__flipping_only__mean_std.png"
                legend_path = heatmap_dir / "class_change_heatmap_legend__MUTAG__Rnd__class_0__flipping_only.png"
                horizontal_legend_path = (
                    heatmap_dir / "class_change_heatmap_legend_horizontal__MUTAG__Rnd__class_0__flipping_only.png"
                )
                self.assertTrue(png_path.exists())
                self.assertTrue(no_numbers_path.exists())
                self.assertTrue(mean_std_path.exists())
                self.assertTrue(legend_path.exists())
                self.assertTrue(horizontal_legend_path.exists())

    def test_class_change_heatmaps_use_shared_scale_and_minimal_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"

            rows = []
            rows.append(_base_row(row_idx="0", val_id="0", path_idx_int="90", step_id="0", operation_str="EDGE INSERT", is_flipping="0", class_0="0.1", class_1="0.9"))
            rows.append(_base_row(row_idx="1", val_id="0", path_idx_int="90", step_id="1", operation_str="EDGE INSERT", is_flipping="1", class_0="0.3", class_1="0.7"))
            rows.append(_base_row(row_idx="2", val_id="0", path_idx_int="90", step_id="2", operation_str="EDGE INSERT", is_flipping="0", class_0="0.9", class_1="0.1"))
            rows.append(_base_row(row_idx="3", val_id="0", path_idx_int="90", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1"))

            rows.append(_base_row(row_idx="4", val_id="1", path_idx_int="91", step_id="0", operation_str="EDGE INSERT", is_flipping="0", class_0="0.2", class_1="0.8"))
            rows.append(_base_row(row_idx="5", val_id="1", path_idx_int="91", step_id="1", operation_str="EDGE INSERT", is_flipping="1", class_0="0.4", class_1="0.6"))
            rows.append(_base_row(row_idx="6", val_id="1", path_idx_int="91", step_id="2", operation_str="EDGE INSERT", is_flipping="0", class_0="1.0", class_1="0.0"))
            rows.append(_base_row(row_idx="7", val_id="1", path_idx_int="91", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1"))
            rows.append(_base_row(row_idx="8", val_id="0", path_idx_int="92", step_id="0", gnn_algorithm="GraphSAGE", operation_str="EDGE INSERT", is_flipping="0", class_0="0.1", class_1="0.9"))
            rows.append(_base_row(row_idx="9", val_id="0", path_idx_int="92", step_id="1", gnn_algorithm="GraphSAGE", operation_str="EDGE INSERT", is_flipping="1", class_0="0.2", class_1="0.8"))
            rows.append(_base_row(row_idx="10", val_id="0", path_idx_int="92", step_id="1", gnn_algorithm="GraphSAGE", is_target="1", is_path="0", operation_str="NONE", operation="-1"))

            _write_rows(data_path, rows)

            figures = []
            fake_matplotlib, fake_pyplot = _build_fake_matplotlib_modules(figures)
            fake_numpy = _build_fake_numpy_module()

            runner = CliRunner()
            with mock.patch.dict(
                sys.modules,
                {
                    "matplotlib": fake_matplotlib,
                    "matplotlib.cm": fake_matplotlib.cm,
                    "matplotlib.colors": fake_matplotlib.colors,
                    "matplotlib.pyplot": fake_pyplot,
                    "numpy": fake_numpy,
                },
            ):
                result = runner.invoke(
                    main,
                    [
                        "--data-path",
                        str(data_path),
                        "--output-dir",
                        str(output_dir),
                        "--experiments",
                        "class_change_heatmaps,class_change_heatmaps_flipping_only",
                    ],
                )

            self.assertEqual(result.exit_code, 0, msg=result.output)

            class0_all = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap__MUTAG__Rnd__class_0__all_operations.png"
            )
            class0_all_no_numbers = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap__MUTAG__Rnd__class_0__all_operations__no_numbers.png"
            )
            class0_all_mean_std = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap__MUTAG__Rnd__class_0__all_operations__mean_std.png"
            )
            class0_flipping = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap__MUTAG__Rnd__class_0__flipping_only.png"
            )
            class0_flipping_no_numbers = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap__MUTAG__Rnd__class_0__flipping_only__no_numbers.png"
            )
            class0_flipping_mean_std = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap__MUTAG__Rnd__class_0__flipping_only__mean_std.png"
            )
            class0_all_legend = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap_legend__MUTAG__Rnd__class_0__all_operations.png"
            )
            class0_all_legend_horizontal = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap_legend_horizontal__MUTAG__Rnd__class_0__all_operations.png"
            )
            class0_flipping_legend = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap_legend__MUTAG__Rnd__class_0__flipping_only.png"
            )
            class0_flipping_legend_horizontal = next(
                figure
                for figure in figures
                if figure.saved_path is not None
                and figure.saved_path.name == "class_change_heatmap_legend_horizontal__MUTAG__Rnd__class_0__flipping_only.png"
            )

            all_kwargs = class0_all.axes[0].imshow_calls[0]["kwargs"]
            flipping_kwargs = class0_flipping.axes[0].imshow_calls[0]["kwargs"]
            self.assertEqual(all_kwargs["cmap"], "viridis")
            self.assertEqual(flipping_kwargs["cmap"], "viridis")
            self.assertAlmostEqual(all_kwargs["vmin"], 0.0)
            self.assertAlmostEqual(all_kwargs["vmax"], 0.4)
            self.assertAlmostEqual(flipping_kwargs["vmin"], all_kwargs["vmin"])
            self.assertAlmostEqual(flipping_kwargs["vmax"], all_kwargs["vmax"])

            all_axis = class0_all.axes[0]
            flipping_axis = class0_flipping.axes[0]
            self.assertIsNone(all_axis.xlabel)
            self.assertIsNone(all_axis.ylabel)
            self.assertIsNone(all_axis.title)
            self.assertIsNone(flipping_axis.xlabel)
            self.assertIsNone(flipping_axis.ylabel)
            self.assertIsNone(flipping_axis.title)
            self.assertEqual(all_axis.xtick_kwargs["fontsize"], 20)
            self.assertEqual(all_axis.ytick_kwargs["fontsize"], 20)
            self.assertIn("GSAGE", all_axis.yticklabels)
            self.assertNotIn("GraphSAGE", all_axis.yticklabels)
            self.assertEqual(flipping_axis.xtick_kwargs["fontsize"], 20)
            self.assertEqual(flipping_axis.ytick_kwargs["fontsize"], 20)
            self.assertEqual(class0_all.colorbar_calls, 0)
            self.assertEqual(class0_flipping.colorbar_calls, 0)
            self.assertEqual(class0_all_legend.colorbar_calls, 1)
            self.assertEqual(class0_flipping_legend.colorbar_calls, 1)
            self.assertEqual(class0_all_legend_horizontal.colorbar_calls, 1)
            self.assertEqual(class0_flipping_legend_horizontal.colorbar_calls, 1)
            self.assertEqual(class0_all_legend.colorbars[0].label, "Mean Class Change")
            self.assertEqual(class0_flipping_legend.colorbars[0].label, "Mean Class Change")
            self.assertEqual(class0_all_legend.colorbars[0].label_kwargs["fontsize"], 22)
            self.assertEqual(class0_all_legend.colorbars[0].label_kwargs["labelpad"], 14)
            self.assertEqual(class0_all_legend.colorbars[0].ticks, [0.0, 0.2, 0.4])
            self.assertEqual(class0_all_legend.colorbars[0].ticklabels, ["0.00", "0.20", "0.40"])
            self.assertEqual(class0_all_legend.colorbars[0].ax.tick_params_kwargs["labelsize"], 20)
            self.assertEqual(class0_all_legend.add_axes_calls[0], [0.18, 0.12, 0.10, 0.76])
            self.assertEqual(class0_all_legend.figure_kwargs["figsize"], (2.6, 4.2))
            self.assertEqual(class0_all_legend.savefig_kwargs["bbox_inches"], "tight")
            self.assertEqual(class0_all_legend.savefig_kwargs["pad_inches"], 0.12)
            self.assertEqual(class0_all_legend_horizontal.colorbars[0].label, "Mean Class Change")
            self.assertEqual(class0_all_legend_horizontal.colorbars[0].label_kwargs["fontsize"], 22)
            self.assertEqual(class0_all_legend_horizontal.colorbars[0].ticks, [0.0, 0.2, 0.4])
            self.assertEqual(class0_all_legend_horizontal.colorbars[0].ticklabels, ["0.00", "0.20", "0.40"])
            self.assertEqual(class0_all_legend_horizontal.colorbars[0].ax.tick_params_kwargs["labelsize"], 20)
            self.assertEqual(class0_all_legend_horizontal.colorbar_call_kwargs[0]["orientation"], "horizontal")
            self.assertEqual(class0_all_legend_horizontal.add_axes_calls[0], [0.12, 0.48, 0.76, 0.18])
            self.assertEqual(class0_all_legend_horizontal.savefig_kwargs["bbox_inches"], "tight")
            self.assertEqual(class0_all_legend_horizontal.savefig_kwargs["pad_inches"], 0.12)
            self.assertEqual(all_axis.text_calls[0]["args"][2], "0.40")
            self.assertEqual(all_axis.text_calls[0]["kwargs"]["fontsize"], 24)
            self.assertNotIn("+-", all_axis.text_calls[0]["args"][2])
            self.assertEqual(class0_all_no_numbers.axes[0].text_calls, [])
            self.assertEqual(class0_all_mean_std.axes[0].text_calls[0]["args"][2], "0.40\n+- 0.00")
            self.assertEqual(class0_all_mean_std.axes[0].text_calls[0]["kwargs"]["fontsize"], 24)
            self.assertEqual(flipping_axis.text_calls[0]["args"][2], "0.20")
            self.assertEqual(flipping_axis.text_calls[0]["kwargs"]["fontsize"], 24)
            self.assertNotIn("+-", flipping_axis.text_calls[0]["args"][2])
            self.assertEqual(class0_flipping_no_numbers.axes[0].text_calls, [])
            self.assertEqual(class0_flipping_mean_std.axes[0].text_calls[0]["args"][2], "0.20\n+- 0.00")
            self.assertEqual(class0_flipping_mean_std.axes[0].text_calls[0]["kwargs"]["fontsize"], 24)


if __name__ == "__main__":
    unittest.main()
