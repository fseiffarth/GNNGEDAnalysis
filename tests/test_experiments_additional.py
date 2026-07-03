import csv
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from experiments_additional import ALL_EXPERIMENTS, DEFAULT_EXPERIMENTS, main


def _has_matplotlib():
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:
        return False


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
        "class_0": "0.1",
        "class_1": "0.9",
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


class AdditionalExperimentsCliTests(unittest.TestCase):
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
                    "unknown_experiment",
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
                _base_row(row_idx="0", path_idx_int="11", step_id="0", operation_str="EDGE INSERT", is_flipping="1"),
                _base_row(row_idx="1", path_idx_int="11", step_id="1", operation_str="EDGE DELETE", is_flipping="0"),
                _base_row(row_idx="2", path_idx_int="11", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, ["--data-path", str(data_path), "--output-dir", str(output_dir), "--min-transition-count", "1"])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            self.assertTrue((output_dir / "run_filters.txt").exists())
            self.assertTrue((output_dir / "experiments_flip_position_sensitivity").exists())
            self.assertTrue((output_dir / "experiments_flip_streaks_and_recovery").exists())
            self.assertTrue((output_dir / "experiments_margin_to_flip_analysis").exists())
            self.assertTrue((output_dir / "experiments_operation_transition_instability").exists())
            self.assertTrue((output_dir / "experiments_path_consistency_score").exists())

    def test_flip_position_sensitivity_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"
            rows = [
                _base_row(row_idx="0", path_idx_int="20", step_id="0", operation_str="EDGE INSERT", is_flipping="1"),
                _base_row(row_idx="1", path_idx_int="20", step_id="1", operation_str="EDGE DELETE", is_flipping="0"),
                _base_row(row_idx="2", path_idx_int="20", step_id="2", operation_str="NODE INSERT", is_flipping="1"),
                _base_row(row_idx="3", path_idx_int="20", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, [
                "--data-path", str(data_path), "--output-dir", str(output_dir), "--experiments", "flip_position_sensitivity"
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            fold_csv = output_dir / "experiments_flip_position_sensitivity" / "flip_position_sensitivity_fold_counts.csv"
            self.assertTrue(fold_csv.exists())
            with fold_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            target = next(
                row for row in rows
                if row["position_bin"] == "early" and row["operation_str"] == "EDGE INSERT" and row["split"] == "validation"
            )
            self.assertEqual(target["decision_change_count"], "1")
            self.assertEqual(target["operation_count"], "1")
            self.assertAlmostEqual(float(target["flip_rate"]), 1.0)

            if _has_matplotlib():
                plot_path = output_dir / "experiments_flip_position_sensitivity" / "flip_position_sensitivity_summary__MUTAG__GIN__Rnd__validation.png"
                self.assertTrue(plot_path.exists())

    def test_flip_streaks_and_recovery_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"
            rows = [
                _base_row(row_idx="0", path_idx_int="30", step_id="0", operation_str="EDGE INSERT", is_flipping="0", predicted_label="1"),
                _base_row(row_idx="1", path_idx_int="30", step_id="1", operation_str="EDGE DELETE", is_flipping="1", predicted_label="0"),
                _base_row(row_idx="2", path_idx_int="30", step_id="2", operation_str="NODE INSERT", is_flipping="1", predicted_label="0"),
                _base_row(row_idx="3", path_idx_int="30", step_id="3", operation_str="NODE DELETE", is_flipping="0", predicted_label="1"),
                _base_row(row_idx="4", path_idx_int="30", step_id="3", is_target="1", is_path="0", operation_str="NONE", operation="-1", predicted_label="1"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, [
                "--data-path", str(data_path), "--output-dir", str(output_dir), "--experiments", "flip_streaks_and_recovery"
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            path_csv = output_dir / "experiments_flip_streaks_and_recovery" / "flip_streaks_and_recovery_path_metrics.csv"
            with path_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            target = next(row for row in rows if row["path_idx_int"] == "30" and row["split"] == "validation")
            self.assertEqual(target["num_flips"], "2")
            self.assertEqual(target["max_flip_streak"], "2")
            self.assertEqual(target["recovered_to_initial_label"], "1")
            self.assertEqual(target["steps_to_recovery"], "2")

            if _has_matplotlib():
                dist_plot = output_dir / "experiments_flip_streaks_and_recovery" / "flip_streak_distribution__MUTAG__GIN__Rnd__validation.png"
                recovery_plot = output_dir / "experiments_flip_streaks_and_recovery" / "flip_streak_recovery__MUTAG__GIN__Rnd__validation.png"
                self.assertTrue(dist_plot.exists())
                self.assertTrue(recovery_plot.exists())

    def test_margin_to_flip_analysis_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"
            rows = [
                _base_row(row_idx="0", path_idx_int="40", step_id="0", operation_str="EDGE INSERT", is_flipping="1", class_0="0.49", class_1="0.51"),
                _base_row(row_idx="1", path_idx_int="40", step_id="1", operation_str="EDGE DELETE", is_flipping="0", class_0="0.1", class_1="0.9"),
                _base_row(row_idx="2", path_idx_int="40", step_id="1", is_target="1", is_path="0", operation_str="NONE", operation="-1"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, [
                "--data-path", str(data_path), "--output-dir", str(output_dir),
                "--experiments", "margin_to_flip_analysis",
                "--margin-bins", "0,0.05,0.5,1.0",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            fold_csv = output_dir / "experiments_margin_to_flip_analysis" / "margin_to_flip_calibration_fold.csv"
            with fold_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            target = next(
                row for row in rows
                if row["operation_str"] == "EDGE INSERT" and row["margin_bin"] == "[0,0.05)" and row["split"] == "validation"
            )
            self.assertEqual(target["step_count"], "1")
            self.assertEqual(target["flip_count"], "1")
            self.assertAlmostEqual(float(target["flip_rate"]), 1.0)

            if _has_matplotlib():
                calibration_plot = output_dir / "experiments_margin_to_flip_analysis" / "margin_to_flip_calibration__MUTAG__GIN__Rnd__validation.png"
                distribution_plot = output_dir / "experiments_margin_to_flip_analysis" / "margin_to_flip_distribution__MUTAG__GIN__Rnd__validation.png"
                self.assertTrue(calibration_plot.exists())
                self.assertTrue(distribution_plot.exists())

    def test_operation_transition_instability_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"
            rows = [
                _base_row(row_idx="0", path_idx_int="50", step_id="0", operation_str="EDGE INSERT", is_flipping="0"),
                _base_row(row_idx="1", path_idx_int="50", step_id="1", operation_str="EDGE DELETE", is_flipping="1"),
                _base_row(row_idx="2", path_idx_int="50", step_id="2", operation_str="NODE INSERT", is_flipping="0"),
                _base_row(row_idx="3", path_idx_int="50", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1"),
                _base_row(row_idx="4", path_idx_int="51", step_id="0", operation_str="EDGE INSERT", is_flipping="0"),
                _base_row(row_idx="5", path_idx_int="51", step_id="1", operation_str="EDGE DELETE", is_flipping="1"),
                _base_row(row_idx="6", path_idx_int="51", step_id="2", operation_str="NODE INSERT", is_flipping="0"),
                _base_row(row_idx="7", path_idx_int="51", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, [
                "--data-path", str(data_path), "--output-dir", str(output_dir),
                "--experiments", "operation_transition_instability",
                "--min-transition-count", "1",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            bigram_csv = output_dir / "experiments_operation_transition_instability" / "operation_transition_bigrams_fold.csv"
            with bigram_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            target = next(
                row for row in rows
                if row["operation_prev"] == "EDGE INSERT" and row["operation_curr"] == "EDGE DELETE" and row["split"] == "validation"
            )
            self.assertEqual(target["sequence_count"], "2")
            self.assertEqual(target["sequence_flip_count"], "2")
            self.assertAlmostEqual(float(target["sequence_flip_rate"]), 1.0)

            if _has_matplotlib():
                rate_plot = output_dir / "experiments_operation_transition_instability" / "operation_transition_bigrams_rate__MUTAG__GIN__Rnd__validation.png"
                enrich_plot = output_dir / "experiments_operation_transition_instability" / "operation_transition_bigrams_enrichment__MUTAG__GIN__Rnd__validation.png"
                self.assertTrue(rate_plot.exists())
                self.assertTrue(enrich_plot.exists())

    def test_path_consistency_score_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"
            rows = [
                _base_row(row_idx="0", path_idx_int="60", step_id="0", operation_str="EDGE INSERT", predicted_label="1", class_0="0.1", class_1="0.9"),
                _base_row(row_idx="1", path_idx_int="60", step_id="1", operation_str="EDGE DELETE", predicted_label="0", class_0="0.8", class_1="0.2"),
                _base_row(row_idx="2", path_idx_int="60", step_id="2", operation_str="NODE INSERT", predicted_label="1", class_0="0.2", class_1="0.8"),
                _base_row(row_idx="3", path_idx_int="60", step_id="2", is_target="1", is_path="0", operation_str="NONE", operation="-1", predicted_label="1"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(main, [
                "--data-path", str(data_path), "--output-dir", str(output_dir), "--experiments", "path_consistency_score"
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            metrics_csv = output_dir / "experiments_path_consistency_score" / "path_consistency_path_metrics.csv"
            with metrics_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            target = next(row for row in rows if row["path_idx_int"] == "60" and row["split"] == "validation")
            self.assertEqual(target["label_switch_count"], "2")
            self.assertEqual(target["endpoint_disagreement"], "0")
            self.assertGreaterEqual(float(target["consistency_score"]), 0.0)
            self.assertLessEqual(float(target["consistency_score"]), 1.0)

            ranking_csv = output_dir / "experiments_path_consistency_score" / "path_consistency_configuration_ranking.csv"
            self.assertTrue(ranking_csv.exists())

            if _has_matplotlib():
                score_plot = output_dir / "experiments_path_consistency_score" / "path_consistency_scores__MUTAG__GIN__Rnd__validation.png"
                ranking_plot = output_dir / "experiments_path_consistency_score" / "path_consistency_ranking.png"
                self.assertTrue(score_plot.exists())
                self.assertTrue(ranking_plot.exists())

    def test_margin_bins_validation_rejects_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "all_results.csv"
            output_dir = tmp_path / "out"
            rows = [
                _base_row(row_idx="0", path_idx_int="70", step_id="0", operation_str="EDGE INSERT"),
                _base_row(row_idx="1", path_idx_int="70", step_id="0", is_target="1", is_path="0", operation_str="NONE", operation="-1"),
            ]
            _write_rows(data_path, rows)

            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--data-path", str(data_path), "--output-dir", str(output_dir),
                    "--experiments", "margin_to_flip_analysis",
                    "--margin-bins", "0.2,0.1",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("margin-bins must be strictly increasing", result.output)


if __name__ == "__main__":
    unittest.main()
