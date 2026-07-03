import tempfile
import unittest
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from statistical_validation import RESULT_COLUMNS, main


def _write_relative_fold_counts(root: Path) -> None:
    """Synthetic per-fold relative flip counts with a clear operation ordering."""
    d = root / "Experiments" / "experiments_flips_per_operation_relative"
    d.mkdir(parents=True, exist_ok=True)
    ops = {
        "EDGE INSERT": 0.30,
        "EDGE DELETE": 0.25,
        "NODE INSERT": 0.10,
        "NODE DELETE": 0.08,
        "EDGE RELABEL": 0.05,
        "NODE RELABEL": 0.04,
    }
    rows = []
    for gnn in ("GIN", "GAT", "GCN"):
        for fold in range(10):
            for op, base in ops.items():
                rate = base + 0.001 * fold + (0.01 if gnn == "GIN" else 0.0)
                rows.append({
                    "dataset": "TOY",
                    "gnn_algorithm": gnn,
                    "path_strategy": "Rnd",
                    "split": "all",
                    "val_id": fold,
                    "operation_str": op,
                    "decision_change_count": int(rate * 1000),
                    "operation_count": 1000,
                    "relative_decision_change_count": rate,
                })
    pd.DataFrame(rows).to_csv(d / "flips_per_operation_relative_fold_counts.csv", index=False)


class TestStatisticalValidationCli(unittest.TestCase):
    def test_fold_tier_emits_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            _write_relative_fold_counts(root)
            out = Path(tmp) / "out"

            runner = CliRunner()
            result = runner.invoke(main, [
                "--results-dir", str(root),
                "--output-dir", str(out),
                "--dataset", "TOY",
                "--tier", "fold",
                "--split", "all",
                "--n-boot", "200",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)

            op_csv = out / "operation_effects_tests.csv"
            self.assertTrue(op_csv.exists())
            df = pd.read_csv(op_csv)
            self.assertEqual(list(df.columns), RESULT_COLUMNS)

            # Friedman omnibus present and significant for the strong ordering.
            omni = df[(df.test == "friedman") & (df.gnn_algorithm == "GIN")]
            self.assertFalse(omni.empty)
            self.assertTrue(bool(omni.iloc[0]["significant"]))
            # Bootstrap CIs are emitted for each of the six operations per cell.
            boots = df[(df.test == "bootstrap") & (df.gnn_algorithm == "GIN")]
            self.assertEqual(len(boots), 6)
            self.assertTrue(boots["ci_low"].notna().all())

            # GNN-difference omnibus (GIN flips more) should also be produced.
            gnn_df = pd.read_csv(out / "gnn_difference_tests.csv")
            self.assertEqual(list(gnn_df.columns), RESULT_COLUMNS)
            self.assertTrue((gnn_df.test == "friedman").any())

            self.assertTrue((out / "report.md").exists())

    def test_missing_dataset_is_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            _write_relative_fold_counts(root)
            out = Path(tmp) / "out"
            runner = CliRunner()
            result = runner.invoke(main, [
                "--results-dir", str(root),
                "--output-dir", str(out),
                "--dataset", "DOES_NOT_EXIST",
                "--tier", "fold",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            df = pd.read_csv(out / "operation_effects_tests.csv")
            self.assertEqual(len(df), 0)
            self.assertEqual(list(df.columns), RESULT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
