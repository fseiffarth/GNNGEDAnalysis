import importlib
import sys
import types
from pathlib import Path

from click.testing import CliRunner


def _import_train_models_with_stubbed_framework(monkeypatch):
    stub_module = types.ModuleType("simplegnn.framework.core")

    class _StubFrameworkMain:
        def __init__(self, _config_path):
            pass

        def preprocessing(self, num_threads):
            return None

        def run_configurations(self, num_threads):
            return None

        def evaluate_results(self, evaluate_validation_only=True):
            return None

    stub_module.FrameworkMain = _StubFrameworkMain
    monkeypatch.setitem(sys.modules, "simplegnn.framework.core", stub_module)

    if "train_models" in sys.modules:
        del sys.modules["train_models"]

    return importlib.import_module("train_models")


def test_multi_db_cli_dispatches_in_order(monkeypatch):
    train_models = _import_train_models_with_stubbed_framework(monkeypatch)

    calls = []

    def _fake_train_ged(num_threads=-1, dbs=("MUTAG",)):
        calls.append((num_threads, list(dbs)))

    monkeypatch.setattr(train_models, "train_ged", _fake_train_ged)

    runner = CliRunner()
    result = runner.invoke(
        train_models.main,
        ["--num_threads", "4", "--db", "MUTAG", "--db", "NCI1"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(4, ["MUTAG", "NCI1"])]


def test_train_ged_fails_fast_for_missing_db(monkeypatch):
    train_models = _import_train_models_with_stubbed_framework(monkeypatch)

    class _ShouldNotBeCalled:
        def __init__(self, _config_path):
            raise AssertionError("FrameworkMain should not be constructed for missing db")

    monkeypatch.setattr(train_models, "_import_framework_main", lambda: _ShouldNotBeCalled)

    runner = CliRunner()
    with runner.isolated_filesystem():
        root = Path(".")
        configs_dir = root / "configs"
        (configs_dir / "MUTAG").mkdir(parents=True)
        (configs_dir / "MUTAG" / "main_config.yml").write_text("datasets: []\n", encoding="utf-8")
        monkeypatch.setattr(train_models, "CONFIGS_ROOT", configs_dir)

        result = runner.invoke(
            train_models.main,
            ["--db", "MUTAG", "--db", "MISSING_DB"],
        )

    assert result.exit_code != 0
    assert "Missing config(s) for db: MISSING_DB" in result.output
    assert "Available db options: MUTAG" in result.output


def test_db_help_mentions_repeatable_usage(monkeypatch):
    train_models = _import_train_models_with_stubbed_framework(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(train_models.main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--all" in result.output
    assert "Repeat --db to train multiple datasets" in result.output
    assert "Available options from configs:" in result.output


def test_all_flag_dispatches_all_configs_in_sorted_order(monkeypatch):
    train_models = _import_train_models_with_stubbed_framework(monkeypatch)

    calls = []

    def _fake_train_ged(num_threads=-1, dbs=("MUTAG",)):
        calls.append((num_threads, list(dbs)))

    monkeypatch.setattr(train_models, "train_ged", _fake_train_ged)

    runner = CliRunner()
    with runner.isolated_filesystem():
        root = Path(".")
        configs_dir = root / "configs"
        for db in ["PTC_FR", "MUTAG", "NCI1"]:
            (configs_dir / db).mkdir(parents=True)
            (configs_dir / db / "main_config.yml").write_text("datasets: []\n", encoding="utf-8")
        monkeypatch.setattr(train_models, "CONFIGS_ROOT", configs_dir)

        result = runner.invoke(train_models.main, ["--all"])

    assert result.exit_code == 0, result.output
    assert calls == [(-1, ["MUTAG", "NCI1", "PTC_FR"])]


def test_all_flag_conflicts_with_explicit_db(monkeypatch):
    train_models = _import_train_models_with_stubbed_framework(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(train_models.main, ["--all", "--db", "MUTAG"])

    assert result.exit_code != 0
    assert "Options '--all' and '--db' are mutually exclusive." in result.output


def test_all_flag_fails_when_no_configs_available(monkeypatch):
    train_models = _import_train_models_with_stubbed_framework(monkeypatch)

    runner = CliRunner()
    with runner.isolated_filesystem():
        root = Path(".")
        configs_dir = root / "configs"
        configs_dir.mkdir(parents=True)
        monkeypatch.setattr(train_models, "CONFIGS_ROOT", configs_dir)

        result = runner.invoke(train_models.main, ["--all"])

    assert result.exit_code != 0
    assert "No dataset configs found for '--all'" in result.output
