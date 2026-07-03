import importlib
import sys
import types
from pathlib import Path


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


def test_prepare_dynamic_assets_generates_missing_ogb_splits(monkeypatch, tmp_path):
    train_models = _import_train_models_with_stubbed_framework(monkeypatch)

    configs_dir = tmp_path / "configs"
    config_dir = configs_dir / "ogbg-molhiv"
    config_dir.mkdir(parents=True)
    (config_dir / "main_config.yml").write_text(
        "\n".join(
            [
                "datasets:",
                "  - {",
                '      name: "ogbg-molhiv",',
                '      source: "OGB_GraphProp",',
                '      task: "graph_classification",',
                "      paths: {",
                '        data: "data/OGB/",',
                '        results: "results/GIN/",',
                '        models: "configs/ogbg-molhiv/models_GIN.yml",',
                '        hyperparameters: "configs/ogbg-molhiv/parameters.yml",',
                '        splits: "splits/ogbg-molhiv_splits.json",',
                "      }",
                "    }",
            ]
        ),
        encoding="utf-8",
    )

    calls = []

    def _fake_ensure(dataset_name, splits_path, dataset_root):
        calls.append((dataset_name, splits_path, dataset_root))

    monkeypatch.setattr(train_models, "CONFIGS_ROOT", configs_dir)
    monkeypatch.setattr(train_models, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(train_models, "ensure_ogb_graphprop_split_file", _fake_ensure)

    train_models._prepare_dynamic_assets("ogbg-molhiv", configs_root=configs_dir)

    assert calls == [
        (
            "ogbg-molhiv",
            tmp_path / "splits" / "ogbg-molhiv_splits.json",
            tmp_path / "data" / "OGB",
        )
    ]
