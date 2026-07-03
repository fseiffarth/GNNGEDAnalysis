import json
from pathlib import Path

import pytest

from utils.ogb_splits import (
    build_simplegnn_splits,
    ensure_ogb_graphprop_split_file,
    write_simplegnn_splits,
)


def test_build_simplegnn_splits_uses_expected_shape():
    splits = build_simplegnn_splits([0, 1], [2], [3, 4])

    assert splits == [
        {
            "test": [3, 4],
            "model_selection": [{"train": [0, 1], "validation": [2]}],
        }
    ]


def test_write_simplegnn_splits_persists_json(tmp_path):
    output_path = tmp_path / "splits" / "ogbg-molhiv_splits.json"

    write_simplegnn_splits(output_path, [0, 1], [2], [3])

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {
            "test": [3],
            "model_selection": [{"train": [0, 1], "validation": [2]}],
        }
    ]


def test_ensure_ogb_graphprop_split_file_requires_ogb_when_missing(monkeypatch, tmp_path):
    def _fake_import(*_args, **_kwargs):
        raise ImportError("missing ogb")

    monkeypatch.setattr("builtins.__import__", _fake_import)

    with pytest.raises(ImportError, match="OGB is required to generate official split files"):
        ensure_ogb_graphprop_split_file(
            dataset_name="ogbg-molhiv",
            splits_path=tmp_path / "ogbg-molhiv_splits.json",
            dataset_root=tmp_path / "data",
        )
