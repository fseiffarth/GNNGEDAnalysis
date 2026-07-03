import json
from collections import Counter

import pytest

from utils.stratified_splits import (
    build_rotating_splits,
    build_stratified_folds,
    load_tu_graph_labels,
    validate_splits,
    write_rotating_splits,
)


BALANCED_LABELS = [0] * 50 + [1] * 50


def test_build_stratified_folds_balances_classes():
    folds = build_stratified_folds(BALANCED_LABELS, folds=10, seed=42)

    assert len(folds) == 10
    for fold in folds:
        assert len(fold) == 10
        labels_in_fold = Counter(BALANCED_LABELS[index] for index in fold)
        assert labels_in_fold == {0: 5, 1: 5}


def test_build_stratified_folds_rejects_small_classes():
    with pytest.raises(ValueError, match="need at least 10"):
        build_stratified_folds([0] * 5 + [1] * 50, folds=10)


def test_build_rotating_splits_gives_10_10_80():
    splits = build_rotating_splits(BALANCED_LABELS, folds=10, seed=42)

    assert len(splits) == 10
    for fold in splits:
        train = fold["model_selection"][0]["train"]
        validation = fold["model_selection"][0]["validation"]
        test = fold["test"]
        assert len(train) == 80
        assert len(validation) == 10
        assert len(test) == 10
        assert not set(validation) & set(test)
        assert not set(train) & (set(validation) | set(test))


def test_build_splits_test_mode_none_gives_10_90():
    splits = build_rotating_splits(BALANCED_LABELS, folds=10, seed=42, test_mode="none")

    assert len(splits) == 10
    validation_seen = []
    for fold in splits:
        train = fold["model_selection"][0]["train"]
        validation = fold["model_selection"][0]["validation"]
        assert fold["test"] == []
        assert len(train) == 90
        assert len(validation) == 10
        assert not set(train) & set(validation)
        labels_in_validation = Counter(BALANCED_LABELS[index] for index in validation)
        assert labels_in_validation == {0: 5, 1: 5}
        validation_seen.extend(validation)
    assert sorted(validation_seen) == list(range(100))


def test_validate_splits_accepts_test_mode_none():
    splits = build_rotating_splits(BALANCED_LABELS, folds=10, seed=42, test_mode="none")
    validate_splits(splits, BALANCED_LABELS, folds=10)


def test_build_splits_rejects_unknown_test_mode():
    with pytest.raises(ValueError, match="Unknown test_mode"):
        build_rotating_splits(BALANCED_LABELS, folds=10, test_mode="fixed")


def test_validate_splits_accepts_rotating_splits():
    splits = build_rotating_splits(BALANCED_LABELS, folds=10, seed=42)
    validate_splits(splits, BALANCED_LABELS, folds=10)


def test_validate_splits_rejects_duplicate_validation_fold():
    splits = build_rotating_splits(BALANCED_LABELS, folds=10, seed=42)
    splits[1]["model_selection"][0]["validation"] = splits[0]["model_selection"][0][
        "validation"
    ]

    with pytest.raises(AssertionError):
        validate_splits(splits, BALANCED_LABELS, folds=10)


def test_write_rotating_splits_persists_json(tmp_path):
    output_path = tmp_path / "splits" / "TRIANGLE_SQUARE_splits.json"

    write_rotating_splits(BALANCED_LABELS, output_path, folds=10, seed=42)

    assert output_path.exists()
    splits = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(splits) == 10
    assert set(splits[0]) == {"test", "model_selection"}


def test_load_tu_graph_labels(tmp_path):
    labels_file = tmp_path / "DB_graph_labels.txt"
    labels_file.write_text("0\n1\n1\n0\n", encoding="utf-8")

    assert load_tu_graph_labels(labels_file) == [0, 1, 1, 0]
