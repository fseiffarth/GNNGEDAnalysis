import json
from pathlib import Path


def build_simplegnn_splits(train_idx, valid_idx, test_idx):
    return [
        {
            "test": [int(index) for index in test_idx],
            "model_selection": [
                {
                    "train": [int(index) for index in train_idx],
                    "validation": [int(index) for index in valid_idx],
                }
            ],
        }
    ]


def write_simplegnn_splits(splits_path, train_idx, valid_idx, test_idx):
    splits_path = Path(splits_path)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits = build_simplegnn_splits(train_idx, valid_idx, test_idx)
    with splits_path.open("w", encoding="utf-8") as file_obj:
        json.dump(splits, file_obj, separators=(",", ":"))
    return splits_path


def ensure_ogb_graphprop_split_file(dataset_name, splits_path, dataset_root):
    splits_path = Path(splits_path)
    if splits_path.exists():
        return splits_path

    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError as exc:
        raise ImportError(
            "OGB is required to generate official split files for "
            f"{dataset_name}. Install the 'ogb' package and rerun training."
        ) from exc

    dataset = PygGraphPropPredDataset(name=dataset_name, root=str(dataset_root))
    split_idx = dataset.get_idx_split()
    return write_simplegnn_splits(
        splits_path=splits_path,
        train_idx=split_idx["train"].tolist(),
        valid_idx=split_idx["valid"].tolist(),
        test_idx=split_idx["test"].tolist(),
    )
