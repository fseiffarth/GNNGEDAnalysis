"""Generate stratified k-fold splits with rotating validation/test folds.

Builds split files in the simplegnn format (see ``utils/ogb_splits.py``):
a JSON list with one entry per fold, each entry holding a ``test`` id list and
a ``model_selection`` list with ``train``/``validation`` id lists.

The labels are stratified into ``folds`` equally sized folds F0..F(k-1).
Fold ``i`` uses F_i as validation, F_((i+1) % k) as test, and the remaining
graphs as train, so for 10 folds every graph appears exactly once as
validation and once as test while train/validation/test is 80/10/10 with
class parity preserved in every split.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import click


def load_tu_graph_labels(labels_path):
    labels_path = Path(labels_path)
    with labels_path.open("r", encoding="utf-8") as file_obj:
        return [int(float(line.strip())) for line in file_obj if line.strip()]


def build_stratified_folds(labels, folds=10, seed=42):
    """Split indices into ``folds`` stratified folds of (near-)equal size."""
    if folds < 3:
        raise ValueError("Rotating validation/test folds require at least 3 folds.")

    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        if count < folds:
            raise ValueError(
                f"Class {label} has only {count} samples; "
                f"need at least {folds} for {folds}-fold stratification."
            )

    rng = random.Random(seed)
    indices_per_class = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_class[label].append(index)

    fold_indices = [[] for _ in range(folds)]
    for label in sorted(indices_per_class):
        class_indices = list(indices_per_class[label])
        rng.shuffle(class_indices)
        for position, index in enumerate(class_indices):
            fold_indices[position % folds].append(index)

    return [sorted(fold) for fold in fold_indices]


def build_rotating_splits(labels, folds=10, seed=42, test_mode="rotating"):
    """Build simplegnn splits from stratified folds.

    test_mode 'rotating': fold i uses F_i as validation and F_(i+1) as test.
    test_mode 'none': fold i uses F_i as validation and an empty test set
    (standard TU-style k-fold cross-validation).
    """
    if test_mode not in ("rotating", "none"):
        raise ValueError(f"Unknown test_mode: {test_mode!r}")

    fold_indices = build_stratified_folds(labels, folds=folds, seed=seed)

    splits = []
    for fold_id in range(folds):
        validation_ids = set(fold_indices[fold_id])
        if test_mode == "rotating":
            test_ids = set(fold_indices[(fold_id + 1) % folds])
        else:
            test_ids = set()
        train_ids = [
            index
            for index in range(len(labels))
            if index not in validation_ids and index not in test_ids
        ]
        splits.append(
            {
                "test": sorted(test_ids),
                "model_selection": [
                    {
                        "train": sorted(train_ids),
                        "validation": sorted(validation_ids),
                    }
                ],
            }
        )
    return splits


def validate_splits(splits, labels, folds):
    """Sanity-check disjointness, coverage, and per-split class parity."""
    num_samples = len(labels)
    has_test = any(fold["test"] for fold in splits)
    validation_seen = Counter()
    test_seen = Counter()
    for fold in splits:
        train = fold["model_selection"][0]["train"]
        validation = fold["model_selection"][0]["validation"]
        test = fold["test"]

        all_ids = train + validation + test
        if sorted(all_ids) != list(range(num_samples)):
            raise AssertionError("Train/validation/test do not partition the dataset.")

        for split_name, split_ids in (("validation", validation), ("test", test)):
            if not split_ids:
                continue
            split_labels = Counter(labels[index] for index in split_ids)
            counts = sorted(split_labels.values())
            if max(counts) - min(counts) > 1:
                raise AssertionError(
                    f"Class parity violated in {split_name} split: {dict(split_labels)}"
                )

        validation_seen.update(validation)
        test_seen.update(test)

    if set(validation_seen) != set(range(num_samples)) or set(validation_seen.values()) != {1}:
        raise AssertionError("Each sample must appear exactly once as validation.")
    if has_test and (
        set(test_seen) != set(range(num_samples)) or set(test_seen.values()) != {1}
    ):
        raise AssertionError("Each sample must appear exactly once as test.")


def write_rotating_splits(labels, splits_path, folds=10, seed=42, test_mode="rotating"):
    splits = build_rotating_splits(labels, folds=folds, seed=seed, test_mode=test_mode)
    validate_splits(splits, labels, folds)
    splits_path = Path(splits_path)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as file_obj:
        json.dump(splits, file_obj, separators=(",", ":"))
    return splits_path


@click.command()
@click.option(
    "--labels-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="TU-format graph labels file (one integer label per graph).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output splits JSON path, e.g. splits/<DB>_splits.json.",
)
@click.option("--folds", default=10, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--test-mode",
    type=click.Choice(["rotating", "none"]),
    default="rotating",
    show_default=True,
    help="'rotating': fold i+1 is the test fold; 'none': empty test sets (TU-style k-fold).",
)
def main(labels_file, output, folds, seed, test_mode):
    labels = load_tu_graph_labels(labels_file)
    splits_path = write_rotating_splits(
        labels, output, folds=folds, seed=seed, test_mode=test_mode
    )
    fold0 = json.loads(Path(splits_path).read_text())[0]
    print(f"Wrote {folds}-fold splits for {len(labels)} graphs to {splits_path}")
    print(
        f"Fold 0 sizes: train={len(fold0['model_selection'][0]['train'])}, "
        f"validation={len(fold0['model_selection'][0]['validation'])}, "
        f"test={len(fold0['test'])}"
    )


if __name__ == "__main__":
    main()
