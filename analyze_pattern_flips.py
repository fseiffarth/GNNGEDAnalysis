"""Analyze decision flips along TRIANGLE_SQUARE edit paths by pattern phase.

Reads the pattern-enriched results table (produced by
utils/pattern_tracking.py) and counts, per path, how many decision flips
occur in five phases of the edit path:

1. before:           steps before the source's planted pattern is destroyed
2. destroying:       the single step that destroys the source pattern
3. after_destroying: steps between the destroying and the restoring operation
4. restoring:        the single step that creates the target pattern
5. after_restoring:  steps from the restoring operation to the end of the path

Phase boundaries per path use the destroying step d = src_destroyed_step (the
first step at which the source pattern is no longer intact) and the restoring
step c = tgt_created_step (the first step of the trailing run in which the
target pattern stays complete). The single-step destroying/restoring phases
take priority, then before = steps < min(d, c), after_destroying = remaining
steps < c, and after_restoring everything else. This always partitions the
path, also for the degenerate cases (pattern never destroyed, so the
destroying/after_destroying phases are empty, or target pattern created
before the source pattern is destroyed).

Same-class paths (triangle->triangle, square->square) get the same treatment
- under the strict pendant-intactness definition their patterns can also be
destroyed and re-created - and are kept as a separate grouping dimension.

Outputs (in results/pattern_flip_analysis/):
- flip_phase_per_path.csv: per-path flip counts and segment lengths
- flip_phase_summary.csv:  aggregated over paths and validation folds
- flip_phases_<strategy>_<class>.png: one single plot per strategy/class pair,
                           mean flips per phase per GNN
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

SEGMENTS = [
    "before",
    "destroying",
    "after_destroying",
    "restoring",
    "after_restoring",
]

SEGMENT_LABELS = {
    "before": "before",
    "destroying": "destroying",
    "after_destroying": "no pattern",
    "restoring": "restoring",
    "after_restoring": "after",
}

NEVER_DESTROYED = 2**31


def compute_per_path(input_csv: Path, dataset: str) -> pl.DataFrame:
    lf = (
        pl.scan_csv(str(input_csv))
        .filter(pl.col("dataset") == dataset)
        .with_columns(
            pl.col("step_id").cast(pl.Float64).cast(pl.Int64),
            pl.col("source_id").cast(pl.Float64).cast(pl.Int64),
            pl.col("target_id").cast(pl.Float64).cast(pl.Int64),
        )
        .with_columns(
            pl.col("src_destroyed_step").alias("_d"),
            pl.col("tgt_created_step").alias("_c"),
            pl.min_horizontal(
                pl.col("src_destroyed_step").fill_null(NEVER_DESTROYED),
                pl.col("tgt_created_step"),
            ).alias("_b1"),
        )
        .with_columns(
            pl.when(pl.col("_d").is_not_null() & (pl.col("step_id") == pl.col("_d")))
            .then(pl.lit("destroying"))
            .when(pl.col("step_id") == pl.col("_c"))
            .then(pl.lit("restoring"))
            .when(pl.col("step_id") < pl.col("_b1"))
            .then(pl.lit("before"))
            .when(pl.col("step_id") < pl.col("_c"))
            .then(pl.lit("after_destroying"))
            .otherwise(pl.lit("after_restoring"))
            .alias("segment")
        )
    )

    path_keys = [
        "gnn_algorithm",
        "path_strategy",
        "val_id",
        "config_id",
        "source_id",
        "target_id",
    ]
    aggregations = [
        pl.col("same_class_path").first(),
        pl.col("src_destroyed_step").first(),
        pl.col("tgt_created_step").first(),
        pl.col("is_train_path").first(),
        pl.col("is_validation_path").first(),
        pl.col("is_flipping").sum().alias("total_flips"),
        (pl.col("step_id") >= 0).sum().alias("path_length"),
    ]
    for segment in SEGMENTS:
        in_segment = pl.col("segment") == segment
        aggregations.append(
            (pl.col("is_flipping") * in_segment).sum().alias(f"flips_{segment}")
        )
        aggregations.append(
            (in_segment & (pl.col("step_id") >= 0)).sum().alias(f"steps_{segment}")
        )

    per_path = lf.group_by(path_keys).agg(aggregations).collect(engine="streaming")

    flip_sum = sum(pl.col(f"flips_{segment}") for segment in SEGMENTS)
    mismatched = per_path.filter(flip_sum != pl.col("total_flips")).height
    if mismatched > 0:
        raise ValueError(
            f"Segment flip counts do not sum to total flips for {mismatched} paths."
        )
    step_sum = sum(pl.col(f"steps_{segment}") for segment in SEGMENTS)
    mismatched = per_path.filter(step_sum != pl.col("path_length")).height
    if mismatched > 0:
        raise ValueError(
            f"Segment lengths do not sum to path length for {mismatched} paths."
        )

    return per_path.with_columns(
        pl.when(pl.col("is_train_path") == 1)
        .then(pl.lit("train"))
        .when(pl.col("is_validation_path") == 1)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("mixed"))
        .alias("path_split"),
        (
            pl.col("src_destroyed_step").is_not_null()
            & (pl.col("tgt_created_step") < pl.col("src_destroyed_step"))
        )
        .cast(pl.Int8)
        .alias("created_before_destroyed"),
        pl.col("src_destroyed_step").is_null().cast(pl.Int8).alias("never_destroyed"),
    )


def summarize(per_path: pl.DataFrame) -> pl.DataFrame:
    def aggregate(frame: pl.DataFrame, keys: list) -> pl.DataFrame:
        aggregations = [
            pl.len().alias("n_paths"),
            pl.col("path_length").mean().alias("mean_path_length"),
            pl.col("total_flips").mean().alias("mean_total_flips"),
            pl.col("never_destroyed").mean().alias("frac_never_destroyed"),
            pl.col("created_before_destroyed").mean().alias("frac_created_before_destroyed"),
        ]
        for segment in SEGMENTS:
            aggregations.extend(
                [
                    pl.col(f"flips_{segment}").mean().alias(f"mean_flips_{segment}"),
                    pl.col(f"flips_{segment}").std().alias(f"std_flips_{segment}"),
                    pl.col(f"flips_{segment}").sum().alias(f"sum_flips_{segment}"),
                    pl.col(f"steps_{segment}").mean().alias(f"mean_steps_{segment}"),
                    (
                        pl.col(f"flips_{segment}").sum()
                        / pl.col(f"steps_{segment}").sum()
                    ).alias(f"flip_rate_{segment}"),
                ]
            )
        return frame.group_by(keys).agg(aggregations).sort(keys)

    base_keys = ["gnn_algorithm", "path_strategy", "same_class_path"]
    by_split = aggregate(per_path, base_keys + ["path_split"])
    overall = aggregate(per_path, base_keys).with_columns(
        pl.lit("all").alias("path_split")
    )
    return pl.concat([overall.select(by_split.columns), by_split]).sort(
        base_keys + ["path_split"]
    )


def plot_flip_phases(summary: pl.DataFrame, output_dir: Path) -> list[Path]:
    """Write one single plot per (strategy, class) pair; return the file paths."""
    data = summary.filter(pl.col("path_split") == "all")
    strategies = data.get_column("path_strategy").unique().sort().to_list()
    available_gnns = data.get_column("gnn_algorithm").unique().to_list()
    gnns = [g for g in ["GCN", "GIN"] if g in available_gnns]

    bar_width = 0.15
    class_slugs = {0: "cross_class", 1: "same_class"}

    written: list[Path] = []
    for same_class in [0, 1]:
        for strategy in strategies:
            subset = data.filter(
                (pl.col("path_strategy") == strategy)
                & (pl.col("same_class_path") == same_class)
            )
            if subset.is_empty():
                continue

            fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(gnns)), 5.0))
            values = {
                g: {
                    segment: subset.filter(pl.col("gnn_algorithm") == g)
                    .get_column(f"flip_rate_{segment}")
                    .to_list()
                    for segment in SEGMENTS
                }
                for g in gnns
            }
            offset0 = (len(SEGMENTS) - 1) / 2
            for seg_idx, segment in enumerate(SEGMENTS):
                heights = [
                    (values[g][segment][0] or 0.0) if values[g][segment] else 0.0
                    for g in gnns
                ]
                positions = [
                    i + (seg_idx - offset0) * bar_width for i in range(len(gnns))
                ]
                ax.bar(
                    positions,
                    heights,
                    width=bar_width,
                    label=SEGMENT_LABELS[segment],
                )
            ax.set_xticks(range(len(gnns)))
            ax.set_xticklabels(gnns, fontsize=10)
            ax.set_ylabel("decision flips per operation")
            ax.legend(fontsize=8)

            output_png = (
                output_dir
                / f"flip_phases_{strategy}_{class_slugs[same_class]}.png"
            )
            fig.tight_layout()
            fig.savefig(output_png, dpi=150)
            plt.close(fig)
            written.append(output_png)
    return written


@click.command()
@click.option(
    "--input-csv",
    default="results/all_results_pattern.csv",
    show_default=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Pattern-enriched results table from utils/pattern_tracking.py.",
)
@click.option(
    "--output-dir",
    default="results/pattern_flip_analysis",
    show_default=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Directory for analysis outputs.",
)
@click.option(
    "--dataset",
    default="TRIANGLE_SQUARE",
    show_default=True,
    help="Dataset name to analyze.",
)
def main(input_csv: Path, output_dir: Path, dataset: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    click.echo(f"Computing per-path flip counts from '{input_csv}'.")
    per_path = compute_per_path(input_csv, dataset)
    per_path_csv = output_dir / "flip_phase_per_path.csv"
    per_path.sort(
        ["gnn_algorithm", "path_strategy", "val_id", "source_id", "target_id"]
    ).write_csv(str(per_path_csv))
    click.echo(f"Per-path flip counts written to '{per_path_csv}' ({per_path.height} paths).")

    summary = summarize(per_path)
    summary_csv = output_dir / "flip_phase_summary.csv"
    summary.write_csv(str(summary_csv))
    click.echo(f"Summary written to '{summary_csv}'.")

    plots = plot_flip_phases(summary, output_dir)
    for plot_png in plots:
        click.echo(f"Plot written to '{plot_png}'.")


if __name__ == "__main__":
    main()
