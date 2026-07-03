"""Statistical validation of the GNN/GED decision-flip findings.

This CLI turns the descriptive per-fold experiment outputs into defensible
claims with p-values, multiple-comparison control (Benjamini-Hochberg FDR),
effect sizes, and cluster-bootstrap confidence intervals — respecting the nested
data structure where **the CV fold is the unit of replication**.

It does not recompute flip aggregations. Tier 1 consumes the per-fold
``*_fold_counts.csv`` / ``*_fold.csv`` files emitted by ``experiments.py`` and
``experiments_additional.py``; Tier 2 (the GLMM/GEE confirmatory model) streams
the per-step rows from ``results/all_results.csv``.

See ``STATISTICAL_VALIDATION_PLAN.md`` for the full design.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import click
import numpy as np
import pandas as pd

from utils.stats import (
    bh_fdr,
    cluster_bootstrap_ci,
    friedman_with_posthoc,
    one_sample_wilcoxon,
    page_trend_test,
    paired_diff_bootstrap_ci,
    wilcoxon_paired,
)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "dataset",
    "gnn_algorithm",
    "path_strategy",
    "split",
    "finding",
    "comparison",
    "test",
    "n",
    "statistic",
    "p_value",
    "p_value_fdr",
    "effect_size",
    "effect_size_name",
    "ci_low",
    "ci_high",
    "significant",
]

OPERATION_ORDER = (
    "EDGE INSERT",
    "EDGE DELETE",
    "NODE INSERT",
    "NODE DELETE",
    "EDGE RELABEL",
    "NODE RELABEL",
)

# Ordered low edge of each margin bin, used to sort the calibration bins by
# increasing prediction margin before the trend test.
MARGIN_BIN_ORDER = {
    "[0,0.05)": 0,
    "[0.05,0.1)": 1,
    "[0.1,0.2)": 2,
    "[0.2,0.4)": 3,
    "[0.4,1]": 4,
    "(1,inf)": 5,
}

POSITION_BIN_ORDER = {"early": 0, "mid": 1, "late": 2}

MIN_FOLDS_FOR_PVALUE = 5
MIN_TRANSITION_COUNT = 10


@dataclass
class Row:
    dataset: str
    gnn_algorithm: str
    path_strategy: str
    split: str
    finding: str
    comparison: str
    test: str
    n: int
    statistic: float = float("nan")
    p_value: float = float("nan")
    p_value_fdr: float = float("nan")
    effect_size: float = float("nan")
    effect_size_name: str = ""
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    significant: Optional[bool] = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def _filter(df: pd.DataFrame, **eq) -> pd.DataFrame:
    out = df
    for col, val in eq.items():
        if val is None:
            continue
        out = out[out[col] == val]
    return out


def _cells(df: pd.DataFrame, keys: Sequence[str]):
    """Iterate over unique combinations of ``keys`` yielding (key_tuple, frame)."""
    for key, frame in df.groupby(list(keys), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        yield key, frame


def _ordered_present(values, order: Sequence[str]) -> List[str]:
    present = set(values)
    return [v for v in order if v in present]


def _pivot_folds(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """fold (val_id) x ``column`` matrix of ``value`` (mean over duplicates)."""
    return frame.pivot_table(
        index="val_id", columns=column, values=value, aggfunc="mean"
    )


def _significant(p_fdr: float, alpha: float) -> Optional[bool]:
    if p_fdr is None or (isinstance(p_fdr, float) and math.isnan(p_fdr)):
        return None
    return bool(p_fdr < alpha)


# ---------------------------------------------------------------------------
# Finding 1 — operation-type flip effects
# ---------------------------------------------------------------------------


def finding_operation_effects(
    results_dir: Path,
    dataset: Optional[str],
    n_boot: int,
    seed: int,
) -> List[Row]:
    path = (
        results_dir
        / "Experiments"
        / "experiments_flips_per_operation_relative"
        / "flips_per_operation_relative_fold_counts.csv"
    )
    df = _read_csv(path)
    rows: List[Row] = []
    if df is None:
        return rows
    df = _filter(df, dataset=dataset)
    if df.empty:
        return rows

    keys = ("dataset", "gnn_algorithm", "path_strategy", "split")
    for (ds, gnn, strat, split), frame in _cells(df, keys):
        pivot = _pivot_folds(frame, "operation_str", "relative_decision_change_count")
        ops = _ordered_present(pivot.columns, OPERATION_ORDER)
        pivot = pivot[ops]
        n_folds = int(pivot.dropna(how="any").shape[0])

        # Per-operation point estimate + bootstrap CI (always emitted).
        for op in ops:
            vals = pivot[op].dropna().to_numpy()
            point, lo, hi = cluster_bootstrap_ci(vals, n_reps=n_boot, seed=seed)
            rows.append(
                Row(ds, gnn, strat, split, "operation_effects",
                    f"mean[{op}]", "bootstrap", int(vals.size),
                    statistic=point, effect_size=point,
                    effect_size_name="mean_relative_flip", ci_low=lo, ci_high=hi)
            )

        if len(ops) < 3 or n_folds < MIN_FOLDS_FOR_PVALUE:
            continue

        data_by_group = {op: pivot[op].to_numpy() for op in ops}
        omnibus, posthoc = friedman_with_posthoc(data_by_group, adjust="holm")
        rows.append(
            Row(ds, gnn, strat, split, "operation_effects", "omnibus(operations)",
                omnibus.test, omnibus.n, statistic=omnibus.statistic,
                p_value=omnibus.p_value, effect_size=omnibus.effect_size,
                effect_size_name=omnibus.effect_size_name)
        )
        for pc in posthoc:
            _, lo, hi = paired_diff_bootstrap_ci(
                pivot[pc.group_a].to_numpy(), pivot[pc.group_b].to_numpy(),
                n_reps=n_boot, seed=seed,
            )
            rows.append(
                Row(ds, gnn, strat, split, "operation_effects",
                    f"{pc.group_a} vs {pc.group_b}", pc.test, pc.n,
                    statistic=pc.statistic, p_value=pc.p_value,
                    effect_size=pc.effect_size, effect_size_name=pc.effect_size_name,
                    ci_low=lo, ci_high=hi)
            )
    return rows


# ---------------------------------------------------------------------------
# Finding 2 — GNN architecture differences
# ---------------------------------------------------------------------------


def finding_gnn_differences(
    results_dir: Path,
    dataset: Optional[str],
    n_boot: int,
    seed: int,
) -> List[Row]:
    path = (
        results_dir
        / "Experiments"
        / "experiments_flips_per_operation_relative"
        / "flips_per_operation_relative_fold_counts.csv"
    )
    df = _read_csv(path)
    rows: List[Row] = []
    if df is None:
        return rows
    df = _filter(df, dataset=dataset)
    if df.empty:
        return rows

    # Overall per-(gnn, fold) relative flip rate = sum(flips) / sum(operations).
    grouped = (
        df.groupby(["dataset", "path_strategy", "split", "gnn_algorithm", "val_id"])
        .agg(flips=("decision_change_count", "sum"),
             ops=("operation_count", "sum"))
        .reset_index()
    )
    grouped["overall_relative_flip"] = grouped["flips"] / grouped["ops"].replace(0, np.nan)

    for (ds, strat, split), frame in _cells(grouped, ("dataset", "path_strategy", "split")):
        pivot = frame.pivot_table(
            index="val_id", columns="gnn_algorithm", values="overall_relative_flip"
        )
        gnns = sorted(pivot.columns)
        pivot = pivot[gnns]
        n_folds = int(pivot.dropna(how="any").shape[0])

        for gnn in gnns:
            vals = pivot[gnn].dropna().to_numpy()
            point, lo, hi = cluster_bootstrap_ci(vals, n_reps=n_boot, seed=seed)
            rows.append(
                Row(ds, gnn, strat, split, "gnn_differences", f"mean[{gnn}]",
                    "bootstrap", int(vals.size), statistic=point, effect_size=point,
                    effect_size_name="mean_relative_flip", ci_low=lo, ci_high=hi)
            )

        if len(gnns) < 3 or n_folds < MIN_FOLDS_FOR_PVALUE:
            continue

        data_by_group = {gnn: pivot[gnn].to_numpy() for gnn in gnns}
        omnibus, posthoc = friedman_with_posthoc(data_by_group, adjust="holm")
        rows.append(
            Row(ds, "ALL", strat, split, "gnn_differences", "omnibus(gnns)",
                omnibus.test, omnibus.n, statistic=omnibus.statistic,
                p_value=omnibus.p_value, effect_size=omnibus.effect_size,
                effect_size_name=omnibus.effect_size_name)
        )
        for pc in posthoc:
            _, lo, hi = paired_diff_bootstrap_ci(
                pivot[pc.group_a].to_numpy(), pivot[pc.group_b].to_numpy(),
                n_reps=n_boot, seed=seed,
            )
            rows.append(
                Row(ds, f"{pc.group_a}|{pc.group_b}", strat, split, "gnn_differences",
                    f"{pc.group_a} vs {pc.group_b}", pc.test, pc.n,
                    statistic=pc.statistic, p_value=pc.p_value,
                    effect_size=pc.effect_size, effect_size_name=pc.effect_size_name,
                    ci_low=lo, ci_high=hi)
            )
    return rows


# ---------------------------------------------------------------------------
# Finding 3 — margin vs flips
# ---------------------------------------------------------------------------


def finding_margin_to_flip(
    results_dir: Path,
    dataset: Optional[str],
    n_boot: int,
    seed: int,
) -> List[Row]:
    base = results_dir / "ExperimentsAdditional" / "experiments_margin_to_flip_analysis"
    rows: List[Row] = []

    # (a) monotone-decreasing flip rate across increasing-margin bins (Page trend).
    calib = _read_csv(base / "margin_to_flip_calibration_fold.csv")
    if calib is not None:
        calib = _filter(calib, dataset=dataset)
        if not calib.empty:
            # Aggregate over operations: flip rate per (fold, margin_bin) weighted by steps.
            calib = calib.copy()
            calib["flip_count_w"] = calib["flip_rate"] * calib["step_count"]
            agg = (
                calib.groupby(
                    ["dataset", "gnn_algorithm", "path_strategy", "split",
                     "val_id", "margin_bin"]
                )
                .agg(flips=("flip_count_w", "sum"), steps=("step_count", "sum"))
                .reset_index()
            )
            agg["flip_rate"] = agg["flips"] / agg["steps"].replace(0, np.nan)
            keys = ("dataset", "gnn_algorithm", "path_strategy", "split")
            for (ds, gnn, strat, split), frame in _cells(agg, keys):
                pivot = frame.pivot_table(
                    index="val_id", columns="margin_bin", values="flip_rate"
                )
                bins = sorted(
                    [b for b in pivot.columns if b in MARGIN_BIN_ORDER],
                    key=lambda b: MARGIN_BIN_ORDER[b],
                )
                if len(bins) < 3:
                    continue
                pivot = pivot[bins]
                n_folds = int(pivot.dropna(how="any").shape[0])
                if n_folds < MIN_FOLDS_FOR_PVALUE:
                    continue
                # Decreasing trend across increasing-margin bins.
                res = page_trend_test(pivot.to_numpy(), ascending=False)
                rows.append(
                    Row(ds, gnn, strat, split, "margin_to_flip",
                        "monotone_decreasing(margin_bins)", res.test, res.n,
                        statistic=res.statistic, p_value=res.p_value,
                        effect_size=res.effect_size, effect_size_name=res.effect_size_name)
                )

    # (b) flipping vs non-flipping margin (paired Wilcoxon by fold).
    dist = _read_csv(base / "margin_to_flip_distribution_summary.csv")
    if dist is not None:
        dist = _filter(dist, dataset=dataset)
        keys = ("dataset", "gnn_algorithm", "path_strategy", "split")
        for (ds, gnn, strat, split), frame in _cells(dist, keys):
            frame = frame.dropna(subset=["mean_margin_flipping", "mean_margin_non_flipping"])
            x = frame["mean_margin_non_flipping"].to_numpy()  # expected larger
            y = frame["mean_margin_flipping"].to_numpy()
            n_folds = int(min(len(x), len(y)))
            point, lo, hi = paired_diff_bootstrap_ci(x, y, n_reps=n_boot, seed=seed)
            if n_folds < MIN_FOLDS_FOR_PVALUE:
                rows.append(
                    Row(ds, gnn, strat, split, "margin_to_flip",
                        "non_flip_minus_flip_margin", "bootstrap", n_folds,
                        statistic=point, effect_size=point,
                        effect_size_name="mean_margin_difference", ci_low=lo, ci_high=hi)
                )
                continue
            res = wilcoxon_paired(x, y)
            rows.append(
                Row(ds, gnn, strat, split, "margin_to_flip",
                    "non_flip_minus_flip_margin", res.test, res.n,
                    statistic=res.statistic, p_value=res.p_value,
                    effect_size=point, effect_size_name="mean_margin_difference",
                    ci_low=lo, ci_high=hi)
            )
    return rows


# ---------------------------------------------------------------------------
# Finding 4 — sequence / position effects
# ---------------------------------------------------------------------------


def finding_sequence_position(
    results_dir: Path,
    dataset: Optional[str],
    n_boot: int,
    seed: int,
    min_transition_count: int,
) -> List[Row]:
    base = results_dir / "ExperimentsAdditional"
    rows: List[Row] = []

    # (a) flip position sensitivity — Friedman across early/mid/late.
    pos = _read_csv(base / "experiments_flip_position_sensitivity"
                    / "flip_position_sensitivity_fold_counts.csv")
    if pos is not None:
        pos = _filter(pos, dataset=dataset)
        if not pos.empty:
            pos = pos.copy()
            pos["flips_w"] = pos["flip_rate"] * pos["operation_count"]
            agg = (
                pos.groupby(["dataset", "gnn_algorithm", "path_strategy", "split",
                             "val_id", "position_bin"])
                .agg(flips=("flips_w", "sum"), ops=("operation_count", "sum"))
                .reset_index()
            )
            agg["flip_rate"] = agg["flips"] / agg["ops"].replace(0, np.nan)
            keys = ("dataset", "gnn_algorithm", "path_strategy", "split")
            for (ds, gnn, strat, split), frame in _cells(agg, keys):
                pivot = frame.pivot_table(
                    index="val_id", columns="position_bin", values="flip_rate"
                )
                bins = sorted(
                    [b for b in pivot.columns if b in POSITION_BIN_ORDER],
                    key=lambda b: POSITION_BIN_ORDER[b],
                )
                if len(bins) < 3:
                    continue
                pivot = pivot[bins]
                for b in bins:
                    vals = pivot[b].dropna().to_numpy()
                    point, lo, hi = cluster_bootstrap_ci(vals, n_reps=n_boot, seed=seed)
                    rows.append(
                        Row(ds, gnn, strat, split, "position_sensitivity",
                            f"mean[{b}]", "bootstrap", int(vals.size),
                            statistic=point, effect_size=point,
                            effect_size_name="mean_flip_rate", ci_low=lo, ci_high=hi)
                    )
                n_folds = int(pivot.dropna(how="any").shape[0])
                if n_folds < MIN_FOLDS_FOR_PVALUE:
                    continue
                data_by_group = {b: pivot[b].to_numpy() for b in bins}
                omnibus, posthoc = friedman_with_posthoc(data_by_group, adjust="holm")
                rows.append(
                    Row(ds, gnn, strat, split, "position_sensitivity",
                        "omnibus(position_bins)", omnibus.test, omnibus.n,
                        statistic=omnibus.statistic, p_value=omnibus.p_value,
                        effect_size=omnibus.effect_size,
                        effect_size_name=omnibus.effect_size_name)
                )
                for pc in posthoc:
                    rows.append(
                        Row(ds, gnn, strat, split, "position_sensitivity",
                            f"{pc.group_a} vs {pc.group_b}", pc.test, pc.n,
                            statistic=pc.statistic, p_value=pc.p_value,
                            effect_size=pc.effect_size,
                            effect_size_name=pc.effect_size_name)
                    )

    # (b) operation-transition instability — one-sample Wilcoxon on log enrichment.
    big = _read_csv(base / "experiments_operation_transition_instability"
                    / "operation_transition_bigrams_fold.csv")
    if big is not None:
        big = _filter(big, dataset=dataset)
        big = big[(big["sequence_count"] >= min_transition_count)
                  & (big["enrichment_ratio"] > 0)].copy()
        if not big.empty:
            big["log_enrichment"] = np.log(big["enrichment_ratio"])
            keys = ("dataset", "gnn_algorithm", "path_strategy", "split",
                    "operation_prev", "operation_curr")
            for (ds, gnn, strat, split, prev, curr), frame in _cells(big, keys):
                vals = frame["log_enrichment"].to_numpy()
                point, lo, hi = cluster_bootstrap_ci(vals, n_reps=n_boot, seed=seed)
                comparison = f"log_enrichment[{prev}->{curr}]"
                if int(np.size(vals)) < MIN_FOLDS_FOR_PVALUE:
                    rows.append(
                        Row(ds, gnn, strat, split, "transition_instability",
                            comparison, "bootstrap", int(np.size(vals)),
                            statistic=point, effect_size=point,
                            effect_size_name="mean_log_enrichment",
                            ci_low=lo, ci_high=hi)
                    )
                    continue
                res = one_sample_wilcoxon(vals, popmean=0.0)
                rows.append(
                    Row(ds, gnn, strat, split, "transition_instability",
                        comparison, res.test, res.n, statistic=res.statistic,
                        p_value=res.p_value, effect_size=point,
                        effect_size_name="mean_log_enrichment", ci_low=lo, ci_high=hi)
                )

    # (c) flip streaks and recovery — Friedman of per-fold recovery rate across GNNs.
    pm = _read_csv(base / "experiments_flip_streaks_and_recovery"
                   / "flip_streaks_and_recovery_path_metrics.csv")
    if pm is not None:
        pm = _filter(pm, dataset=dataset)
        flipped = pm[pm["num_flips"] > 0]
        if not flipped.empty:
            rec = (
                flipped.groupby(["dataset", "path_strategy", "split",
                                 "gnn_algorithm", "val_id"])
                .agg(recovery_rate=("recovered_to_initial_label", "mean"))
                .reset_index()
            )
            for (ds, strat, split), frame in _cells(rec, ("dataset", "path_strategy", "split")):
                pivot = frame.pivot_table(
                    index="val_id", columns="gnn_algorithm", values="recovery_rate"
                )
                gnns = sorted(pivot.columns)
                pivot = pivot[gnns]
                for gnn in gnns:
                    vals = pivot[gnn].dropna().to_numpy()
                    point, lo, hi = cluster_bootstrap_ci(vals, n_reps=n_boot, seed=seed)
                    rows.append(
                        Row(ds, gnn, strat, split, "flip_recovery",
                            f"mean_recovery[{gnn}]", "bootstrap", int(vals.size),
                            statistic=point, effect_size=point,
                            effect_size_name="mean_recovery_rate",
                            ci_low=lo, ci_high=hi)
                    )
                n_folds = int(pivot.dropna(how="any").shape[0])
                if len(gnns) < 3 or n_folds < MIN_FOLDS_FOR_PVALUE:
                    continue
                data_by_group = {gnn: pivot[gnn].to_numpy() for gnn in gnns}
                omnibus, posthoc = friedman_with_posthoc(data_by_group, adjust="holm")
                rows.append(
                    Row(ds, "ALL", strat, split, "flip_recovery",
                        "omnibus(recovery~gnn)", omnibus.test, omnibus.n,
                        statistic=omnibus.statistic, p_value=omnibus.p_value,
                        effect_size=omnibus.effect_size,
                        effect_size_name=omnibus.effect_size_name)
                )
                for pc in posthoc:
                    rows.append(
                        Row(ds, f"{pc.group_a}|{pc.group_b}", strat, split,
                            "flip_recovery", f"{pc.group_a} vs {pc.group_b}",
                            pc.test, pc.n, statistic=pc.statistic,
                            p_value=pc.p_value, effect_size=pc.effect_size,
                            effect_size_name=pc.effect_size_name)
                    )
    return rows


# ---------------------------------------------------------------------------
# Tier 2 — GEE / GLMM confirmatory model
# ---------------------------------------------------------------------------

GLMM_USECOLS = [
    "operation_str", "is_flipping", "is_path", "is_validation_path",
    "val_id", "path_idx_int", "gnn_algorithm", "dataset", "path_strategy",
]


def _load_glmm_frame(
    data_path: Path,
    dataset: str,
    split: str,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Stream ``all_results.csv`` and collect path steps for one dataset."""
    parts: List[pd.DataFrame] = []
    for chunk in pd.read_csv(data_path, usecols=GLMM_USECOLS, chunksize=chunksize):
        sub = chunk[(chunk["dataset"] == dataset)
                    & (chunk["is_path"] == 1)
                    & (chunk["operation_str"] != "NONE")]
        if split == "validation":
            sub = sub[sub["is_validation_path"] == 1]
        if not sub.empty:
            parts.append(sub.copy())
    if not parts:
        return pd.DataFrame(columns=GLMM_USECOLS)
    return pd.concat(parts, ignore_index=True)


def fit_glmm(
    frame: pd.DataFrame,
    dataset: str,
    gnn: str,
    split: str,
    max_paths: int,
    seed: int,
    alpha: float,
) -> List[dict]:
    """Population-averaged GEE (cluster-robust) for one dataset/gnn."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    sub = frame[frame["gnn_algorithm"] == gnn].copy()
    if sub.empty:
        return []
    sub["cluster"] = (
        sub["path_strategy"].astype(str) + ":"
        + sub["val_id"].astype(str) + ":"
        + sub["path_idx_int"].astype(str)
    )

    # Subsample whole clusters (paths) for tractability on the multi-million-row CSV.
    clusters = sub["cluster"].unique()
    if len(clusters) > max_paths:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(clusters, size=max_paths, replace=False))
        sub = sub[sub["cluster"].isin(keep)]

    if sub["is_flipping"].nunique() < 2:
        return []
    present_ops = _ordered_present(sub["operation_str"].unique(), OPERATION_ORDER)
    if len(present_ops) < 2:
        return []
    # Reference = most frequent operation.
    reference = sub["operation_str"].value_counts().idxmax()
    sub["operation_str"] = pd.Categorical(
        sub["operation_str"], categories=[reference] + [o for o in present_ops if o != reference]
    )

    model = smf.gee(
        "is_flipping ~ C(operation_str)",
        groups="cluster",
        data=sub,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    )
    res = model.fit()

    out: List[dict] = []
    params = res.params
    conf = res.conf_int()
    raw_p: List[float] = []
    op_terms: List[Tuple[str, float, float, float, float, float]] = []
    for name in params.index:
        if name == "Intercept":
            continue
        op = name.split("[T.")[-1].rstrip("]") if "[T." in name else name
        coef = float(params[name])
        lo, hi = float(conf.loc[name][0]), float(conf.loc[name][1])
        p = float(res.pvalues[name])
        se = float(res.bse[name])
        op_terms.append((op, coef, se, lo, hi, p))
        raw_p.append(p)

    p_fdr = bh_fdr(raw_p)
    n_clusters = int(sub["cluster"].nunique())
    for (op, coef, se, lo, hi, p), pf in zip(op_terms, p_fdr):
        out.append({
            "dataset": dataset,
            "gnn_algorithm": gnn,
            "path_strategy": "ALL",
            "split": split,
            "operation_str": op,
            "reference": reference,
            "n_obs": int(sub.shape[0]),
            "n_clusters": n_clusters,
            "odds_ratio": math.exp(coef),
            "ci_low": math.exp(lo),
            "ci_high": math.exp(hi),
            "coef": coef,
            "std_err": se,
            "p_value": p,
            "p_value_fdr": pf,
            "significant": _significant(pf, alpha),
        })
    return out


# ---------------------------------------------------------------------------
# Assembly / output
# ---------------------------------------------------------------------------


def _apply_family_fdr(rows: List[Row], alpha: float) -> None:
    """Family-wide BH-FDR across every row that carries a raw p-value."""
    idx = [i for i, r in enumerate(rows) if not math.isnan(r.p_value)]
    p_fdr = bh_fdr([rows[i].p_value for i in idx])
    for i, pf in zip(idx, p_fdr):
        rows[i].p_value_fdr = pf
        rows[i].significant = _significant(pf, alpha)


def _write_rows(rows: List[Row], path: Path) -> int:
    frame = pd.DataFrame([asdict(r) for r in rows], columns=RESULT_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return len(frame)


def _summarise(name: str, rows: List[Row]) -> List[str]:
    lines = [f"### {name}", ""]
    tested = [r for r in rows if not math.isnan(r.p_value)]
    sig = [r for r in tested if r.significant]
    lines.append(f"- comparisons with a p-value: **{len(tested)}**")
    lines.append(f"- significant after FDR (alpha): **{len(sig)}**")
    for r in sorted(sig, key=lambda r: r.p_value_fdr)[:8]:
        lines.append(
            f"  - `{r.dataset}/{r.gnn_algorithm}/{r.path_strategy}/{r.split}` "
            f"{r.comparison} ({r.test}): p_fdr={r.p_value_fdr:.4g}, "
            f"effect={r.effect_size:.3g} [{r.effect_size_name}]"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--results-dir", type=click.Path(path_type=Path), default=Path("results"),
              show_default=True, help="Directory holding Experiments/ and ExperimentsAdditional/.")
@click.option("--output-dir", type=click.Path(path_type=Path),
              default=Path("results/StatisticalValidation"), show_default=True)
@click.option("--data-path", type=click.Path(path_type=Path),
              default=Path("results/all_results.csv"), show_default=True,
              help="Per-step rows for the Tier-2 GLMM/GEE.")
@click.option("--dataset", "datasets", multiple=True,
              help="Datasets to validate. Defaults to all present in the fold files.")
@click.option("--split", type=click.Choice(["all", "validation"]), default="all",
              show_default=True, help="Path split scope for the analyses.")
@click.option("--tier", type=click.Choice(["fold", "glmm", "all"]), default="fold",
              show_default=True)
@click.option("--gnn", "gnns", multiple=True,
              help="Restrict the GLMM tier to these GNNs. Defaults to all.")
@click.option("--n-boot", type=int, default=2000, show_default=True,
              help="Bootstrap replicates for confidence intervals.")
@click.option("--glmm-max-paths", type=int, default=4000, show_default=True,
              help="Cap on clusters (paths) per dataset/gnn for the GEE fit.")
@click.option("--alpha", type=float, default=0.05, show_default=True)
@click.option("--min-transition-count", type=int, default=MIN_TRANSITION_COUNT,
              show_default=True, help="Minimum sequence count for transition bigrams.")
@click.option("--seed", type=int, default=42, show_default=True)
def main(
    results_dir: Path,
    output_dir: Path,
    data_path: Path,
    datasets: Tuple[str, ...],
    split: str,
    tier: str,
    gnns: Tuple[str, ...],
    n_boot: int,
    glmm_max_paths: int,
    alpha: float,
    min_transition_count: int,
    seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen = list(datasets) if datasets else [None]
    report: List[str] = ["# Statistical validation report", ""]
    report.append(f"- split scope: `{split}`  ·  alpha: `{alpha}`  ·  "
                  f"bootstrap reps: `{n_boot}`  ·  seed: `{seed}`")
    report.append(f"- tier: `{tier}`")
    report.append("")

    def split_filter(rows: List[Row]) -> List[Row]:
        return [r for r in rows if r.split == split]

    if tier in ("fold", "all"):
        families = {
            "operation_effects_tests.csv": ("Finding 1 — operation-type flip effects", []),
            "gnn_difference_tests.csv": ("Finding 2 — GNN architecture differences", []),
            "margin_to_flip_tests.csv": ("Finding 3 — margin vs flips", []),
            "sequence_position_tests.csv": ("Finding 4 — sequence/position effects", []),
        }
        for ds in chosen:
            families["operation_effects_tests.csv"][1].extend(
                finding_operation_effects(results_dir, ds, n_boot, seed))
            families["gnn_difference_tests.csv"][1].extend(
                finding_gnn_differences(results_dir, ds, n_boot, seed))
            families["margin_to_flip_tests.csv"][1].extend(
                finding_margin_to_flip(results_dir, ds, n_boot, seed))
            families["sequence_position_tests.csv"][1].extend(
                finding_sequence_position(results_dir, ds, n_boot, seed, min_transition_count))

        report.append("## Tier 1 — fold-level nonparametric tests")
        report.append("")
        for filename, (title, rows) in families.items():
            rows = split_filter(rows)
            _apply_family_fdr(rows, alpha)
            written = _write_rows(rows, output_dir / filename)
            click.echo(f"  {filename}: {written} rows")
            report.extend(_summarise(title, rows))

        # Single-fold / under-powered notice.
        skipped = sum(
            1 for _, (_, rows) in families.items()
            for r in split_filter(rows)
            if math.isnan(r.p_value) and r.test != "bootstrap"
        )
        if skipped:
            report.append(f"> {skipped} cells had too few folds (<{MIN_FOLDS_FOR_PVALUE}) "
                          f"for a p-value; point estimate + bootstrap CI reported instead.")
            report.append("")

    if tier in ("glmm", "all"):
        report.append("## Tier 2 — GEE (cluster-robust logistic) confirmatory model")
        report.append("")
        if not data_path.exists():
            raise click.ClickException(f"Results file not found: {data_path}")
        glmm_datasets = [d for d in chosen if d is not None]
        if not glmm_datasets:
            raise click.ClickException(
                "Tier 2 requires at least one explicit --dataset (the per-step CSV is large).")
        for ds in glmm_datasets:
            click.echo(f"  loading per-step rows for {ds} ...")
            frame = _load_glmm_frame(data_path, ds, split)
            if frame.empty:
                report.append(f"- `{ds}`: no path steps found in {data_path.name}.")
                continue
            target_gnns = list(gnns) if gnns else sorted(frame["gnn_algorithm"].unique())
            for gnn in target_gnns:
                try:
                    records = fit_glmm(frame, ds, gnn, split, glmm_max_paths, seed, alpha)
                except Exception as exc:  # pragma: no cover - model convergence guard
                    report.append(f"- `{ds}/{gnn}`: GEE failed ({exc}).")
                    continue
                if not records:
                    report.append(f"- `{ds}/{gnn}`: insufficient variation for the GEE.")
                    continue
                out_path = output_dir / f"glmm_operation_{ds}_{gnn}.csv"
                pd.DataFrame(records).to_csv(out_path, index=False)
                click.echo(f"  {out_path.name}: {len(records)} operation terms")
                sig = [r for r in records if r["significant"]]
                ranked = sorted(records, key=lambda r: r["odds_ratio"], reverse=True)
                ordering = ", ".join(f"{r['operation_str']}({r['odds_ratio']:.2f})" for r in ranked)
                report.append(
                    f"- `{ds}/{gnn}` (ref=`{records[0]['reference']}`, "
                    f"n={records[0]['n_obs']}, clusters={records[0]['n_clusters']}): "
                    f"{len(sig)} significant ORs. OR ranking: {ordering}")
        report.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n")
    click.echo(f"Report written: {report_path}")
    click.echo(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
