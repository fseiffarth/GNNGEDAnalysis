"""Answer the significance questions raised as comments in the DS 2026 paper.

The paper (sections/graph-neural-network-decisions.tex) raises three questions
and leaves their statistical verification to "future work":

  Q1  How often does a model's prediction change along an edit path?
      -> claim: same-label-endpoint paths flip less than different-label paths,
         and the "0 flips / exactly 1 flip" cases dominate (~50-60%).

  Q2  Which edit operations are most likely to trigger a decision change, and
      (tex comment, l.51) are there significant differences for edit operations
      of one type (regarding flips) BETWEEN ALGORITHMS?

  Q3  Are Q1/Q2 affected by the structural properties of intermediate graphs,
      i.e. (tex comment, l.49) is the flips-per-operation pattern significantly
      different between path strategies vs. between models on the same dataset?

This script reuses utils/stats.py and the per-fold experiment outputs.  The CV
fold is the unit of replication (n=10 per cell).  Output -> paper tmp/ folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils.stats import (
    friedman_with_posthoc,
    wilcoxon_paired,
    kendalls_w,
    rank_biserial,
    bh_fdr,
)

RESULTS = Path("results")
OUT = Path("/home/florian/Documents/Work/Forschung/EigeneForschung/2026/DS_GNN_GED/tmp/significance")
OUT.mkdir(parents=True, exist_ok=True)

SPLIT = "all"
ALPHA = 0.05
GNNS = ["GCN", "GraphSAGE", "GIN", "GAT", "GATv2"]
OPS = ["NODE INSERT", "NODE DELETE", "NODE RELABEL",
       "EDGE INSERT", "EDGE DELETE", "EDGE RELABEL"]
STRAT_LABEL = {"Rnd": "Rnd", "i-E_d-IsoN": "+E", "d-E_d-IsoN": "-E"}

# Restrict all significance tests to the six molecular benchmark datasets and the
# Rnd / +E path strategies.  We exclude the synthetic TRIANGLE_SQUARE sanity-check
# dataset, the auxiliary Rnd_d-IsoN strategy, and the -E strategy: -E paths exist
# for only two datasets (MUTAG, PTC_FM), too few for a cross-dataset comparison, so
# we drop -E everywhere to keep the architecture- and dataset-level analyses on the
# same strategy set.
MOLECULAR_DATASETS = ["DHFR", "MUTAG", "Mutagenicity", "NCI1", "NCI109", "PTC_FM"]
STRATEGIES_SRC = {"Rnd", "i-E_d-IsoN"}  # Rnd and +E only


def _restrict(frame):
    """Keep only the six molecular datasets and the Rnd/+E/-E strategies."""
    return frame[frame.dataset.isin(MOLECULAR_DATASETS)
                 & frame.path_strategy.isin(STRATEGIES_SRC)]


def _pivot(frame, index, column, value):
    """folds(index) x groups(column) matrix; keep only complete blocks."""
    p = frame.pivot_table(index=index, columns=column, values=value, aggfunc="mean")
    return p.dropna(axis=0, how="any")


def rmcorr(x, y, groups):
    """Repeated-measures correlation (Bakdash & Marusich, 2017).

    Common within-group association between x and y after removing each group's
    mean (i.e. the partial correlation of x and y controlling for a categorical
    group factor).  Equivalent to the t-test on the x slope in the OLS model
    ``y ~ C(group) + x``.  Returns (r_rm, p_value, dof, n, n_groups).

    The group is whatever we want to hold fixed: for the fold-level test it is
    the (dataset, gnn, strategy) cell, so the correlation reflects how a model's
    accuracy and flip behaviour co-vary *across its own CV folds*; for the
    architecture-level test it is the dataset, so it reflects how, *within a
    dataset*, the better-classifying architectures differ in flip behaviour.
    """
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float),
                      "g": np.asarray(groups)}).dropna()
    # keep only groups with >=2 observations (a singleton contributes no df)
    d = d[d.groupby("g")["g"].transform("size") >= 2]
    n, k = len(d), d["g"].nunique()
    dof = n - k - 1
    if dof < 3 or k < 1:
        return np.nan, np.nan, dof, n, k
    xc = d["x"] - d.groupby("g")["x"].transform("mean")
    yc = d["y"] - d.groupby("g")["y"].transform("mean")
    sx, sy = np.sqrt((xc ** 2).sum()), np.sqrt((yc ** 2).sum())
    if sx == 0 or sy == 0:
        return np.nan, np.nan, dof, n, k
    r = float((xc * yc).sum() / (sx * sy))
    r = max(min(r, 1.0), -1.0)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * np.sqrt(dof / (1 - r * r))
        p = float(2 * stats.t.sf(abs(t), dof))
    return r, p, dof, n, k


# ---------------------------------------------------------------------------
# Q2 (l.51): per operation type, do the five GNN architectures differ?
# ---------------------------------------------------------------------------
def q2_gnn_differences_per_operation():
    f = (RESULTS / "Experiments" / "experiments_flips_per_operation_relative"
         / "flips_per_operation_relative_fold_counts.csv")
    df = pd.read_csv(f)
    df = _restrict(df[df.split == SPLIT])
    rows = []
    for (ds, strat, op), cell in df.groupby(["dataset", "path_strategy", "operation_str"]):
        mat = _pivot(cell, "val_id", "gnn_algorithm", "relative_decision_change_count")
        mat = mat[[g for g in GNNS if g in mat.columns]]
        if mat.shape[1] < 3 or mat.shape[0] < 3:
            continue
        groups = [mat[c].values for c in mat.columns]
        try:
            stat, p = stats.friedmanchisquare(*groups)
        except ValueError:
            continue
        w = kendalls_w([list(b) for b in zip(*groups)])
        means = mat.mean(axis=0)
        rows.append(dict(
            dataset=ds, path_strategy=STRAT_LABEL.get(strat, strat), operation=op,
            n_folds=mat.shape[0], test="friedman_across_gnns",
            statistic=stat, p_value=p, kendalls_w=w,
            most_sensitive=means.idxmax(), least_sensitive=means.idxmin(),
            **{f"mean_{g}": means.get(g, np.nan) for g in GNNS},
        ))
    res = pd.DataFrame(rows)
    res["p_value_fdr"] = bh_fdr(res["p_value"].tolist())
    res["significant"] = res["p_value_fdr"] < ALPHA
    res = res.sort_values(["dataset", "path_strategy", "operation"])
    res.to_csv(OUT / "sig_Q2_gnn_differences_per_operation.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Q3 (l.49): does the path strategy change the flips-per-operation pattern?
#   - strategy effect: Friedman across the 3 strategies (per dataset/gnn/op)
#   - compared against the model effect (Q2) on the SAME dataset.
# ---------------------------------------------------------------------------
def q3_strategy_effect_per_operation():
    f = (RESULTS / "Experiments" / "experiments_flips_per_operation_relative"
         / "flips_per_operation_relative_fold_counts.csv")
    df = pd.read_csv(f)
    df = _restrict(df[df.split == SPLIT])
    rows = []
    for (ds, gnn, op), cell in df.groupby(["dataset", "gnn_algorithm", "operation_str"]):
        mat = _pivot(cell, "val_id", "path_strategy", "relative_decision_change_count")
        if mat.shape[1] < 2 or mat.shape[0] < 3:
            continue
        groups = [mat[c].values for c in mat.columns]
        if mat.shape[1] >= 3:
            try:
                stat, p = stats.friedmanchisquare(*groups)
            except ValueError:
                continue
            test = "friedman_across_strategies"
        else:  # exactly 2 strategies (Rnd vs +E) -> paired Wilcoxon
            try:
                stat, p = stats.wilcoxon(groups[0], groups[1])
            except ValueError:
                stat, p = np.nan, np.nan
            test = "wilcoxon_across_strategies"
        w = kendalls_w([list(b) for b in zip(*groups)])
        means = mat.mean(axis=0)
        rows.append(dict(
            dataset=ds, gnn_algorithm=gnn, operation=op, n_folds=mat.shape[0],
            n_strategies=mat.shape[1], test=test, statistic=stat, p_value=p, kendalls_w=w,
            **{f"mean_{STRAT_LABEL.get(s, s)}": means.get(s, np.nan)
               for s in ["Rnd", "i-E_d-IsoN", "d-E_d-IsoN"]},
        ))
    res = pd.DataFrame(rows)
    res["p_value_fdr"] = bh_fdr(res["p_value"].tolist())
    res["significant"] = res["p_value_fdr"] < ALPHA
    res = res.sort_values(["dataset", "gnn_algorithm", "operation"])
    res.to_csv(OUT / "sig_Q3_strategy_effect_per_operation.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Q2/Q3 cross-cutting: which factor (operation, model, strategy) explains the
# most variance in flip rate?  Report median Kendall's W per factor & dataset.
# ---------------------------------------------------------------------------
def factor_effect_sizes():
    f = (RESULTS / "Experiments" / "experiments_flips_per_operation_relative"
         / "flips_per_operation_relative_fold_counts.csv")
    df = pd.read_csv(f)
    df = _restrict(df[df.split == SPLIT])
    rows = []
    for ds, dcell in df.groupby("dataset"):
        # operation effect (within gnn/strategy) -> Finding 1 / Q2-operations
        w_op = []
        for _, cell in dcell.groupby(["gnn_algorithm", "path_strategy"]):
            mat = _pivot(cell, "val_id", "operation_str", "relative_decision_change_count")
            if mat.shape[1] >= 3 and mat.shape[0] >= 3:
                w_op.append(kendalls_w([list(b) for b in zip(*[mat[c].values for c in mat.columns])]))
        # gnn effect (within operation/strategy)
        w_gnn = []
        for _, cell in dcell.groupby(["operation_str", "path_strategy"]):
            mat = _pivot(cell, "val_id", "gnn_algorithm", "relative_decision_change_count")
            if mat.shape[1] >= 3 and mat.shape[0] >= 3:
                w_gnn.append(kendalls_w([list(b) for b in zip(*[mat[c].values for c in mat.columns])]))
        # strategy effect (within operation/gnn)
        w_strat = []
        for _, cell in dcell.groupby(["operation_str", "gnn_algorithm"]):
            mat = _pivot(cell, "val_id", "path_strategy", "relative_decision_change_count")
            if mat.shape[1] >= 2 and mat.shape[0] >= 3:
                w_strat.append(kendalls_w([list(b) for b in zip(*[mat[c].values for c in mat.columns])]))
        def _med(xs):
            xs = [x for x in xs if np.isfinite(x)]
            return (np.median(xs), len(xs)) if xs else (np.nan, 0)
        w_op_m, n_op = _med(w_op)
        w_gnn_m, n_gnn = _med(w_gnn)
        w_strat_m, n_strat = _med(w_strat)
        rows.append(dict(
            dataset=ds,
            W_operation_median=w_op_m, n_op_cells=n_op,
            W_gnn_median=w_gnn_m, n_gnn_cells=n_gnn,
            W_strategy_median=w_strat_m, n_strat_cells=n_strat,
        ))
    res = pd.DataFrame(rows).sort_values("dataset")
    res.to_csv(OUT / "sig_factor_effect_sizes.csv", index=False)
    return res


def factor_effect_magnitudes():
    """Magnitude (not concordance) of the flip-rate change caused by each factor.

    Kendall's W only measures whether an ordering is *consistent*; a tiny but
    stable strategy difference still gives W~1.  Here we quantify how *large*
    the flip-rate spread is: for each held-fixed context we take the range
    (max-min) of the fold-averaged relative flip rate across the factor's
    levels, then report the median range over all contexts (in percentage
    points).  This is the measure relevant to the paper's "minor effect" claim.
    """
    f = (RESULTS / "Experiments" / "experiments_flips_per_operation_relative"
         / "flips_per_operation_relative_fold_counts.csv")
    df = pd.read_csv(f)
    df = _restrict(df[df.split == SPLIT])
    # fold-averaged flip rate per (dataset, gnn, strategy, operation)
    m = (df.groupby(["dataset", "gnn_algorithm", "path_strategy", "operation_str"])
           .relative_decision_change_count.mean().reset_index())
    rows = []
    for ds, d in m.groupby("dataset"):
        # operation factor: vary operation within (gnn, strategy)
        r_op = d.groupby(["gnn_algorithm", "path_strategy"]).relative_decision_change_count.agg(
            lambda x: x.max() - x.min())
        # gnn factor: vary gnn within (operation, strategy)
        r_gnn = d.groupby(["operation_str", "path_strategy"]).relative_decision_change_count.agg(
            lambda x: x.max() - x.min())
        # strategy factor: vary strategy within (operation, gnn)
        r_strat = d.groupby(["operation_str", "gnn_algorithm"]).relative_decision_change_count.agg(
            lambda x: x.max() - x.min())
        rows.append(dict(
            dataset=ds,
            range_operation_pp=100 * r_op.median(),
            range_gnn_pp=100 * r_gnn.median(),
            range_strategy_pp=100 * r_strat.median(),
            strategy_vs_operation_ratio=r_strat.median() / r_op.median() if r_op.median() else np.nan,
        ))
    res = pd.DataFrame(rows).sort_values("dataset")
    res.to_csv(OUT / "sig_factor_effect_magnitudes.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Q1: decision changes along edit paths (needs per-path data from all_results)
# ---------------------------------------------------------------------------
def load_paths():
    """Stream the 14 GB all_results.csv once and reduce it to per-path flips and
    per-fold classification accuracy.

    Returns ``(paths, accuracy)``.

    ``paths`` has one row per correctly-classified-endpoint path (is_correct_path
    == 1) with columns
        dataset, gnn_algorithm, path_strategy, val_id, path_idx_int,
        n_flips (# decision changes along the path),
        n_steps (# path operations = path length),
        same_label (1 if endpoints share the true label).

    ``accuracy`` has one row per (dataset, gnn_algorithm, val_id) with the
    held-out classification accuracy on that fold, computed from the unique
    validation *endpoint graphs* (the real dataset graphs at path ends), deduped
    by graph id so each graph is counted once.  Accuracy is independent of the
    path strategy (the model classifies each graph once), so the strategy is not
    part of the key.
    """
    src = RESULTS / "all_results.csv"
    usecols = ["dataset", "gnn_algorithm", "path_strategy", "val_id",
               "path_idx_int", "is_flipping", "is_path", "is_correct_path",
               "same_endpoint_labels_true",
               "is_source", "is_target", "source_id", "target_id",
               "is_validation", "is_correct"]
    acc = {}       # path key -> [flipsum, n_steps, samelabel]
    perf = {}      # (dataset, gnn, val_id, graph_id) -> is_correct (0/1)
    reader = pd.read_csv(src, usecols=usecols, chunksize=3_000_000)
    n_rows = 0
    for chunk in reader:
        chunk = chunk[chunk.dataset.isin(MOLECULAR_DATASETS)
                      & chunk.path_strategy.isin(STRATEGIES_SRC)]

        # --- per-fold accuracy: unique held-out endpoint graphs -------------
        ep = chunk[((chunk.is_source == 1) | (chunk.is_target == 1))
                   & (chunk.is_validation == 1)
                   & chunk.is_correct.isin([0, 1])]
        if len(ep):
            gid = np.where(ep.is_source == 1, ep.source_id, ep.target_id)
            for ds, gnn, vid, g, ok in zip(ep.dataset, ep.gnn_algorithm,
                                           ep.val_id, gid, ep.is_correct):
                perf[(ds, gnn, vid, g)] = ok  # deterministic per graph & fold

        # --- per-path flip counts (correct-endpoint paths only) ------------
        pth = chunk[(chunk.is_path == 1) & (chunk.is_correct_path == 1)]
        n_rows += len(pth)
        g = pth.groupby(["dataset", "gnn_algorithm", "path_strategy", "val_id",
                         "path_idx_int"]).agg(
            flips=("is_flipping", "sum"),
            steps=("is_flipping", "size"),
            same=("same_endpoint_labels_true", "max"))
        for key, row in zip(g.index, g.itertuples(index=False)):
            if key in acc:
                acc[key][0] += row.flips
                acc[key][1] += row.steps
                acc[key][2] = max(acc[key][2], row.same)
            else:
                acc[key] = [row.flips, row.steps, row.same]
    print(f"  processed {n_rows:,} path-step rows, {len(acc):,} unique paths, "
          f"{len(perf):,} unique (fold, endpoint graph) classifications")
    recs = [(*k, v[0], v[1], v[2]) for k, v in acc.items()]
    paths = pd.DataFrame(recs, columns=["dataset", "gnn_algorithm", "path_strategy",
                                        "val_id", "path_idx_int",
                                        "n_flips", "n_steps", "same_label"])
    paths["rel_flips"] = paths.n_flips / paths.n_steps.where(paths.n_steps > 0)

    perf_recs = [(ds, gnn, vid, g, ok) for (ds, gnn, vid, g), ok in perf.items()]
    perf_df = pd.DataFrame(perf_recs, columns=["dataset", "gnn_algorithm",
                                               "val_id", "graph_id", "is_correct"])
    accuracy = (perf_df.groupby(["dataset", "gnn_algorithm", "val_id"])
                .is_correct.agg(accuracy="mean", n_graphs="size").reset_index())
    return paths, accuracy


def q1_decision_changes(paths):
    # (a) distribution claim: fraction 0-flip (same-label) + 1-flip (diff-label)
    dist_rows = []
    for (ds, gnn, strat), cell in paths.groupby(["dataset", "gnn_algorithm", "path_strategy"]):
        same = cell[cell.same_label == 1]
        diff = cell[cell.same_label == 0]
        n = len(cell)
        frac_same0 = (same.n_flips == 0).sum() / n if n else np.nan
        frac_diff1 = (diff.n_flips == 1).sum() / n if n else np.nan
        dist_rows.append(dict(
            dataset=ds, gnn_algorithm=gnn, path_strategy=STRAT_LABEL.get(strat, strat),
            n_paths=n, n_same=len(same), n_diff=len(diff),
            frac_same_0flip=frac_same0, frac_diff_1flip=frac_diff1,
            frac_canonical=frac_same0 + frac_diff1,
            mean_flips_same=same.n_flips.mean(), mean_flips_diff=diff.n_flips.mean(),
        ))
    dist = pd.DataFrame(dist_rows).sort_values(["dataset", "gnn_algorithm", "path_strategy"])
    dist.to_csv(OUT / "sig_Q1_flip_distribution.csv", index=False)

    # (b) per-fold paired test: do different-label paths flip more than same-label?
    test_rows = []
    for (ds, gnn, strat), cell in paths.groupby(["dataset", "gnn_algorithm", "path_strategy"]):
        per_fold = cell.groupby(["val_id", "same_label"]).n_flips.mean().unstack("same_label")
        if 0 not in per_fold.columns or 1 not in per_fold.columns:
            continue
        per_fold = per_fold.dropna()
        if len(per_fold) < 3:
            continue
        diff_v = per_fold[0].values  # different-label endpoints
        same_v = per_fold[1].values  # same-label endpoints
        try:
            w_stat, p = stats.wilcoxon(diff_v, same_v)
        except ValueError:
            p, w_stat = np.nan, np.nan
        rb = rank_biserial(diff_v, same_v)
        test_rows.append(dict(
            dataset=ds, gnn_algorithm=gnn, path_strategy=STRAT_LABEL.get(strat, strat),
            n_folds=len(per_fold), test="wilcoxon_diff_vs_same_endpoint",
            mean_flips_diff=diff_v.mean(), mean_flips_same=same_v.mean(),
            mean_diff=(diff_v - same_v).mean(), statistic=w_stat, p_value=p,
            rank_biserial=rb,
        ))
    res = pd.DataFrame(test_rows)
    res["p_value_fdr"] = bh_fdr(res["p_value"].tolist())
    res["significant"] = res["p_value_fdr"] < ALPHA
    res = res.sort_values(["dataset", "gnn_algorithm", "path_strategy"])
    res.to_csv(OUT / "sig_Q1_endpoint_flip_test.csv", index=False)
    return dist, res


# ---------------------------------------------------------------------------
# Q1 (reframed): are there significant differences in the number of flips along
# paths BETWEEN ALGORITHMS?  Same CV partition is shared across architectures
# (same val_id -> same graphs and identical paths), so this is a repeated-
# measures comparison: Friedman across the five GNNs blocked by fold, per
# (dataset, strategy).  Because all models walk identical paths, the raw flip
# count per path is directly comparable; we test the per-fold mean.
# ---------------------------------------------------------------------------
def q1_algorithm_differences(paths):
    rows = []
    for (ds, strat), cell in paths.groupby(["dataset", "path_strategy"]):
        # per-fold mean number of flips per path, one column per architecture
        mat = _pivot(cell.groupby(["val_id", "gnn_algorithm"]).n_flips.mean().reset_index(),
                     "val_id", "gnn_algorithm", "n_flips")
        mat = mat[[g for g in GNNS if g in mat.columns]]
        if mat.shape[1] < 3 or mat.shape[0] < 3:
            continue
        groups = {c: mat[c].values for c in mat.columns}
        omni, posthoc = friedman_with_posthoc(groups, adjust="holm")
        means = mat.mean(axis=0)
        rows.append(dict(
            dataset=ds, path_strategy=STRAT_LABEL.get(strat, strat),
            n_folds=mat.shape[0], n_gnns=mat.shape[1],
            test="friedman_across_gnns", statistic=omni.statistic,
            p_value=omni.p_value, kendalls_w=omni.effect_size,
            most_flips=means.idxmax(), least_flips=means.idxmin(),
            max_minus_min=float(means.max() - means.min()),
            **{f"mean_flips_{g}": means.get(g, np.nan) for g in GNNS},
        ))
    res = pd.DataFrame(rows)
    res["p_value_fdr"] = bh_fdr(res["p_value"].tolist())
    res["significant"] = res["p_value_fdr"] < ALPHA
    res = res.sort_values(["dataset", "path_strategy"])
    res.to_csv(OUT / "sig_Q1_algorithm_differences.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Q1 (reframed): are there significant differences in the number of flips along
# paths BETWEEN DATASETS?  Datasets are independent (different graphs, separate
# CV partitions), so this is an unpaired comparison: Kruskal-Wallis across
# datasets per (gnn, strategy), with the CV fold as the unit of replication.
# Path length differs between datasets, so we compare the RELATIVE flip rate
# (flips per path operation) to remove the path-length confound.  Effect size
# is epsilon-squared.
# ---------------------------------------------------------------------------
def _epsilon_squared(h, n, k):
    """Epsilon-squared effect size for Kruskal-Wallis H (0..1)."""
    if not np.isfinite(h) or n <= k:
        return np.nan
    return float((h - k + 1) / (n - k))


def q1_dataset_differences(paths):
    rows = []
    # per-fold mean relative flip rate per (dataset, gnn, strategy, fold)
    per_fold = (paths.groupby(["dataset", "gnn_algorithm", "path_strategy", "val_id"])
                .rel_flips.mean().reset_index())
    for (gnn, strat), cell in per_fold.groupby(["gnn_algorithm", "path_strategy"]):
        groups, labels = [], []
        for ds, g in cell.groupby("dataset"):
            v = g.rel_flips.dropna().values
            if len(v) >= 3:                       # need >=3 folds per dataset
                groups.append(v)
                labels.append(ds)
        if len(groups) < 3:
            continue
        try:
            stat, p = stats.kruskal(*groups)
        except ValueError:
            continue
        n_total = sum(len(g) for g in groups)
        eps2 = _epsilon_squared(stat, n_total, len(groups))
        means = {lab: float(np.mean(g)) for lab, g in zip(labels, groups)}
        order = sorted(means, key=means.get, reverse=True)
        rows.append(dict(
            gnn_algorithm=gnn, path_strategy=STRAT_LABEL.get(strat, strat),
            n_datasets=len(groups), n_total_folds=n_total,
            test="kruskal_across_datasets", statistic=stat, p_value=p,
            epsilon_squared=eps2,
            most_flips=order[0], least_flips=order[-1],
            ranking=" > ".join(f"{d}({means[d]:.3f})" for d in order),
        ))
    res = pd.DataFrame(rows)
    res["p_value_fdr"] = bh_fdr(res["p_value"].tolist())
    res["significant"] = res["p_value_fdr"] < ALPHA
    res = res.sort_values(["gnn_algorithm", "path_strategy"])
    res.to_csv(OUT / "sig_Q1_dataset_differences.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Q4 (new): is classification PERFORMANCE associated with FLIP BEHAVIOUR?
#   No existing test relates how well a model classifies to how much / how it
#   flips along edit paths.  We pair per-fold held-out accuracy with three
#   per-fold flip metrics and test the association at two levels:
#     - fold level:  rmcorr within each (dataset, gnn, strategy) cell
#                    (does accuracy co-vary with flips across a model's folds?)
#     - architecture level: rmcorr within each dataset across architectures, and
#                    a plain Spearman across all (dataset, gnn) configurations
#                    (do better-classifying configs flip differently?).
# ---------------------------------------------------------------------------
FLIP_METRICS = {
    "mean_flips_per_path": "mean number of decision changes per path",
    "frac_canonical": "fraction of canonical paths (same&0-flip or diff&1-flip)",
    "mean_rel_flips": "mean flips per path operation (length-normalised)",
}


def _fold_flip_metrics(paths):
    """Per (dataset, gnn, strategy, fold) flip metrics, matching Q1's path set."""
    p = paths.copy()
    p["canonical"] = (((p.same_label == 1) & (p.n_flips == 0))
                      | ((p.same_label == 0) & (p.n_flips == 1))).astype(float)
    return (p.groupby(["dataset", "gnn_algorithm", "path_strategy", "val_id"])
            .agg(mean_flips_per_path=("n_flips", "mean"),
                 frac_canonical=("canonical", "mean"),
                 mean_rel_flips=("rel_flips", "mean"),
                 n_paths=("n_flips", "size")).reset_index())


def q4_performance_vs_flips(paths, accuracy):
    fold = _fold_flip_metrics(paths).merge(
        accuracy, on=["dataset", "gnn_algorithm", "val_id"], how="inner")
    fold["cell"] = (fold.dataset + "|" + fold.gnn_algorithm + "|" + fold.path_strategy)
    # architecture-level: average folds & strategies -> one point per (dataset, gnn)
    config = (fold.groupby(["dataset", "gnn_algorithm"])
              .agg(accuracy=("accuracy", "mean"),
                   **{m: (m, "mean") for m in FLIP_METRICS}).reset_index())

    rows = []
    for metric in FLIP_METRICS:
        # (1) fold-level rmcorr, holding the (dataset, gnn, strategy) cell fixed
        r, p, dof, n, k = rmcorr(fold.accuracy, fold[metric], fold.cell)
        rows.append(dict(level="fold", scope="within-cell (folds)", metric=metric,
                         test="rmcorr", r=r, p_value=p, dof=dof, n=n, n_groups=k))
        # (2) architecture-level rmcorr, holding the dataset fixed
        r, p, dof, n, k = rmcorr(config.accuracy, config[metric], config.dataset)
        rows.append(dict(level="architecture", scope="within-dataset (gnns)",
                         metric=metric, test="rmcorr", r=r, p_value=p,
                         dof=dof, n=n, n_groups=k))
        # (3) overall Spearman across the (dataset, gnn) configurations
        rho, p = stats.spearmanr(config.accuracy, config[metric])
        rows.append(dict(level="overall", scope="across configs (dataset x gnn)",
                         metric=metric, test="spearman", r=float(rho), p_value=float(p),
                         dof=np.nan, n=len(config), n_groups=np.nan))
    res = pd.DataFrame(rows)
    res["p_value_fdr"] = bh_fdr(res["p_value"].tolist())
    res["significant"] = res["p_value_fdr"] < ALPHA
    res.to_csv(OUT / "sig_Q4_performance_vs_flips.csv", index=False)
    return res, fold, config


def q4_performance_vs_flips_per_operation(accuracy):
    """Per operation type: does accuracy track that operation's flip rate?"""
    f = (RESULTS / "Experiments" / "experiments_flips_per_operation_relative"
         / "flips_per_operation_relative_fold_counts.csv")
    df = pd.read_csv(f)
    df = _restrict(df[df.split == SPLIT])
    df = df.merge(accuracy, on=["dataset", "gnn_algorithm", "val_id"], how="inner")
    df["cell"] = df.dataset + "|" + df.gnn_algorithm + "|" + df.path_strategy
    rows = []
    for op, cell in df.groupby("operation_str"):
        # fold level: rmcorr within (dataset, gnn, strategy)
        r, p, dof, n, k = rmcorr(cell.accuracy, cell.relative_decision_change_count,
                                 cell.cell)
        # architecture level: per (dataset, gnn) mean, rmcorr within dataset
        conf = (cell.groupby(["dataset", "gnn_algorithm"])
                .agg(accuracy=("accuracy", "mean"),
                     rate=("relative_decision_change_count", "mean")).reset_index())
        r2, p2, dof2, n2, k2 = rmcorr(conf.accuracy, conf.rate, conf.dataset)
        rho, prho = stats.spearmanr(conf.accuracy, conf.rate)
        rows.append(dict(
            operation=op,
            fold_r=r, fold_p=p, fold_dof=dof, fold_n=n,
            arch_r=r2, arch_p=p2, arch_dof=dof2, arch_n=n2,
            overall_spearman_r=float(rho), overall_spearman_p=float(prho),
            n_configs=len(conf)))
    res = pd.DataFrame(rows)
    # FDR within each test family across the operations
    res["fold_p_fdr"] = bh_fdr(res["fold_p"].tolist())
    res["arch_p_fdr"] = bh_fdr(res["arch_p"].tolist())
    res["fold_significant"] = res["fold_p_fdr"] < ALPHA
    res["arch_significant"] = res["arch_p_fdr"] < ALPHA
    res = res.sort_values("operation")
    res.to_csv(OUT / "sig_Q4_performance_vs_flips_per_operation.csv", index=False)
    return res


def main():
    print("Q2: GNN differences per operation type (l.51) ...")
    q2 = q2_gnn_differences_per_operation()
    print(f"  {q2.significant.sum()}/{len(q2)} cells significant (FDR)")
    print("Q3: path-strategy effect per operation (l.49) ...")
    q3 = q3_strategy_effect_per_operation()
    print(f"  {q3.significant.sum()}/{len(q3)} cells significant (FDR)")
    print("Factor effect sizes (operation vs gnn vs strategy) ...")
    fe = factor_effect_sizes()
    print(fe.to_string(index=False))
    print("Q1: streaming all_results.csv to per-path flips + per-fold accuracy ...")
    paths, accuracy = load_paths()
    print("Q1a: different- vs same-label endpoint flips ...")
    dist, q1 = q1_decision_changes(paths)
    print(f"  {q1.significant.sum()}/{len(q1)} cells significant (FDR)")
    print("Q1b: flip-count differences BETWEEN ALGORITHMS (Friedman/fold) ...")
    q1_alg = q1_algorithm_differences(paths)
    print(f"  {q1_alg.significant.sum()}/{len(q1_alg)} (dataset,strategy) cells significant (FDR)")
    print(q1_alg[["dataset", "path_strategy", "kendalls_w", "most_flips",
                  "least_flips", "p_value_fdr", "significant"]].to_string(index=False))
    print("Q1c: flip-count differences BETWEEN DATASETS (Kruskal-Wallis/fold) ...")
    q1_ds = q1_dataset_differences(paths)
    print(f"  {q1_ds.significant.sum()}/{len(q1_ds)} (gnn,strategy) cells significant (FDR)")
    print(q1_ds[["gnn_algorithm", "path_strategy", "epsilon_squared", "most_flips",
                 "least_flips", "p_value_fdr", "significant"]].to_string(index=False))
    print("Q4: classification performance vs flip behaviour ...")
    q4, q4_fold, q4_config = q4_performance_vs_flips(paths, accuracy)
    print(f"  per-fold accuracy: {len(accuracy)} (dataset,gnn,fold) cells; "
          f"mean acc {accuracy.accuracy.mean():.3f}")
    print(f"  {q4.significant.sum()}/{len(q4)} association tests significant (FDR)")
    print(q4[["level", "metric", "test", "r", "p_value_fdr",
              "significant"]].to_string(index=False))
    print("Q4b: performance vs per-operation flip rate ...")
    q4op = q4_performance_vs_flips_per_operation(accuracy)
    print(f"  fold-level: {q4op.fold_significant.sum()}/{len(q4op)} operations significant (FDR);"
          f" arch-level: {q4op.arch_significant.sum()}/{len(q4op)} significant (FDR)")
    print(q4op[["operation", "fold_r", "fold_p_fdr", "fold_significant",
                "arch_r", "arch_p_fdr", "arch_significant"]].to_string(index=False))
    print("Done. CSVs in", OUT)


if __name__ == "__main__":
    main()
