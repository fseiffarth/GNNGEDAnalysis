"""Diagnostic: is the Q4 *fold-level* accuracy<->flips correlation a
per-fold composition / selection artefact?

The Q4 fold-level rmcorr pairs, within each (dataset, gnn, strategy) cell, the
per-fold held-out accuracy with per-fold flip metrics.  Two reasons it might be
spurious rather than a genuine "more accurate -> more flips" link:

  (A) SELECTION.  The flip metrics in answer_significance_questions.py are
      computed only on correct-endpoint paths (is_correct_path == 1).  A more
      accurate fold has more correct endpoints, so a different/larger set of
      paths enters the metric.

  (B) COMPOSITION.  Different-label-endpoint paths flip more (Q1) and longer
      paths flip more (raw count).  If accuracy co-varies with the per-fold mix
      of same/different-label paths or with mean path length, the flip
      correlation is a Simpson-style confound, not a performance effect.

This script streams all_results.csv ONCE keeping ALL paths (is_path == 1) with
an is_correct_path flag and the path length, caches the per-path table to the
scratchpad as parquet, and then runs:

  1. composition vs accuracy        (rmcorr accuracy <-> frac_diff / mean_steps / n_paths)
  2. selection test                 (accuracy<->flips on correct-endpoint paths vs ALL paths)
  3. stratified test                (accuracy<->flips within same-label / diff-label paths)
  4. partial rmcorr                 (accuracy<->flips controlling for frac_diff + mean_steps)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from answer_significance_questions import (
    rmcorr, MOLECULAR_DATASETS, STRATEGIES_SRC, RESULTS, OUT,
)

CACHE = Path("/tmp/claude-1000/-home-florian-Documents-Work-Forschung-EigeneForschung-2026-DS-GNN-GED"
             "/1e5fc180-403f-40fb-8fc3-d21beeda671d/scratchpad")
CACHE.mkdir(parents=True, exist_ok=True)
PATHS_PARQUET = CACHE / "q4_all_paths.pkl"
ACC_PARQUET = CACHE / "q4_accuracy.pkl"


def stream_all_paths():
    """All paths (not just correct-endpoint) + per-fold accuracy, single pass."""
    src = RESULTS / "all_results.csv"
    usecols = ["dataset", "gnn_algorithm", "path_strategy", "val_id",
               "path_idx_int", "is_flipping", "is_path", "is_correct_path",
               "same_endpoint_labels_true",
               "is_source", "is_target", "source_id", "target_id",
               "is_validation", "is_correct"]
    acc = {}   # path key -> [flipsum, n_steps, samelabel, is_correct_path]
    perf = {}  # (dataset, gnn, val_id, graph_id) -> is_correct
    reader = pd.read_csv(src, usecols=usecols, chunksize=3_000_000)
    n_rows = 0
    for chunk in reader:
        chunk = chunk[chunk.dataset.isin(MOLECULAR_DATASETS)
                      & chunk.path_strategy.isin(STRATEGIES_SRC)]
        ep = chunk[((chunk.is_source == 1) | (chunk.is_target == 1))
                   & (chunk.is_validation == 1)
                   & chunk.is_correct.isin([0, 1])]
        if len(ep):
            gid = np.where(ep.is_source == 1, ep.source_id, ep.target_id)
            for ds, gnn, vid, g, ok in zip(ep.dataset, ep.gnn_algorithm,
                                           ep.val_id, gid, ep.is_correct):
                perf[(ds, gnn, vid, g)] = ok

        pth = chunk[chunk.is_path == 1]          # <-- ALL paths, no correctness filter
        n_rows += len(pth)
        g = pth.groupby(["dataset", "gnn_algorithm", "path_strategy", "val_id",
                         "path_idx_int"]).agg(
            flips=("is_flipping", "sum"),
            steps=("is_flipping", "size"),
            same=("same_endpoint_labels_true", "max"),
            corr=("is_correct_path", "max"))
        for key, row in zip(g.index, g.itertuples(index=False)):
            if key in acc:
                acc[key][0] += row.flips
                acc[key][1] += row.steps
                acc[key][2] = max(acc[key][2], row.same)
                acc[key][3] = max(acc[key][3], row.corr)
            else:
                acc[key] = [row.flips, row.steps, row.same, row.corr]
    print(f"  processed {n_rows:,} path-step rows, {len(acc):,} unique paths "
          f"(ALL endpoints), {len(perf):,} endpoint-graph classifications")
    recs = [(*k, v[0], v[1], v[2], v[3]) for k, v in acc.items()]
    paths = pd.DataFrame(recs, columns=["dataset", "gnn_algorithm", "path_strategy",
                                        "val_id", "path_idx_int", "n_flips",
                                        "n_steps", "same_label", "is_correct_path"])
    paths["rel_flips"] = paths.n_flips / paths.n_steps.where(paths.n_steps > 0)
    perf_recs = [(ds, gnn, vid, g, ok) for (ds, gnn, vid, g), ok in perf.items()]
    perf_df = pd.DataFrame(perf_recs, columns=["dataset", "gnn_algorithm",
                                               "val_id", "graph_id", "is_correct"])
    accuracy = (perf_df.groupby(["dataset", "gnn_algorithm", "val_id"])
                .is_correct.agg(accuracy="mean", n_graphs="size").reset_index())
    return paths, accuracy


def get_data():
    if PATHS_PARQUET.exists() and ACC_PARQUET.exists():
        print("loading cached per-path table ...")
        return pd.read_pickle(PATHS_PARQUET), pd.read_pickle(ACC_PARQUET)
    print("streaming all_results.csv (one pass) ...")
    paths, accuracy = stream_all_paths()
    paths.to_pickle(PATHS_PARQUET)
    accuracy.to_pickle(ACC_PARQUET)
    return paths, accuracy


def _fold_table(paths, accuracy, subset):
    """Per (dataset, gnn, strategy, fold) flip metrics + composition + accuracy."""
    p = paths
    if subset == "correct":
        p = p[p.is_correct_path == 1]
    elif subset == "diff":
        p = p[(p.is_correct_path == 1) & (p.same_label == 0)]
    elif subset == "same":
        p = p[(p.is_correct_path == 1) & (p.same_label == 1)]
    # subset == "all" keeps every path
    p = p.copy()
    p["canonical"] = (((p.same_label == 1) & (p.n_flips == 0))
                      | ((p.same_label == 0) & (p.n_flips == 1))).astype(float)
    fold = (p.groupby(["dataset", "gnn_algorithm", "path_strategy", "val_id"])
            .agg(mean_flips_per_path=("n_flips", "mean"),
                 frac_canonical=("canonical", "mean"),
                 mean_rel_flips=("rel_flips", "mean"),
                 mean_steps=("n_steps", "mean"),
                 frac_diff=("same_label", lambda s: float((s == 0).mean())),
                 n_paths=("n_flips", "size")).reset_index())
    fold = fold.merge(accuracy, on=["dataset", "gnn_algorithm", "val_id"], how="inner")
    fold["cell"] = fold.dataset + "|" + fold.gnn_algorithm + "|" + fold.path_strategy
    return fold


def partial_rmcorr(x, y, covars, groups):
    """rmcorr of x,y within `groups`, additionally partialling out covariates.

    Within-group-center everything, regress the centred x and y on the centred
    covariates, then correlate the residuals.  df = n - k - 1 - n_covars.
    """
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float),
                      "g": np.asarray(groups)})
    for i, c in enumerate(covars):
        d[f"c{i}"] = np.asarray(c, float)
    d = d.dropna()
    d = d[d.groupby("g")["g"].transform("size") >= 2]
    ccols = [f"c{i}" for i in range(len(covars))]
    for col in ["x", "y"] + ccols:
        d[col] = d[col] - d.groupby("g")[col].transform("mean")
    n, k = len(d), d["g"].nunique()
    dof = n - k - 1 - len(covars)
    if dof < 3:
        return np.nan, np.nan, dof, n, k
    C = np.column_stack([d[c].values for c in ccols]) if ccols else np.empty((n, 0))
    def resid(v):
        if C.shape[1] == 0:
            return v
        beta, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ beta
    rx, ry = resid(d["x"].values), resid(d["y"].values)
    sx, sy = np.sqrt((rx ** 2).sum()), np.sqrt((ry ** 2).sum())
    if sx == 0 or sy == 0:
        return np.nan, np.nan, dof, n, k
    r = float((rx * ry).sum() / (sx * sy))
    r = max(min(r, 1.0), -1.0)
    t = r * np.sqrt(dof / (1 - r * r)) if abs(r) < 1 else np.inf
    p = float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else 0.0
    return r, p, dof, n, k


METRICS = ["mean_flips_per_path", "frac_canonical", "mean_rel_flips"]


def main():
    paths, accuracy = get_data()
    corr = _fold_table(paths, accuracy, "correct")
    alln = _fold_table(paths, accuracy, "all")
    diff = _fold_table(paths, accuracy, "diff")
    same = _fold_table(paths, accuracy, "same")
    print(f"\ncorrect-endpoint paths: {paths.is_correct_path.mean():.3f} of "
          f"{len(paths):,}; fold cells: {len(corr)}")

    rows = []

    # 1. composition vs accuracy (within cell)
    print("\n[1] Does accuracy co-vary with fold COMPOSITION? (rmcorr within cell, correct-endpoint set)")
    for comp in ["frac_diff", "mean_steps", "n_paths"]:
        r, p, dof, n, k = rmcorr(corr.accuracy, corr[comp], corr.cell)
        print(f"  accuracy <-> {comp:11s}: r={r:+.3f}  p={p:.2e}  (n={n}, df={dof})")
        rows.append(dict(test="composition", target=comp, subset="correct",
                         r=r, p_value=p, dof=dof, n=n))

    # 2. selection test: accuracy<->flips on correct-endpoint vs ALL paths
    print("\n[2] SELECTION: accuracy<->flips on correct-endpoint paths vs on ALL paths")
    for metric in METRICS:
        rc, pc, dc, nc, kc = rmcorr(corr.accuracy, corr[metric], corr.cell)
        ra, pa, da, na, ka = rmcorr(alln.accuracy, alln[metric], alln.cell)
        print(f"  {metric:20s}: correct r={rc:+.3f} (p={pc:.1e}) | "
              f"ALL r={ra:+.3f} (p={pa:.1e})")
        rows.append(dict(test="selection", target=metric, subset="correct",
                         r=rc, p_value=pc, dof=dc, n=nc))
        rows.append(dict(test="selection", target=metric, subset="all",
                         r=ra, p_value=pa, dof=da, n=na))

    # 3. stratified within same/diff label (correct-endpoint set)
    print("\n[3] STRATIFIED: accuracy<->flips within same-label and diff-label paths")
    for metric in METRICS:
        rd, pd_, dd, nd, kd = rmcorr(diff.accuracy, diff[metric], diff.cell)
        rs, ps, ds_, ns, ks = rmcorr(same.accuracy, same[metric], same.cell)
        print(f"  {metric:20s}: diff-label r={rd:+.3f} (p={pd_:.1e}) | "
              f"same-label r={rs:+.3f} (p={ps:.1e})")
        rows.append(dict(test="stratified", target=metric, subset="diff",
                         r=rd, p_value=pd_, dof=dd, n=nd))
        rows.append(dict(test="stratified", target=metric, subset="same",
                         r=rs, p_value=ps, dof=ds_, n=ns))

    # 4. partial rmcorr controlling for composition (correct-endpoint set)
    print("\n[4] PARTIAL: accuracy<->flips controlling for [frac_diff, mean_steps]")
    for metric in METRICS:
        r0, p0, *_ = rmcorr(corr.accuracy, corr[metric], corr.cell)
        r, p, dof, n, k = partial_rmcorr(
            corr.accuracy, corr[metric], [corr.frac_diff, corr.mean_steps], corr.cell)
        print(f"  {metric:20s}: raw r={r0:+.3f} -> partial r={r:+.3f} (p={p:.1e})")
        rows.append(dict(test="partial", target=metric, subset="correct",
                         r=r, p_value=p, dof=dof, n=n))

    res = pd.DataFrame(rows)
    out = OUT / "sig_Q4_fold_confound_check.csv"
    res.to_csv(out, index=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
