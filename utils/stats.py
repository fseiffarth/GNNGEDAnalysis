"""Reusable statistical helpers for the validation layer.

These functions back ``statistical_validation.py``. They operate on plain
sequences / numpy arrays so they can be unit-tested in isolation without any of
the experiment-output plumbing.

Conventions
-----------
* Every test helper returns a small dataclass with at least a ``statistic`` and a
  ``p_value`` field (``p_value`` may be ``float('nan')`` when a test is not
  applicable, e.g. too few groups / replication units).
* Effect-size helpers return plain floats.
* ``cluster_bootstrap_ci`` resamples the *replication units* (folds, or paths for
  single-split data) with replacement and is the single source of confidence
  intervals for point estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Result of an omnibus or paired test."""

    test: str
    statistic: float
    p_value: float
    n: int
    effect_size: float = float("nan")
    effect_size_name: str = ""
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class PosthocResult:
    """One pairwise post-hoc comparison."""

    group_a: str
    group_b: str
    test: str
    statistic: float
    p_value: float
    p_value_adjusted: float
    n: int
    effect_size: float
    effect_size_name: str


# ---------------------------------------------------------------------------
# Multiple-comparison corrections
# ---------------------------------------------------------------------------


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR adjusted p-values.

    NaN p-values are passed through unchanged and excluded from the ranking.
    Returns a list aligned with the input order; adjusted values are clipped to
    ``[0, 1]`` and enforced monotone non-decreasing in rank order.
    """

    p = np.asarray(list(p_values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = ~np.isnan(p)
    valid = p[mask]
    m = valid.size
    if m == 0:
        return out.tolist()

    order = np.argsort(valid, kind="mergesort")
    ranked = valid[order]
    ranks = np.arange(1, m + 1)
    adj = ranked * m / ranks
    # enforce monotonicity from the largest p downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    adjusted_valid = np.empty(m, dtype=float)
    adjusted_valid[order] = adj
    out[mask] = adjusted_valid
    return out.tolist()


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (step-down).

    NaN p-values are passed through and excluded from the family size.
    """

    p = np.asarray(list(p_values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = ~np.isnan(p)
    valid = p[mask]
    m = valid.size
    if m == 0:
        return out.tolist()

    order = np.argsort(valid, kind="mergesort")
    ranked = valid[order]
    factors = np.arange(m, 0, -1)
    adj = ranked * factors
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0.0, 1.0)

    adjusted_valid = np.empty(m, dtype=float)
    adjusted_valid[order] = adj
    out[mask] = adjusted_valid
    return out.tolist()


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def kendalls_w(blocks: Sequence[Sequence[float]]) -> float:
    """Kendall's W (coefficient of concordance) for a Friedman layout.

    ``blocks`` is a sequence of blocks (folds); each block holds one value per
    treatment, in a consistent treatment order. Returns a value in ``[0, 1]``.
    Ties are handled via average ranks with the standard tie correction.
    """

    matrix = np.asarray(blocks, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("blocks must be a 2D array-like (block x treatment)")
    n, k = matrix.shape  # n blocks (raters), k treatments (items)
    if n < 1 or k < 2:
        return float("nan")

    # Rank treatments within each block.
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    s = np.sum((rank_sums - rank_sums.mean()) ** 2)

    # Tie correction term.
    tie_term = 0.0
    for block in ranks:
        _, counts = np.unique(block, return_counts=True)
        tie_term += np.sum(counts**3 - counts)

    denom = (n**2) * (k**3 - k) - n * tie_term
    if denom <= 0:
        return float("nan")
    return float(12.0 * s / denom)


def rank_biserial(x: Sequence[float], y: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test.

    Defined as ``(W+ - W-) / (W+ + W-)`` on the ranks of non-zero differences.
    Sign convention: positive means ``x > y`` tends to hold. Range ``[-1, 1]``.
    """

    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[diff != 0]
    if diff.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    total = r_plus + r_minus
    if total == 0:
        return float("nan")
    return float((r_plus - r_minus) / total)


# ---------------------------------------------------------------------------
# Omnibus / paired tests
# ---------------------------------------------------------------------------


def friedman_with_posthoc(
    data_by_group: Dict[str, Sequence[float]],
    adjust: str = "holm",
) -> Tuple[TestResult, List[PosthocResult]]:
    """Friedman omnibus test plus pairwise Wilcoxon post-hoc comparisons.

    ``data_by_group`` maps a treatment label to its per-block (per-fold) values.
    All groups must be the same length (paired by block / fold). Groups are
    aligned by position: index ``i`` across every group is block ``i``.

    Returns ``(TestResult, [PosthocResult, ...])``. Post-hoc p-values are
    adjusted within this family using ``adjust`` (``"holm"`` or ``"bh"``). When
    fewer than 3 groups or fewer than 2 complete blocks exist, the omnibus
    p-value is NaN and no post-hoc tests are produced.
    """

    labels = list(data_by_group.keys())
    arrays = [np.asarray(data_by_group[label], dtype=float) for label in labels]

    # Keep only blocks that are complete (no NaN) across every group.
    if arrays:
        lengths = {a.size for a in arrays}
        if len(lengths) != 1:
            raise ValueError("all groups must have the same number of blocks")
    stacked = np.column_stack(arrays) if arrays else np.empty((0, 0))
    if stacked.size:
        complete = ~np.isnan(stacked).any(axis=1)
        stacked = stacked[complete]

    k = len(labels)
    n = stacked.shape[0]

    if k < 3 or n < 2:
        result = TestResult(
            test="friedman",
            statistic=float("nan"),
            p_value=float("nan"),
            n=n,
            effect_size=kendalls_w(stacked) if n >= 1 and k >= 2 else float("nan"),
            effect_size_name="kendalls_w",
        )
        return result, []

    columns = [stacked[:, i] for i in range(k)]
    statistic, p_value = stats.friedmanchisquare(*columns)
    result = TestResult(
        test="friedman",
        statistic=float(statistic),
        p_value=float(p_value),
        n=n,
        effect_size=kendalls_w(stacked),
        effect_size_name="kendalls_w",
    )

    # Pairwise post-hoc Wilcoxon signed-rank.
    raw: List[PosthocResult] = []
    raw_p: List[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            pair = wilcoxon_paired(columns[i], columns[j])
            raw.append(
                PosthocResult(
                    group_a=labels[i],
                    group_b=labels[j],
                    test=pair.test,
                    statistic=pair.statistic,
                    p_value=pair.p_value,
                    p_value_adjusted=float("nan"),
                    n=pair.n,
                    effect_size=pair.effect_size,
                    effect_size_name=pair.effect_size_name,
                )
            )
            raw_p.append(pair.p_value)

    adjusted = holm_correction(raw_p) if adjust == "holm" else bh_fdr(raw_p)
    for item, adj_p in zip(raw, adjusted):
        item.p_value_adjusted = adj_p
    return result, raw


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> TestResult:
    """Wilcoxon signed-rank test for paired samples with rank-biserial effect.

    Pairs containing NaN are dropped. If every difference is zero or fewer than
    one non-zero pair remains, the p-value is NaN.
    """

    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    nonzero = diff[diff != 0]
    n = int(a.size)

    effect = rank_biserial(a, b)
    if nonzero.size < 1:
        return TestResult(
            test="wilcoxon",
            statistic=float("nan"),
            p_value=float("nan"),
            n=n,
            effect_size=effect,
            effect_size_name="rank_biserial",
        )
    try:
        statistic, p_value = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        statistic, p_value = float("nan"), float("nan")
    return TestResult(
        test="wilcoxon",
        statistic=float(statistic),
        p_value=float(p_value),
        n=n,
        effect_size=effect,
        effect_size_name="rank_biserial",
        extra={"mean_difference": float(np.mean(diff)) if diff.size else float("nan")},
    )


def page_trend_test(
    blocks: Sequence[Sequence[float]],
    ascending: bool = True,
) -> TestResult:
    """Page's L trend test for an ordered alternative across treatments.

    ``blocks`` is block x treatment (fold x ordered-bin); columns must already be
    in the hypothesised order. ``ascending=True`` tests for an increasing trend
    across columns; set ``ascending=False`` to test a decreasing trend (the
    columns are reversed internally). Uses the normal approximation for the
    p-value (valid for the fold counts here).
    """

    matrix = np.asarray(blocks, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("blocks must be 2D (block x treatment)")
    # Drop incomplete blocks.
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    n, k = matrix.shape
    if n < 1 or k < 3:
        return TestResult(
            test="page_trend",
            statistic=float("nan"),
            p_value=float("nan"),
            n=n,
            effect_size=float("nan"),
            effect_size_name="page_z",
        )

    if not ascending:
        matrix = matrix[:, ::-1]

    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    weights = np.arange(1, k + 1)
    L = float(np.sum(weights * rank_sums))

    # Normal approximation (Page, 1963).
    mean_L = n * k * (k + 1) ** 2 / 4.0
    var_L = n * k**2 * (k + 1) * (k**2 - 1) / 144.0
    if var_L <= 0:
        return TestResult(
            test="page_trend",
            statistic=L,
            p_value=float("nan"),
            n=n,
            effect_size=float("nan"),
            effect_size_name="page_z",
        )
    z = (L - mean_L) / math.sqrt(var_L)
    p_value = float(stats.norm.sf(z))  # one-sided: trend in the hypothesised direction
    return TestResult(
        test="page_trend",
        statistic=L,
        p_value=p_value,
        n=n,
        effect_size=float(z),
        effect_size_name="page_z",
    )


def one_sample_wilcoxon(values: Sequence[float], popmean: float = 0.0) -> TestResult:
    """One-sample Wilcoxon signed-rank test of ``values`` against ``popmean``."""

    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    diff = a - popmean
    nonzero = diff[diff != 0]
    n = int(a.size)
    effect = rank_biserial(a, np.full_like(a, popmean))
    if nonzero.size < 1:
        return TestResult(
            test="wilcoxon_one_sample",
            statistic=float("nan"),
            p_value=float("nan"),
            n=n,
            effect_size=effect,
            effect_size_name="rank_biserial",
        )
    try:
        statistic, p_value = stats.wilcoxon(diff, zero_method="wilcox")
    except ValueError:
        statistic, p_value = float("nan"), float("nan")
    return TestResult(
        test="wilcoxon_one_sample",
        statistic=float(statistic),
        p_value=float(p_value),
        n=n,
        effect_size=effect,
        effect_size_name="rank_biserial",
        extra={"median": float(np.median(a)) if a.size else float("nan")},
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def cluster_bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_reps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI by resampling replication units with replacement.

    ``values`` are the per-unit observations (one per fold, or one per path for
    single-split data). Returns ``(point_estimate, ci_low, ci_high)``. With fewer
    than 2 units the CI bounds are NaN but the point estimate is still returned.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(arr))
    if arr.size < 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_reps, arr.size))
    samples = arr[idx]
    reps = np.array([statistic(row) for row in samples], dtype=float)
    lo = float(np.percentile(reps, 100 * (alpha / 2)))
    hi = float(np.percentile(reps, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_diff_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_reps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the paired mean difference ``mean(x - y)``."""

    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    diff = a[mask] - b[mask]
    return cluster_bootstrap_ci(diff, np.mean, n_reps=n_reps, alpha=alpha, seed=seed)
