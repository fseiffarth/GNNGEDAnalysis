# Documentation

## Statistical validation

The descriptive flip findings produced by `experiments.py` and
`experiments_additional.py` (per-fold counts and `mean/std` summaries) carry no
significance testing, no confidence intervals, and no handling of the nested data
structure (edit step → path → CV fold → trained model). Treating each step as an
independent observation would massively overstate significance. The
`statistical_validation.py` CLI adds a validation layer that turns those numbers
into defensible claims with p-values, multiple-comparison control, effect sizes,
and confidence intervals — respecting the nested structure where **the CV fold is
the unit of replication**. See `STATISTICAL_VALIDATION_PLAN.md` for the design.

### Layout

- `utils/stats.py` — reusable, unit-tested helpers: `friedman_with_posthoc`,
  `wilcoxon_paired`, `one_sample_wilcoxon`, `page_trend_test`, `kendalls_w`,
  `rank_biserial`, `cluster_bootstrap_ci`, `paired_diff_bootstrap_ci`, `bh_fdr`,
  `holm_correction`.
- `statistical_validation.py` — `click` CLI mirroring `experiments.py`.
- `results/StatisticalValidation/` — one tidy CSV per finding family plus
  `report.md`.
- `tests/test_stats.py` — unit tests for the helpers.

### Two tiers

**Tier 1 — fold-level nonparametric tests (headline claims).** Consumes the
per-fold `*_fold_counts.csv` / `*_fold.csv` files already on disk; never
recomputes flip aggregations. Per `(dataset, gnn, strategy, split)` cell it runs:

1. **Operation-type flip effects** — Friedman across operation types blocked by
   fold on `relative_decision_change_count`; Kendall's W effect size; pairwise
   Wilcoxon signed-rank post-hoc.
2. **GNN architecture differences** — Friedman across GNNs blocked by fold on the
   overall per-fold relative flip rate; pairwise Wilcoxon post-hoc.
3. **Margin vs flips** — Page's trend test for a monotone *decreasing* flip rate
   across increasing-margin bins, plus a paired Wilcoxon of non-flipping vs
   flipping mean margin (with a bootstrap CI on the paired difference).
4. **Sequence/position effects** — Friedman across early/mid/late position bins;
   one-sample Wilcoxon of per-fold `log(enrichment_ratio)` vs 0 per transition
   bigram (filtered by `--min-transition-count`); Friedman of per-fold recovery
   rate across GNNs.

**Tier 2 — GEE / GLMM confirmatory model.** Streams the per-step rows of
`results/all_results.csv` and fits a population-averaged logistic GEE
`is_flipping ~ C(operation_str)` grouped on a composite path id
(`strategy:val_id:path_idx_int`) with cluster-robust SEs and an independence
working correlation. Output per dataset/gnn: operation odds-ratios, 95% CIs, and
FDR-adjusted p-values, with the reference operation noted. `--glmm-max-paths`
subsamples whole paths for tractability on the multi-million-row CSV.

### Cross-cutting mechanics

- **Multiple comparisons:** every raw p-value within a finding family (one output
  CSV) is BH-FDR adjusted into the `p_value_fdr` column; `significant` is
  `p_value_fdr < --alpha`. Within-cell post-hoc families additionally use Holm.
- **Effect sizes + CIs:** Kendall's W (Friedman), rank-biserial (Wilcoxon), and
  percentile **cluster bootstrap** CIs (default 2000 reps, resampling folds; paths
  for single-split data) for every reported point estimate.
- **Guard rails:** cells with fewer than 5 folds report a point estimate +
  bootstrap CI but no p-value; Friedman is never emitted on fewer than 3 groups.
  Single-split datasets (e.g. ogbg-molhiv) automatically skip fold-level p-values
  and fall back to bootstrap CIs + the GEE.

### Tidy output schema

Each Tier-1 CSV row:
`dataset, gnn_algorithm, path_strategy, split, finding, comparison, test, n,
statistic, p_value, p_value_fdr, effect_size, effect_size_name, ci_low, ci_high,
significant`.

### Usage

```bash
# Install the analysis dependencies (into the shared GNNGED venv).
../GNNGED/venv/bin/python -m pip install -r requirements.txt

# Tier 1 on one dataset.
../GNNGED/venv/bin/python statistical_validation.py --dataset DHFR --tier fold

# Tier 2 (GEE) on one dataset; restrict GNNs to keep it fast.
../GNNGED/venv/bin/python statistical_validation.py --dataset MUTAG --tier glmm --gnn GIN

# Both tiers, validation-split scope.
../GNNGED/venv/bin/python statistical_validation.py --dataset DHFR --tier all --split validation
```

Outputs land in `results/StatisticalValidation/` (`operation_effects_tests.csv`,
`gnn_difference_tests.csv`, `margin_to_flip_tests.csv`,
`sequence_position_tests.csv`, `glmm_operation_<dataset>_<gnn>.csv`, `report.md`).
