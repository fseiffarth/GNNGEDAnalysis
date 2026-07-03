import math
import unittest

import numpy as np
from scipy import stats

from utils.stats import (
    bh_fdr,
    cluster_bootstrap_ci,
    friedman_with_posthoc,
    holm_correction,
    kendalls_w,
    one_sample_wilcoxon,
    page_trend_test,
    paired_diff_bootstrap_ci,
    rank_biserial,
    wilcoxon_paired,
)


class TestCorrections(unittest.TestCase):
    def test_bh_fdr_matches_statsmodels(self):
        try:
            from statsmodels.stats.multitest import multipletests
        except Exception:  # pragma: no cover
            self.skipTest("statsmodels not available")
        p = [0.001, 0.008, 0.039, 0.041, 0.9]
        expected = multipletests(p, method="fdr_bh")[1]
        got = bh_fdr(p)
        np.testing.assert_allclose(got, expected, rtol=1e-9)

    def test_holm_matches_statsmodels(self):
        try:
            from statsmodels.stats.multitest import multipletests
        except Exception:  # pragma: no cover
            self.skipTest("statsmodels not available")
        p = [0.001, 0.008, 0.039, 0.041, 0.9]
        expected = multipletests(p, method="holm")[1]
        got = holm_correction(p)
        np.testing.assert_allclose(got, expected, rtol=1e-9)

    def test_nan_passthrough(self):
        p = [0.01, float("nan"), 0.5]
        got = bh_fdr(p)
        self.assertTrue(math.isnan(got[1]))
        self.assertFalse(math.isnan(got[0]))
        # family size excludes the NaN -> m=2
        self.assertAlmostEqual(got[0], min(0.01 * 2 / 1, 0.5 * 2 / 2), places=9)

    def test_empty(self):
        self.assertEqual(bh_fdr([]), [])
        self.assertEqual(holm_correction([]), [])


class TestEffectSizes(unittest.TestCase):
    def test_kendalls_w_perfect_concordance(self):
        # Every block ranks the treatments in the same order -> W == 1.
        blocks = [[1, 2, 3], [4, 5, 6], [0, 10, 20]]
        self.assertAlmostEqual(kendalls_w(blocks), 1.0, places=9)

    def test_kendalls_w_range(self):
        rng = np.random.default_rng(0)
        blocks = rng.normal(size=(8, 4))
        w = kendalls_w(blocks)
        self.assertGreaterEqual(w, 0.0)
        self.assertLessEqual(w, 1.0)

    def test_rank_biserial_all_positive(self):
        x = [5, 6, 7, 8]
        y = [1, 2, 3, 4]
        self.assertAlmostEqual(rank_biserial(x, y), 1.0, places=9)

    def test_rank_biserial_all_negative(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]
        self.assertAlmostEqual(rank_biserial(x, y), -1.0, places=9)


class TestPairedTests(unittest.TestCase):
    def test_wilcoxon_matches_scipy(self):
        x = [1.2, 2.4, 3.1, 4.8, 5.0, 6.7, 7.1, 8.2]
        y = [1.0, 2.0, 3.5, 4.0, 5.6, 6.0, 7.9, 8.0]
        res = wilcoxon_paired(x, y)
        stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
        self.assertAlmostEqual(res.statistic, stat, places=9)
        self.assertAlmostEqual(res.p_value, p, places=9)
        self.assertEqual(res.n, 8)

    def test_wilcoxon_all_equal_is_nan(self):
        res = wilcoxon_paired([1, 2, 3], [1, 2, 3])
        self.assertTrue(math.isnan(res.p_value))

    def test_one_sample_wilcoxon(self):
        vals = [0.2, 0.4, -0.1, 0.5, 0.3, 0.6]
        res = one_sample_wilcoxon(vals, popmean=0.0)
        stat, p = stats.wilcoxon(np.array(vals))
        self.assertAlmostEqual(res.statistic, stat, places=9)
        self.assertAlmostEqual(res.p_value, p, places=9)


class TestFriedman(unittest.TestCase):
    def test_friedman_matches_scipy(self):
        g1 = [1, 2, 3, 4, 5]
        g2 = [2, 3, 4, 5, 6]
        g3 = [1, 1, 2, 2, 3]
        res, posthoc = friedman_with_posthoc({"a": g1, "b": g2, "c": g3})
        stat, p = stats.friedmanchisquare(g1, g2, g3)
        self.assertAlmostEqual(res.statistic, stat, places=9)
        self.assertAlmostEqual(res.p_value, p, places=9)
        # 3 groups -> 3 pairwise comparisons.
        self.assertEqual(len(posthoc), 3)
        for item in posthoc:
            self.assertTrue(0.0 <= item.p_value_adjusted <= 1.0 or math.isnan(item.p_value_adjusted))

    def test_too_few_groups_no_pvalue(self):
        res, posthoc = friedman_with_posthoc({"a": [1, 2, 3], "b": [2, 3, 4]})
        self.assertTrue(math.isnan(res.p_value))
        self.assertEqual(posthoc, [])

    def test_unequal_lengths_raises(self):
        with self.assertRaises(ValueError):
            friedman_with_posthoc({"a": [1, 2], "b": [1, 2, 3], "c": [1, 2, 3]})


class TestPageTrend(unittest.TestCase):
    def test_monotone_increasing_detected(self):
        # Strong increasing trend across 4 ordered columns.
        blocks = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [0, 2, 4, 6],
            [1, 3, 3, 4],
            [0, 1, 2, 3],
        ]
        res = page_trend_test(blocks, ascending=True)
        self.assertLess(res.p_value, 0.05)

    def test_decreasing_with_descending_flag(self):
        blocks = [
            [4, 3, 2, 1],
            [5, 3, 2, 1],
            [6, 4, 2, 0],
            [4, 3, 3, 1],
        ]
        res = page_trend_test(blocks, ascending=False)
        self.assertLess(res.p_value, 0.05)

    def test_too_few_columns(self):
        res = page_trend_test([[1, 2], [3, 4]], ascending=True)
        self.assertTrue(math.isnan(res.p_value))


class TestBootstrap(unittest.TestCase):
    def test_ci_brackets_mean(self):
        rng = np.random.default_rng(1)
        vals = rng.normal(loc=5.0, scale=1.0, size=40)
        point, lo, hi = cluster_bootstrap_ci(vals, n_reps=1000, seed=7)
        self.assertAlmostEqual(point, float(np.mean(vals)), places=9)
        self.assertLess(lo, point)
        self.assertGreater(hi, point)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)

    def test_single_value_nan_bounds(self):
        point, lo, hi = cluster_bootstrap_ci([3.0])
        self.assertEqual(point, 3.0)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_deterministic_with_seed(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = cluster_bootstrap_ci(vals, n_reps=500, seed=99)
        b = cluster_bootstrap_ci(vals, n_reps=500, seed=99)
        self.assertEqual(a, b)

    def test_paired_diff_ci(self):
        x = [2.0, 3.0, 4.0, 5.0, 6.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        point, lo, hi = paired_diff_bootstrap_ci(x, y, n_reps=500, seed=3)
        self.assertAlmostEqual(point, 1.0, places=9)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)


if __name__ == "__main__":
    unittest.main()
