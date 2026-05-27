import math
import random
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from cross_validation import kfold_cv, ridge_cv_score, ridge_lambda_search
from matrix_ops import all_close, diag, matmul, matvec, trace, transpose
from ols_implementation import coef_inference, hat_matrix, model_metrics, ols_fit, vif
from residual_analysis import residual_plots
from ridge_lasso import lasso_fit_cd, plot_ridge_trace, ridge_fit


def norm2(values):
    return math.sqrt(sum(v * v for v in values))


def make_design(seed=123, n=80):
    rng = random.Random(seed)
    X = []
    y = []
    beta = [1.5, 2.0, -1.0]
    for _ in range(n):
        x1 = rng.gauss(0.0, 1.0)
        x2 = rng.gauss(0.0, 1.0)
        row = [1.0, x1, x2]
        X.append(row)
        y.append(sum(row[j] * beta[j] for j in range(len(beta))) + rng.gauss(0.0, 0.25))
    return X, y, beta


class TestOlsFit(unittest.TestCase):
    def test_simple_line(self):
        X = [[1, 1], [1, 2], [1, 3], [1, 4]]
        y = [3, 5, 7, 9]
        beta, sigma2 = ols_fit(X, y)
        self.assertTrue(all_close(beta, [1.0, 2.0], atol=1e-10))
        self.assertAlmostEqual(sigma2, 0.0, places=10)

    def test_intercept_only(self):
        X = [[1], [1], [1], [1], [1]]
        y = [2, 4, 4, 5, 10]
        beta, _ = ols_fit(X, y)
        self.assertTrue(all_close(beta, [sum(y) / len(y)], atol=1e-10))


class TestHatMatrix(unittest.TestCase):
    def setUp(self):
        self.X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        self.H = hat_matrix(self.X)

    def test_symmetric_and_idempotent(self):
        self.assertTrue(all_close(self.H, transpose(self.H), atol=1e-10))
        self.assertTrue(all_close(matmul(self.H, self.H), self.H, atol=1e-10))

    def test_projection_and_trace(self):
        y = [1, 3, 5, 7]
        beta, _ = ols_fit(self.X, y)
        self.assertTrue(all_close(matvec(self.H, y), matvec(self.X, beta), atol=1e-10))
        self.assertAlmostEqual(trace(self.H), len(self.X[0]), places=10)


class TestModelMetrics(unittest.TestCase):
    def test_perfect_fit(self):
        y = [2, 4, 6, 8]
        RSS, TSS, r2, adj_r2, f_stat = model_metrics(y, y, p=1)
        self.assertAlmostEqual(RSS, 0.0)
        self.assertAlmostEqual(TSS, 20.0)
        self.assertAlmostEqual(r2, 1.0)
        self.assertAlmostEqual(adj_r2, 1.0)
        self.assertTrue(math.isinf(f_stat))

    def test_constant_y_has_undefined_r2(self):
        y = [4, 4, 4, 4]
        _, _, r2, _, _ = model_metrics(y, y, p=1)
        self.assertTrue(math.isnan(r2))


class TestCoefInference(unittest.TestCase):
    def test_standard_errors_and_ci(self):
        X, y, beta_true = make_design()
        beta_hat, sigma2 = ols_fit(X, y)
        se, _, _, ci_lower, ci_upper = coef_inference(X, y, beta_hat, sigma2)
        self.assertTrue(all(v > 0 for v in se))
        self.assertTrue(all(ci_upper[j] > ci_lower[j] for j in range(len(beta_hat))))
        self.assertTrue(all(ci_lower[j] <= beta_hat[j] <= ci_upper[j] for j in range(len(beta_hat))))
        self.assertTrue(all(abs(beta_hat[j] - beta_true[j]) < 0.15 for j in range(len(beta_true))))


class TestVif(unittest.TestCase):
    def test_independent_features_near_one(self):
        rng = random.Random(7)
        X = [[1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(120)]
        scores = vif(X)
        self.assertTrue(all(0.9 <= value <= 1.2 for value in scores))

    def test_multicollinearity_large(self):
        rng = random.Random(8)
        X = []
        for _ in range(120):
            x = rng.gauss(0.0, 1.0)
            X.append([1.0, x, x + rng.gauss(0.0, 0.01)])
        scores = vif(X)
        self.assertTrue(scores[0] > 1000)
        self.assertTrue(scores[1] > 1000)


class TestRidgeFit(unittest.TestCase):
    def test_lambda_zero_matches_ols(self):
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [2, 5, 8, 11]
        beta_ridge = ridge_fit(X, y, lam=0.0)
        beta_ols, _ = ols_fit(X, y)
        self.assertTrue(all_close(beta_ridge, beta_ols, atol=1e-10))

    def test_large_lambda_shrinks_coefficients(self):
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [2, 5, 8, 11]
        beta_small = ridge_fit(X, y, lam=0.1)
        beta_large = ridge_fit(X, y, lam=100.0)
        self.assertLess(norm2(beta_large), norm2(beta_small))

    def test_plot_ridge_trace_runs(self):
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [2, 5, 8, 11]
        with patch("matplotlib.pyplot.show"):
            plot_ridge_trace(X, y)


class TestLassoFitCd(unittest.TestCase):
    def test_lambda_zero_matches_ols_for_simple_line(self):
        X = [[1, -1], [1, 0], [1, 1], [1, 2]]
        y = [-1, 2, 5, 8]
        beta = lasso_fit_cd(X, y, lam=0.0, max_iter=5000)
        self.assertTrue(all_close(beta, [2.0, 3.0], atol=1e-6))

    def test_large_lambda_sets_slopes_to_zero(self):
        X = [[1, -2], [1, -1], [1, 0], [1, 1], [1, 2]]
        y = [7, 4, 1, -2, -5]
        beta = lasso_fit_cd(X, y, lam=1000.0, max_iter=5000)
        self.assertAlmostEqual(beta[0], sum(y) / len(y), places=6)
        self.assertTrue(all_close(beta[1:], [0.0], atol=1e-6))


class TestResidualPlots(unittest.TestCase):
    def test_residual_plots_runs(self):
        X, y, _ = make_design(seed=42, n=40)
        beta, _ = ols_fit(X, y)
        with patch("matplotlib.pyplot.show"):
            residual_plots(X, y, beta)


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        rng = random.Random(2024)
        self.X = []
        self.y = []
        for _ in range(45):
            x = rng.gauss(0.0, 1.0)
            row = [1.0, x]
            self.X.append(row)
            self.y.append(1.0 + 2.5 * x + rng.gauss(0.0, 0.1))

    def test_kfold_is_deterministic(self):
        first = kfold_cv(self.X, self.y, k=5)
        second = kfold_cv(self.X, self.y, k=5)
        self.assertAlmostEqual(first, second, places=12)

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            kfold_cv(self.X, self.y, k=1)
        with self.assertRaises(ValueError):
            kfold_cv(self.X, self.y, k=len(self.y) + 1)

    def test_ridge_lambda_search_returns_minimum(self):
        _, scores, best_lam, best_score = ridge_lambda_search(self.X, self.y, k=5)
        self.assertIn(best_lam, ridge_lambda_search(self.X, self.y, k=5)[0])
        self.assertAlmostEqual(best_score, min(scores))

    def test_ridge_cv_rejects_negative_lambda(self):
        with self.assertRaises(ValueError):
            ridge_cv_score(self.X, self.y, k=5, lam=-0.1)


if __name__ == "__main__":
    unittest.main()
