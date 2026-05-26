import contextlib
import io
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cross_validation import kfold_cv, ridge_cv_score, ridge_lambda_search
from ols_implementation import (
    coef_inference,
    hat_matrix,
    model_metrics,
    ols_fit,
    verify_solution,
    vif,
)
from residual_analysis import residual_plots
from ridge_lasso import lasso_fit_cd, plot_ridge_trace, ridge_fit


class TestOlsFit(unittest.TestCase):
    def test_exact_line_coefficients(self):
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
        y = np.array([3, 5, 7, 9], dtype=float)
        beta, sigma2 = ols_fit(X, y)

        np.testing.assert_allclose(beta, [1.0, 2.0], atol=1e-10)
        self.assertLess(sigma2, 1e-10)

    def test_intercept_only_matches_mean(self):
        X = np.ones((5, 1))
        y = np.array([2, 4, 4, 5, 10], dtype=float)
        beta, sigma2 = ols_fit(X, y)

        np.testing.assert_allclose(beta, [np.mean(y)], atol=1e-10)
        self.assertGreater(sigma2, 0.0)


class TestHatMatrix(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)

    def test_hat_matrix_is_symmetric_and_idempotent(self):
        H = hat_matrix(self.X)

        np.testing.assert_allclose(H, H.T, atol=1e-10)
        np.testing.assert_allclose(H @ H, H, atol=1e-10)

    def test_hat_matrix_projects_to_fitted_values(self):
        y = np.array([1, 3, 5, 7], dtype=float)
        beta, _ = ols_fit(self.X, y)
        H = hat_matrix(self.X)

        np.testing.assert_allclose(H @ y, self.X @ beta, atol=1e-10)
        self.assertAlmostEqual(float(np.trace(H)), self.X.shape[1])


class TestModelMetrics(unittest.TestCase):
    def test_perfect_fit_has_r2_one(self):
        y = np.array([2, 4, 6, 8], dtype=float)
        rss, tss, r2, adj_r2, f_stat = model_metrics(y, y, p=1)

        self.assertAlmostEqual(rss, 0.0)
        self.assertAlmostEqual(r2, 1.0)
        self.assertAlmostEqual(adj_r2, 1.0)
        self.assertTrue(np.isinf(f_stat))
        self.assertGreater(tss, 0.0)

    def test_mean_prediction_has_r2_zero(self):
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        y_hat = np.full_like(y, np.mean(y))
        _, _, r2, adj_r2, _ = model_metrics(y, y_hat, p=1)

        self.assertAlmostEqual(r2, 0.0)
        self.assertLess(adj_r2, r2)


class TestCoefInference(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        n = 500
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        self.X = np.column_stack([np.ones(n), x1, x2])
        self.beta_true = np.array([1.5, 2.0, -1.0])
        self.y = self.X @ self.beta_true + rng.normal(scale=0.15, size=n)
        self.beta_hat, self.sigma2_hat = ols_fit(self.X, self.y)

    def test_inference_shapes_and_positive_standard_errors(self):
        se, t_stats, p_values, ci_lower, ci_upper = coef_inference(
            self.X, self.y, self.beta_hat, self.sigma2_hat
        )

        self.assertEqual(se.shape, self.beta_hat.shape)
        self.assertEqual(t_stats.shape, self.beta_hat.shape)
        self.assertEqual(p_values.shape, self.beta_hat.shape)
        self.assertTrue(np.all(se > 0))
        self.assertTrue(np.all(ci_upper > ci_lower))

    def test_confidence_intervals_contain_true_coefficients(self):
        _, _, p_values, ci_lower, ci_upper = coef_inference(
            self.X, self.y, self.beta_hat, self.sigma2_hat
        )

        self.assertTrue(np.all(ci_lower <= self.beta_true))
        self.assertTrue(np.all(self.beta_true <= ci_upper))
        self.assertLess(p_values[1], 0.001)
        self.assertLess(p_values[2], 0.001)


class TestVif(unittest.TestCase):
    def test_independent_features_have_low_vif(self):
        rng = np.random.default_rng(7)
        X = np.column_stack([np.ones(120), rng.normal(size=120), rng.normal(size=120)])
        scores = vif(X)

        self.assertEqual(len(scores), 2)
        self.assertTrue(all(score < 2.0 for score in scores))

    def test_collinear_features_have_high_vif(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=120)
        X = np.column_stack([np.ones(120), x, x + rng.normal(scale=0.01, size=120)])
        scores = vif(X)

        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], 100.0)
        self.assertGreater(scores[1], 100.0)


class TestRidgeFit(unittest.TestCase):
    def test_lambda_zero_matches_ols(self):
        X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
        y = np.array([2, 5, 8, 11], dtype=float)
        beta_ols, _ = ols_fit(X, y)
        beta_ridge = ridge_fit(X, y, lam=0.0)

        np.testing.assert_allclose(beta_ridge, beta_ols, atol=1e-10)

    def test_large_lambda_shrinks_coefficients(self):
        X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
        y = np.array([2, 5, 8, 11], dtype=float)
        beta_small = ridge_fit(X, y, lam=0.1)
        beta_large = ridge_fit(X, y, lam=1_000.0)

        self.assertLess(np.linalg.norm(beta_large), np.linalg.norm(beta_small))


class TestLassoFitCd(unittest.TestCase):
    def test_lambda_zero_recovers_simple_ols_solution(self):
        X = np.array([[1, -1], [1, 0], [1, 1], [1, 2]], dtype=float)
        y = 2 + 3 * X[:, 1]
        beta = lasso_fit_cd(X, y, lam=0.0, max_iter=5_000)

        np.testing.assert_allclose(beta, [2.0, 3.0], atol=1e-6)

    def test_large_lambda_zeros_non_intercept_terms(self):
        X = np.array([[1, -2], [1, -1], [1, 0], [1, 1], [1, 2]], dtype=float)
        y = 4 + 5 * X[:, 1]
        beta = lasso_fit_cd(X, y, lam=1_000.0, max_iter=5_000)

        self.assertAlmostEqual(beta[0], np.mean(y), places=6)
        np.testing.assert_allclose(beta[1:], [0.0], atol=1e-6)


class TestRidgeTracePlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_ridge_trace_creates_one_axis(self):
        X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
        y = np.array([2, 5, 8, 11], dtype=float)

        with patch("matplotlib.pyplot.show"):
            plot_ridge_trace(X, y)

        self.assertEqual(len(plt.gcf().axes), 1)

    def test_plot_ridge_trace_uses_log_x_axis(self):
        X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
        y = np.array([2, 5, 8, 11], dtype=float)

        with patch("matplotlib.pyplot.show"):
            plot_ridge_trace(X, y)

        self.assertEqual(plt.gcf().axes[0].get_xscale(), "log")


class TestResidualPlots(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def _sample_data(self):
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
        beta = np.array([1.0, 2.0, -1.5])
        y = X @ beta + rng.normal(scale=0.2, size=40)
        beta_hat, _ = ols_fit(X, y)
        return X, y, beta_hat

    def test_residual_plots_create_four_axes(self):
        X, y, beta_hat = self._sample_data()

        with patch("matplotlib.pyplot.show"):
            residual_plots(X, y, beta_hat)

        self.assertEqual(len(plt.gcf().axes), 4)

    def test_residual_plots_have_expected_titles(self):
        X, y, beta_hat = self._sample_data()

        with patch("matplotlib.pyplot.show"):
            residual_plots(X, y, beta_hat)

        titles = [ax.get_title() for ax in plt.gcf().axes]
        self.assertIn("Residuals vs Fitted", titles)
        self.assertIn("Cook's Distance", titles)


class TestKFoldCv(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2024)
        x = rng.normal(size=45)
        self.X = np.column_stack([np.ones(45), x])
        self.y = 1.0 + 2.0 * x + rng.normal(scale=0.1, size=45)

    def test_default_signature_returns_float(self):
        score = kfold_cv(self.X, self.y, k=5)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_internal_seed_is_reproducible(self):
        score_a = kfold_cv(self.X, self.y, k=5)
        score_b = kfold_cv(self.X, self.y, k=5)

        self.assertAlmostEqual(score_a, score_b)

    def test_ridge_cv_score_accepts_explicit_lambda(self):
        score = ridge_cv_score(self.X, self.y, k=5, lam=0.1)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_ridge_lambda_search_returns_best_score(self):
        lambdas, scores, best_lam, best_score = ridge_lambda_search(self.X, self.y, k=5)

        self.assertEqual(len(lambdas), len(scores))
        self.assertIn(best_lam, lambdas)
        self.assertAlmostEqual(best_score, float(np.min(scores)))

    def test_invalid_k_raises_value_error(self):
        with self.assertRaises(ValueError):
            kfold_cv(self.X, self.y, k=1)

        with self.assertRaises(ValueError):
            kfold_cv(self.X, self.y, k=len(self.y) + 1)


class TestVerifySolution(unittest.TestCase):
    def test_verify_solution_prints_checks(self):
        X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
        y = np.array([1, 3, 5, 7], dtype=float)
        beta, _ = ols_fit(X, y)
        H = hat_matrix(X)

        with contextlib.redirect_stdout(io.StringIO()) as output:
            verify_solution(X, y, beta, H)

        self.assertGreater(len(output.getvalue()), 0)

    def test_verify_solution_handles_missing_beta(self):
        X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
        y = np.array([1, 3, 5, 7], dtype=float)
        H = hat_matrix(X)

        with contextlib.redirect_stdout(io.StringIO()) as output:
            verify_solution(X, y, None, H)

        self.assertGreater(len(output.getvalue()), 0)


if __name__ == "__main__":
    unittest.main()
