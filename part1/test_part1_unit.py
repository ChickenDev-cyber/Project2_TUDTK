import math
import random
import unittest
from unittest.mock import patch
import sys

import matplotlib
matplotlib.use("Agg")

# Cấu hình encoding utf-8 cho terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cross_validation import kfold_cv, ridge_cv_score, ridge_lambda_search
from matrix_ops import all_close, diag, matmul, matvec, trace, transpose, max_abs_diff, mean
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
        print("\nKiểm Thử: OLS FIT (HỒI QUY TUYẾN TÍNH ĐƠN GIẢN)")
        X = [[1, 1], [1, 2], [1, 3], [1, 4]]
        y = [3, 5, 7, 9]
        beta, sigma2 = ols_fit(X, y)
        
        # So sánh với thư viện chuẩn sklearn
        from sklearn.linear_model import LinearRegression
        sk_model = LinearRegression(fit_intercept=False).fit(X, y)
        sk_coef = sk_model.coef_.tolist()
        
        print(f"Dữ liệu đầu vào X: {X}")
        print(f"Dữ liệu đầu vào y: {y}")
        print(f"Hệ số Beta tự cài đặt: {[round(v, 4) for v in beta]}")
        print(f"Hệ số Beta từ sklearn   : {[round(v, 4) for v in sk_coef]}")
        print(f"Sai lệch lớn nhất: {max_abs_diff(beta, sk_coef):.2e}")
        
        self.assertTrue(all_close(beta, [1.0, 2.0], atol=1e-10))
        self.assertAlmostEqual(sigma2, 0.0, places=10)
        self.assertTrue(all_close(beta, sk_coef, atol=1e-8))
        print("-> ĐẠT: Thuật toán OLS trùng khớp hoàn toàn với sklearn.")

    def test_intercept_only(self):
        print("\nKiểm Thử: OLS FIT (CHỈ CÓ HỆ SỐ CHẶN INTERCEPT)")
        X = [[1], [1], [1], [1], [1]]
        y = [2, 4, 4, 5, 10]
        beta, _ = ols_fit(X, y)
        expected = sum(y) / len(y)
        
        print(f"Dữ liệu y: {y}")
        print(f"Beta tự cài đặt (Intercept): {beta[0]:.4f}")
        print(f"Kỳ vọng lý thuyết (Mean y): {expected:.4f}")
        print(f"Độ lệch: {abs(beta[0] - expected):.2e}")
        
        self.assertTrue(all_close(beta, [expected], atol=1e-10))
        print("-> ĐẠT: Hệ số chặn khi không có đặc trưng bằng đúng trung bình mẫu.")


class TestHatMatrix(unittest.TestCase):
    def setUp(self):
        self.X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        self.H = hat_matrix(self.X)

    def test_symmetric_and_idempotent(self):
        print("\nKiểm Thử: HAT MATRIX (TÍNH ĐỐI XỨNG & LŨY ĐẲNG)")
        H_T = transpose(self.H)
        H_sq = matmul(self.H, self.H)
        
        diff_sym = max_abs_diff(self.H, H_T)
        diff_idem = max_abs_diff(H_sq, self.H)
        
        print(f"Độ lệch tính đối xứng (H - H^T)  : {diff_sym:.2e} (Kỳ vọng = 0)")
        print(f"Độ lệch tính lũy đẳng (H^2 - H)  : {diff_idem:.2e} (Kỳ vọng = 0)")
        
        self.assertTrue(all_close(self.H, H_T, atol=1e-10))
        self.assertTrue(all_close(H_sq, self.H, atol=1e-10))
        print("-> ĐẠT: Hat Matrix H đối xứng và lũy đẳng hoàn hảo.")

    def test_projection_and_trace(self):
        print("\nKiểm Thử: HAT MATRIX (PHÉP CHIẾU & TRACE)")
        y = [1, 3, 5, 7]
        beta, _ = ols_fit(self.X, y)
        y_hat_direct = matvec(self.H, y)
        y_hat_regression = matvec(self.X, beta)
        
        diff_proj = max_abs_diff(y_hat_direct, y_hat_regression)
        tr_H = trace(self.H)
        p_plus_1 = len(self.X[0])
        
        print(f"Độ lệch phép chiếu y_hat (H*y vs X*beta): {diff_proj:.2e} (Kỳ vọng = 0)")
        print(f"Vết ma trận trace(H): {tr_H:.4f} | Số tham số p+1: {p_plus_1}")
        
        self.assertTrue(all_close(y_hat_direct, y_hat_regression, atol=1e-10))
        self.assertAlmostEqual(tr_H, p_plus_1, places=10)
        print("-> ĐẠT: Phép chiếu Hy = X*beta và trace(H) = p+1 thỏa mãn.")


class TestModelMetrics(unittest.TestCase):
    def test_perfect_fit(self):
        print("\nKiểm Thử: METRICS ĐÁNH GIÁ MÔ HÌNH (FIT HOÀN HẢO)")
        y = [2, 4, 6, 8]
        RSS, TSS, r2, adj_r2, f_stat, p_val_f = model_metrics(y, y, p=1)
        
        print(f"RSS: {RSS:.4f} (Kỳ vọng = 0)")
        print(f"TSS: {TSS:.4f} | R^2: {r2:.4f} (Kỳ vọng = 1.0)")
        print(f"Trị số F-statistic: {f_stat} (Kỳ vọng = Vô hạn/inf)")
        print(f"Giá trị p-value F  : {p_val_f:.4e} (Kỳ vọng = 0.0)")
        
        self.assertAlmostEqual(RSS, 0.0)
        self.assertAlmostEqual(TSS, 20.0)
        self.assertAlmostEqual(r2, 1.0)
        self.assertAlmostEqual(adj_r2, 1.0)
        self.assertTrue(math.isinf(f_stat))
        self.assertAlmostEqual(p_val_f, 0.0)
        print("-> ĐẠT: Các chỉ số R^2, Adj R^2, F-test hoạt động chính xác khi khớp 100%.")

    def test_constant_y_has_undefined_r2(self):
        print("\nKiểm Thử: METRICS ĐÁNH GIÁ (Y HẰNG SỐ - KHÔNG XÁC ĐỊNH R^2)")
        y = [4, 4, 4, 4]
        _, _, r2, _, _, _ = model_metrics(y, y, p=1)
        print(f"R^2 khi y hằng số: {r2} (Kỳ vọng = nan)")
        self.assertTrue(math.isnan(r2))
        print("-> ĐẠT: Hàm nhận diện chính xác trường hợp mẫu không có biến thiên mẫu (TSS=0).")


class TestCoefInference(unittest.TestCase):
    def test_standard_errors_and_ci(self):
        print("\nKiểm Thử: COEF INFERENCE (SAI SỐ CHUẨN SE & KHOẢNG TIN CẬY 95%)")
        X, y, beta_true = make_design()
        beta_hat, sigma2 = ols_fit(X, y)
        se, t_stats, p_values, ci_lower, ci_upper = coef_inference(X, y, beta_hat, sigma2)
        
        print("So sánh hệ số ước lượng với giá trị thực tế:")
        for j in range(len(beta_hat)):
            print(f"  Hệ số beta_{j}: Thật = {beta_true[j]:.2f} | Ước lượng = {beta_hat[j]:.4f} | SE = {se[j]:.4f}")
            print(f"    Khoảng tin cậy 95%: [{ci_lower[j]:.4f}, {ci_upper[j]:.4f}]")
            
        self.assertTrue(all(v > 0 for v in se))
        self.assertTrue(all(ci_upper[j] > ci_lower[j] for j in range(len(beta_hat))))
        self.assertTrue(all(ci_lower[j] <= beta_hat[j] <= ci_upper[j] for j in range(len(beta_hat))))
        self.assertTrue(all(abs(beta_hat[j] - beta_true[j]) < 0.15 for j in range(len(beta_true))))
        print("-> ĐẠT: Ước lượng khoảng tin cậy 95% chứa đúng giá trị thực tế của tham số.")


class TestVif(unittest.TestCase):
    def test_independent_features_near_one(self):
        print("\nKiểm Thử: HỆ SỐ PHÓNG ĐẠI PHƯƠNG SAI VIF (DỮ LIỆU ĐỘC LẬP)")
        rng = random.Random(7)
        X = [[1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(120)]
        scores = vif(X)
        print(f"Điểm VIF tự tính cho các biến độc lập: {[round(v, 4) for v in scores]}")
        self.assertTrue(all(0.9 <= value <= 1.2 for value in scores))
        print("-> ĐẠT: VIF gần bằng 1 khi các đặc trưng hoàn toàn độc lập tuyến tính.")

    def test_multicollinearity_large(self):
        print("\nKiểm Thử: HỆ SỐ VIF (DỮ LIỆU ĐA CỘNG TUYẾN CAO)")
        rng = random.Random(8)
        X = []
        for _ in range(120):
            x = rng.gauss(0.0, 1.0)
            X.append([1.0, x, x + rng.gauss(0.0, 0.001)])  # X2 rất gần X1
        scores = vif(X)
        print(f"Điểm VIF tự tính khi đa cộng tuyến nghiêm trọng: {[round(v, 2) for v in scores]}")
        self.assertTrue(scores[0] > 1000)
        self.assertTrue(scores[1] > 1000)
        print("-> ĐẠT: Điểm VIF tăng vọt (> 1000) phản ánh chính xác đa cộng tuyến giữa X1 và X2.")


class TestRidgeFit(unittest.TestCase):
    def test_lambda_zero_matches_ols(self):
        print("\nKiểm Thử: HỒI QUY RIDGE (KHI LAMBDA = 0 KHỚP OLS)")
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [2, 5, 8, 11]
        beta_ridge = ridge_fit(X, y, lam=0.0)
        beta_ols, _ = ols_fit(X, y)
        
        print(f"Hệ số Beta OLS  : {[round(v, 4) for v in beta_ols]}")
        print(f"Hệ số Beta Ridge: {[round(v, 4) for v in beta_ridge]}")
        print(f"Độ lệch tuyệt đối cực đại: {max_abs_diff(beta_ridge, beta_ols):.2e}")
        
        self.assertTrue(all_close(beta_ridge, beta_ols, atol=1e-10))
        print("-> ĐẠT: Ridge hồi quy với lambda = 0 đồng nhất tuyệt đối với OLS.")

    def test_large_lambda_shrinks_coefficients(self):
        print("\nKiểm Thử: HỒI QUY RIDGE (HIỆU ỨNG CO RÚT L2 KHI LAMBDA LỚN)")
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [2, 5, 8, 11]
        beta_small = ridge_fit(X, y, lam=0.1)
        beta_large = ridge_fit(X, y, lam=100.0)
        
        norm_small = norm2(beta_small)
        norm_large = norm2(beta_large)
        
        print(f"Độ dài L2 hệ số khi lambda = 0.1: {norm_small:.4f}")
        print(f"Độ dài L2 hệ số khi lambda = 100: {norm_large:.4f}")
        print(f"Tỷ lệ co nhỏ: {norm_large / norm_small * 100:.2f}%")
        
        self.assertLess(norm_large, norm_small)
        print("-> ĐẠT: Phạt L2 làm giảm chuẩn của vector hệ số đúng như lý thuyết.")

    def test_plot_ridge_trace_runs(self):
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [2, 5, 8, 11]
        with patch("matplotlib.pyplot.show"):
            plot_ridge_trace(X, y)


class TestLassoFitCd(unittest.TestCase):
    def test_lambda_zero_matches_ols_for_simple_line(self):
        print("\nKiểm Thử: HỒI QUY LASSO (KHI LAMBDA = 0 KHỚP OLS)")
        X = [[1, -1], [1, 0], [1, 1], [1, 2]]
        y = [-1, 2, 5, 8]
        beta_lasso = lasso_fit_cd(X, y, lam=0.0, max_iter=5000)
        beta_ols, _ = ols_fit(X, y)
        
        print(f"Hệ số Beta OLS    : {[round(v, 4) for v in beta_ols]}")
        print(f"Hệ số Beta Lasso CD: {[round(v, 4) for v in beta_lasso]}")
        print(f"Sai khác OLS vs Lasso CD: {max_abs_diff(beta_lasso, beta_ols):.2e}")
        
        self.assertTrue(all_close(beta_lasso, [2.0, 3.0], atol=1e-6))
        self.assertTrue(all_close(beta_lasso, beta_ols, atol=1e-6))
        print("-> ĐẠT: Thuật toán coordinate descent cho Lasso với lambda = 0 hội tụ về nghiệm OLS.")

    def test_large_lambda_sets_slopes_to_zero(self):
        print("\nKiểm Thử: HỒI QUY LASSO (TẠO NGHIỆM THƯA L1 KHI LAMBDA LỚN)")
        X = [[1, -2], [1, -1], [1, 0], [1, 1], [1, 2]]
        y = [7, 4, 1, -2, -5]
        beta = lasso_fit_cd(X, y, lam=1000.0, max_iter=5000)
        
        print(f"Hệ số Beta thu được: {[round(v, 4) for v in beta]}")
        print(f"Đặc trưng X1 (Độ dốc) bị triệt tiêu hoàn toàn về 0: {abs(beta[1]) < 1e-6}")
        
        self.assertAlmostEqual(beta[0], sum(y) / len(y), places=6)
        self.assertTrue(all_close(beta[1:], [0.0], atol=1e-6))
        print("-> ĐẠT: Phạt L1 triệt tiêu hoàn toàn hệ số độ dốc về đúng 0.0.")


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
        print("\nKiểm Thử: K-FOLD CROSS VALIDATION (TÍNH TÁI LẬP KẾT QUẢ)")
        first = kfold_cv(self.X, self.y, k=5)
        second = kfold_cv(self.X, self.y, k=5)
        
        print(f"Chạy lần 1 - Lỗi CV-MSE: {first:.8f}")
        print(f"Chạy lần 2 - Lỗi CV-MSE: {second:.8f}")
        print(f"Độ chênh lệch: {abs(first - second):.2e}")
        
        self.assertAlmostEqual(first, second, places=12)
        print("-> ĐẠT: K-Fold CV bảo đảm kết quả ổn định và tái lập nhờ random state cố định.")

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            kfold_cv(self.X, self.y, k=1)
        with self.assertRaises(ValueError):
            kfold_cv(self.X, self.y, k=len(self.y) + 1)

    def test_ridge_lambda_search_returns_minimum(self):
        print("\nKiểm Thử: RIDGE LAMBDA SEARCH (LỰA CHỌN SIÊU THAM SỐ TỐI ƯU)")
        lambdas, scores, best_lam, best_score = ridge_lambda_search(self.X, self.y, k=5)
        
        print(f"Hệ số lambda tối ưu chọn được: {best_lam:.6f}")
        print(f"Lỗi CV-MSE tương ứng nhỏ nhất : {best_score:.6f}")
        print(f"Lỗi CV-MSE trung bình dải quét  : {mean(scores):.6f}")
        
        self.assertIn(best_lam, ridge_lambda_search(self.X, self.y, k=5)[0])
        self.assertAlmostEqual(best_score, min(scores))
        print("-> ĐẠT: Hàm quét chính xác và lựa chọn siêu tham số lambda tối thiểu hóa lỗi CV-MSE.")

    def test_ridge_cv_rejects_negative_lambda(self):
        with self.assertRaises(ValueError):
            ridge_cv_score(self.X, self.y, k=5, lam=-0.1)


if __name__ == "__main__":
    import io
    runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
    unittest.main(testRunner=runner)
