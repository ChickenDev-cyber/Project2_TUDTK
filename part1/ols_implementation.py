import numpy as np
from scipy import stats
import warnings
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _solve_normal_equation(X, y):
    gram = X.T @ X
    rhs = X.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def _gram_inverse(X):
    gram = X.T @ X
    identity = np.eye(gram.shape[0])
    try:
        return np.linalg.solve(gram, identity)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram)


def ols_fit(X, y):
    beta_hat = _solve_normal_equation(X, y)
    
    y_hat = X @ beta_hat
    phan_du = y - y_hat
    RSS = np.sum(phan_du**2)
    
    n = X.shape[0]
    p_cong_1 = X.shape[1]
    
    sigma2_hat = RSS / (n - p_cong_1)
    
    return beta_hat, sigma2_hat

def hat_matrix(X):
    X_T = X.T
    H = X @ _gram_inverse(X) @ X_T
    return H

def model_metrics(y, y_hat, p):
    n = len(y)
    RSS = np.sum((y - y_hat)**2)
    
    y_bar = np.mean(y)
    TSS = np.sum((y - y_bar)**2)
    
    r2 = np.nan if np.isclose(TSS, 0.0) else 1 - (RSS / TSS)
    adj_r2 = np.nan if n - p - 1 <= 0 else 1 - ((n - 1) / (n - p - 1)) * (1 - r2)

    if p <= 0 or n - p - 1 <= 0:
        f_stat = np.nan
    elif np.isclose(RSS, 0.0):
        f_stat = np.inf
    else:
        f_stat = ((TSS - RSS) / p) / (RSS / (n - p - 1))
    
    return RSS, TSS, r2, adj_r2, f_stat

def coef_inference(X, y, beta_hat, sigma2_hat):
    n = X.shape[0]
    p_cong_1 = X.shape[1]
    df = n - p_cong_1
    
    cov_matrix = sigma2_hat * _gram_inverse(X)
    se = np.sqrt(np.diag(cov_matrix))
    
    t_stats = beta_hat / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))
    
    t_critical = stats.t.ppf(0.975, df=df)
    ci_lower = beta_hat - t_critical * se
    ci_upper = beta_hat + t_critical * se
    
    return se, t_stats, p_values, ci_lower, ci_upper

def vif(X):
    dac_trung = X[:, 1:] if np.all(X[:, 0] == 1) else X
    so_luong_bien = dac_trung.shape[1]
    vif_scores = []
    
    for j in range(so_luong_bien):
        y_tam = dac_trung[:, j]
        X_tam = np.delete(dac_trung, j, axis=1)
        X_tam = np.column_stack([np.ones(X_tam.shape[0]), X_tam])
        
        beta_tam = _solve_normal_equation(X_tam, y_tam)
        y_tam_hat = X_tam @ beta_tam
        
        RSS_tam = np.sum((y_tam - y_tam_hat)**2)
        TSS_tam = np.sum((y_tam - np.mean(y_tam))**2)
        R2_tam = 1 - (RSS_tam / TSS_tam)
        
        if R2_tam == 1.0:
            vif_scores.append(float('inf'))
        else:
            vif_scores.append(1 / (1 - R2_tam))
            
    return vif_scores

def verify_solution(X, y, beta_hat, H):
    from sklearn.linear_model import LinearRegression
    
    if beta_hat is None: 
        print("Kiểm chứng: Không có hệ số để đối chiếu.")
        return
    
    try:
        if np.allclose(H @ H, H, atol=1e-5):
            print("Kiểm tra tính chất Idempotent (H^2 = H): Giống")
        else:
            print("Kiểm tra tính chất Idempotent (H^2 = H): Khác")
            
        sk_model = LinearRegression(fit_intercept=False).fit(X, y)
        if np.allclose(beta_hat, sk_model.coef_, atol=1e-5):
            print("Đối chiếu hệ số Beta với sklearn: Giống")
        else:
            print("Đối chiếu hệ số Beta với sklearn: Khác")
            
    except Exception as e:
        print(f"Lỗi xảy ra trong quá trình đối chiếu thư viện: {e}")


def test_ols_fit():
    print("--- Phân tích OLS Fit ---")
    X_small = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]], dtype=float)
    y_small = np.array([3, 5, 7, 9, 11], dtype=float)
    
    X_intercept = np.ones((6, 1))
    y_intercept = np.array([2, 4, 4, 4, 5, 7], dtype=float)
    
    beta, sigma2 = ols_fit(X_small, y_small)
    beta2, sigma2_2 = ols_fit(X_intercept, y_intercept)
    
    # 1. Kiểm tra Beta đúng với y = 1 + 2x không nhiễu
    if np.allclose(beta, [1.0, 2.0], atol=1e-8):
        print("Kiểm tra Beta ước lượng: Giống")
    else:
        print("Kiểm tra Beta ước lượng: Khác")
        
    if sigma2 < 1e-10:
        print("Kiểm tra sigma2 xấp xỉ 0: Giống")
    else:
        print("Kiểm tra sigma2 xấp xỉ 0: Khác")
        
    # 2. Kiểm tra Intercept-only, beta = mean(y)
    if np.allclose(beta2, [np.mean(y_intercept)], atol=1e-8):
        print("Kiểm tra Beta = mean(y): Giống")
    else:
        print("Kiểm tra Beta = mean(y): Khác")
        
    if np.allclose(sigma2_2, np.var(y_intercept, ddof=1), atol=1e-8):
        print("Kiểm tra sigma2 = variance mẫu: Giống")
    else:
        print("Kiểm tra sigma2 = variance mẫu: Khác")
        
def test_hat_matrix():
    print("\n--- Phân tích Hat Matrix ---")
    X_small = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]], dtype=float)
    y_small = np.array([3, 5, 7, 9, 11], dtype=float)
    
    H = hat_matrix(X_small)
    rank_H = np.linalg.matrix_rank(H)
    
    if np.allclose(H @ H, H, atol=1e-8):
        print("Kiểm tra tính chất Idempotent (H^2 = H): Giống")
    else:
        print("Kiểm tra tính chất Idempotent (H^2 = H): Khác")
        
    if np.allclose(H, H.T, atol=1e-8):
        print("Kiểm tra tính đối xứng (H = H.T): Giống")
    else:
        print("Kiểm tra tính đối xứng (H = H.T): Khác")
        
    if rank_H == 2:
        print("Kiểm tra hạng ma trận (rank = p+1): Giống")
    else:
        print("Kiểm tra hạng ma trận (rank = p+1): Khác")
        
    if np.allclose(H @ y_small, y_small, atol=1e-8):
        print("Kiểm tra tính chiếu (Hy = y): Giống")
    else:
        print("Kiểm tra tính chiếu (Hy = y): Khác")

def test_model_metrics():
    print("\n--- Phân tích Model Metrics ---")
    X_small = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]], dtype=float)
    y_small = np.array([3, 5, 7, 9, 11], dtype=float)
    
    beta, _ = ols_fit(X_small, y_small)
    y_hat_small = X_small @ beta
    RSS, TSS, r2, adj_r2, f_stat = model_metrics(y_small, y_hat_small, p=1)
    
    y_const = np.array([1, 2, 3, 4, 5], dtype=float)
    y_hat_mean = np.full_like(y_const, np.mean(y_const))
    _, _, r2_zero, _, _ = model_metrics(y_const, y_hat_mean, p=1)
    
    if RSS < 1e-10:
        print("Kiểm tra RSS xấp xỉ 0 khi fit hoàn hảo: Giống")
    else:
        print("Kiểm tra RSS xấp xỉ 0 khi fit hoàn hảo: Khác")
        
    if np.isclose(r2, 1.0, atol=1e-8):
        print("Kiểm tra R^2 = 1.0: Giống")
    else:
        print("Kiểm tra R^2 = 1.0: Khác")
        
    if np.isclose(r2_zero, 0.0, atol=1e-8):
        print("Kiểm tra R^2 = 0 khi y_hat = mean(y): Giống")
    else:
        print("Kiểm tra R^2 = 0 khi y_hat = mean(y): Khác")

def test_coef_inference():
    print("\n--- Phân tích Coef Inference ---")
    np.random.seed(42)
    n_inf = 200
    X_inf_feat = np.random.randn(n_inf, 2)
    X_inf = np.column_stack([np.ones(n_inf), X_inf_feat])
    # β0 = 5, β1 = 3, β2 = 0 (tạo biến không ý nghĩa)
    y_inf = 5 + 3 * X_inf_feat[:, 0] + np.random.randn(n_inf)
    
    beta_inf, sigma2_inf = ols_fit(X_inf, y_inf)
    se, t_stats, p_values, ci_lower, ci_upper = coef_inference(X_inf, y_inf, beta_inf, sigma2_inf)
    
    if p_values[1] < 0.01:
        print("Kiểm tra p-value < 0.01 (biến có ý nghĩa): Giống")
    else:
        print("Kiểm tra p-value < 0.01 (biến có ý nghĩa): Khác")
        
    if p_values[2] > 0.05:
        print("Kiểm tra p-value > 0.05 (biến không ý nghĩa): Giống")
    else:
        print("Kiểm tra p-value > 0.05 (biến không ý nghĩa): Khác")
        
    if ci_lower[0] <= 5.0 <= ci_upper[0]:
        print("Kiểm tra CI chứa Beta_0 = 5: Giống")
    else:
        print("Kiểm tra CI chứa Beta_0 = 5: Khác")
        
    if ci_lower[1] <= 3.0 <= ci_upper[1]:
        print("Kiểm tra CI chứa Beta_1 = 3: Giống")
    else:
        print("Kiểm tra CI chứa Beta_1 = 3: Khác")

def test_vif():
    print("\n--- Phân tích VIF ---")
    np.random.seed(7)
    n_vif = 100
    X_indep = np.column_stack([np.ones(n_vif), np.random.randn(n_vif), np.random.randn(n_vif)])
    x_base = np.random.randn(n_vif)
    X_collinear = np.column_stack([np.ones(n_vif), x_base, x_base + np.random.randn(n_vif)*0.01, np.random.randn(n_vif)])
    
    vif_indep = vif(X_indep)
    vif_collinear = vif(X_collinear)
    
    if all(v < 2.0 for v in vif_indep):
        print("Kiểm tra VIF < 2 cho biến độc lập: Giống")
    else:
        print("Kiểm tra VIF < 2 cho biến độc lập: Khác")
        
    if vif_collinear[0] > 100 or vif_collinear[1] > 100:
        print("Kiểm tra VIF > 100 cho biến đa cộng tuyến: Giống")
    else:
        print("Kiểm tra VIF > 100 cho biến đa cộng tuyến: Khác")
        
    if vif_collinear[2] < 5.0:
        print("Kiểm tra VIF < 5 cho biến không liên quan: Giống")
    else:
        print("Kiểm tra VIF < 5 cho biến không liên quan: Khác")

def run_all_unit_tests():
    warnings.filterwarnings('ignore')
    print("--------------- Unit Test ---------------")
    test_ols_fit()
    test_hat_matrix()
    test_model_metrics()
    test_coef_inference()
    test_vif()
    print("-----------------------------------------------")

def test_integration_simple_regression():
    print("--- Integration Test: Hồi quy tuyến tính đơn ---")
    np.random.seed(42)
    n = 50
    X_features = np.random.rand(n, 1) * 10
    X = np.column_stack([np.ones(n), X_features])
    y = 5 + 3 * X_features[:, 0] + np.random.randn(n) * 0.5
    
    print(f"Kích thước X: {X.shape}, y: {y.shape}")
    beta_hat, sigma2_hat = ols_fit(X, y)
    
    print(f"Vector hệ số Beta: {[round(float(v), 4) for v in beta_hat]}")
    print(f"Phương sai nhiễu (sigma^2): {sigma2_hat:.4f}")
    
    y_hat = X @ beta_hat
    _, _, r2, _, _ = model_metrics(y, y_hat, p=1)
    print(f"Hệ số xác định R^2: {r2:.4f}")
    
    H = hat_matrix(X)
    verify_solution(X, y, beta_hat, H)

def test_integration_multiple_regression():
    print("\n--- Integration Test: Hồi quy tuyến tính bội ---")
    np.random.seed(42)
    n = 100
    X_features = np.random.rand(n, 3) * 5
    X = np.column_stack([np.ones(n), X_features])
    y = 2.5 - 1.5 * X_features[:, 0] + 4 * X_features[:, 1] - 0.8 * X_features[:, 2] + np.random.randn(n) * 1.2
    
    print(f"Kích thước X: {X.shape}, y: {y.shape}")
    beta_hat, sigma2_hat = ols_fit(X, y)
    
    print(f"Vector hệ số Beta: {[round(float(v), 4) for v in beta_hat]}")
    print(f"Phương sai nhiễu (sigma^2): {sigma2_hat:.4f}")
    
    y_hat = X @ beta_hat
    _, _, r2, _, _ = model_metrics(y, y_hat, p=3)
    print(f"Hệ số xác định R^2: {r2:.4f}")
    
    vif_scores = vif(X)
    print(f"Chỉ số VIF của các biến: {[round(float(v), 4) for v in vif_scores]}")
    
    H = hat_matrix(X)
    verify_solution(X, y, beta_hat, H)

def run_all_integration_tests():
    print("-------------- Integration Test --------------")
    test_integration_simple_regression()
    test_integration_multiple_regression()
    print("-----------------------------------------------")


if __name__ == "__main__":
    
    # Chạy Unit Tests mẫu của nhóm
    run_all_unit_tests()
    
    # Chạy thử nghiệm tích hợp trên tập dữ liệu mô phỏng
    run_all_integration_tests()
