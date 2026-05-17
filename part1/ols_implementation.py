import numpy as np
from scipy import stats
import warnings

def ols_fit(X, y):
    X_T = X.T
    beta_hat = np.linalg.inv(X_T @ X) @ X_T @ y
    
    y_hat = X @ beta_hat
    phan_du = y - y_hat
    RSS = np.sum(phan_du**2)
    
    n = X.shape[0]
    p_cong_1 = X.shape[1]
    
    sigma2_hat = RSS / (n - p_cong_1)
    
    return beta_hat, sigma2_hat

def hat_matrix(X):
    X_T = X.T
    H = X @ np.linalg.inv(X_T @ X) @ X_T
    return H

def model_metrics(y, y_hat, p):
    n = len(y)
    RSS = np.sum((y - y_hat)**2)
    
    y_bar = np.mean(y)
    TSS = np.sum((y - y_bar)**2)
    
    r2 = 1 - (RSS / TSS)
    adj_r2 = 1 - ((n - 1) / (n - p - 1)) * (1 - r2)
    
    f_stat = ((TSS - RSS) / p) / (RSS / (n - p - 1))
    
    return RSS, TSS, r2, adj_r2, f_stat

def coef_inference(X, y, beta_hat, sigma2_hat):
    n = X.shape[0]
    p_cong_1 = X.shape[1]
    df = n - p_cong_1
    
    cov_matrix = sigma2_hat * np.linalg.inv(X.T @ X)
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
        
        beta_tam = np.linalg.inv(X_tam.T @ X_tam) @ X_tam.T @ y_tam
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
    from sklearn.metrics import r2_score
    
    if beta_hat is None: 
        print("Không có hệ số để kiểm chứng.")
        return
    
    try:
        if np.allclose(H @ H, H, atol=1e-5):
            print("-> Verify OK: Ma trận chiều H thỏa tính chất Idempotent (H^2 = H).")
        else:
            print("-> SAI: Ma trận chiều H không thỏa tính chất Idempotent.")
            
        sk_model = LinearRegression(fit_intercept=False).fit(X, y)
        if np.allclose(beta_hat, sk_model.coef_, atol=1e-5):
            print("-> Verify OK: Hệ số Beta chính xác hoàn toàn với thư viện sklearn.")
        else:
            print("-> SAI: Hệ số Beta không khớp với thư viện sklearn!")
            
    except Exception as e:
        print(f"Xảy ra lỗi khi kiểm chứng OLS: {e}")

def run_tests(test_cases):
    warnings.filterwarnings('ignore')
    
    for idx, test in enumerate(test_cases, 1):
        X = np.array(test['X'], dtype=float)
        y = np.array(test['y'], dtype=float)
        p = X.shape[1] - 1
        
        print("-" * 65)
        print(f"TEST CASE {idx}: {test['name']}")
        print(f"Kích thước ma trận (X): {X.shape}")
        print(f"Kích thước vector (y): {y.shape}")

        try:
            beta_hat, sigma2_hat = ols_fit(X, y)
            
            if beta_hat is not None:
                formatted_beta = ", ".join([f"{v:.4f}" for v in beta_hat])
                print(f"   Vector hệ số Beta: [{formatted_beta}]")
                print(f"   Phương sai nhiễu (sigma^2): {sigma2_hat:.4f}")
                
                y_hat = X @ beta_hat
                RSS, TSS, r2, adj_r2, f_stat = model_metrics(y, y_hat, p)
                print(f"   Hệ số xác định R^2: {r2:.4f}")
                
                if p > 1:
                    vif_scores = vif(X)
                    formatted_vif = ", ".join([f"{v:.4f}" for v in vif_scores])
                    print(f"   Chỉ số VIF của các biến: [{formatted_vif}]")
                
                H = hat_matrix(X)
                verify_solution(X, y, beta_hat, H)
            else:
                print("   => Không tìm được nghiệm OLS do ma trận X^T X bị suy biến.")
                
        except Exception as e:
            print(f"   [OLS] Lỗi chương trình: {e}")

        #print("-" * 65 + "\n")


# UNIT TESTS — kiểm tra từng hàm với expected output đã biết
def run_unit_tests():
    warnings.filterwarnings('ignore')

    unit_test_cases = []

    # Dữ liệu dùng chung
    X_small = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]], dtype=float)
    y_small = np.array([3, 5, 7, 9, 11], dtype=float)  # y = 1 + 2x, không nhiễu

    X_intercept = np.ones((6, 1))
    y_intercept = np.array([2, 4, 4, 4, 5, 7], dtype=float)

    np.random.seed(0)
    n_big = 50
    X_big = np.column_stack([np.ones(n_big), np.random.randn(n_big, 5)])
    y_big = 3 + 2 * X_big[:, 1] + np.random.randn(n_big)

    np.random.seed(42)
    n_inf = 200
    X_inf_feat = np.random.randn(n_inf, 2)
    X_inf = np.column_stack([np.ones(n_inf), X_inf_feat])
    y_inf = 5 + 3 * X_inf_feat[:, 0] + np.random.randn(n_inf)

    np.random.seed(7)
    n_vif = 100
    X_indep = np.column_stack([np.ones(n_vif), np.random.randn(n_vif), np.random.randn(n_vif)])
    x_base = np.random.randn(n_vif)
    X_collinear = np.column_stack([np.ones(n_vif), x_base, x_base + np.random.randn(n_vif)*0.01, np.random.randn(n_vif)])

    # ols_fit
    beta, sigma2 = ols_fit(X_small, y_small)
    beta2, sigma2_2 = ols_fit(X_intercept, y_intercept)

    unit_test_cases.append({
        'name': 'ols_fit — TC1: Beta đúng với y = 1 + 2x không nhiễu',
        'checks': [
            f"   Beta ước lượng:  {np.round(beta, 4).tolist()}",
            f"   Beta kỳ vọng:    [1.0, 2.0]",
            f"-> {'Verify OK' if np.allclose(beta, [1.0, 2.0], atol=1e-8) else 'SAI'}: beta = [1.0, 2.0]",
            f"   sigma²:          {sigma2:.2e}",
            f"-> {'Verify OK' if sigma2 < 1e-10 else 'SAI'}: sigma² ≈ 0 khi không có nhiễu",
        ]
    })

    unit_test_cases.append({
        'name': 'ols_fit — TC2: Intercept-only, beta = mean(y)',
        'checks': [
            f"   Beta ước lượng:  {beta2[0]:.4f}",
            f"   mean(y):         {np.mean(y_intercept):.4f}",
            f"-> {'Verify OK' if np.allclose(beta2, [np.mean(y_intercept)], atol=1e-8) else 'SAI'}: beta = mean(y)",
            f"   sigma²:          {sigma2_2:.4f}",
            f"   sigma² kỳ vọng: {np.var(y_intercept, ddof=1):.4f}",
            f"-> {'Verify OK' if np.allclose(sigma2_2, np.var(y_intercept, ddof=1), atol=1e-8) else 'SAI'}: sigma² = variance mẫu",
        ]
    })

    # hat_matrix
    H = hat_matrix(X_small)
    rank_H = np.linalg.matrix_rank(H)

    unit_test_cases.append({
        'name': 'hat_matrix — TC1: Tính chất Idempotent và đối xứng',
        'checks': [
            f"-> {'Verify OK' if np.allclose(H @ H, H, atol=1e-8) else 'SAI'}: H² = H (Idempotent)",
            f"-> {'Verify OK' if np.allclose(H, H.T, atol=1e-8) else 'SAI'}: H = Hᵀ (Đối xứng)",
        ]
    })

    unit_test_cases.append({
        'name': 'hat_matrix — TC2: rank(H) và tính chiếu Hy = y',
        'checks': [
            f"   rank(H):         {rank_H}",
            f"   rank kỳ vọng:    2  (= p+1)",
            f"-> {'Verify OK' if rank_H == 2 else 'SAI'}: rank(H) = p+1",
            f"-> {'Verify OK' if np.allclose(H @ y_small, y_small, atol=1e-8) else 'SAI'}: Hy = y khi y ∈ col(X)",
        ]
    })

    # model_metrics
    y_hat_small = X_small @ beta
    RSS, TSS, r2, adj_r2, f_stat = model_metrics(y_small, y_hat_small, p=1)

    y_const = np.array([1, 2, 3, 4, 5], dtype=float)
    y_hat_mean = np.full_like(y_const, np.mean(y_const))
    _, _, r2_zero, _, _ = model_metrics(y_const, y_hat_mean, p=1)

    beta_big, _ = ols_fit(X_big, y_big)
    _, _, r2_b, adj_r2_b, _ = model_metrics(y_big, X_big @ beta_big, p=5)

    unit_test_cases.append({
        'name': 'model_metrics — TC1: R² = 1 khi fit hoàn hảo',
        'checks': [
            f"   RSS:             {RSS:.2e}",
            f"   R²:              {r2:.6f}",
            f"-> {'Verify OK' if RSS < 1e-10 else 'SAI'}: RSS ≈ 0",
            f"-> {'Verify OK' if np.isclose(r2, 1.0, atol=1e-8) else 'SAI'}: R² = 1.0",
        ]
    })

    unit_test_cases.append({
        'name': 'model_metrics — TC2: R² = 0 khi ŷ = mean(y), adj_R² ≤ R²',
        'checks': [
            f"   R² (ŷ=mean):     {r2_zero:.6f}",
            f"-> {'Verify OK' if np.isclose(r2_zero, 0.0, atol=1e-8) else 'SAI'}: R² = 0 khi ŷ = mean(y)",
            f"   R²:              {r2_b:.4f}  |  adj_R²: {adj_r2_b:.4f}",
            f"-> {'Verify OK' if adj_r2_b <= r2_b + 1e-10 else 'SAI'}: adj_R² ≤ R²",
        ]
    })

    # coef_inference
    beta_inf, sigma2_inf = ols_fit(X_inf, y_inf)
    se, t_stats, p_values, ci_lower, ci_upper = coef_inference(X_inf, y_inf, beta_inf, sigma2_inf)

    unit_test_cases.append({
        'name': 'coef_inference — TC1: p-value phân biệt biến có/không ý nghĩa',
        'checks': [
            f"   p-value β₁ (≈3): {p_values[1]:.4f}",
            f"-> {'Verify OK' if p_values[1] < 0.01 else 'SAI'}: p-value < 0.01 (biến có ý nghĩa)",
            f"   p-value β₂ (≈0): {p_values[2]:.4f}",
            f"-> {'Verify OK' if p_values[2] > 0.05 else 'SAI'}: p-value > 0.05 (biến không ý nghĩa)",
        ]
    })

    unit_test_cases.append({
        'name': 'coef_inference — TC2: CI 95% chứa beta thật',
        'checks': [
            f"   CI β₀: [{ci_lower[0]:.3f}, {ci_upper[0]:.3f}]",
            f"-> {'Verify OK' if ci_lower[0] <= 5.0 <= ci_upper[0] else 'SAI'}: CI chứa β₀ = 5",
            f"   CI β₁: [{ci_lower[1]:.3f}, {ci_upper[1]:.3f}]",
            f"-> {'Verify OK' if ci_lower[1] <= 3.0 <= ci_upper[1] else 'SAI'}: CI chứa β₁ = 3",
        ]
    })

    # vif
    vif_indep = vif(X_indep)
    vif_collinear = vif(X_collinear)

    unit_test_cases.append({
        'name': 'vif — TC1: VIF ≈ 1 khi các biến độc lập nhau',
        'checks': [
            f"   VIF:             {[round(float(v),3) for v in vif_indep]}",
            f"-> {'Verify OK' if all(v < 2.0 for v in vif_indep) else 'SAI'}: Tất cả VIF < 2",
        ]
    })

    unit_test_cases.append({
        'name': 'vif — TC2: VIF lớn khi có đa cộng tuyến',
        'checks': [
            f"   VIF:             {[round(float(v),1) for v in vif_collinear]}",
            f"-> {'Verify OK' if vif_collinear[0] > 100 or vif_collinear[1] > 100 else 'SAI'}: VIF > 100 cho biến đa cộng tuyến",
            f"-> {'Verify OK' if vif_collinear[2] < 5.0 else 'SAI'}: VIF[X3] = {vif_collinear[2]:.3f} < 5 (biến không liên quan)",
        ]
    })

    for idx, test in enumerate(unit_test_cases, 1):
        print("-" * 65)
        print(f"UNIT TEST {idx}: {test['name']}")
        for line in test['checks']:
            print(line)
    print("-" * 65 + "\n")


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    # Test case 1: Hồi quy tuyến tính đơn (1 biến X)
    n1 = 50
    X1_features = np.random.rand(n1, 1) * 10
    X1 = np.column_stack([np.ones(n1), X1_features])
    y1 = 5 + 3 * X1_features[:, 0] + np.random.randn(n1) * 0.5
    
    # Test case 2: Hồi quy tuyến tính bội (3 biến X)
    n2 = 100
    X2_features = np.random.rand(n2, 3) * 5
    X2 = np.column_stack([np.ones(n2), X2_features])
    y2 = 2.5 - 1.5 * X2_features[:, 0] + 4 * X2_features[:, 1] - 0.8 * X2_features[:, 2] + np.random.randn(n2) * 1.2

    test_cases = [
        {
            'name': 'Hồi quy tuyến tính đơn (Simple Linear Regression)',
            'X': X1.tolist(),
            'y': y1.tolist()
        },
        {
            'name': 'Hồi quy tuyến tính bội (Multiple Linear Regression)',
            'X': X2.tolist(),
            'y': y2.tolist()
        }
    ]
    
    run_unit_tests()
    
    print("Integration Test")
    run_tests(test_cases)
