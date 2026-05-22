import numpy as np
import matplotlib.pyplot as plt
import sys


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Tính toán trọng số Ridge Regression bằng công thức đóng.
    Lý luận: lam giúp kiểm soát độ lớn trọng số, ngăn chặn overfitting.
    """
    # Đảm bảo y có dạng cột (N, 1) để phép nhân ma trận nhất quán
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    n_features = X.shape[1]
    I = np.eye(n_features)
    
    # Giải phương trình: (X^T @ X + lam * I) @ w = X^T @ y
    A = X.T @ X + lam * I
    b = X.T @ y
    
    # Sử dụng np.linalg.solve thay vì inv() để tăng độ chính xác số học
    w = np.linalg.solve(A, b)
    return w.flatten()

def plot_ridge_trace(X: np.ndarray, y: np.ndarray):
    """
    Vẽ biểu đồ Ridge Trace hiển thị sự biến thiên của các hệ số hồi quy theo Lambda.
    Mục đích: Giúp nhà nghiên cứu quan sát trực quan quá trình co rút (shrinkage) 
              của trọng số để đưa ra lý luận chọn vùng Lambda phù hợp.
    """
    # 1. Khởi tạo dải giá trị Lambda theo cấp số nhân (từ 10^-3 đến 10^5)
    # Quét rộng giúp nhìn rõ điểm gãy và vùng trọng số bị triệt tiêu
    lambdas = np.logspace(-3, 5, 200)
    
    # 2. Tính toán trọng số w ứng với mỗi giá trị Lambda
    weights = []
    for lam in lambdas:
        w = ridge_fit(X, y, lam)
        weights.append(w)
    
    # Chuyển thành ma trận NumPy để dễ trích xuất theo cột (từng đặc trưng)
    weights = np.array(weights)
    
    # 3. Khởi tạo cấu trúc biểu đồ chuẩn kỹ thuật
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Vẽ từng đường trọng số ứng với mỗi đặc trưng
    n_features = X.shape[1]
    for i in range(n_features):
        plt.plot(lambdas, weights[:, i], linewidth=2, label=f'Đặc trưng {i+1}')
    
    # 4. Định dạng cấu hình biểu đồ
    plt.xscale('log')  
    
    
    plt.title('Biểu đồ Ridge Trace', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Hệ số điều chỉnh Lambda', fontsize=11, labelpad=10)
    plt.ylabel('Giá trị của Hệ số hồi quy', fontsize=11, labelpad=10)
    
    # Hiển thị lưới đồ thị (grid) cho cả hai trục để dễ gióng giá trị
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    
    # Chú thích phân biệt các đường đặc trưng
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10, shadow=True)
    
    # Tối ưu hóa không gian hiển thị, tránh mất chữ ở rìa ảnh
    plt.tight_layout()
    plt.show()


def lasso_fit_cd(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Cài đặt Lasso Regression bằng thuật toán Coordinate Descent từ đầu.

    Lý luận:
        Lasso (L1) không có closed-form vì |β|₁ không khả vi tại 0.
        Coordinate Descent cập nhật lần lượt từng toạ độ β_j, giữ nguyên
        các toạ độ còn lại, sử dụng toán tử soft-thresholding:
            S(ρ, λ) = sign(ρ) · max(|ρ| - λ, 0)

    Công thức cập nhật tại mỗi bước j:
        ρ_j   = X_j^T (y - X @ β + β_j * X_j)     (partial residual)
        z_j   = X_j^T X_j                          (chuẩn hoá)
        β_j   = S(ρ_j, λ) / z_j

    Args:
        X        : Ma trận đặc trưng (n, p+1), bao gồm cột intercept.
        y        : Vector nhãn (n,).
        lam      : Hệ số chính quy hoá λ ≥ 0. Nếu lam=0 → OLS.
        max_iter : Số vòng lặp tối đa qua toàn bộ tọa độ.
        tol      : Ngưỡng hội tụ — dừng khi max_j |Δβ_j| < tol.

    Returns:
        w : Vector hệ số (p+1,), đã hội tụ hoặc sau max_iter vòng.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n, p1 = X.shape  # p1 = p + 1
    w = np.zeros((p1, 1))  # Khởi tạo β = 0

    def _soft_threshold(rho: float, threshold: float) -> float:
        """Toán tử soft-thresholding S(ρ, λ)."""
        if rho > threshold:
            return rho - threshold
        elif rho < -threshold:
            return rho + threshold
        else:
            return 0.0

    for iteration in range(max_iter):
        w_old = w.copy()

        for j in range(p1):
            # Partial residual: phần dư khi loại bỏ đóng góp của β_j
            r_j = y - X @ w + w[j] * X[:, j:j+1]

            # Tính ρ_j = X_j^T r_j (dot product)
            rho_j = float(X[:, j] @ r_j.flatten())

            # z_j = ‖X_j‖²
            z_j = float(X[:, j] @ X[:, j])

            if z_j < 1e-12:  # Bảo vệ chia cho 0
                w[j] = 0.0
                continue

            # Không regularize intercept (cột 0 thường là cột hằng)
            penalty = lam if j > 0 else 0.0

            w[j] = _soft_threshold(rho_j, penalty) / z_j

        # Kiểm tra hội tụ
        if np.max(np.abs(w - w_old)) < tol:
            break

    return w.flatten()

    
def test_ridge_fit():
    """
    Unit tests kiểm tra tính đúng đắn của hàm ridge_fit theo style nhóm.
    """
    np.random.seed(42)

    # --- Test 1: Lambda = 0 thì Ridge = OLS ---
    X1 = np.array([[1, 0], [0, 1]])
    y1 = np.array([1, 2])
    expected_w1 = np.array([1.0, 2.0])
    w1 = ridge_fit(X1, y1, lam=0)

    if np.allclose(w1, expected_w1):
        print("[Ridge] Lambda=0 khớp OLS: Giống")
    else:
        print("[Ridge] Lambda=0 khớp OLS: Khác")

    # --- Test 2: Lambda rất lớn thì hệ số co về 0 ---
    X2 = np.random.randn(10, 3)
    y2 = np.random.randn(10)
    w2 = ridge_fit(X2, y2, lam=1e10)

    if np.allclose(w2, np.zeros(3), atol=1e-5):
        print("[Ridge] Lambda cực lớn -> beta ~ 0: Giống")
    else:
        print("[Ridge] Lambda cực lớn -> beta ~ 0: Khác")


def test_lasso_fit_cd():
    """
    Unit tests kiểm tra tính đúng đắn của hàm lasso_fit_cd.
    So sánh kết quả với sklearn.linear_model.Lasso.
    """
    from sklearn.linear_model import Lasso as SkLasso
    np.random.seed(42)

    n, p = 200, 4
    X_feat = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_feat])
    # Giá trị thực: intercept=2.0, coeffs=[3.0, 0.0, -2.0, 0.0] (2 hệ số = 0)
    beta_true = np.array([2.0, 3.0, 0.0, -2.0, 0.0])
    y = X @ beta_true + np.random.randn(n) * 0.5

    lam = 0.1
    w_cd = lasso_fit_cd(X, y, lam=lam, max_iter=3000)

    # Kiểm chứng với sklearn (fit_intercept=False vì X đã có cột 1)
    sk = SkLasso(alpha=lam, fit_intercept=False, max_iter=10000)
    sk.fit(X, y)
    w_sk = sk.coef_

    # --- Test 1: Hệ số gần với sklearn (rtol=10%) ---
    if np.allclose(w_cd, w_sk, atol=0.1):
        print("[Lasso-CD] Hệ số gần với sklearn (atol=0.1): Giống")
    else:
        print("[Lasso-CD] Hệ số gần với sklearn (atol=0.1): Khác")
        print(f"  CD  : {np.round(w_cd, 4)}")
        print(f"  sk  : {np.round(w_sk, 4)}")

    # --- Test 2: Lasso tạo ra sparse solution (hệ số triệt tiêu) ---
    n_zero_cd = np.sum(np.abs(w_cd[1:]) < 1e-2)   # bỏ intercept
    if n_zero_cd >= 1:
        print(f"[Lasso-CD] Có {n_zero_cd} hệ số bị triệt tiêu (sparse): Giống")
    else:
        print("[Lasso-CD] Không có hệ số bị triệt tiêu: Khác")

    # --- Test 3: Lambda rất lớn -> tất cả hệ số (không intercept) -> 0 ---
    w_big = lasso_fit_cd(X, y, lam=1000.0)
    if np.allclose(w_big[1:], np.zeros(p), atol=1e-3):
        print("[Lasso-CD] Lambda cực lớn -> beta[1:] ~ 0: Giống")
    else:
        print("[Lasso-CD] Lambda cực lớn -> beta[1:] ~ 0: Khác")
        
if __name__ == "__main__":
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    print("===== Unit Tests: Ridge & Lasso =====")
    test_ridge_fit()
    print()
    test_lasso_fit_cd()
    print("=====================================")