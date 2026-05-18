import numpy as np

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

def test_ridge_fit():
    """
    Unit tests đảm bảo tính đúng đắn của hàm ridge_fit.
    Yêu cầu: Ít nhất 2 unit test trên dữ liệu đã biết.
    """
    # Thiết lập seed để đảm bảo tính tái lập trong các test có tính ngẫu nhiên
    np.random.seed(42)

    # --- Test 1: Kiểm tra tính đúng đắn với OLS (lambda = 0) ---
    X1 = np.array([[1, 0], [0, 1]])
    y1 = np.array([1, 2])
    w1 = ridge_fit(X1, y1, lam=0)
    # Kỳ vọng: w = [1, 2]
    assert np.allclose(w1, [1, 2]), f"Test 1 thất bại: Mong đợi [1, 2], nhận được {w1}"

    # --- Test 2: Kiểm tra tính co rút (shrinkage) với lambda cực lớn ---
    X2 = np.random.randn(10, 3)
    y2 = np.random.randn(10)
    w2 = ridge_fit(X2, y2, lam=1e10)
    # Kỳ vọng: Các trọng số phải cực kỳ gần 0
    expected_zeros = np.zeros(3)
    assert np.allclose(w2, expected_zeros, atol=1e-5), f"Test 2 thất bại: Trọng số không tiến về 0. Nhận được: {w2}"

    print("✅ Ridge Fit: Tất cả Unit Tests đã vượt qua (Passed)!")

if __name__ == "__main__":
    test_ridge_fit()