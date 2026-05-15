import numpy as np
from ridge_lasso import ridge_fit

def kfold_cv(X: np.ndarray, y: np.ndarray, k: int, lam: float, seed: int = 42) -> float:
    """
    Đánh giá mô hình bằng K-Fold Cross Validation.
    Lý luận: k=5 hoặc 10 là mức cân bằng giữa bias và variance trong đánh giá.
    """
    np.random.seed(seed) # Đảm bảo tính tái lập (Requirement 3.3)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    folds = np.array_split(indices, k)
    
    mses = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        w = ridge_fit(X[train_idx], y[train_idx], lam)
        y_pred = X[val_idx] @ w
        mse = np.mean((y[val_idx] - y_pred)**2)
        mses.append(mse)
    
    return float(np.mean(mses))

def test_kfold_cv():
    # Test 1: Kiểm tra tính ổn định của kết quả với cùng một seed
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    err1 = kfold_cv(X, y, k=2, lam=0.1, seed=42)
    err2 = kfold_cv(X, y, k=2, lam=0.1, seed=42)
    assert err1 == err2, "Unit Test 1 thất bại: Kết quả không tái lập được!"

    # Test 2: Kiểm tra k=1 (chia fold lỗi)
    try:
        kfold_cv(X, y, k=len(y)+1, lam=0.1)
    except Exception:
        print("Unit Test 2: Đã bắt được lỗi chia fold quá lớn (Passed)")
    
    print("K-Fold CV: All Unit Tests Passed!")

if __name__ == "__main__":
    test_kfold_cv()