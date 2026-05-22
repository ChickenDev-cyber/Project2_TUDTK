import numpy as np
from ridge_lasso import ridge_fit
import sys

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
    """
    Unit tests kiểm tra tính ổn định của hàm kfold_cv theo style nhóm.
    """
    np.random.seed(100)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    # --- Test 1: Kiểm tra tính ổn định với cùng một seed ---
    err1 = kfold_cv(X, y, k=3, lam=0.1, seed=42)
    err2 = kfold_cv(X, y, k=3, lam=0.1, seed=42)
    
    if np.isclose(err1, err2):
        print("So sánh kết quả MSE khi trùng seed: Giống")
    else:
        print("So sánh kết quả MSE khi trùng seed: Khác")

    # --- Test 2: Kiểm tra tính ngẫu nhiên khi đổi seed ---
    err3 = kfold_cv(X, y, k=3, lam=0.1, seed=123)
    
    # Kỳ vọng 2 kết quả phải KHÁC nhau khi đổi seed
    if not np.isclose(err1, err3):
        print("So sánh độ lệch MSE khi đổi seed (kỳ vọng khác nhau): Giống")
    else:
        print("So sánh độ lệch MSE khi đổi seed (kỳ vọng khác nhau): Khác")

if __name__ == "__main__":
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    test_kfold_cv()