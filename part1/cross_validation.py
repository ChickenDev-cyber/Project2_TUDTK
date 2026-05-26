import numpy as np
from ridge_lasso import ridge_fit
import sys


DEFAULT_RANDOM_STATE = 42
DEFAULT_LAMBDAS = np.logspace(-3, 3, 60)


def _make_folds(n_samples: int, k: int):
    if k < 2 or k > n_samples:
        raise ValueError("k must be between 2 and the number of samples")

    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    indices = rng.permutation(n_samples)
    return np.array_split(indices, k)


def _ridge_cv_score(X: np.ndarray, y: np.ndarray, k: int, lam: float) -> float:
    if lam < 0:
        raise ValueError("lam must be non-negative")

    folds = _make_folds(len(y), k)
    mses = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        w = ridge_fit(X[train_idx], y[train_idx], lam)
        y_pred = X[val_idx] @ w
        mse = np.mean((y[val_idx] - y_pred) ** 2)
        mses.append(mse)

    return float(np.mean(mses))


def kfold_cv(X: np.ndarray, y: np.ndarray, k: int) -> float:
    """
    Đánh giá mô hình OLS bằng K-Fold Cross Validation.
    Lý luận: k=5 hoặc 10 là mức cân bằng giữa bias và variance trong đánh giá.
    DEFAULT_RANDOM_STATE được cố định trong module để kết quả tái lập được.
    """
    return _ridge_cv_score(X, y, k, lam=0.0)


def ridge_cv_score(X: np.ndarray, y: np.ndarray, k: int, lam: float) -> float:
    """Tính CV-MSE cho Ridge với lambda chỉ định, dùng cùng fold với kfold_cv."""
    return _ridge_cv_score(X, y, k, lam)


def ridge_lambda_search(X: np.ndarray, y: np.ndarray, k: int):
    """Quét dải lambda mặc định và trả về lambda có CV-MSE nhỏ nhất."""
    cv_scores = np.array([ridge_cv_score(X, y, k, lam) for lam in DEFAULT_LAMBDAS])
    best_idx = int(np.argmin(cv_scores))
    best_lam = float(DEFAULT_LAMBDAS[best_idx])
    best_score = float(cv_scores[best_idx])

    return DEFAULT_LAMBDAS.copy(), cv_scores, best_lam, best_score

def test_kfold_cv():
    """
    Unit tests kiểm tra tính ổn định của hàm kfold_cv theo style nhóm.
    """
    np.random.seed(100)
    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    # --- Test 1: Kiểm tra tính ổn định với random_state cố định trong hàm ---
    err1 = kfold_cv(X, y, k=3)
    err2 = kfold_cv(X, y, k=3)
    
    if np.isclose(err1, err2):
        print("So sánh kết quả MSE khi dùng random_state cố định: Giống")
    else:
        print("So sánh kết quả MSE khi dùng random_state cố định: Khác")

    # --- Test 2: Kiểm tra helper chọn lambda Ridge ---
    lambdas, scores, best_lam, best_score = ridge_lambda_search(X, y, k=3)
    if best_lam in lambdas and np.isclose(best_score, np.min(scores)):
        print("Kiểm tra chọn lambda tốt nhất cho Ridge: Giống")
    else:
        print("Kiểm tra chọn lambda tốt nhất cho Ridge: Khác")

if __name__ == "__main__":
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    test_kfold_cv()
