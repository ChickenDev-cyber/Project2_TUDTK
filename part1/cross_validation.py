import random
import sys

from matrix_ops import Vector, as_matrix, as_vector, logspace, matvec, take_rows, take_values
from ridge_lasso import ridge_fit


DEFAULT_RANDOM_STATE = 42
DEFAULT_LAMBDAS = logspace(-3, 3, 60)


def _make_folds(n_samples, k):
    if k < 2 or k > n_samples:
        raise ValueError("k must be between 2 and the number of samples")

    indices = list(range(n_samples))
    rng = random.Random(DEFAULT_RANDOM_STATE)
    rng.shuffle(indices)

    base_size = n_samples // k
    remainder = n_samples % k
    folds = []
    start = 0
    for i in range(k):
        size = base_size + (1 if i < remainder else 0)
        folds.append(indices[start:start + size])
        start += size
    return folds


def _ridge_cv_score(X, y, k, lam):
    if lam < 0:
        raise ValueError("lam must be non-negative")

    X = as_matrix(X)
    y = as_vector(y)
    folds = _make_folds(len(y), k)
    mses = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = [idx for j, fold in enumerate(folds) if j != i for idx in fold]

        w = ridge_fit(take_rows(X, train_idx), take_values(y, train_idx), lam)
        y_pred = matvec(take_rows(X, val_idx), w)
        y_val = take_values(y, val_idx)
        mse = sum((y_val[j] - y_pred[j]) ** 2 for j in range(len(y_val))) / len(y_val)
        mses.append(mse)

    return sum(mses) / len(mses)


def kfold_cv(X, y, k):
    """
    Đánh giá mô hình OLS bằng K-Fold Cross Validation.
    DEFAULT_RANDOM_STATE được cố định trong module để kết quả tái lập được.
    """
    return _ridge_cv_score(X, y, k, lam=0.0)


def ridge_cv_score(X, y, k, lam):
    """Tính CV-MSE cho Ridge với lambda chỉ định, dùng cùng fold với kfold_cv."""
    return _ridge_cv_score(X, y, k, lam)


def ridge_lambda_search(X, y, k):
    """Quét dải lambda mặc định và trả về lambda có CV-MSE nhỏ nhất."""
    cv_scores = Vector(ridge_cv_score(X, y, k, lam) for lam in DEFAULT_LAMBDAS)
    best_idx = min(range(len(cv_scores)), key=lambda i: cv_scores[i])
    best_lam = float(DEFAULT_LAMBDAS[best_idx])
    best_score = float(cv_scores[best_idx])
    return Vector(DEFAULT_LAMBDAS), cv_scores, best_lam, best_score


def test_kfold_cv():
    X = [[1, i, i % 3] for i in range(20)]
    y = [2 + 0.5 * row[1] - row[2] for row in X]

    err1 = kfold_cv(X, y, k=3)
    err2 = kfold_cv(X, y, k=3)

    if abs(err1 - err2) < 1e-12:
        print("So sánh kết quả MSE khi dùng random_state cố định: Giống")
    else:
        print("So sánh kết quả MSE khi dùng random_state cố định: Khác")

    lambdas, scores, best_lam, best_score = ridge_lambda_search(X, y, k=3)
    if best_lam in lambdas and abs(best_score - min(scores)) < 1e-12:
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
