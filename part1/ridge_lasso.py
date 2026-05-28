import matplotlib.pyplot as plt
import random

from matrix_ops import (
    Vector,
    add_to_diagonal,
    all_close,
    as_matrix,
    as_vector,
    column,
    logspace,
    matmul,
    matvec,
    max_abs_diff,
    mean,
    solve,
    transpose,
)


def _solve_linear_system(A, b):
    return solve(A, b)


def ridge_fit(X, y, lam):
    """
    Tính hệ số Ridge bằng công thức đóng, dùng đại số tuyến tính tự cài đặt.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")

    X = as_matrix(X)
    y = as_vector(y)
    
    is_intercept = True
    for row in X:
        if abs(row[0] - 1.0) > 1e-12:
            is_intercept = False
            break
            
    Xt = transpose(X)
    A = add_to_diagonal(matmul(Xt, X), lam, skip_first=is_intercept)
    b = matvec(Xt, y)
    return _solve_linear_system(A, b)


def plot_ridge_trace(X, y):
    """
    Vẽ Ridge Trace để quan sát hệ số thay đổi như thế nào khi lambda tăng.
    """
    X = as_matrix(X)
    y = as_vector(y)
    lambdas = logspace(-3, 5, 200)
    weights = [ridge_fit(X, y, lam) for lam in lambdas]

    plt.figure(figsize=(10, 6), dpi=100)
    n_features = len(X[0])
    for i in range(n_features):
        plt.plot(lambdas, [w[i] for w in weights], linewidth=2, label=f'Đặc trưng {i+1}')

    plt.xscale('log')
    plt.title('Biểu đồ Ridge Trace', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Hệ số điều chỉnh Lambda', fontsize=11, labelpad=10)
    plt.ylabel('Giá trị của Hệ số hồi quy', fontsize=11, labelpad=10)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10, shadow=True)
    plt.tight_layout()
    plt.show()


def lasso_fit_cd(X, y, lam, max_iter=2000, tol=1e-6):
    """
    Cài đặt Lasso Regression bằng Coordinate Descent.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")

    X = as_matrix(X)
    y = as_vector(y)
    n, p1 = len(X), len(X[0])
    w = Vector([0.0] * p1)

    def _soft_threshold(rho, threshold):
        if rho > threshold:
            return rho - threshold
        if rho < -threshold:
            return rho + threshold
        return 0.0

    for _ in range(max_iter):
        w_old = Vector(w)

        for j in range(p1):
            rho_j = 0.0
            z_j = 0.0
            for i in range(n):
                prediction_without_j = sum(X[i][k] * w[k] for k in range(p1) if k != j)
                rho_j += X[i][j] * (y[i] - prediction_without_j)
                z_j += X[i][j] ** 2

            if z_j < 1e-12:
                w[j] = 0.0
                continue

            penalty = lam if j > 0 else 0.0
            w[j] = _soft_threshold(rho_j, penalty) / z_j

        if max_abs_diff(w, w_old) < tol:
            break

    return w


def test_ridge_fit():
    print("--- Kiểm tra Ridge Regression ---")

    X1 = [[1, 0], [0, 1]]
    y1 = [1, 2]
    expected_w1 = [1.0, 2.0]
    w1 = ridge_fit(X1, y1, lam=0.0)
    print("[Ridge] Lambda=0 khớp OLS:", "Giống" if all_close(w1, expected_w1) else "Khác")

    X2 = [[1, 0, 2], [1, 1, 3], [1, 2, 4], [1, 3, 5]]
    y2 = [1, 2, 3, 4]
    w_small = ridge_fit(X2, y2, lam=0.1)
    w_big = ridge_fit(X2, y2, lam=1e9)
    print("[Ridge] Lambda cực lớn -> beta ~ 0:", "Giống" if all(abs(v) < 1e-4 for v in w_big) else "Khác")
    print(f"  ||beta small||^2 = {sum(v*v for v in w_small):.6f}")


def test_lasso_fit_cd():
    """
    Unit tests kiểm tra tính đúng đắn của hàm lasso_fit_cd.
    sklearn chỉ dùng để kiểm chứng kết quả.
    """
    from sklearn.linear_model import Lasso as SkLasso

    rng = random.Random(42)
    n, p = 200, 4
    X_feat = [[rng.gauss(0.0, 1.0) for _ in range(p)] for _ in range(n)]
    X = [[1.0] + row for row in X_feat]
    beta_true = [2.0, 3.0, 0.0, -2.0, 0.0]
    y = [sum(row[j] * beta_true[j] for j in range(len(beta_true))) + rng.gauss(0.0, 0.5) for row in X]

    alpha = 0.1
    w_cd = lasso_fit_cd(X, y, lam=alpha * n, max_iter=3000)

    sk = SkLasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    sk.fit(X_feat, y)
    w_sk = [float(sk.intercept_)] + [float(v) for v in sk.coef_]

    if all_close(w_cd, w_sk, atol=0.1):
        print("[Lasso-CD] Hệ số gần với sklearn (atol=0.1): Giống")
    else:
        print("[Lasso-CD] Hệ số gần với sklearn (atol=0.1): Khác")
        print(f"  CD  : {[round(v, 4) for v in w_cd]}")
        print(f"  sk  : {[round(v, 4) for v in w_sk]}")

    n_zero_cd = sum(abs(v) < 1e-2 for v in w_cd[1:])
    print("[Lasso-CD] Tạo nghiệm thưa:", "Giống" if n_zero_cd >= 1 else "Khác")

    w_big = lasso_fit_cd(X, y, lam=1e6, max_iter=1000)
    print("[Lasso-CD] Lambda cực lớn -> beta[1:] ~ 0:", "Giống" if all(abs(v) < 1e-4 for v in w_big[1:]) else "Khác")


if __name__ == "__main__":
    test_ridge_fit()
    test_lasso_fit_cd()
