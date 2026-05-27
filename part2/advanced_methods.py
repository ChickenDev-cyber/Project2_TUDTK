import math
import os
import sys
from statistics import NormalDist

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot

# Import các hàm đại số tuyến tính tự cài đặt từ Part 1
_PART1_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part1'))
if _PART1_DIR not in sys.path:
    sys.path.insert(0, _PART1_DIR)

from matrix_ops import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    identity,
    inverse,
    matmul,
    matvec,
    solve,
    transpose,
    zeros,
)


# ──────────────────────────────────────────────
# Hàm tiện ích
# ──────────────────────────────────────────────

def _to_matrix_rows(data):
    """Chuyển DataFrame / array / list sang list-of-lists thuần Python."""
    if hasattr(data, 'values'):
        return data.values.tolist()
    if isinstance(data, (Matrix, list)):
        return [list(row) for row in data]
    return [list(row) for row in data]


def _to_flat_list(data):
    """Chuyển Series / array / Vector sang list phẳng."""
    if hasattr(data, 'values'):
        v = data.values
        return v.flatten().tolist() if hasattr(v, 'flatten') else list(v)
    if isinstance(data, (list, Vector)):
        return list(data)
    return list(data)


def _add_intercept_rows(rows):
    """Thêm cột 1 (intercept) vào đầu mỗi hàng."""
    return as_matrix([[1.0] + [float(v) for v in row] for row in rows])


def _scalar_matrix(scalar, M):
    """Nhân vô hướng với ma trận: scalar * M."""
    return Matrix([Vector(scalar * M[i][j] for j in range(len(M[0]))) for i in range(len(M))])


def _sq_euclidean_distances(X1, X2):
    """Tính ma trận khoảng cách Euclidean bình phương giữa hai tập điểm."""
    n1, n2 = len(X1), len(X2)
    d = len(X1[0])
    result = zeros(n1, n2)
    for i in range(n1):
        for j in range(n2):
            dist = sum((float(X1[i][k]) - float(X2[j][k])) ** 2 for k in range(d))
            result[i][j] = dist
    return result


# ──────────────────────────────────────────────
# Bayesian Linear Regression
# ──────────────────────────────────────────────

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.m_N = None   # Vector: posterior mean
        self.S_N = None   # Matrix: posterior covariance

    def fit(self, X, y):
        X_rows = _to_matrix_rows(X)
        y_vec = as_vector(_to_flat_list(y))

        X_design = _add_intercept_rows(X_rows)
        M = len(X_design[0])

        # S_0_inv = alpha * I_M
        S_0_inv = _scalar_matrix(self.alpha, identity(M))

        # S_N = inv(S_0_inv + beta * X^T X)
        XtX = matmul(transpose(X_design), X_design)
        beta_XtX = _scalar_matrix(self.beta, XtX)

        # S_0_inv + beta * X^T X
        sum_mat = Matrix([
            Vector(S_0_inv[i][j] + beta_XtX[i][j] for j in range(M))
            for i in range(M)
        ])
        self.S_N = inverse(sum_mat)

        # m_N = beta * S_N @ X^T y
        Xty = matvec(transpose(X_design), y_vec)
        self.m_N = Vector(
            self.beta * v for v in matvec(self.S_N, Xty)
        )
        return self

    def get_credible_interval(self, X):
        X_rows = _to_matrix_rows(X)
        X_design = _add_intercept_rows(X_rows)
        n = len(X_design)
        M = len(X_design[0])

        # y_pred = X_design @ m_N
        y_pred_log = matvec(X_design, self.m_N)

        # pred_variance[i] = 1/beta + X_design[i]^T @ S_N @ X_design[i]
        # = 1/beta + sum_j (XS[i][j] * X_design[i][j])
        XS = matmul(X_design, self.S_N)
        pred_variance = Vector(
            1.0 / self.beta + sum(XS[i][j] * X_design[i][j] for j in range(M))
            for i in range(n)
        )
        pred_std = Vector(math.sqrt(max(v, 0.0)) for v in pred_variance)

        z_score = 1.96
        lower_log = Vector(y_pred_log[i] - z_score * pred_std[i] for i in range(n))
        upper_log = Vector(y_pred_log[i] + z_score * pred_std[i] for i in range(n))

        return y_pred_log, lower_log, upper_log


# ──────────────────────────────────────────────
# Kernel Ridge Regression (RBF Kernel)
# ──────────────────────────────────────────────

class KernelRidgeRegression:
    def __init__(self, lam=1.0, gamma=0.1):
        self.lam = lam
        self.gamma = gamma
        self.X_train = None   # Matrix (part1)
        self.y_train = None   # Vector (part1)
        self.alpha_coef = None  # Vector (part1)

    def _compute_rbf_kernel(self, X1, X2):
        """Công thức RBF Kernel: K(x,y) = exp(-gamma * ||x-y||^2)"""
        sq_dists = _sq_euclidean_distances(X1, X2)
        n1, n2 = len(sq_dists), len(sq_dists[0])
        return Matrix([
            Vector(math.exp(-self.gamma * sq_dists[i][j]) for j in range(n2))
            for i in range(n1)
        ])

    def fit(self, X, y):
        self.X_train = as_matrix(_to_matrix_rows(X))
        self.y_train = as_vector(_to_flat_list(y))

        n_samples = len(self.X_train)
        K = self._compute_rbf_kernel(self.X_train, self.X_train)

        # A = K + lambda * I
        A = Matrix([
            Vector(
                K[i][j] + (self.lam if i == j else 0.0)
                for j in range(n_samples)
            )
            for i in range(n_samples)
        ])

        # alpha = A^{-1} y  (giải hệ phương trình tuyến tính)
        self.alpha_coef = solve(A, self.y_train)
        return self

    def predict(self, X):
        X_test = as_matrix(_to_matrix_rows(X))
        K_test = self._compute_rbf_kernel(X_test, self.X_train)
        return matvec(K_test, self.alpha_coef)


# ──────────────────────────────────────────────
# Biểu đồ chẩn đoán phần dư cho Kernel RBF
# ──────────────────────────────────────────────

def plot_kernel_diagnostics(y_true_log, y_pred_log, X_train=None, X_test=None, gamma=None, lam=None):
    """
    Biểu đồ chẩn đoán phần dư cho Kernel RBF.
    Gồm Scale-Location và Cook's Distance theo yêu cầu.
    """
    y_t = _to_flat_list(y_true_log)
    y_p = _to_flat_list(y_pred_log)

    y_true_real = [math.expm1(v) for v in y_t]
    y_pred_real = [math.expm1(v) for v in y_p]
    n = len(y_true_real)

    residuals = [y_true_real[i] - y_pred_real[i] for i in range(n)]
    res_mean = sum(residuals) / n
    res_std = math.sqrt(sum((r - res_mean) ** 2 for r in residuals) / n) + 1e-10
    std_residuals = [(r - res_mean) / res_std for r in residuals]

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Chẩn đoán Phần dư (Residual Diagnostics) - Mô hình Tối ưu Kernel RBF',
                 fontsize=18, fontweight='bold', y=1.02)

    # 1. Residuals vs Predicted
    sns.scatterplot(x=y_pred_real, y=residuals, ax=axes[0, 0], color='crimson', alpha=0.5)
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[0, 0].set_title('1. Phần dư vs Dự đoán (Residuals vs Fitted)')
    axes[0, 0].set_xlabel('Giá trị dự đoán (Predicted AQI)')
    axes[0, 0].set_ylabel('Phần dư (Residuals)')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    # 2. Q-Q Plot (standardized residuals + đường hồi quy fit — giống stats.probplot)
    sorted_res = sorted(std_residuals)
    normal = NormalDist()
    theoretical = [normal.inv_cdf((i + 0.5) / n) for i in range(n)]
    axes[0, 1].scatter(theoretical, sorted_res, alpha=0.5, color='crimson', edgecolors='crimson')
    # Fit đường hồi quy OLS qua dữ liệu (giống stats.probplot)
    t_mean = sum(theoretical) / n
    r_mean = sum(sorted_res) / n
    slope = sum((theoretical[i] - t_mean) * (sorted_res[i] - r_mean) for i in range(n)) / \
            (sum((theoretical[i] - t_mean) ** 2 for i in range(n)) + 1e-10)
    intercept = r_mean - slope * t_mean
    t_min, t_max = theoretical[0], theoretical[-1]
    axes[0, 1].plot([t_min, t_max], [intercept + slope * t_min, intercept + slope * t_max],
                    color='black', linestyle='--', linewidth=2)
    axes[0, 1].set_title('2. Biểu đồ Q-Q (Kiểm tra Phân phối chuẩn)')
    axes[0, 1].set_xlabel('Theoretical Quantiles')
    axes[0, 1].set_ylabel('Standardized Residuals')
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # 3. Histogram of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='crimson', bins=40)
    axes[1, 0].set_title('3. Phân phối Phần dư (Histogram of Residuals)')
    axes[1, 0].set_xlabel('Giá trị Phần dư (Residuals)')
    axes[1, 0].set_ylabel('Tần suất')

    # 4. Actual vs Predicted
    sns.scatterplot(x=y_true_real, y=y_pred_real, ax=axes[1, 1], color='crimson', alpha=0.5)
    max_val = max(max(y_true_real), max(y_pred_real))
    axes[1, 1].plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=2)
    axes[1, 1].set_title('4. Thực tế vs Dự đoán (Actual vs Predicted)')
    axes[1, 1].set_xlabel('Thực tế (Actual AQI)')
    axes[1, 1].set_ylabel('Dự đoán (Predicted AQI)')
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    # 5. Residuals Sequence
    axes[2, 0].plot(residuals, color='crimson', alpha=0.7, linewidth=1.5)
    axes[2, 0].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[2, 0].set_title('5. Chuỗi Phần dư theo Thứ tự (Residuals Sequence)')
    axes[2, 0].set_xlabel('Thứ tự mẫu thử (Index)')
    axes[2, 0].set_ylabel('Phần dư (Residuals)')
    axes[2, 0].grid(True, linestyle=':', alpha=0.6)

    # 6. Autocorrelation Plot (ACF)
    import pandas as pd
    autocorrelation_plot(pd.Series(residuals), ax=axes[2, 1], color='crimson')
    axes[2, 1].set_title('6. Biểu đồ Tự tương quan của Phần dư (ACF)')
    axes[2, 1].set_xlabel('Độ trễ (Lag)')
    axes[2, 1].set_ylabel('Hệ số tự tương quan')

    # 7. Scale-Location
    scale_loc = [math.sqrt(abs(v)) for v in std_residuals]
    axes[3, 0].scatter(y_pred_real, scale_loc, alpha=0.5, color='crimson')
    axes[3, 0].set_title('7. Scale-Location (Kiểm tra phương sai đồng đều)')
    axes[3, 0].set_xlabel('Fitted values')
    axes[3, 0].set_ylabel('Sqrt(|Standardized Residuals|)')
    axes[3, 0].grid(True, linestyle=':', alpha=0.6)

    # 8. Cook's Distance
    # Xấp xỉ Cook's Distance cho mô hình phi tuyến Kernel RBF
    p_approx = len(X_train[0]) if X_train is not None else 10

    if X_train is not None and X_test is not None and gamma is not None and lam is not None:
        X_tr = as_matrix(_to_matrix_rows(X_train))
        X_te = as_matrix(_to_matrix_rows(X_test))
        n_tr = len(X_tr)

        K_test = _sq_euclidean_distances(X_te, X_tr)
        K_test = Matrix([
            Vector(math.exp(-gamma * K_test[i][j]) for j in range(n_tr))
            for i in range(len(X_te))
        ])

        K_train = _sq_euclidean_distances(X_tr, X_tr)
        K_train = Matrix([
            Vector(math.exp(-gamma * K_train[i][j]) for j in range(n_tr))
            for i in range(n_tr)
        ])

        # A = K_train + lam * I
        A = Matrix([
            Vector(
                K_train[i][j] + (lam if i == j else 0.0)
                for j in range(n_tr)
            )
            for i in range(n_tr)
        ])
        A_inv = inverse(A)

        # H_approx = K_test @ A_inv @ K_test^T
        H_approx = matmul(matmul(K_test, A_inv), transpose(K_test))
        leverage = [max(0.0, min(H_approx[i][i], 0.99)) for i in range(n)]
    else:
        leverage = [p_approx / n] * n

    cooks_d = [
        (std_residuals[i] ** 2 / (p_approx + 1)) * (leverage[i] / ((1 - leverage[i] + 1e-10) ** 2))
        for i in range(n)
    ]
    axes[3, 1].bar(range(n), cooks_d, alpha=0.6, color='crimson')
    axes[3, 1].axhline(y=4 / n, color='black', linestyle='--', label='Ngưỡng 4/n')
    axes[3, 1].set_title("8. Khoảng cách Cook (Cook's Distance)")
    axes[3, 1].set_xlabel('Observation Index')
    axes[3, 1].set_ylabel("Cook's Distance")
    axes[3, 1].legend()
    axes[3, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
