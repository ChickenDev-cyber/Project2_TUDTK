import math
import sys
from statistics import NormalDist

import matplotlib.pyplot as plt

from matrix_ops import as_matrix, as_vector, diag, matvec, mean, sum_squares
from ols_implementation import hat_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _residual_diagnostics(X, y, beta_hat):
    X = as_matrix(X)
    y = as_vector(y)
    beta_hat = as_vector(beta_hat)

    n, p_plus_1 = len(X), len(X[0])
    p = p_plus_1 - 1
    y_hat = matvec(X, beta_hat)
    residuals = y - y_hat

    H = hat_matrix(X)
    leverage = diag(H)

    RSS = sum_squares(residuals)
    sigma2_hat = RSS / (n - p - 1)

    std_residuals = [
        residuals[i] / math.sqrt(sigma2_hat * (1 - leverage[i] + 1e-10))
        for i in range(n)
    ]
    cooks_d = [
        (std_residuals[i] ** 2 / (p + 1)) * (leverage[i] / (1 - leverage[i] + 1e-10))
        for i in range(n)
    ]
    return y_hat, residuals, std_residuals, cooks_d


def residual_plots(X, y, beta_hat):
    """
    Vẽ 4 biểu đồ phân tích phần dư.
    Phần tính toán chẩn đoán dùng các hàm đại số tự cài đặt; matplotlib chỉ dùng để vẽ.
    """
    y_hat, residuals, std_residuals, cooks_d = _residual_diagnostics(X, y, beta_hat)
    n = len(y_hat)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Analysis Plots', fontsize=16)

    axs[0, 0].scatter(y_hat, residuals, alpha=0.6, edgecolors='k')
    axs[0, 0].axhline(0, color='red', linestyle='dashed')
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')

    sorted_res = sorted(std_residuals)
    normal = NormalDist()
    theoretical = [normal.inv_cdf((i + 0.5) / n) for i in range(n)]
    axs[0, 1].scatter(theoretical, sorted_res, alpha=0.6, edgecolors='k')
    line_min = min(min(theoretical), min(sorted_res))
    line_max = max(max(theoretical), max(sorted_res))
    axs[0, 1].plot([line_min, line_max], [line_min, line_max], color='red', linestyle='dashed')
    axs[0, 1].set_title('Normal Q-Q Plot')
    axs[0, 1].set_xlabel('Theoretical Quantiles')
    axs[0, 1].set_ylabel('Standardized Residuals')

    scale_location = [math.sqrt(abs(v)) for v in std_residuals]
    axs[1, 0].scatter(y_hat, scale_location, alpha=0.6, edgecolors='k')
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('Sqrt(|Standardized Residuals|)')

    axs[1, 1].bar(range(n), cooks_d, alpha=0.6, color='b')
    axs[1, 1].axhline(y=4 / n, color='red', linestyle='dashed', label='Ngưỡng tham chiếu (4/n)')
    axs[1, 1].set_title("Cook's Distance")
    axs[1, 1].set_xlabel('Observation Index')
    axs[1, 1].set_ylabel("Cook's Distance")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def test_residual_analysis():
    n = 10
    p = 2
    X = [[1, i, i % 3] for i in range(n)]
    beta_hat = [1.0, 2.0, -1.0]
    y = [sum(X[i][j] * beta_hat[j] for j in range(len(beta_hat))) + (0.1 if i % 2 else -0.1) for i in range(n)]

    H = hat_matrix(X)
    leverage = diag(H)

    sum_leverage = sum(leverage)
    if abs(sum_leverage - (p + 1)) < 1e-8:
        print("Kiểm tra tổng leverage: Giống")
    else:
        print("Kiểm tra tổng leverage: Khác")

    if all(v >= -1e-10 and v <= 1.0 + 1e-10 for v in leverage):
        print("Kiểm tra giá trị leverage: Giống")
    else:
        print("Kiểm tra giá trị leverage: Khác")

    _, _, _, cooks_d = _residual_diagnostics(X, y, beta_hat)
    if all(v >= 0 for v in cooks_d):
        print("Kiểm tra Cook's distance: Giống")
    else:
        print("Kiểm tra Cook's distance: Khác")


if __name__ == "__main__":
    test_residual_analysis()

    n_mock = 200
    X_mock = [[1, i / 20, (i % 11) / 5, (i % 7) / 3] for i in range(n_mock)]
    beta_hat_mock = [2.5, 1.2, -0.8, 3.0]
    y_mock = [
        sum(X_mock[i][j] * beta_hat_mock[j] for j in range(len(beta_hat_mock)))
        + 1.5 * math.sin(i)
        for i in range(n_mock)
    ]

    print("\nĐang vẽ biểu đồ...")
    residual_plots(X_mock, y_mock, beta_hat_mock)
    print("Vẽ biểu đồ hoàn tất!")
