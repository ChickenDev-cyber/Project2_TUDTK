
import numpy as np
import sys
import os
part1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'part1'))
if part1_path not in sys.path:
    sys.path.append(part1_path)

from matrix_ops import (
    inverse, solve, transpose, matmul, identity, matrix_add, as_matrix, as_vector
)
import math

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pandas.plotting import autocorrelation_plot

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None

    def fit(self, X, y):
        # Chuyển sang list 
        X_mat = X.values.tolist() if hasattr(X, 'values') else X.tolist()
        
        if hasattr(y, 'values'):
            y_vec = y.values.flatten().tolist()
        else:
            if hasattr(y, 'flatten'):
                y_vec = y.flatten().tolist()
            else:
                y_vec = list(y)
        
        X_design = [[1.0] + list(row) for row in X_mat]
        N = len(X_design)
        M = len(X_design[0])
        
        S_0_inv = identity(M)
        S_0_inv = [[self.alpha * val for val in row] for row in S_0_inv]
        
        X_T = transpose(X_design)
        XtX = matmul(X_T, X_design)
        
        beta_XtX = [[self.beta * val for val in row] for row in XtX]
        
        S_N_array = inverse(matrix_add(S_0_inv, beta_XtX))
        self.S_N = S_N_array.tolist() if hasattr(S_N_array, 'tolist') else S_N_array
        
        X_T_y = matmul(X_T, as_matrix([[v] for v in y_vec]))
        S_N_X_T_y = matmul(self.S_N, X_T_y)
        if hasattr(S_N_X_T_y, 'tolist'):
            S_N_X_T_y = S_N_X_T_y.tolist()
        
        self.m_N = [self.beta * row[0] for row in S_N_X_T_y]
            
        return self

    def get_credible_interval(self, X):
        X_mat = X.values.tolist() if hasattr(X, 'values') else X.tolist()
        X_design = [[1.0] + list(row) for row in X_mat]
        
        m_N_mat = as_matrix([[v] for v in self.m_N])
        y_pred_log_arr = matmul(X_design, m_N_mat)
        y_pred_log = [row[0] for row in y_pred_log_arr]
        
        X_S_arr = matmul(X_design, self.S_N)
        X_S = X_S_arr.tolist() if hasattr(X_S_arr, 'tolist') else X_S_arr
        
        pred_variance = []
        for i in range(len(X_design)):
            row_sum = 0.0
            for j in range(len(X_design[0])):
                row_sum += X_S[i][j] * X_design[i][j]
            pred_variance.append((1.0 / self.beta) + row_sum)
            
        pred_std = [math.sqrt(v) for v in pred_variance]
        
        z_score = 1.96
        lower_log = [y_pred_log[i] - z_score * pred_std[i] for i in range(len(y_pred_log))]
        upper_log = [y_pred_log[i] + z_score * pred_std[i] for i in range(len(y_pred_log))]
        
        return y_pred_log, pred_std, (lower_log, upper_log)

class KernelRidgeRegression:
    def __init__(self, lam=1.0, gamma=0.1):
        self.lam = lam
        self.gamma = gamma
        self.X_train = None
        self.y_train = None
        self.alpha_coef = None

    def _compute_rbf_kernel(self, X1, X2):
        n1 = len(X1)
        n2 = len(X2)
        K = [[0.0 for _ in range(n2)] for _ in range(n1)]
        for i in range(n1):
            for j in range(n2):
                sq_dist = sum((X1[i][k] - X2[j][k])**2 for k in range(len(X1[0])))
                K[i][j] = math.exp(-self.gamma * sq_dist)
        return K

    def fit(self, X, y):
        self.X_train = X.values.tolist() if hasattr(X, 'values') else X.tolist()
        
        if hasattr(y, 'values'):
            self.y_train = y.values.flatten().tolist()
        else:
            if hasattr(y, 'flatten'):
                self.y_train = y.flatten().tolist()
            else:
                self.y_train = list(y)
        
        n_samples = len(self.X_train)
        K = self._compute_rbf_kernel(self.X_train, self.X_train)
        
        I = identity(n_samples)
        I_scaled = [[self.lam * val for val in row] for row in I]
        A = matrix_add(K, I_scaled)
        
        alpha_arr = solve(A, self.y_train)
        self.alpha_coef = alpha_arr.tolist() if hasattr(alpha_arr, 'tolist') else alpha_arr
            
        return self

    def predict(self, X):
        X_test = X.values.tolist() if hasattr(X, 'values') else X.tolist()
        K_test = self._compute_rbf_kernel(X_test, self.X_train)
        
        alpha_mat = as_matrix([[v] for v in self.alpha_coef])
        pred_arr = matmul(K_test, alpha_mat)
        pred_list = [row[0] for row in pred_arr]
            
        return pred_list


def plot_kernel_diagnostics(y_true_log, y_pred_log, X_train=None, X_test=None, gamma=None, lam=None):
    """
    Vẽ 8 biểu đồ chẩn đoán phần dư cho mô hình phi tuyến (Kernel RBF)
    Đầu vào: y_true_log, y_pred_log (giá trị thực tế và dự đoán ở không gian Logarit)
    X_train, X_test, gamma, lam: Các tham số tùy chọn để xấp xỉ Cook's Distance cho Kernel
    """
    y_true_real = np.expm1(np.array(y_true_log).flatten())
    y_pred_real = np.expm1(np.array(y_pred_log).flatten())
    residuals = y_true_real - y_pred_real
    std_residuals = residuals / (np.std(residuals) + 1e-10)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('Chẩn đoán Phần dư (Residual Diagnostics) - Mô hình Tối ưu Kernel RBF', fontsize=18, fontweight='bold', y=1.02)

    # 1. Residuals vs Predicted
    sns.scatterplot(x=y_pred_real, y=residuals, ax=axes[0, 0], color='crimson', alpha=0.5)
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[0, 0].set_title('1. Phần dư vs Dự đoán (Residuals vs Fitted)')
    axes[0, 0].set_xlabel('Giá trị dự đoán (Predicted AQI)')
    axes[0, 0].set_ylabel('Phần dư (Residuals)')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_markerfacecolor('crimson')
    axes[0, 1].get_lines()[0].set_markeredgecolor('crimson')
    axes[0, 1].get_lines()[0].set_alpha(0.5)
    axes[0, 1].set_title('2. Biểu đồ Q-Q (Kiểm tra Phân phối chuẩn)')
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # 3. Histogram of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='crimson', bins=40)
    axes[1, 0].set_title('3. Phân phối Phần dư (Histogram of Residuals)')
    axes[1, 0].set_xlabel('Giá trị Phần dư (Residuals)')
    axes[1, 0].set_ylabel('Tần suất')

    # 4. Actual vs Predicted
    sns.scatterplot(x=y_true_real, y=y_pred_real, ax=axes[1, 1], color='crimson', alpha=0.5)
    max_val = max(y_true_real.max(), y_pred_real.max())
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
    axes[3, 0].scatter(y_pred_real, np.sqrt(np.abs(std_residuals)), alpha=0.5, color='crimson')
    axes[3, 0].set_title('7. Scale-Location (Kiểm tra phương sai đồng đều)')
    axes[3, 0].set_xlabel('Fitted values')
    axes[3, 0].set_ylabel('Sqrt(|Standardized Residuals|)')
    axes[3, 0].grid(True, linestyle=':', alpha=0.6)

    # 8. Cook's Distance
    # Do Kernel RBF là mô hình phi tuyến, xấp xỉ Cook's Distance dựa trên phần dư chuẩn hóa
    # Nếu được cung cấp đủ ma trận, sẽ tính xấp xỉ Leverage, ngược lại dùng xấp xỉ đơn giản.
    n = len(residuals)
    p_approx = X_train.shape[1] if X_train is not None else 10
    
    if X_train is not None and X_test is not None and gamma is not None and lam is not None:
        from scipy.spatial.distance import cdist
        # Sử dụng hàm chuẩn để có độ chính xác cao nhất khi đánh giá (Kiểm chứng)
        K_test = np.exp(-gamma * cdist(X_test, X_train, metric='sqeuclidean'))
        K_train = np.exp(-gamma * cdist(X_train, X_train, metric='sqeuclidean'))
        A_inv = np.linalg.pinv(K_train + lam * np.eye(len(X_train)))
        H_approx = K_test @ A_inv @ K_test.T
        leverage = np.clip(np.diag(H_approx), 0, 0.99)
    else:
        leverage = np.full(n, p_approx / n)
        
    cooks_d = (std_residuals**2 / (p_approx + 1)) * (leverage / ((1 - leverage + 1e-10)**2))
    axes[3, 1].bar(range(n), cooks_d, alpha=0.6, color='crimson')
    axes[3, 1].axhline(y=4/n, color='black', linestyle='--', label='Ngưỡng 4/n')
    axes[3, 1].set_title("8. Khoảng cách Cook (Cook's Distance)")
    axes[3, 1].set_xlabel('Observation Index')
    axes[3, 1].set_ylabel("Cook's Distance")
    axes[3, 1].legend()
    axes[3, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
