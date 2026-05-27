import numpy as np
from scipy.spatial.distance import cdist # Chỉ dùng để tính khoảng cách hình học
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
        X_mat = X.values if hasattr(X, 'values') else X
        y_vec = y.values.flatten() if hasattr(y, 'values') else y.flatten()
        
        X_design = np.c_[np.ones((X_mat.shape[0], 1)), X_mat]
        N, M = X_design.shape
        
        S_0_inv = self.alpha * np.eye(M)
        self.S_N = np.linalg.inv(S_0_inv + self.beta * (X_design.T @ X_design))
        self.m_N = self.beta * (self.S_N @ (X_design.T @ y_vec))
        return self

    def get_credible_interval(self, X):
        X_mat = X.values if hasattr(X, 'values') else X
        X_design = np.c_[np.ones((X_mat.shape[0], 1)), X_mat]
        
        y_pred_log = X_design @ self.m_N
        pred_variance = (1 / self.beta) + np.sum((X_design @ self.S_N) * X_design, axis=1)
        pred_std = np.sqrt(pred_variance)
        
        z_score = 1.96
        lower_log = y_pred_log - z_score * pred_std
        upper_log = y_pred_log + z_score * pred_std
        
        return y_pred_log, lower_log, upper_log



class KernelRidgeRegression:
    def __init__(self, lam=1.0, gamma=0.1):
        self.lam = lam
        self.gamma = gamma
        self.X_train = None
        self.y_train = None
        self.alpha_coef = None

    def _compute_rbf_kernel(self, X1, X2):
        # Công thức RBF Kernel thuần toán học: K(x,y) = exp(-gamma * ||x-y||^2)
        sq_dists = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        self.X_train = X.values if hasattr(X, 'values') else X
        self.y_train = y.values.flatten() if hasattr(y, 'values') else y.flatten()
        
        n_samples = self.X_train.shape[0]
        K = self._compute_rbf_kernel(self.X_train, self.X_train)
        
        # Giải hệ phương trình tuyến tính tìm hệ số alpha
        A = K + self.lam * np.eye(n_samples)
        self.alpha_coef = np.linalg.solve(A, self.y_train)
        return self

    def predict(self, X):
        X_test = X.values if hasattr(X, 'values') else X
        K_test = self._compute_rbf_kernel(X_test, self.X_train)
        return K_test @ self.alpha_coef
    



def plot_kernel_diagnostics(y_true_log, y_pred_log, X_train=None, X_test=None, gamma=None, lam=None):
    """
    biểu đồ chẩn đoán phần dư cho Kernel RBF
    gồm Scale-Location và Cook's Distance theo yêu cầu.
    input: y_true_log, y_pred_log (giá trị thực tế và dự đoán ở không gian Logarit)
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
    # Nếu được cung cấp đủ ma trận, sẽ tính xấp xỉ Leverage, ngược lại dùng xấp xỉ đơn giản
    n = len(residuals)
    p_approx = X_train.shape[1] if X_train is not None else 10
    
    if X_train is not None and X_test is not None and gamma is not None and lam is not None:
        from scipy.spatial.distance import cdist
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

