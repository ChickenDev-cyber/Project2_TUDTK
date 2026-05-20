import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        """
        alpha: Trọng số phân phối tiền nghiệm của hệ số hồi quy (Prior precision)
        beta: Độ chính xác của thành phần nhiễu dữ liệu (Noise precision)
        """
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None

    def fit(self, X, y):
        X_mat = X.values if hasattr(X, 'values') else X
        y_vec = y.values.flatten() if hasattr(y, 'values') else y.flatten()
        
        # Thêm cột hệ số chặn (Intercept) vào ma trận thiết kế
        X_design = np.c_[np.ones((X_mat.shape[0], 1)), X_mat]
        N, M = X_design.shape
        
        # Tính toán ma trận hiệp phương sai hậu nghiệm S_N
        S_0_inv = self.alpha * np.eye(M)
        self.S_N = np.linalg.inv(S_0_inv + self.beta * (X_design.T @ X_design))
        
        # Tính toán vector kỳ vọng hậu nghiệm m_N
        self.m_N = self.beta * (self.S_N @ (X_design.T @ y_vec))
        return self

    def get_credible_interval(self, X, confidence=0.95):
        X_mat = X.values if hasattr(X, 'values') else X
        X_design = np.c_[np.ones((X_mat.shape[0], 1)), X_mat]
        
        # Dự đoán giá trị trung bình trong không gian Log
        y_pred_log = X_design @ self.m_N
        
        # Tính toán phương sai phân phối dự đoán phi tuyến của từng mẫu
        pred_variance = (1.0 / self.beta) + np.sum((X_design @ self.S_N) * X_design, axis=1)
        pred_std = np.sqrt(pred_variance)
        
        # Khoảng tin cậy đối xứng (mức ý nghĩa 95% tương ứng Z = 1.96)
        z_score = 1.96
        lower_log = y_pred_log - z_score * pred_std
        upper_log = y_pred_log + z_score * pred_std
        
        return y_pred_log, lower_log, upper_log

class KernelRidgeRegression:
    def __init__(self, lam=1.0, gamma=0.1):
        """
        lam (lambda): Hệ số phạt điều chuẩn không gian đối ngẫu
        gamma: Hệ số co giãn của hàm cấu trúc RBF Kernel
        """
        self.lam = lam
        self.gamma = gamma
        self.X_train = None
        self.y_train = None
        self.alpha_coef = None

    def fit(self, X, y):
        self.X_train = X.values if hasattr(X, 'values') else X
        self.y_train = y.values.flatten() if hasattr(y, 'values') else y.flatten()
        
        n_samples = self.X_train.shape[0]
        # Xây dựng ma trận cấu trúc Gram K
        K = rbf_kernel(self.X_train, self.X_train, gamma=self.gamma)
        
        A = K + self.lam * np.eye(n_samples)
        # Giải hệ phương trình tuyến tính đối ngẫu tối ưu tốc độ tránh nghịch đảo thủ công
        self.alpha_coef = np.linalg.solve(A, self.y_train)
        return self

    def predict(self, X):
        X_test = X.values if hasattr(X, 'values') else X
        K_trans = rbf_kernel(X_test, self.X_train, gamma=self.gamma)
        return K_trans @ self.alpha_coef
