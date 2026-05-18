import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 
# from part1.ols_implementation import hat_matrix

# Hàm hat_matrix tạm
def mock_hat_matrix(X):
    return X @ np.linalg.inv(X.T @ X) @ X.T

# Vẽ 4 biểu đồ phân tích phần dư.
def residual_plots(X, y, beta_hat):
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1 
    
    # 1. TÍNH TOÁN CÁC THÔNG SỐ CƠ BẢN
    y_hat = X @ beta_hat
    residuals = y - y_hat
    
    # 2. TÍNH LEVERAGE (Đường chéo của Hat matrix)
    H = mock_hat_matrix(X) 
    leverage = np.diag(H)
    
    # 3. TÍNH PHẦN DƯ CHUẨN HÓA 
    RSS = np.sum(residuals**2)
    sigma2_hat = RSS / (n - p - 1)
    
    std_residuals = residuals / np.sqrt(sigma2_hat * (1 - leverage + 1e-10))
    
    # 4. TÍNH COOK'S DISTANCE
    cooks_d = (std_residuals**2 / (p + 1)) * (leverage / (1 - leverage + 1e-10))
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Analysis Plots', fontsize=16)

    # Đồ thị 1: Residuals vs Fitted
    axs[0, 0].scatter(y_hat, residuals, alpha=0.6, edgecolors='k')
    axs[0, 0].axhline(0, color='red', linestyle='dashed') # Đường tham chiếu số 0
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')

    # Đồ thị 2: Normal Q-Q Plot
    stats.probplot(std_residuals, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q Plot')
    axs[0, 1].set_xlabel('Theoretical Quantiles')
    axs[0, 1].set_ylabel('Standardized Residuals')

    # Đồ thị 3: Scale-Location
    axs[1, 0].scatter(y_hat, np.sqrt(np.abs(std_residuals)), alpha=0.6, edgecolors='k')
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('Sqrt(|Standardized Residuals|)')

    # Đồ thị 4: Cook's Distance
    axs[1, 1].bar(range(n), cooks_d, alpha=0.6, color='b')
    axs[1, 1].axhline(y=4/n, color='red', linestyle='dashed', label='Ngưỡng tham chiếu (4/n)')
    axs[1, 1].set_title("Cook's Distance")
    axs[1, 1].set_xlabel('Observation Index')
    axs[1, 1].set_ylabel("Cook's Distance")
    axs[1, 1].legend() 

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()


if __name__ == "__main__":
    print("Đang khởi tạo dữ liệu giả lập để test residual_plots...")
    
    # 1. Cố định random_state để kết quả không đổi mỗi lần chạy
    np.random.seed(42)
    
    # 2. Tạo ma trận X giả (Ví dụ n=200 dòng, p=3 biến theo tiêu chí đồ án)
    n = 200
    p = 3
    # X phải có cột đầu tiên là số 1 (intercept)
    X_mock = np.column_stack((np.ones(n), np.random.randn(n, p)))
    
    # 3. Tạo vector hệ số beta_hat giả (gồm 4 hệ số vì có intercept)
    beta_hat_mock = np.array([2.5, 1.2, -0.8, 3.0])
    
    # 4. Tạo vector y giả (y = X*beta + nhiễu ngẫu nhiên)
    y_mock = X_mock @ beta_hat_mock + np.random.normal(0, 1.5, n)
    
    # 5. Gọi hàm để vẽ thử
    print("Dữ liệu đã sẵn sàng. Đang vẽ biểu đồ...")
    residual_plots(X_mock, y_mock, beta_hat_mock)
    print("Test hoàn tất!")