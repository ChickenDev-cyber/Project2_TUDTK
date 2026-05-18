import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 
from ols_implementation import hat_matrix

# Vẽ 4 biểu đồ phân tích phần dư.
def residual_plots(X, y, beta_hat):
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1 
    
    # 1. TÍNH TOÁN CÁC THÔNG SỐ CƠ BẢN
    y_hat = X @ beta_hat
    residuals = y - y_hat
    
    # 2. TÍNH LEVERAGE (Đường chéo của Hat matrix)
    H = hat_matrix(X) 
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


def test_residual_analysis():
    # Unit test cho phân tích phần dư
    np.random.seed(42)
    n = 10
    p = 2
    X = np.column_stack((np.ones(n), np.random.randn(n, p)))
    beta_hat = np.array([1.0, 2.0, -1.0])
    y = X @ beta_hat + np.random.normal(0, 0.1, n)
    
    H = hat_matrix(X)
    leverage = np.diag(H)
    
    # 1. So sánh tổng leverage với p + 1
    sum_leverage = np.sum(leverage)
    if np.isclose(sum_leverage, p + 1):
        print("Kiểm tra tổng leverage: Giống")
    else:
        print("Kiểm tra tổng leverage: Khác")
        
    # 2. Kiểm tra leverage hợp lệ
    if np.all(leverage >= -1e-10) and np.all(leverage <= 1.0 + 1e-10):
        print("Kiểm tra giá trị leverage: Giống")
    else:
        print("Kiểm tra giá trị leverage: Khác")
        
    # 3. Kiểm tra Cook's distance không âm
    y_hat = X @ beta_hat
    residuals = y - y_hat
    RSS = np.sum(residuals**2)
    sigma2_hat = RSS / (n - p - 1)
    std_residuals = residuals / np.sqrt(sigma2_hat * (1 - leverage + 1e-10))
    cooks_d = (std_residuals**2 / (p + 1)) * (leverage / (1 - leverage + 1e-10))
    if np.all(cooks_d >= 0):
        print("Kiểm tra Cook's distance: Giống")
    else:
        print("Kiểm tra Cook's distance: Khác")

if __name__ == "__main__":
    # Chạy test
    test_residual_analysis()
    
    # Vẽ đồ thị demo
    np.random.seed(42)
    n_mock = 200
    p_mock = 3
    X_mock = np.column_stack((np.ones(n_mock), np.random.randn(n_mock, p_mock)))
    beta_hat_mock = np.array([2.5, 1.2, -0.8, 3.0])
    y_mock = X_mock @ beta_hat_mock + np.random.normal(0, 1.5, n_mock)
    
    print("\nĐang vẽ biểu đồ...")
    residual_plots(X_mock, y_mock, beta_hat_mock)
    print("Vẽ biểu đồ hoàn tất!")