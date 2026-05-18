import numpy as np
import matplotlib.pyplot as plt

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Tính toán trọng số Ridge Regression bằng công thức đóng.
    Lý luận: lam giúp kiểm soát độ lớn trọng số, ngăn chặn overfitting.
    """
    # Đảm bảo y có dạng cột (N, 1) để phép nhân ma trận nhất quán
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    n_features = X.shape[1]
    I = np.eye(n_features)
    
    # Giải phương trình: (X^T @ X + lam * I) @ w = X^T @ y
    A = X.T @ X + lam * I
    b = X.T @ y
    
    # Sử dụng np.linalg.solve thay vì inv() để tăng độ chính xác số học
    w = np.linalg.solve(A, b)
    return w.flatten()

def plot_ridge_trace(X: np.ndarray, y: np.ndarray):
    """
    Vẽ biểu đồ Ridge Trace hiển thị sự biến thiên của các hệ số hồi quy theo Lambda.
    Mục đích: Giúp nhà nghiên cứu quan sát trực quan quá trình co rút (shrinkage) 
              của trọng số để đưa ra lý luận chọn vùng Lambda phù hợp.
    """
    # 1. Khởi tạo dải giá trị Lambda theo cấp số nhân (từ 10^-3 đến 10^5)
    # Quét rộng giúp nhìn rõ điểm gãy và vùng trọng số bị triệt tiêu
    lambdas = np.logspace(-3, 5, 200)
    
    # 2. Tính toán trọng số w ứng với mỗi giá trị Lambda
    weights = []
    for lam in lambdas:
        w = ridge_fit(X, y, lam)
        weights.append(w)
    
    # Chuyển thành ma trận NumPy để dễ trích xuất theo cột (từng đặc trưng)
    weights = np.array(weights)
    
    # 3. Khởi tạo cấu trúc biểu đồ chuẩn kỹ thuật
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Vẽ từng đường trọng số ứng với mỗi đặc trưng
    n_features = X.shape[1]
    for i in range(n_features):
        plt.plot(lambdas, weights[:, i], linewidth=2, label=f'Đặc trưng {i+1}')
    
    # 4. Định dạng cấu hình biểu đồ
    plt.xscale('log')  
    
    
    plt.title('Biểu đồ Ridge Trace', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Hệ số điều chỉnh Lambda', fontsize=11, labelpad=10)
    plt.ylabel('Giá trị của Hệ số hồi quy', fontsize=11, labelpad=10)
    
    # Hiển thị lưới đồ thị (grid) cho cả hai trục để dễ gióng giá trị
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    
    # Chú thích phân biệt các đường đặc trưng
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10, shadow=True)
    
    # Tối ưu hóa không gian hiển thị, tránh mất chữ ở rìa ảnh
    plt.tight_layout()
    plt.show()
    
def test_ridge_fit():
    """
    Unit tests kiểm tra tính đúng đắn của hàm ridge_fit theo style nhóm.
    """
    # Thiết lập seed cố định dữ liệu ngẫu nhiên
    np.random.seed(42)

    # --- Test 1: So sánh hệ số với trường hợp lý tưởng OLS ---
    X1 = np.array([[1, 0], [0, 1]])
    y1 = np.array([1, 2])
    expected_w1 = np.array([1.0, 2.0])
    w1 = ridge_fit(X1, y1, lam=0)
    
    if np.allclose(w1, expected_w1):
        print("So sánh hệ số beta với OLS: Giống")
    else:
        print("So sánh hệ số beta với OLS: Khác")

    # --- Test 2: So sánh hệ số khi lambda tiến ra vô cùng ---
    X2 = np.random.randn(10, 3)
    y2 = np.random.randn(10)
    expected_w2 = np.zeros(3)
    w2 = ridge_fit(X2, y2, lam=1e10)
    
    if np.allclose(w2, expected_w2, atol=1e-5):
        print("So sánh hệ số khi lambda cực lớn: Giống")
    else:
        print("So sánh hệ số khi lambda cực lớn: Khác")
        
if __name__ == "__main__":
    test_ridge_fit()