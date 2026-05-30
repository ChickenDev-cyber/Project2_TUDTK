import math
import sys

from matrix_ops import (
    Matrix,
    Vector,
    add_to_diagonal,
    all_close,
    as_matrix,
    as_vector,
    column,
    column_stack,
    diag,
    frobenius_norm,
    identity,
    inverse,
    matmul,
    matvec,
    matrix_sub,
    mean,
    solve,
    sum_squares,
    trace,
    transpose,
    without_column,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _solve_normal_equation(X, y):
    X = as_matrix(X)
    y = as_vector(y)
    gram = matmul(transpose(X), X)
    rhs = matvec(transpose(X), y)
    return solve(gram, rhs)


def _gram_inverse(X):
    X = as_matrix(X)
    gram = matmul(transpose(X), X)
    return inverse(gram)


def ols_fit(X, y):
    # Chuyen doi du lieu dau vao thanh ma tran va vector tuong thich.
    X = as_matrix(X)
    y = as_vector(y)
    # Giai he phuong trinh chuan de uoc luong beta_hat.
    beta_hat = _solve_normal_equation(X, y)

    # Tinh gia tri du bao va phan du thiet yeu.
    y_hat = matvec(X, beta_hat)
    residuals = y - y_hat
    # Tinh tong binh phuong cac phan du (RSS).
    RSS = sum_squares(residuals)

    # Lay so quan sat (n) va so he so (p+1).
    n = len(X)
    p_plus_1 = len(X[0])
    # Uoc luong khong chech cua phuong sai nhieu sigma^2.
    sigma2_hat = RSS / (n - p_plus_1)

    return beta_hat, sigma2_hat


def hat_matrix(X):
    X = as_matrix(X)
    return matmul(matmul(X, _gram_inverse(X)), transpose(X))


def model_metrics(y, y_hat, p):
    y = as_vector(y)
    y_hat = as_vector(y_hat)
    n = len(y)
    RSS = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    y_bar = mean(y)
    TSS = sum((yi - y_bar) ** 2 for yi in y)

    r2 = math.nan if abs(TSS) < 1e-12 else 1 - (RSS / TSS)
    adj_r2 = math.nan if n - p - 1 <= 0 else 1 - ((n - 1) / (n - p - 1)) * (1 - r2)

    if p <= 0 or n - p - 1 <= 0:
        f_stat = math.nan
        p_val_f = math.nan
    elif abs(RSS) < 1e-12:
        f_stat = math.inf
        p_val_f = 0.0
    else:
        f_stat = ((TSS - RSS) / p) / (RSS / (n - p - 1))
        p_val_f = f_test_p_value(f_stat, p, n - p - 1)

    return RSS, TSS, r2, adj_r2, f_stat, p_val_f


def _student_t_pdf(x, df):
    coeff = math.exp(
        math.lgamma((df + 1) / 2)
        - math.lgamma(df / 2)
        - 0.5 * (math.log(df) + math.log(math.pi))
    )
    return coeff * (1 + (x * x) / df) ** (-(df + 1) / 2)


def _adaptive_simpson(f, a, b, eps=1e-8, depth=16):
    c = (a + b) / 2
    fa, fb, fc = f(a), f(b), f(c)
    whole = (b - a) * (fa + 4 * fc + fb) / 6

    def recurse(left, right, f_left, f_mid, f_right, area, current_depth):
        mid = (left + right) / 2
        left_mid = (left + mid) / 2
        right_mid = (mid + right) / 2
        f_left_mid = f(left_mid)
        f_right_mid = f(right_mid)
        left_area = (mid - left) * (f_left + 4 * f_left_mid + f_mid) / 6
        right_area = (right - mid) * (f_mid + 4 * f_right_mid + f_right) / 6
        refined = left_area + right_area
        if current_depth <= 0 or abs(refined - area) <= 15 * eps:
            return refined + (refined - area) / 15
        return (
            recurse(left, mid, f_left, f_left_mid, f_mid, left_area, current_depth - 1)
            + recurse(mid, right, f_mid, f_right_mid, f_right, right_area, current_depth - 1)
        )

    return recurse(a, b, fa, fc, fb, whole, depth)


def _student_t_cdf(x, df):
    if df <= 0:
        return math.nan
    if x == 0:
        return 0.5
    if x > 12:
        return 1.0
    if x < -12:
        return 0.0
    area = _adaptive_simpson(lambda t: _student_t_pdf(t, df), 0.0, abs(x))
    return 0.5 + area if x > 0 else 0.5 - area


def _student_t_ppf(probability, df):
    if not 0 < probability < 1:
        raise ValueError("probability must be between 0 and 1")
    if probability == 0.5:
        return 0.0
    sign = 1.0 if probability > 0.5 else -1.0
    target = probability if probability > 0.5 else 1 - probability
    low, high = 0.0, 12.0
    for _ in range(80):
        mid = (low + high) / 2
        if _student_t_cdf(mid, df) < target:
            low = mid
        else:
            high = mid
    return sign * (low + high) / 2


def _f_pdf(x, df1, df2):
    if x <= 0:
        return 0.0
    a = df1 / 2.0
    b = df2 / 2.0
    # Tinh log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    try:
        log_val = (a * math.log(df1) + b * math.log(df2) - log_beta
                   + (a - 1.0) * math.log(x) - (a + b) * math.log(df1 * x + df2))
        return math.exp(log_val)
    except (ValueError, OverflowError):
        return 0.0


def f_test_p_value(f_stat, df1, df2):
    # Tinh toan p-value cho kiem dinh F tong the bang phuong phap Simpson thich ung.
    # df1 = p (so dac trung), df2 = n - p - 1 (bac tu do phan du).
    if df1 <= 0 or df2 <= 0 or f_stat < 0:
        return math.nan
    if f_stat == 0:
        return 1.0

    if df1 == 1:
        # Moi quan he giua phan phoi F(1, d) va t(d):
        # Bieu dien exact qua CDF t-Student de tranh diem ki di tai x=0.
        cdf_f = 2 * _student_t_cdf(math.sqrt(f_stat), df2) - 1.0
        p_val = 1.0 - cdf_f
    else:
        # Tich phan PDF tu 0.0 den f_stat bang Simpson
        limit = min(f_stat, 100.0)
        cdf_f = _adaptive_simpson(lambda t: _f_pdf(t, df1, df2), 0.0, limit)
        p_val = 1.0 - cdf_f

    return max(0.0, min(p_val, 1.0))


def coef_inference(X, y, beta_hat, sigma2_hat):
    # Dam bao dinh dang dau vao cua cac ma tran va vector.
    X = as_matrix(X)
    beta_hat = as_vector(beta_hat)
    n = len(X)
    p_plus_1 = len(X[0])
    # Tinh bac tu do cho phan phoi t-Student.
    df = n - p_plus_1

    # Tinh ma tran hiep phuong sai cua cac he so: sigma^2 * (X^T X)^(-1).
    gram_inv = _gram_inverse(X)
    cov_matrix = Matrix([[sigma2_hat * value for value in row] for row in gram_inv])
    # Lay sai so chuan (SE) tu duong cheo cua cov_matrix.
    se = Vector(math.sqrt(value) for value in diag(cov_matrix))

    # Tinh tri so thong ke kiem dinh t cho tung he so.
    t_stats = Vector(beta_hat[j] / se[j] for j in range(len(beta_hat)))
    # Tinh p-value qua ham t-CDF hai duoi.
    p_values = Vector(2 * (1 - _student_t_cdf(abs(t), df)) for t in t_stats)

    # Tim gia tri critical cua t ung voi muc tin cay 95% (alpha = 0.05).
    t_critical = _student_t_ppf(0.975, df)
    # Tinh khoang tin cay 95% cho tung he so.
    ci_lower = Vector(beta_hat[j] - t_critical * se[j] for j in range(len(beta_hat)))
    ci_upper = Vector(beta_hat[j] + t_critical * se[j] for j in range(len(beta_hat)))

    return se, t_stats, p_values, ci_lower, ci_upper


def vif(X):
    X = as_matrix(X)
    has_intercept = all(abs(row[0] - 1.0) < 1e-12 for row in X)
    features = Matrix([row[1:] if has_intercept else row[:] for row in X])
    n_features = len(features[0])
    vif_scores = []

    for j in range(n_features):
        y_temp = column(features, j)
        other_features = without_column(features, j)
        X_temp = column_stack([[1.0] * len(features)] + [column(other_features, k) for k in range(len(other_features[0]))])

        try:
            beta_temp = _solve_normal_equation(X_temp, y_temp)
            y_temp_hat = matvec(X_temp, beta_temp)
        except ValueError:
            vif_scores.append(math.inf)
            continue

        RSS_temp = sum((y_temp[i] - y_temp_hat[i]) ** 2 for i in range(len(y_temp)))
        y_bar = mean(y_temp)
        TSS_temp = sum((yi - y_bar) ** 2 for yi in y_temp)
        if abs(TSS_temp) < 1e-12:
            vif_scores.append(math.inf)
            continue
        R2_temp = 1 - (RSS_temp / TSS_temp)
        vif_scores.append(math.inf if abs(1 - R2_temp) < 1e-12 else 1 / (1 - R2_temp))

    return Vector(vif_scores)


def verify_solution(X, y, beta_hat, H):
    from sklearn.linear_model import LinearRegression

    if beta_hat is None:
        print("Kiểm chứng: Không có hệ số để đối chiếu.")
        return

    try:
        if all_close(matmul(H, H), H, atol=1e-5):
            print("Kiểm tra tính chất Idempotent (H^2 = H): Giống")
        else:
            print("Kiểm tra tính chất Idempotent (H^2 = H): Khác")

        sk_model = LinearRegression(fit_intercept=False).fit(X, y)
        if all_close(beta_hat, sk_model.coef_, atol=1e-5):
            print("Đối chiếu hệ số Beta với sklearn: Giống")
        else:
            print("Đối chiếu hệ số Beta với sklearn: Khác")

    except Exception as e:
        print(f"Lỗi xảy ra trong quá trình đối chiếu thư viện: {e}")


def test_ols_fit():
    print(" Phân tích OLS Fit ")
    X_small = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
    y_small = [3, 5, 7, 9, 11]

    X_intercept = [[1], [1], [1], [1], [1], [1]]
    y_intercept = [2, 4, 4, 4, 5, 7]

    beta, sigma2 = ols_fit(X_small, y_small)
    beta2, sigma2_2 = ols_fit(X_intercept, y_intercept)

    if all_close(beta, [1.0, 2.0], atol=1e-8):
        print("Kiểm tra Beta ước lượng: Giống")
    else:
        print("Kiểm tra Beta ước lượng: Khác")

    if sigma2 < 1e-10:
        print("Kiểm tra sigma2 xấp xỉ 0: Giống")
    else:
        print("Kiểm tra sigma2 xấp xỉ 0: Khác")

    if all_close(beta2, [mean(y_intercept)], atol=1e-8):
        print("Kiểm tra Beta = mean(y): Giống")
    else:
        print("Kiểm tra Beta = mean(y): Khác")


def test_hat_matrix():
    print("\n Phân tích Hat Matrix ")
    X_small = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
    y_small = [3, 5, 7, 9, 11]

    H = hat_matrix(X_small)

    # (i) Tính lũy đẳng: H^2 = H
    if all_close(matmul(H, H), H, atol=1e-8):
        print("Kiểm tra (i) H^2 = H (lũy đẳng): Giống")
    else:
        print("Kiểm tra (i) H^2 = H (lũy đẳng): Khác")

    # (ii) Tính đối xứng: H^T = H
    if all_close(H, transpose(H), atol=1e-8):
        print("Kiểm tra (ii) H^T = H (đối xứng): Giống")
    else:
        print("Kiểm tra (ii) H^T = H (đối xứng): Khác")

    # (iii) Trị riêng chỉ là 0 hoặc 1
    # Hệ quả toán học: Ma trận đối xứng lũy đẳng (H²=H, H=Hᵀ) có trị riêng chỉ là 0 hoặc 1.
    # Chứng minh: Nếu Hv = λv thì H²v = λ²v = Hv = λv → λ² = λ → λ ∈ {0, 1}.
    # Kiểm tra gián tiếp qua trace: trace(H) = tổng trị riêng = số trị riêng bằng 1.
    n = len(H)
    p_plus_1 = len(X_small[0])
    tr = trace(H)
    eigenvalues_valid = abs(tr - p_plus_1) < 1e-8 and abs(tr - round(tr)) < 1e-8
    if eigenvalues_valid:
        print(f"Kiểm tra (iii) Trị riêng ∈ {{0,1}} (trace = {tr:.6f} = p+1 = {p_plus_1}): Giống")
    else:
        print(f"Kiểm tra (iii) Trị riêng ∈ {{0,1}} (trace = {tr:.6f}): Khác")

    # (iv) rank(H) = p+1
    rank_H = round(tr)
    if rank_H == p_plus_1:
        print(f"Kiểm tra (iv) rank(H) = {rank_H} = p+1 = {p_plus_1}: Giống")
    else:
        print(f"Kiểm tra (iv) rank(H) = {rank_H}, kỳ vọng {p_plus_1}: Khác")

    # (v) Giá trị dự báo: ŷ = Hy
    if all_close(matvec(H, y_small), y_small, atol=1e-8):
        print("Kiểm tra (v) ŷ = H @ y: Giống")
    else:
        print("Kiểm tra (v) ŷ = H @ y: Khác")


def test_model_metrics():
    print("\n Phân tích Metrics ")
    y = [2, 4, 6, 8]
    y_hat = [2, 4, 6, 8]
    _, _, r2, _, f_stat, _ = model_metrics(y, y_hat, p=1)

    y_const = [1, 2, 3, 4, 5]
    y_hat_mean = [mean(y_const)] * len(y_const)
    _, _, r2_zero, _, _, _ = model_metrics(y_const, y_hat_mean, p=1)

    print("Kiểm tra R^2 = 1 khi fit hoàn hảo:", "Giống" if abs(r2 - 1.0) < 1e-8 else "Khác")
    print("Kiểm tra F-stat vô hạn khi fit hoàn hảo:", "Giống" if math.isinf(f_stat) else "Khác")
    print("Kiểm tra R^2 = 0 khi y_hat = mean(y):", "Giống" if abs(r2_zero) < 1e-8 else "Khác")

    # Kiem tra tinh dung dan cua f_test_p_value
    p_val_inf = f_test_p_value(f_stat, df1=1, df2=2)
    p_val_zero = f_test_p_value(0.0, df1=1, df2=3)
    p_val_normal = f_test_p_value(4.0, df1=1, df2=10)
    print("Kiểm tra p-value F (F=inf):", "Giống" if p_val_inf == 0.0 else "Khác")
    print("Kiểm tra p-value F (F=0):", "Giống" if p_val_zero == 1.0 else "Khác")
    print(f"Kiểm tra p-value F (F=4.0, df1=1, df2=10): {p_val_normal:.6f} (Kỳ vọng ~ 0.073)")


def test_coef_inference():
    print("\n Phân tích Coef Inference ")
    X = [[1, i, i % 3] for i in range(1, 31)]
    y = [5 + 3 * row[1] - 2 * row[2] + (0.1 if i % 2 else -0.1) for i, row in enumerate(X)]
    beta, sigma2 = ols_fit(X, y)
    se, t_stats, p_values, ci_lower, ci_upper = coef_inference(X, y, beta, sigma2)

    print("Số lượng SE khớp số hệ số:", "Giống" if len(se) == len(beta) else "Khác")
    print("Khoảng tin cậy hợp lệ:", "Giống" if all(ci_upper[i] > ci_lower[i] for i in range(len(beta))) else "Khác")


def test_vif():
    print("\n Phân tích VIF ")
    X_indep = [[1, i, (i * 7) % 11, (i * 5) % 13] for i in range(1, 80)]
    x_base = [i / 10 for i in range(80)]
    X_collinear = [[1, x, x + 0.001 * ((i % 2) - 0.5), (i % 7)] for i, x in enumerate(x_base)]

    vif_indep = vif(X_indep)
    vif_coll = vif(X_collinear)
    print("VIF dữ liệu bình thường thấp:", "Giống" if max(vif_indep) < 5 else "Khác")
    print("VIF dữ liệu đa cộng tuyến cao:", "Giống" if max(vif_coll) > 10 else "Khác")


def run_all_unit_tests():
    test_ols_fit()
    test_hat_matrix()
    test_model_metrics()
    test_coef_inference()
    test_vif()


def test_integration_simple_regression():
    print("\n Kiểm thử tích hợp: Simple Regression ")
    X = [[1, i] for i in range(1, 21)]
    y = [5 + 3 * i + (0.2 if i % 2 else -0.2) for i in range(1, 21)]
    beta_hat, sigma2_hat = ols_fit(X, y)
    y_hat = matvec(X, beta_hat)
    H = hat_matrix(X)
    print("Beta gần [5, 3]:", "Giống" if all_close(beta_hat, [5, 3], atol=0.25) else "Khác")
    print("Hat matrix hợp lệ:", "Giống" if all_close(matmul(H, H), H, atol=1e-8) else "Khác")
    print(f"sigma2_hat = {sigma2_hat:.6f}, RSS = {sum_squares(as_vector(y) - y_hat):.6f}")


def test_integration_multiple_regression():
    print("\n Kiểm thử tích hợp: Multiple Regression ")
    X = [[1, i, i % 5, (i * i) % 7] for i in range(1, 41)]
    y = [2.5 - 1.5 * row[1] + 4 * row[2] + 0.7 * row[3] for row in X]
    beta_hat, sigma2_hat = ols_fit(X, y)
    y_hat = matvec(X, beta_hat)
    RSS, _, r2, _, _, _ = model_metrics(y, y_hat, p=3)
    print("R^2 gần 1:", "Giống" if abs(r2 - 1.0) < 1e-8 else "Khác")
    print(f"Beta = {[round(v, 4) for v in beta_hat]}, RSS = {RSS:.6f}, sigma2 = {sigma2_hat:.6f}")


def run_all_integration_tests():
    test_integration_simple_regression()
    test_integration_multiple_regression()


if __name__ == "__main__":
    run_all_unit_tests()
    run_all_integration_tests()
