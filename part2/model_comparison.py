import math
import os
import sys

import pandas as pd

# Import các hàm đại số tuyến tính tự cài đặt từ Part 1
_PART1_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part1'))
if _PART1_DIR not in sys.path:
    sys.path.insert(0, _PART1_DIR)

from matrix_ops import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    matvec,
)
from ols_implementation import ols_fit, vif
from ridge_lasso import ridge_fit
from cross_validation import ridge_cv_score


# ──────────────────────────────────────────────
# Hàm tiện ích chuyển đổi kiểu dữ liệu
# ──────────────────────────────────────────────

def _to_list(data):
    """Chuyển pandas/Vector/list sang list thuần Python."""
    if isinstance(data, pd.Series):
        return data.tolist()
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0].tolist()
    if hasattr(data, 'tolist'):
        return data.tolist()
    return list(data)


def _to_vector(data):
    """Chuyển pandas/list sang Vector (part1)."""
    return as_vector(_to_list(data))


def _to_matrix_rows(data):
    """Chuyển DataFrame/array sang list-of-lists thuần Python."""
    if isinstance(data, pd.DataFrame):
        return data.values.tolist()
    if isinstance(data, (Matrix, list)):
        return [list(row) for row in data]
    if hasattr(data, 'tolist'):
        return data.tolist()
    return [list(row) for row in data]


# ──────────────────────────────────────────────
# Thêm cột intercept (cột toàn 1)
# ──────────────────────────────────────────────

def add_intercept(X):
    """Thêm cột 1 (intercept) vào đầu ma trận X. Trả về Matrix (part1)."""
    rows = _to_matrix_rows(X)
    return as_matrix([[1.0] + [float(v) for v in row] for row in rows])


# ──────────────────────────────────────────────
# Huấn luyện OLS cơ bản (dùng ols_fit từ Part 1)
# ──────────────────────────────────────────────

def train_ols(X_train, y_train):
    """OLS: beta = (X^T X)^{-1} X^T y — kế thừa từ Part 1."""
    X_b = add_intercept(X_train)
    y = _to_vector(y_train)
    beta_hat, _sigma2 = ols_fit(X_b, y)
    return beta_hat


# ──────────────────────────────────────────────
# OLS với chọn biến theo VIF (dùng vif từ Part 1)
# ──────────────────────────────────────────────

def train_ols_selected(X_train, y_train, threshold=10.0, num_cols_count=6):
    """
    Lọc biến bằng VIF (kế thừa Part 1). Chỉ quét VIF trên các cột số thực
    (num_cols_count cột đầu), giữ nguyên cột One-Hot để tránh Dummy Variable Trap.
    """
    X_df = X_train.copy()
    numeric_cols = list(X_df.columns[:num_cols_count])

    while len(numeric_cols) > 0:
        # Xây ma trận design có intercept cho VIF
        X_num_list = X_df[numeric_cols].values.tolist()
        X_with_intercept = as_matrix(
            [[1.0] + [float(v) for v in row] for row in X_num_list]
        )

        try:
            vif_scores = vif(X_with_intercept)  # Từ part1/ols_implementation.py
        except ValueError:
            # Ma trận suy biến → loại biến đầu tiên
            col_to_drop = numeric_cols[0]
            X_df = X_df.drop(col_to_drop, axis=1)
            numeric_cols.remove(col_to_drop)
            continue

        max_vif_idx = max(range(len(vif_scores)), key=lambda i: vif_scores[i])
        max_vif = vif_scores[max_vif_idx]

        if max_vif > threshold:
            col_to_drop = numeric_cols[max_vif_idx]
            X_df = X_df.drop(col_to_drop, axis=1)
            numeric_cols.remove(col_to_drop)
        else:
            break

    # Hồi quy OLS với các biến đã lọc
    beta_hat = train_ols(X_df, y_train)
    return beta_hat, X_df.columns.tolist()


# ──────────────────────────────────────────────
# Ridge / Lasso với Cross-Validation (kế thừa Part 1)
# ──────────────────────────────────────────────

def train_ridge_lasso(X_train, y_train, k=5):
    """Chọn lambda tốt nhất bằng k-fold CV, sau đó fit Ridge."""
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_cv_score = float('inf')

    X_b = add_intercept(X_train)
    y = _to_vector(y_train)

    # Chuyển sang dạng list thuần cho các hàm Part 1
    X_list = [list(row) for row in X_b]
    y_list = list(y)

    for lam in lambdas:
        cv_score = ridge_cv_score(X_list, y_list, k, lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    beta_hat_ridge = ridge_fit(X_list, y_list, best_lambda)
    return beta_hat_ridge, best_lambda


# ──────────────────────────────────────────────
# Đánh giá mô hình (MAE, RMSE, R²) — thuần Python
# ──────────────────────────────────────────────

def evaluate_models(y_true, y_pred):
    """Tính MAE, RMSE, R² — không dùng numpy."""
    y_t = _to_list(y_true)
    y_p = _to_list(y_pred)
    n = len(y_t)

    mae = sum(abs(y_t[i] - y_p[i]) for i in range(n)) / n
    mse = sum((y_t[i] - y_p[i]) ** 2 for i in range(n)) / n
    rmse = math.sqrt(mse)

    y_mean = sum(y_t) / n
    tss = sum((v - y_mean) ** 2 for v in y_t)
    rss = sum((y_t[i] - y_p[i]) ** 2 for i in range(n))
    r2 = 1 - (rss / tss) if abs(tss) > 1e-12 else 0.0

    return {'MAE': mae, 'RMSE': rmse, 'R-squared': r2}


def evaluate_model(y_true, y_pred, is_log_transformed=True):
    """
    Tính MAE, RMSE, R² — thuần Python.
    Đảo ngược thang đo Logarit (expm1) để ra sai số thực tế.
    """
    y_t = _to_list(y_true)
    y_p = _to_list(y_pred)

    if is_log_transformed:
        y_t = [math.expm1(v) for v in y_t]
        y_p = [math.expm1(v) for v in y_p]

    n = len(y_t)
    mae = sum(abs(y_t[i] - y_p[i]) for i in range(n)) / n
    mse = sum((y_t[i] - y_p[i]) ** 2 for i in range(n)) / n
    rmse = math.sqrt(mse)

    y_mean = sum(y_t) / n
    tss = sum((v - y_mean) ** 2 for v in y_t)
    rss = sum((y_t[i] - y_p[i]) ** 2 for i in range(n))
    r2 = 1 - (rss / tss) if abs(tss) > 1e-12 else 0.0

    return {'MAE': mae, 'RMSE': rmse, 'R-squared': r2}


def compare_models(predictions_dict, y_true, is_log_transformed=True):
    results = {}
    for model_name, y_pred in predictions_dict.items():
        results[model_name] = evaluate_model(y_true, y_pred, is_log_transformed)

    df_results = pd.DataFrame(results).T
    df_results = df_results[['MAE', 'RMSE', 'R-squared']]
    return df_results


def compare_train_test_models(train_preds, test_preds, y_train, y_test, is_log=True):
    df_train = compare_models(train_preds, y_train, is_log)
    df_train.columns = [f"Train_{c}" for c in df_train.columns]

    df_test = compare_models(test_preds, y_test, is_log)
    df_test.columns = [f"Test_{c}" for c in df_test.columns]

    df_final = pd.concat([df_train, df_test], axis=1)
    return df_final