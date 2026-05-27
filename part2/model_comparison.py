import numpy as np
import pandas as pd
import sys

# Import các hàm từ Part 1
sys.path.append('../part1')
try:
    from cross_validation import kfold_cv
    from ridge_lasso import ridge_fit
except ImportError:
    pass

def add_intercept(X):
    if isinstance(X, pd.DataFrame):
        X = X.values
    return np.c_[np.ones((X.shape[0], 1)), X]

def train_ols(X_train, y_train):
    X_b = add_intercept(X_train)
    y = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train
    
    try:
        beta_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    except np.linalg.LinAlgError:
        # Xử lý nhiễu vi phân (Tikhonov Regularization)
        beta_hat = np.linalg.inv(X_b.T @ X_b + np.eye(X_b.shape[1]) * 1e-8) @ X_b.T @ y
    return beta_hat

def train_ols_selected(X_train, y_train, threshold=10.0, num_cols_count=6):
    """
    num_cols_count: Số lượng cột liên tục (numeric) nằm ở phần đầu của ma trận X_train.
    """
    X_df = X_train.copy()
    
    # Chỉ trích xuất các cột số thực để quét VIF, tránh Dummy Variable Trap từ OneHotEncoder
    numeric_cols = list(X_df.columns[:num_cols_count])
    
    while True:
        X_num = X_df[numeric_cols]
        corr_matrix = np.corrcoef(X_num.values, rowvar=False)
        try:
            inv_corr = np.linalg.inv(corr_matrix)
            vifs = np.diag(inv_corr)
        except np.linalg.LinAlgError:
            # Drop biến đầu tiên nếu ma trận corr suy biến hoàn toàn
            col_to_drop = numeric_cols[0]
            X_df = X_df.drop(col_to_drop, axis=1)
            numeric_cols.remove(col_to_drop)
            continue
            
        max_vif_idx = np.argmax(vifs)
        max_vif = vifs[max_vif_idx]
        
        if max_vif > threshold:
            col_to_drop = numeric_cols[max_vif_idx]
            X_df = X_df.drop(col_to_drop, axis=1)
            numeric_cols.remove(col_to_drop)
        else:
            break
            
    # Hồi quy OLS với các biến số thực đã lọc + toàn bộ biến One-hot
    beta_hat = train_ols(X_df, y_train)
    return beta_hat, X_df.columns.tolist()

def train_ridge_lasso(X_train, y_train, k=5):
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_cv_score = float('inf')
    
    # 1. Thêm Intercept (Cột số 1) để đồng bộ với OLS
    X_b = add_intercept(X_train) 
    
    # 2. Chuyển y sang numpy array
    y_val = y_train.values if isinstance(y_train, pd.Series) else y_train

    for lam in lambdas:
        # Truyền ma trận đã có Intercept (X_b) vào kfold_cv
        cv_score = kfold_cv(X_b, y_val, k=k, lam=lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Truyền ma trận đã có Intercept (X_b) vào ridge_fit
    beta_hat_ridge = ridge_fit(X_b, y_val, best_lambda)
    
    return beta_hat_ridge, best_lambda

def evaluate_models(y_true, y_pred):
    y_t = np.array(y_true).flatten()
    y_p = np.array(y_pred).flatten()
    
    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean((y_t - y_p)**2))
    tss = np.sum((y_t - np.mean(y_t))**2)
    rss = np.sum((y_t - y_p)**2)
    r2 = 1 - (rss / tss)
    
    return {'MAE': mae, 'RMSE': rmse, 'R-squared': r2}
























import numpy as np
import pandas as pd

def mean_absolute_error_numpy(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error_numpy(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score_numpy(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def evaluate_model(y_true, y_pred, is_log_transformed=True):
    """
    Tính toán sai số bằng thuần Numpy. Đảo ngược thang đo Logarit để ra sai số thực tế.
    """
    if is_log_transformed:
        y_true_real = np.expm1(y_true)
        y_pred_real = np.expm1(y_pred)
    else:
        y_true_real = y_true
        y_pred_real = y_pred

    # Dùng hàm tự build thay vì sklearn
    mae = mean_absolute_error_numpy(y_true_real, y_pred_real)
    rmse = root_mean_squared_error_numpy(y_true_real, y_pred_real)
    r2 = r2_score_numpy(y_true_real, y_pred_real)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r2
    }

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