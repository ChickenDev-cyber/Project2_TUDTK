import pandas as pd
import math
import sys
import os

part1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'part1'))
if part1_path not in sys.path:
    sys.path.append(part1_path)
try:
    from cross_validation import kfold_cv, ridge_cv_score
    from ridge_lasso import ridge_fit
    from ols_implementation import ols_fit, vif
except ImportError:
    pass

def add_intercept(X):
    X_list = X.values.tolist() if hasattr(X, 'values') else X.tolist()
    return [[1.0] + list(row) for row in X_list]

def train_ols(X_train, y_train):
    X_b = add_intercept(X_train)
    
    if hasattr(y_train, 'values'):
        y_val = y_train.values.flatten().tolist()
    else:
        if hasattr(y_train, 'flatten'):
            y_val = y_train.flatten().tolist()
        else:
            y_val = list(y_train)
            
    beta_hat, _ = ols_fit(X_b, y_val)
    return beta_hat

def train_ols_selected(X_train, y_train, threshold=10.0, num_cols_count=6):
    X_df = X_train.copy()
    numeric_cols = list(X_df.columns[:num_cols_count])
    
    while True:
        X_num = X_df[numeric_cols].values.tolist()
        vifs = vif(X_num)
        
        has_inf = False
        for i, val in enumerate(vifs):
            if math.isinf(val):
                col_to_drop = numeric_cols[i]
                X_df = X_df.drop(col_to_drop, axis=1)
                numeric_cols.remove(col_to_drop)
                has_inf = True
                break
                
        if has_inf:
            continue
            
        max_vif = max(vifs)
        max_vif_idx = vifs.index(max_vif)
        
        if max_vif > threshold:
            col_to_drop = numeric_cols[max_vif_idx]
            X_df = X_df.drop(col_to_drop, axis=1)
            numeric_cols.remove(col_to_drop)
        else:
            break
            
    beta_hat = train_ols(X_df, y_train)
    return beta_hat, X_df.columns.tolist()

def train_ridge_lasso(X_train, y_train, k=5):
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_cv_score = float('inf')
    
    X_b = add_intercept(X_train) 
    
    if hasattr(y_train, 'values'):
        y_val = y_train.values.flatten().tolist()
    else:
        if hasattr(y_train, 'flatten'):
            y_val = y_train.flatten().tolist()
        else:
            y_val = list(y_train)

    for lam in lambdas:
        cv_score = ridge_cv_score(X_b, y_val, k=k, lam=lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    beta_hat_ridge = ridge_fit(X_b, y_val, best_lambda)
    
    return beta_hat_ridge, best_lambda

def mean_absolute_error_py(y_true, y_pred):
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)

def root_mean_squared_error_py(y_true, y_pred):
    return math.sqrt(sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true))

def r2_score_py(y_true, y_pred):
    mean_true = sum(y_true) / len(y_true)
    ss_res = sum((t - p)**2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_true)**2 for t in y_true)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def evaluate_model(y_true, y_pred, is_log_transformed=True):
    y_t = y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
    y_p = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
    
    if hasattr(y_t[0], '__iter__') and not isinstance(y_t[0], str):
        y_t = [v[0] for v in y_t]
    if hasattr(y_p[0], '__iter__') and not isinstance(y_p[0], str):
        y_p = [v[0] for v in y_p]
        
    if is_log_transformed:
        y_true_real = [math.expm1(v) for v in y_t]
        y_pred_real = [math.expm1(v) for v in y_p]
    else:
        y_true_real = y_t
        y_pred_real = y_p

    mae = mean_absolute_error_py(y_true_real, y_pred_real)
    rmse = root_mean_squared_error_py(y_true_real, y_pred_real)
    r2 = r2_score_py(y_true_real, y_pred_real)

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
    df_test = compare_models(test_preds, y_test, is_log)
    
    df_final = pd.concat([df_train, df_test], axis=1)
    
    columns = pd.MultiIndex.from_product(
        [['Tập Train', 'Tập Test'], ['MAE', 'RMSE', 'R-squared']]
    )
    df_final.columns = columns
    
    return df_final