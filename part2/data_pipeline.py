import pandas as pd
import math
import sys
import os

part1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'part1'))
if part1_path not in sys.path:
    sys.path.append(part1_path)
from ols_implementation import ols_fit
from matrix_ops import matmul, as_matrix

class AirQualityPipeline:
    def __init__(self, num_cols, skewed_cols):
        self.num_cols = num_cols
        self.skewed_cols = skewed_cols
        
        # Regression Imputation
        self.impute_models = {}      # {col: (other_cols, beta)}
        self.impute_medians = {}     # {col: median} dùng làm fallback
        self.cols_with_missing = []  # Cột nào có missing trong tập train
        
        # RobustScaler (x - median) / IQR
        self.scale_medians = None
        self.scale_iqrs = None
        
        # One-Hot Encoding (pd.get_dummies)
        self.ohe_columns = None
        
        self.top_cities = []
        self.imputed_cols = []

    # Regression Imputation
    def _impute_fit_transform(self, X_num):
        X_filled = X_num.copy()
        cols = list(X_num.columns)
        
        for col in cols:
            med = X_num[col].median()
            self.impute_medians[col] = med if pd.notna(med) else 0.0
        
        self.cols_with_missing = [c for c in cols if X_num[c].isna().any()]
        
        for col in self.cols_with_missing:
            missing_mask = X_num[col].isna()
            other_cols = [c for c in cols if c != col]
            
            X_temp = X_filled[other_cols].copy()
            for oc in other_cols:
                X_temp[oc] = X_temp[oc].fillna(self.impute_medians[oc])
            
            train_rows = ~missing_mask
            X_train = X_temp[train_rows].values.tolist()
            y_train = X_num.loc[train_rows, col].values.tolist()
            
            # Thêm cột hệ số chặn
            X_b = [[1.0] + row for row in X_train]
            
            try:
                beta_hat, _ = ols_fit(X_b, y_train)
                self.impute_models[col] = (other_cols, beta_hat)
                
                # Dự đoán giá trị missing
                X_pred = X_temp[missing_mask].values.tolist()
                X_pred_b = [[1.0] + row for row in X_pred]
                
                beta_mat = as_matrix([[b] for b in beta_hat])
                preds = matmul(X_pred_b, beta_mat)
                X_filled.loc[missing_mask, col] = [row[0] for row in preds]
            except Exception:
                self.impute_models[col] = None
                X_filled.loc[missing_mask, col] = self.impute_medians[col]
        
        indicator_df = pd.DataFrame(index=X_num.index)
        for col in self.cols_with_missing:
            indicator_df[f'missingindicator_{col}'] = X_num[col].isna().astype(float)
        
        result = pd.concat([X_filled.reset_index(drop=True), 
                            indicator_df.reset_index(drop=True)], axis=1)
        return result

    def _impute_transform(self, X_num):
        X_filled = X_num.copy()
        cols = list(X_num.columns)
        
        for col in cols:
            missing_mask = X_num[col].isna()
            if not missing_mask.any():
                continue
            
            if col in self.impute_models and self.impute_models[col] is not None:
                other_cols, beta = self.impute_models[col]
                X_temp = X_filled[other_cols].copy()
                for oc in other_cols:
                    X_temp[oc] = X_temp[oc].fillna(self.impute_medians.get(oc, 0.0))
                
                X_pred = X_temp[missing_mask].values.tolist()
                X_pred_b = [[1.0] + row for row in X_pred]
                
                beta_mat = as_matrix([[b] for b in beta])
                preds = matmul(X_pred_b, beta_mat)
                X_filled.loc[missing_mask, col] = [row[0] for row in preds]
            else:
                X_filled.loc[missing_mask, col] = self.impute_medians.get(col, 0.0)
        
        indicator_df = pd.DataFrame(index=X_num.index)
        for col in self.cols_with_missing:
            indicator_df[f'missingindicator_{col}'] = X_num[col].isna().astype(float)
        
        result = pd.concat([X_filled.reset_index(drop=True), 
                            indicator_df.reset_index(drop=True)], axis=1)
        return result

    # RobustScaler
    def _scale_fit(self, X):
        self.scale_medians = X.median()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        self.scale_iqrs = q3 - q1
        self.scale_iqrs = self.scale_iqrs.replace(0, 1.0)

    def _scale_transform(self, X):
        return (X - self.scale_medians) / self.scale_iqrs

    # Feature Engineering
    def _preprocess_features(self, X):
        X_copy = X.copy()
        
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        month = X_copy['Date'].dt.month
        
        X_copy['Month_sin'] = month.apply(lambda x: math.sin(2 * math.pi * x / 12))
        X_copy['Month_cos'] = month.apply(lambda x: math.cos(2 * math.pi * x / 12))
        X_copy = X_copy.drop(columns=['Date'])
        
        X_copy['City_Grouped'] = X_copy['City'].apply(lambda c: c if c in self.top_cities else 'Other')
        
        return X_copy

    def fit(self, X):
        self.top_cities = X['City'].value_counts().nlargest(5).index.tolist()
        
        X_prep = self._preprocess_features(X)
        X_num = X_prep[self.num_cols]
        
        X_num_imputed = self._impute_fit_transform(X_num)
        self.imputed_cols = list(X_num_imputed.columns)
        
        X_num_imputed = X_num_imputed.clip(lower=0)

        for col in self.skewed_cols:
            if col in X_num_imputed.columns:
                X_num_imputed[col] = X_num_imputed[col].apply(math.log1p)
            
        self._scale_fit(X_num_imputed)
        
        cat_encoded = pd.get_dummies(X_prep[['City_Grouped']], drop_first=True, dtype=float)
        self.ohe_columns = list(cat_encoded.columns)
        
        return self

    def transform(self, X):
        X_prep = self._preprocess_features(X)
        X_num = X_prep[self.num_cols]
        X_month_cyclic = X_prep[['Month_sin', 'Month_cos']].reset_index(drop=True)
        
        X_num_imputed = self._impute_transform(X_num)
        X_num_imputed.index = X.index
        X_num_imputed = X_num_imputed.clip(lower=0)
        
        for col in self.skewed_cols:
            if col in X_num_imputed.columns:
                X_num_imputed[col] = X_num_imputed[col].apply(math.log1p)
            
        num_scaled_df = self._scale_transform(X_num_imputed)
        num_scaled_df.columns = self.imputed_cols
        num_scaled_df.index = X.index
        
        num_scaled_df = pd.concat([num_scaled_df, X_month_cyclic.set_index(num_scaled_df.index)], axis=1)
        
        cat_encoded = pd.get_dummies(X_prep[['City_Grouped']], drop_first=True, dtype=float)
        cat_encoded = cat_encoded.reindex(columns=self.ohe_columns, fill_value=0.0)
        cat_encoded_df = pd.DataFrame(cat_encoded.values, columns=self.ohe_columns, index=X.index)
        
        return pd.concat([num_scaled_df, cat_encoded_df], axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)