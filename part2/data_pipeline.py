import pandas as pd
import numpy as np

class AirQualityPipeline:
    def __init__(self, num_cols, skewed_cols):
        self.num_cols = num_cols
        self.skewed_cols = skewed_cols
        
        # Tham số MV3: Regression Imputation, OLS tự cài
        self.impute_models = {}      # {col: (other_cols, beta)}
        self.impute_medians = {}     # {col: median} dùng làm fallback
        self.cols_with_missing = []  # Cột nào có missing trong tập train
        
        # Tham số RobustScaler tự cài đặt: (x - median) / IQR
        self.scale_medians = None
        self.scale_iqrs = None
        
        # Tham số One-Hot Encoding (pd.get_dummies)
        self.ohe_columns = None
        
        self.top_cities = []
        self.imputed_cols = []

    # MV3: REGRESSION IMPUTATION
    # Với mỗi cột bị missing: hồi quy cột đó theo các cột còn lại,
    # dùng giá trị dự đoán OLS để điền giá trị khuyết
    # Công thức: beta = (X^T X + εI)^{-1} X^T y
    def _impute_fit_transform(self, X_num):
        X_filled = X_num.copy()
        cols = list(X_num.columns)
        
        # Lưu median tấ cả các cột (fallback + điền tạm khi hồi quy)
        for col in cols:
            med = X_num[col].median()
            self.impute_medians[col] = med if not pd.isna(med) else 0.0
        
        # Xác định cột nào có missing
        self.cols_with_missing = [c for c in cols if X_num[c].isna().any()]
        
        for col in self.cols_with_missing:
            missing_mask = X_num[col].isna()
            other_cols = [c for c in cols if c != col]
            
            # Điền tạm median cho các cột khác (xử lý nhiều cột missing cùng lúc)
            X_temp = X_filled[other_cols].copy()
            for oc in other_cols:
                X_temp[oc] = X_temp[oc].fillna(self.impute_medians[oc])
            
            train_rows = ~missing_mask
            X_train = X_temp[train_rows].values
            y_train = X_num.loc[train_rows, col].values
            
            # OLS: beta = (X^T X + εI)^{-1} X^T y
            X_b = np.c_[np.ones(X_train.shape[0]), X_train]
            try:
                XtX = X_b.T @ X_b
                beta = np.linalg.solve(XtX + 1e-8 * np.eye(XtX.shape[0]), X_b.T @ y_train)
                self.impute_models[col] = (other_cols, beta)
                
                # Dự đoán giá trị missing
                X_pred = X_temp[missing_mask].values
                X_pred_b = np.c_[np.ones(X_pred.shape[0]), X_pred]
                X_filled.loc[missing_mask, col] = X_pred_b @ beta
            except np.linalg.LinAlgError:
                self.impute_models[col] = None
                X_filled.loc[missing_mask, col] = self.impute_medians[col]
        
        # Tạo cột indicator (0/1) đánh dấu vị trí đã điền khuyết
        indicator_df = pd.DataFrame(index=X_num.index)
        for col in self.cols_with_missing:
            indicator_df[f'missingindicator_{col}'] = X_num[col].isna().astype(float)
        
        result = pd.concat([X_filled.reset_index(drop=True), 
                            indicator_df.reset_index(drop=True)], axis=1)
        return result

    def _impute_transform(self, X_num):
        X_filled = X_num.copy()
        cols = list(X_num.columns)
        
        # Điền missing bằng model đã fit
        for col in cols:
            missing_mask = X_num[col].isna()
            if not missing_mask.any():
                continue
            
            if col in self.impute_models and self.impute_models[col] is not None:
                other_cols, beta = self.impute_models[col]
                X_temp = X_filled[other_cols].copy()
                for oc in other_cols:
                    X_temp[oc] = X_temp[oc].fillna(self.impute_medians.get(oc, 0.0))
                
                X_pred = X_temp[missing_mask].values
                X_pred_b = np.c_[np.ones(X_pred.shape[0]), X_pred]
                X_filled.loc[missing_mask, col] = X_pred_b @ beta
            else:
                X_filled.loc[missing_mask, col] = self.impute_medians.get(col, 0.0)
        
        # Tạo indicator cho CÙNG danh sách cột như khi fit
        indicator_df = pd.DataFrame(index=X_num.index)
        for col in self.cols_with_missing:
            indicator_df[f'missingindicator_{col}'] = X_num[col].isna().astype(float)
        
        result = pd.concat([X_filled.reset_index(drop=True), 
                            indicator_df.reset_index(drop=True)], axis=1)
        return result

    # ROBUST SCALER: (x - median) / IQR
    # Bớt ảnh hưởng bởi outliers hơn z-score chuẩn (dùng median thay mean)
    def _scale_fit(self, X):
        self.scale_medians = X.median()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        self.scale_iqrs = q3 - q1
        self.scale_iqrs = self.scale_iqrs.replace(0, 1.0)  # Tránh chia cho 0

    def _scale_transform(self, X):
        return (X - self.scale_medians) / self.scale_iqrs

    # FEATURE ENGINEERING
    def _preprocess_features(self, X):
        X_copy = X.copy()
        
        # Trích xuất thời gian (Tháng) để thuật toán học tính mùa vụ (Sương mù/Mưa)
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        month = X_copy['Date'].dt.month
        X_copy['Month_sin'] = np.sin(2 * np.pi * month / 12)
        X_copy['Month_cos'] = np.cos(2 * np.pi * month / 12)
        X_copy = X_copy.drop(columns=['Date'])
        
        # Gom nhóm không gian: Top 5 thành phố ô nhiễm/nhiều dữ liệu nhất, còn lại là 'Other'
        X_copy['City_Grouped'] = np.where(X_copy['City'].isin(self.top_cities), X_copy['City'], 'Other')
        
        return X_copy

    # FIT / TRANSFORM / FIT_TRANSFORM
    def fit(self, X):
        # Xác định top 5 city từ tập Train
        self.top_cities = X['City'].value_counts().nlargest(5).index.tolist()
        
        X_prep = self._preprocess_features(X)
        X_num = X_prep[self.num_cols]
        
        # 1. Regression Imputation
        X_num_imputed = self._impute_fit_transform(X_num)
        self.imputed_cols = list(X_num_imputed.columns)
        
        # 2. Chặn giá trị âm (Nồng độ hóa chất ngoài đời không có âm đc)
        X_num_imputed = X_num_imputed.clip(lower=0)

        # 3. Trị phân phối lệch phải (Log1p bảo toàn quan hệ Log-Log với Y)
        for col in self.skewed_cols:
            if col in X_num_imputed.columns:
                X_num_imputed[col] = np.log1p(X_num_imputed[col])
            
        # 4. Chuẩn hóa RobustScaler: (x - median) / IQR
        self._scale_fit(X_num_imputed)
        
        # 5. One-Hot Encoding bằng pd.get_dummies (Pandas — chắc vẫn được phép)
        cat_encoded = pd.get_dummies(X_prep[['City_Grouped']], drop_first=True, dtype=float)
        self.ohe_columns = list(cat_encoded.columns)
        
        return self

    def transform(self, X):
        X_prep = self._preprocess_features(X)
        X_num = X_prep[self.num_cols]
        X_month_cyclic = X_prep[['Month_sin', 'Month_cos']].reset_index(drop=True)
        
        # 1. Regression Imputation (dùng model đã fit)
        X_num_imputed = self._impute_transform(X_num)
        X_num_imputed.index = X.index
        X_num_imputed = X_num_imputed.clip(lower=0)
        
        # 2. Log1p
        for col in self.skewed_cols:
            if col in X_num_imputed.columns:
                X_num_imputed[col] = np.log1p(X_num_imputed[col])
            
        # 3. RobustScaler (dùng median/IQR đã fit)
        num_scaled_df = self._scale_transform(X_num_imputed)
        num_scaled_df.columns = self.imputed_cols
        num_scaled_df.index = X.index
        
        # 4. Nối Month_sin, Month_cos
        num_scaled_df = pd.concat([num_scaled_df, X_month_cyclic.set_index(num_scaled_df.index)], axis=1)
        
        # 5. One-Hot Encoding + reindex đảm bảo cùng cột với train
        cat_encoded = pd.get_dummies(X_prep[['City_Grouped']], drop_first=True, dtype=float)
        cat_encoded = cat_encoded.reindex(columns=self.ohe_columns, fill_value=0.0)
        cat_encoded_df = pd.DataFrame(cat_encoded.values, columns=self.ohe_columns, index=X.index)
        
        # Gộp lại thành ma trận cuối cùng
        return pd.concat([num_scaled_df, cat_encoded_df], axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)