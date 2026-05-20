# data_pipeline.py
import pandas as pd
import numpy as np


# Tính năng thử nghiệm trong SCIKIT-LEARN (Bắt buộc)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

class AirQualityPipeline:
    def __init__(self, num_cols, skewed_cols):
        self.num_cols = num_cols
        self.skewed_cols = skewed_cols
        
        # IterativeImputer cực mạnh, nội suy nồng độ hóa chất dựa trên các hóa chất khác
        self.imputer = IterativeImputer(max_iter=10, random_state=42)
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        
        self.top_cities = []

    def _preprocess_features(self, X):
        X_copy = X.copy()
        
        # Trích xuất thời gian (Tháng) để thuật toán học tính mùa vụ (Sương mù/Mưa)
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        X_copy['Month'] = X_copy['Date'].dt.month.astype(str) 
        
        # Gom nhóm không gian: Top 5 thành phố ô nhiễm/nhiều dữ liệu nhất, còn lại là 'Other'
        if not self.top_cities:
            self.top_cities = X_copy['City'].value_counts().nlargest(5).index.tolist()
        X_copy['City_Grouped'] = np.where(X_copy['City'].isin(self.top_cities), X_copy['City'], 'Other')
        
        return X_copy

    def fit(self, X):
        X_prep = self._preprocess_features(X)
        X_num = X_prep[self.num_cols]
        
        # 1. Điền khuyết (Impute)
        self.imputer.fit(X_num)
        X_num_imputed = pd.DataFrame(self.imputer.transform(X_num), columns=self.num_cols)
        
        # 2. Chặn giá trị âm (Nồng độ hóa chất ngoài đời không thể âm)
        X_num_imputed = X_num_imputed.clip(lower=0)

        # 3. Trị phân phối lệch phải (Log Transform) cho bụi mịn
        for col in self.skewed_cols:
            X_num_imputed[col] = np.log1p(X_num_imputed[col])
            
        # 4. Chuẩn hóa Z-score
        self.scaler.fit(X_num_imputed)
        
        # 5. Fit One-Hot Encoding cho Biến phân loại (City & Month)
        self.ohe.fit(X_prep[['City_Grouped', 'Month']])
        
        return self

    def transform(self, X):
        X_prep = self._preprocess_features(X)
        X_num = X_prep[self.num_cols]
        
        # Xử lý số thực
        X_num_imputed = pd.DataFrame(self.imputer.transform(X_num), columns=self.num_cols, index=X.index)
        X_num_imputed = X_num_imputed.clip(lower=0)
        
        for col in self.skewed_cols:
            X_num_imputed[col] = np.log1p(X_num_imputed[col])
            
        num_scaled = self.scaler.transform(X_num_imputed)
        num_scaled_df = pd.DataFrame(num_scaled, columns=self.num_cols, index=X.index)
        
        # Xử lý Categorical
        cat_encoded = self.ohe.transform(X_prep[['City_Grouped', 'Month']])
        cat_names = self.ohe.get_feature_names_out(['City_Grouped', 'Month'])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_names, index=X.index)
        
        # Gộp lại thành ma trận cuối cùng
        return pd.concat([num_scaled_df, cat_encoded_df], axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
