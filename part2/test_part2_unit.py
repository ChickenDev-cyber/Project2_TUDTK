import unittest
import pandas as pd
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import modules part 2
from data_pipeline import AirQualityPipeline
from model_comparison import mean_absolute_error_py, root_mean_squared_error_py, r2_score_py
from advanced_methods import BayesianLinearRegression


# Test class 1: data pipeline
class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Tạo dữ liệu giả lập cho pipeline
        self.df = pd.DataFrame({
            'City': ['Hanoi', 'Hanoi', 'HCMC', 'HCMC'],
            'Date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
            'PM2.5': [50.0, None, 40.0, 45.0],
            'Temp': [20.0, 25.0, 30.0, None]
        })
        self.pipeline = AirQualityPipeline(num_cols=['PM2.5', 'Temp'], skewed_cols=['PM2.5'])

    def test_fit_transform_no_missing_values(self):
        """Kiểm tra Pipeline có điền khuyết Missing Values (Imputation) không."""
        transformed = self.pipeline.fit_transform(self.df)
        self.assertIn('PM2.5', transformed.columns)
        self.assertEqual(transformed.isna().sum().sum(), 0, "Dữ liệu sau khi qua Pipeline vẫn còn giá trị None!")

    def test_transform_unseen_data(self):
        """Kiểm tra phương thức transform trên tập dữ liệu chưa từng thấy (Test set)."""
        self.pipeline.fit(self.df)
        test_df = pd.DataFrame({
            'City': ['Danang'],
            'Date': ['2023-05-01'],
            'PM2.5': [None],
            'Temp': [35.0]
        })
        test_transformed = self.pipeline.transform(test_df)
        self.assertEqual(test_transformed.isna().sum().sum(), 0, "Transform trên tập Test thất bại!")
        
        # Kiểm tra cột missing indicator có được tạo ra không
        has_indicator = 'missingindicator_PM2.5' in test_transformed.columns or 'missingindicator_Temp' in test_transformed.columns
        self.assertTrue(has_indicator, "Không tạo được cột Missing Indicator.")

# Test class 2: model comparison metrics
class TestModelComparisonMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = [1.0, 2.0, 3.0]
        self.y_pred_perfect = [1.0, 2.0, 3.0]
        self.y_pred_off = [2.0, 3.0, 4.0]

    def test_mean_absolute_error(self):
        """Kiểm tra MAE có trả về 0 nếu dự đoán hoàn hảo và tính đúng sai số tuyệt đối."""
        mae_perfect = mean_absolute_error_py(self.y_true, self.y_pred_perfect)
        self.assertAlmostEqual(mae_perfect, 0.0)

        mae_off = mean_absolute_error_py(self.y_true, self.y_pred_off)
        self.assertAlmostEqual(mae_off, 1.0)

    def test_root_mean_squared_error(self):
        """Kiểm tra RMSE."""
        rmse_perfect = root_mean_squared_error_py(self.y_true, self.y_pred_perfect)
        self.assertAlmostEqual(rmse_perfect, 0.0)

        rmse_off = root_mean_squared_error_py(self.y_true, self.y_pred_off)
        self.assertAlmostEqual(rmse_off, 1.0)

    def test_r2_score(self):
        """Kiểm tra điểm R-squared (Hệ số xác định)."""
        r2_perfect = r2_score_py(self.y_true, self.y_pred_perfect)
        self.assertAlmostEqual(r2_perfect, 1.0)

# Test class 3: advanced methods
class TestAdvancedMethods(unittest.TestCase):
    def setUp(self):
        self.df_X = pd.DataFrame({'f1': [1.0, 2.0], 'f2': [2.0, 3.0]})
        self.df_y = pd.Series([1.0, 2.0])
        self.bayes_model = BayesianLinearRegression()

    def test_bayesian_fit(self):
        """Kiểm tra quá trình Fit của mô hình Bayesian cập nhật phân phối hậu nghiệm."""
        self.bayes_model.fit(self.df_X, self.df_y)
        self.assertIsNotNone(self.bayes_model.m_N, "Vector m_N hậu nghiệm chưa được tính!")
        self.assertIsNotNone(self.bayes_model.S_N, "Ma trận S_N hậu nghiệm chưa được tính!")

    def test_bayesian_credible_interval(self):
        """Kiểm tra phương thức tính khoảng tin cậy (Credible Interval) Bayesian."""
        self.bayes_model.fit(self.df_X, self.df_y)
        y_pred, std, (lower_bound, upper_bound) = self.bayes_model.get_credible_interval(self.df_X)
        
        self.assertEqual(len(y_pred), 2)
        self.assertEqual(len(std), 2)
        self.assertEqual(len(lower_bound), 2)
        self.assertEqual(len(upper_bound), 2)
        
        # Đảm bảo lower_bound luôn nhỏ hơn upper_bound
        for i in range(2):
            self.assertLess(lower_bound[i], upper_bound[i], "Khoảng tin cậy không hợp lệ!")

# Run all unit tests
if __name__ == '__main__':
    unittest.main()
