# Đồ án 2: Data Fitting và Phương pháp OLS

**Môn học:** Toán Ứng Dụng và Thống Kê (MTH00051)  
**Trường:** Đại học Khoa học Tự nhiên, ĐHQG-HCM (FIT-HCMUS)  
**Nhóm:** 02

**Nhóm sinh viên thực hiện:**

| STT | MSSV | Họ và Tên | Phân công chính |
|:---:|:---:|:---|:---|
| 1 | 24120151 | Phạm Minh Trọng | 
| 2 | 24120033 | Đào Tiến Đạt | 
| 3 | 24120167 | Bùi Nhật Bảo | 
| 4 | 24120199 | Trịnh Kim Mai | 
| 5 | 24120221 | Trần Công Quang |

---

## 📝 Giới thiệu Đồ án

Đồ án tập trung vào việc tìm hiểu, tự cài đặt và minh họa các nội dung quan trọng trong **Data Fitting** và **Ordinary Least Squares (OLS)**. Nhóm không chỉ chạy ra kết quả, mà còn cố gắng giải thích công thức, kiểm chứng thuật toán và phân tích ý nghĩa của từng kết quả.

Dự án gồm 2 phần chính:

1. **Phần 1:** Trình bày lý thuyết và tự cài đặt các thuật toán liên quan đến OLS, Hat Matrix, VIF, Ridge, Lasso, Cross-Validation, Residual Analysis và mô phỏng Gauss-Markov.
2. **Phần 2:** Ứng dụng Data Fitting trên bộ dữ liệu thực tế về chất lượng không khí, bao gồm tiền xử lý dữ liệu, xây dựng mô hình và so sánh kết quả.

---

## ⚠️ Lưu ý quan trọng về thư viện

Theo yêu cầu của đề bài và giảng viên, trong **Phần 1**, nhóm không dùng `NumPy`, `SciPy`, `np.array`, `np.linalg` hay `numpy.linalg.lstsq` để thay thế phần cài đặt thuật toán chính.

Để làm được điều đó, nhóm tạo file:

```text
part1/matrix_ops.py
```

File này chứa các phép toán ma trận/vector tự cài bằng Python thuần như nhân ma trận, chuyển vị, giải hệ tuyến tính, nghịch đảo, trace, diag, logspace,... Đây là phần hỗ trợ nội bộ của nhóm, không phải thư viện ngoài.

Lý do tách riêng `matrix_ops.py`:

- Tránh viết lặp lại cùng một đoạn xử lý ma trận trong nhiều file.
- Giúp các file thuật toán như OLS, Ridge, Lasso, Cross-Validation dễ đọc hơn.
- Dễ kiểm tra rằng phần cài đặt chính của Phần 1 không phụ thuộc NumPy/SciPy.

---

## 📂 Sơ đồ Cấu trúc Thư mục

```text
Project2_TUDTK/
├── part1/                         # Lý thuyết và minh họa OLS
│   ├── matrix_ops.py              # Các phép toán ma trận/vector tự cài
│   ├── ols_implementation.py      # OLS, Hat Matrix, Metrics, VIF
│   ├── ridge_lasso.py             # Ridge Regression và Lasso
│   ├── cross_validation.py        # K-Fold Cross-Validation
│   ├── residual_analysis.py       # Phân tích phần dư
│   ├── test_part1_unit.py         # Unit test cho Phần 1
│   └── part1_notebook.ipynb       # Notebook trình bày Phần 1
├── part2/                         # Ứng dụng trên dữ liệu thực tế
│   ├── data/
│   │   └── city_day.csv           # Bộ dữ liệu chất lượng không khí
│   ├── data_pipeline.py           # Pipeline tiền xử lý dữ liệu
│   ├── advanced_methods.py        # Các phương pháp/mô hình nâng cao
│   ├── model_comparison.py        # So sánh mô hình và metric
│   └── part2_notebook.ipynb       # Notebook trình bày Phần 2
├── requirements.txt               # Danh sách thư viện cần cài
├── Toan UDTK_Project_2-Data Fitting va OLS.pdf
└── README.md
```

---

## ⚙️ Hướng dẫn Cài đặt Môi trường

### 1. Yêu cầu hệ thống

- **Python:** phiên bản 3.10 trở lên.
- **Jupyter Notebook:** dùng để mở và chạy notebook.
- **Các thư viện Python:** được liệt kê trong `requirements.txt`.

### 2. Cài đặt thư viện

Mở Terminal/PowerShell tại thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```

Các thư viện như `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `numpy`, `scipy` được dùng cho việc đọc dữ liệu, trực quan hóa, Part 2 hoặc kiểm chứng kết quả. Riêng thuật toán chính của **Part 1** được tự cài bằng Python thuần.

---

## 🚀 Hướng dẫn Chạy Code

### 1. Chạy Notebook Phần 1

Notebook Phần 1 trình bày lý thuyết, gọi các hàm tự cài và minh họa bằng dữ liệu mô phỏng:

```bash
jupyter notebook part1/part1_notebook.ipynb
```

Trong notebook này, các kết quả chính gồm:

- So sánh hệ số OLS tự cài với kết quả kiểm chứng.
- Kiểm tra các tính chất của Hat Matrix.
- Tính VIF để phát hiện đa cộng tuyến.
- Minh họa Ridge, Lasso và chọn lambda bằng Cross-Validation.
- Vẽ các biểu đồ phân tích phần dư.
- Mô phỏng định lý Gauss-Markov bằng Monte Carlo.

### 2. Chạy Unit Test Phần 1

Để kiểm tra nhanh các hàm tự cài:

```bash
cd part1
python -m unittest test_part1_unit.py
```

### 3. Chạy Notebook Phần 2

Notebook Phần 2 trình bày quá trình xử lý dữ liệu thực tế và so sánh mô hình:

```bash
jupyter notebook part2/part2_notebook.ipynb
```

---

## 🔎 Kiểm tra nhanh trước khi nộp

Để kiểm tra các file Python ở Phần 1 có chạy được không:

```bash
cd part1
python -m py_compile matrix_ops.py ols_implementation.py ridge_lasso.py cross_validation.py residual_analysis.py test_part1_unit.py
python -m unittest test_part1_unit.py
```

Để kiểm tra Phần 1 không còn gọi NumPy/SciPy trong code/notebook:

```powershell
Select-String -Path ".\part1\*.py", ".\part1\part1_notebook.ipynb" -Pattern "import numpy|numpy|np\.|scipy|np\.array|np\.linalg|linalg"
```

Nếu lệnh trên không in ra kết quả, nghĩa là Phần 1 không còn dấu vết dùng NumPy/SciPy trong phần cài đặt hiện tại.

---

## 📊 Kết luận chính của Đồ án

- **OLS:** Hệ số tự cài khớp với kết quả kiểm chứng và gần với hệ số thật trên dữ liệu mô phỏng.
- **Hat Matrix:** Các tính chất như đối xứng, lũy đẳng và trace được kiểm tra đúng bằng thực nghiệm.
- **VIF:** Khi tạo dữ liệu có đa cộng tuyến mạnh, VIF tăng rất lớn, cho thấy hệ số hồi quy có thể trở nên nhạy.
- **Ridge và Lasso:** Ridge co nhỏ hệ số, còn Lasso có thể đưa một số hệ số về gần 0 để hỗ trợ chọn biến.
- **Cross-Validation:** Giúp chọn lambda dựa trên lỗi kiểm chứng thay vì chọn thủ công.
- **Residual Analysis:** Các biểu đồ phần dư giúp đánh giá giả thiết tuyến tính, phương sai và điểm ảnh hưởng lớn.
- **Gauss-Markov:** Mô phỏng Monte Carlo cho thấy OLS có phương sai nhỏ hơn so với một ước lượng tuyến tính không chệch thay thế.

---

## ✅ Trạng thái hiện tại

- Phần 1 đã được chỉnh để thuật toán chính không dùng NumPy/SciPy.
- `matrix_ops.py` đã được thêm để tự cài các phép toán ma trận cần thiết.
- Unit test Phần 1 đã chạy ổn.
- Notebook Phần 1 đã được cập nhật để giải thích rõ vai trò của phần tự cài.
- README đã được viết lại để giáo viên dễ đọc và dễ kiểm tra cấu trúc dự án.

