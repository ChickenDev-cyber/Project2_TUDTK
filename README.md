# Project 2 - Data Fitting va Ordinary Least Squares

Repository nay la phan code cho Do an 2 mon **Toan Ung Dung va Thong Ke (MTH00051)**, tap trung vao Data Fitting, Ordinary Least Squares (OLS), cac mo hinh chinh quy hoa va ung dung tren du lieu thuc te.

## Thong tin nhom

**Nhom:** 02

| STT | Ho va ten | MSSV | Phan cong chinh |
|---:|---|---|---|
| 1 | Dao Tien Dat | 24120033 | `ridge_lasso.py`, `cross_validation.py`, ly thuyet Ridge, Lasso va Cross-Validation |
| 2 | Pham Minh Trong | 24120151 | Code Part 2, dataset va EDA |
| 3 | Bui Nhat Bao | 24120167 | Code Part 2, tien xu ly du lieu va mo hinh |
| 4 | Trinh Kim Mai | 24120199 | `ols_implementation.py`, ly thuyet OLS va Hat Matrix |
| 5 | Tran Cong Quang | 24120221 | `residual_analysis.py`, notebook Part 1/Part 2, ket qua va bao cao |

## Muc tieu do an

Do an gom hai phan chinh:

1. **Part 1 - Ly thuyet va minh hoa Data Fitting/OLS**
   - Cai dat OLS tu dau.
   - Kiem tra Hat Matrix va cac tinh chat dai so.
   - Tinh cac chi so danh gia mo hinh: RSS, TSS, R-squared, adjusted R-squared, F-statistic.
   - Suy dien he so hoi quy: standard error, t-statistic, p-value, confidence interval.
   - Phat hien da cong tuyen bang VIF.
   - Cai dat Ridge Regression, Lasso Regression bang Coordinate Descent.
   - Cai dat K-Fold Cross-Validation.
   - Phan tich phan du va minh hoa dinh ly Gauss-Markov bang Monte Carlo.

2. **Part 2 - Ung dung tren du lieu thuc te**
   - Su dung bo du lieu chat luong khong khi `city_day.csv`.
   - Xay dung pipeline tien xu ly du lieu.
   - Xu ly missing values, feature engineering, scaling va encoding.
   - Thu nghiem cac mo hinh hoi quy va so sanh ket qua.
   - Trinh bay ket qua bang notebook va hinh truc quan.

## Luu y quan trong ve yeu cau thu vien

Theo yeu cau cua de bai va phan nhac lai cua giang vien:

> NumPy/SciPy chi duoc dung cho tinh toan ho tro, truc quan hoa, kiem chung hoac phan tich du lieu khi phu hop; khong duoc dung de thay the phan cai dat thuat toan chinh trong Part 1.

Vi vay, trong **Part 1**, cac thuat toan chinh khong goi `numpy`, `scipy`, `np.array`, `np.linalg` hay `numpy.linalg.lstsq`. Nhom tach cac phep toan nen tang vao file `matrix_ops.py` de dung chung.

### Vai tro cua `matrix_ops.py`

`part1/matrix_ops.py` la module tu cai dat cac phep toan ma tran/vector co ban bang Python thuan:

- `Vector`, `Matrix`: lop danh sach mo rong de thao tac gan voi vector va ma tran.
- `transpose`, `matmul`, `matvec`, `dot`: cac phep nhan va chuyen vi.
- `solve`: giai he phuong trinh tuyen tinh bang khu Gauss-Jordan co pivot.
- `inverse`: tinh nghich dao bang cach giai nhieu he tuyen tinh.
- `diag`, `trace`, `identity`, `zeros`: cac ham tien ich cho dai so tuyen tinh.
- `mean`, `variance`, `sum_squares`: cac phep thong ke co ban.
- `take_rows`, `take_values`, `column_stack`, `logspace`: ho tro Cross-Validation va Ridge.

Ly do tach file nay:

- Tranh lap lai cung mot doan code ma tran trong nhieu file.
- Giup cac file thuat toan nhu OLS, Ridge, Lasso, Cross-Validation ngan gon hon.
- De kiem tra viec khong dung NumPy/SciPy trong phan cai dat chinh.
- Neu can sua cach nhan ma tran, giai he tuyen tinh hoac kiem tra sai so, chi can sua mot noi.

Noi ngan gon: `matrix_ops.py` khong phai thu vien ngoai, ma la phan tu cai dat cua nhom de thay cho cac phep ma tran thuong duoc NumPy ho tro.

## Cau truc thu muc

```text
Project2_TUDTK/
|-- README.md
|-- requirements.txt
|-- Toan UDTK_Project_2-Data Fitting va OLS.pdf
|-- _Toan_UDTK_Project_2_extract.txt
|-- part1/
|   |-- matrix_ops.py
|   |-- ols_implementation.py
|   |-- ridge_lasso.py
|   |-- cross_validation.py
|   |-- residual_analysis.py
|   |-- test_part1_unit.py
|   |-- part1_notebook.ipynb
|-- part2/
|   |-- data/
|   |   |-- city_day.csv
|   |-- data_pipeline.py
|   |-- advanced_methods.py
|   |-- model_comparison.py
|   |-- part2_notebook.ipynb
```

## Mo ta file chinh

### Part 1

| File | Vai tro |
|---|---|
| `matrix_ops.py` | Cac phep toan ma tran/vector tu cai dat, dung chung cho Part 1 |
| `ols_implementation.py` | OLS, Hat Matrix, model metrics, suy dien he so, VIF |
| `ridge_lasso.py` | Ridge Regression va Lasso Coordinate Descent |
| `cross_validation.py` | K-Fold Cross-Validation va chon lambda cho Ridge |
| `residual_analysis.py` | Residual plots, standardized residuals, Cook's Distance |
| `test_part1_unit.py` | Unit test cho cac ham Part 1 |
| `part1_notebook.ipynb` | Notebook trinh bay ly thuyet, minh hoa va ket qua Part 1 |

### Part 2

| File | Vai tro |
|---|---|
| `data/city_day.csv` | Bo du lieu chat luong khong khi dung cho ung dung thuc te |
| `data_pipeline.py` | Pipeline tien xu ly: imputation, scaling, feature engineering, encoding |
| `advanced_methods.py` | Cac mo hinh/mo phong nang cao va diagnostic plots |
| `model_comparison.py` | Ham huan luyen, du doan va so sanh metric |
| `part2_notebook.ipynb` | Notebook trinh bay pipeline, mo hinh va ket qua Part 2 |

## Cai dat moi truong

Yeu cau Python 3.10+.

```bash
pip install -r requirements.txt
```

Thu vien trong `requirements.txt` phuc vu cho ca hai phan cua do an:

- `matplotlib`, `seaborn`, `jupyter`: truc quan hoa va notebook.
- `pandas`: doc va xu ly du lieu Part 2.
- `scikit-learn`: doi chieu/kiem chung ket qua khi can.
- `numpy`, `scipy`: ho tro tinh toan va cac thuc nghiem Part 2; khong thay the thuat toan chinh trong Part 1.

## Cach chay Part 1

Chay unit test:

```bash
cd part1
python -m unittest test_part1_unit.py
```

Chay tung file minh hoa:

```bash
python ols_implementation.py
python ridge_lasso.py
python cross_validation.py
python residual_analysis.py
```

Mo notebook:

```bash
jupyter notebook part1/part1_notebook.ipynb
```

## Cach chay Part 2

Mo notebook Part 2:

```bash
jupyter notebook part2/part2_notebook.ipynb
```

Du lieu mac dinh nam tai:

```text
part2/data/city_day.csv
```

## Kiem tra nhanh truoc khi nop

Nhom co the dung cac lenh sau de kiem tra Part 1:

```bash
cd part1
python -m py_compile matrix_ops.py ols_implementation.py ridge_lasso.py cross_validation.py residual_analysis.py test_part1_unit.py
python -m unittest test_part1_unit.py
```

De kiem tra Part 1 khong con import NumPy/SciPy trong code:

```powershell
Select-String -Path ".\part1\*.py", ".\part1\part1_notebook.ipynb" -Pattern "import numpy|numpy|np\.|scipy|np\.array|np\.linalg|linalg"
```

Neu lenh tren khong in ra ket qua, Part 1 khong con dau vet goi NumPy/SciPy trong phan code/notebook hien tai.

## Ghi chu ve tinh tai lap

- Cac vi du mo phong dung seed co dinh de ket qua tai lap duoc.
- Part 1 uu tien giai thich ro cong thuc va minh hoa bang du lieu mo phong.
- Part 2 uu tien quy trinh ung dung thuc te: tien xu ly du lieu, huan luyen mo hinh, danh gia va so sanh.
- Notebook va report co the dung hinh truc quan de giai thich ket qua, nhung phan thuat toan chinh cua Part 1 van duoc viet bang Python thuan.

## Trang thai hien tai

- Part 1 da duoc chinh de khong dung NumPy/SciPy cho cac thuat toan chinh.
- `matrix_ops.py` la module ho tro noi bo do nhom tu cai dat.
- Unit test Part 1 da duoc chuyen sang Python thuan.
- Notebook Part 1 da co ghi chu ve viec khong dung thu vien so de thay the thuat toan chinh.

