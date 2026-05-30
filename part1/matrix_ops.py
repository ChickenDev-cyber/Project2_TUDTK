import math


class Vector(list):
    @property
    def shape(self):
        return (len(self),)

    def flatten(self):
        return Vector(self)

    def tolist(self):
        return list(self)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Vector(x + other for x in self)
        return Vector(x + y for x, y in zip(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Vector(x - other for x in self)
        return Vector(x - y for x, y in zip(self, other))

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Vector(other - x for x in self)
        return Vector(y - x for x, y in zip(self, other))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(x * other for x in self)
        return Vector(x * y for x, y in zip(self, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(x / other for x in self)
        return Vector(x / y for x, y in zip(self, other))

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(other / x for x in self)
        return Vector(y / x for x, y in zip(self, other))

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Vector(x ** other for x in self)
        return Vector(x ** y for x, y in zip(self, other))

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            return Vector(other ** x for x in self)
        return Vector(y ** x for x, y in zip(self, other))


class Matrix(list):
    @property
    def shape(self):
        if not self:
            return (0, 0)
        return (len(self), len(self[0]))

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[v + other for v in row] for row in self])
        return Matrix([[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self, other)])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[v - other for v in row] for row in self])
        return Matrix([[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self, other)])

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[other - v for v in row] for row in self])
        return Matrix([[b - a for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self, other)])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[v * other for v in row] for row in self])
        return Matrix([[a * b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self, other)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[v / other for v in row] for row in self])
        return Matrix([[a / b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(self, other)])


def as_vector(x):
    if isinstance(x, (int, float)):
        return Vector([float(x)])
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        if len(x[0]) == 1:
            return Vector(float(row[0]) for row in x)
        elif len(x) == 1:
            return Vector(float(v) for v in x[0])
        else:
            return Vector(v for row in x for v in row)
    return Vector(float(v) for v in x)


def as_matrix(X):
    if not isinstance(X, list):
        X = list(X)
    if len(X) == 0:
        return Matrix([])
    if not isinstance(X[0], list):
        return Matrix([[float(v)] for v in X])
    return Matrix([[float(v) for v in row] for row in X])


def transpose(A):
    return Matrix([list(row) for row in zip(*A)])


def matmul(A, B):
    B_t = transpose(B)
    return Matrix([[sum(a * b for a, b in zip(row, col)) for col in B_t] for row in A])


def matvec(A, x):
    return Vector([sum(a * b for a, b in zip(row, x)) for row in A])


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def identity(n):
    return Matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])


def add_intercept(X):
    return Matrix([[1.0] + list(row) for row in X])


def zeros(rows, cols):
    return Matrix([[0.0 for _ in range(cols)] for _ in range(rows)])


def copy_matrix(A):
    return Matrix([list(row) for row in A])


def residuals(y, y_hat):
    return Vector(yi - yh for yi, yh in zip(y, y_hat))


def vector_norm2(x):
    return dot(x, x)


def mean(x):
    return sum(x) / len(x)


def center(x):
    m = mean(x)
    return Vector(v - m for v in x)


def scalar_multiply(A, c):
    return Matrix([[c * value for value in row] for row in A])


def matrix_add(A, B):
    return Matrix([[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)])


def matrix_sub(A, B):
    return Matrix([[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)])


def diag(A):
    n = min(len(A), len(A[0]))
    return Vector([A[i][i] for i in range(n)])


def add_to_diagonal(A, lam):
    n = len(A)
    return Matrix([[A[i][j] + (lam if i == j else 0.0) for j in range(n)] for i in range(n)])


def sum_squares(x):
    return sum(v * v for v in x)


def all_close(a, b, atol=1e-8):
    a_flat = as_vector(a)
    b_flat = as_vector(b)
    if len(a_flat) != len(b_flat):
        return False
    return all(abs(x - y) <= atol for x, y in zip(a_flat, b_flat))


def column(A, j):
    return Vector([row[j] for row in A])


def column_stack(cols):
    n = len(cols[0])
    return Matrix([[col[i] for col in cols] for i in range(n)])


def frobenius_norm(A):
    return math.sqrt(sum(v * v for row in A for v in row))


def trace(A):
    return sum(A[i][i] for i in range(len(A)))


def without_column(A, j):
    return Matrix([[row[k] for k in range(len(row)) if k != j] for row in A])


def logspace(start, stop, num):
    if num == 1:
        return Vector([10.0 ** start])
    step = (stop - start) / (num - 1)
    return Vector(10.0 ** (start + i * step) for i in range(num))


def take_rows(A, indices):
    return Matrix([A[i] for i in indices])


def take_values(x, indices):
    return Vector([x[i] for i in indices])


def solve(A, b, tol=1e-12):
    # Giải hệ A x = b bằng khử Gauss--Jordan có chọn pivot.
    n = len(A)
    M = [list(row) + [float(rhs)] for row, rhs in zip(A, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < tol:
            raise ValueError("Singular matrix in linear solve")
        M[col], M[pivot] = M[pivot], M[col]

        pivot_value = M[col][col]
        M[col] = [value / pivot_value for value in M[col]]

        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            M[row] = [a - factor * b for a, b in zip(M[row], M[col])]

    return [M[i][-1] for i in range(n)]


def inverse(A):
    n = len(A)
    columns = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        columns.append(solve(A, e))
    return transpose(columns)


def determinant(A):
    M = copy_matrix(A)
    n = len(M)
    det = 1.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            return 0.0
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            det *= -1.0
        det *= M[col][col]
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for k in range(col, n):
                M[row][k] -= factor * M[col][k]
    return det


def rank(A, tol=1e-10):
    M = copy_matrix(A)
    rows, cols = len(M), len(M[0])
    rank_value = 0
    for col in range(cols):
        pivot = None
        for row in range(rank_value, rows):
            if abs(M[row][col]) > tol:
                pivot = row
                break
        if pivot is None:
            continue
        M[rank_value], M[pivot] = M[pivot], M[rank_value]
        pivot_value = M[rank_value][col]
        M[rank_value] = [v / pivot_value for v in M[rank_value]]
        for row in range(rows):
            if row != rank_value:
                factor = M[row][col]
                M[row] = [a - factor * b for a, b in zip(M[row], M[rank_value])]
        rank_value += 1
    return rank_value


def max_abs_diff(a, b):
    a = as_vector(a)
    b = as_vector(b)
    return max(abs(x - y) for x, y in zip(a, b))