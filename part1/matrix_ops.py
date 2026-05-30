import math
import operator


class Vector(list):
    @property
    def shape(self):
        return (len(self),)

    def flatten(self):
        return Vector(self)

    def tolist(self):
        return list(self)

    def _binary(self, other, op):
        if _is_scalar(other):
            return Vector(op(float(x), float(other)) for x in self)
        other = as_vector(other)
        return Vector(op(float(a), float(b)) for a, b in zip(self, other))

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        if _is_scalar(other):
            return Vector(float(other) - float(x) for x in self)
        other = as_vector(other)
        return Vector(float(a) - float(b) for a, b in zip(other, self))

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __pow__(self, power):
        return Vector(float(x) ** power for x in self)

    def __lt__(self, other):
        return self._binary(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._binary(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._binary(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._binary(other, lambda a, b: a >= b)


class Matrix(list):
    @property
    def shape(self):
        if not self:
            return (0, 0)
        return (len(self), len(self[0]))

    @property
    def T(self):
        return transpose(self)

    def tolist(self):
        return [list(row) for row in self]

    def __matmul__(self, other):
        if is_matrix_like(other):
            return matmul(self, other)
        return matvec(self, other)


def _is_scalar(value):
    return isinstance(value, (int, float))


def _raw_list(data):
    if isinstance(data, (Vector, Matrix)):
        return data.tolist()
    if hasattr(data, "tolist"):
        data = data.tolist()
    return data


def is_matrix_like(data):
    data = _raw_list(data)
    return bool(data) and isinstance(data[0], (list, tuple, Vector))


def as_vector(data):
    if isinstance(data, Vector):
        return data
    data = _raw_list(data)
    if isinstance(data, (int, float)):
        return Vector([float(data)])
    if not data:
        return Vector()
    if isinstance(data[0], (list, tuple, Vector)):
        return Vector(float(row[0]) for row in data)
    return Vector(float(x) for x in data)


def as_matrix(data):
    if isinstance(data, Matrix):
        return data
    data = _raw_list(data)
    if not data:
        return Matrix()
    if not isinstance(data[0], (list, tuple, Vector)):
        return Matrix([[float(x)] for x in data])
    return Matrix(Vector(float(x) for x in row) for row in data)


def zeros(rows, cols):
    return Matrix([[0.0 for _ in range(cols)] for _ in range(rows)])


def identity(n):
    return Matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])


def transpose(A):
    A = as_matrix(A)
    if not A:
        return Matrix()
    return Matrix([[A[i][j] for i in range(len(A))] for j in range(len(A[0]))])


def dot(u, v):
    u = as_vector(u)
    v = as_vector(v)
    return sum(map(operator.mul, u, v))


def matvec(A, v):
    A = as_matrix(A)
    v = as_vector(v)
    return Vector(dot(row, v) for row in A)


def matmul(A, B):
    A = as_matrix(A)
    B = as_matrix(B)
    Bt = transpose(B)
    return Matrix([[dot(row, col) for col in Bt] for row in A])


def matrix_add(A, B):
    A = as_matrix(A)
    B = as_matrix(B)
    return Matrix([[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))])


def matrix_sub(A, B):
    A = as_matrix(A)
    B = as_matrix(B)
    return Matrix([[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))])


def add_to_diagonal(A, value, skip_first=False):
    A = as_matrix(A)
    out = Matrix([Vector(row) for row in A])
    start = 1 if skip_first else 0
    for i in range(start, min(len(out), len(out[0]))):
        out[i][i] += value
    return out


def solve(A, b, tol=1e-12):
    A = as_matrix(A)
    b = as_vector(b)
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be a non-empty square matrix")
    if len(b) != n:
        raise ValueError("b length must match A")

    aug = [list(A[i]) + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < tol:
            raise ValueError("Matrix is singular or nearly singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        aug[col][col:] = [val / pivot_value for val in aug[col][col:]]

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < tol:
                continue
            aug[r][col:] = [ar - factor * ac for ar, ac in zip(aug[r][col:], aug[col][col:])]

    return Vector(aug[i][n] for i in range(n))


def cg_solve(A, b, tol=1e-8, max_iter=2000):
    A = as_matrix(A)
    b = as_vector(b)
    n = len(b)
    if not n:
        return Vector()
    x = Vector([0.0] * n)
    r = Vector(list(b))
    p = Vector(list(r))
    rsold = sum(map(operator.mul, r, r))
    for i in range(max_iter):
        Ap = [sum(map(operator.mul, row, p)) for row in A]
        p_Ap = sum(map(operator.mul, p, Ap))
        if p_Ap == 0:
            break
        alpha = rsold / p_Ap
        x = [xj + alpha * pj for xj, pj in zip(x, p)]
        r = [rj - alpha * apj for rj, apj in zip(r, Ap)]
        rsnew = sum(map(operator.mul, r, r))
        if math.sqrt(rsnew) < tol:
            break
        ratio = rsnew / rsold
        p = [rj + ratio * pj for rj, pj in zip(r, p)]
        rsold = rsnew
    return Vector(x)

def inverse(A):
    A = as_matrix(A)
    n = len(A)
    cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(solve(A, e))
    return transpose(cols)


def mean(values):
    values = as_vector(values)
    return sum(values) / len(values)


def variance(values):
    values = as_vector(values)
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def sum_squares(values):
    values = as_vector(values)
    return sum(map(operator.mul, values, values))


def diag(A):
    A = as_matrix(A)
    return Vector(A[i][i] for i in range(min(len(A), len(A[0]))))


def trace(A):
    return sum(diag(A))


def frobenius_norm(A):
    A = as_matrix(A)
    return math.sqrt(sum(map(operator.mul, (x for row in A for x in row), (x for row in A for x in row))))


def all_close(a, b, atol=1e-8):
    a = _raw_list(a)
    b = _raw_list(b)
    if is_matrix_like(a) or is_matrix_like(b):
        A = as_matrix(a)
        B = as_matrix(b)
        return len(A) == len(B) and all(
            len(A[i]) == len(B[i]) and all(abs(A[i][j] - B[i][j]) <= atol for j in range(len(A[i])))
            for i in range(len(A))
        )
    u = as_vector(a)
    v = as_vector(b)
    return len(u) == len(v) and all(abs(x - y) <= atol for x, y in zip(u, v))


def column(A, j):
    A = as_matrix(A)
    return Vector(row[j] for row in A)


def without_column(A, j):
    A = as_matrix(A)
    return Matrix([[row[k] for k in range(len(row)) if k != j] for row in A])


def column_stack(columns):
    cols = [as_vector(col) for col in columns]
    if not cols:
        return Matrix()
    return Matrix([[col[i] for col in cols] for i in range(len(cols[0]))])


def take_rows(A, indices):
    A = as_matrix(A)
    return Matrix([A[i] for i in indices])


def take_values(v, indices):
    v = as_vector(v)
    return Vector(v[i] for i in indices)


def logspace(start, stop, num):
    if num == 1:
        return Vector([10.0 ** start])
    step = (stop - start) / (num - 1)
    return Vector(10.0 ** (start + i * step) for i in range(num))


def max_abs_diff(a, b):
    a = as_vector(a)
    b = as_vector(b)
    return max(abs(x - y) for x, y in zip(a, b))
