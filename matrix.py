import numpy as np
from typing import List


class Matrix:
    """

    """

    def __init__(self, rows, cols, elements):
        """

        """
        if isinstance(elements, (int, float)):
            self.matrix = np.ones((rows, cols)) * elements
        elif isinstance(elements, (list, np.ndarray, Matrix)):
            elements_to_reshape = elements.matrix if isinstance(
                elements, Matrix) else np.array(elements)
            try:
                self.matrix = np.matrix.reshape(
                    elements_to_reshape, (rows, cols))
                if self.matrix.dtype not in ["float64", "int32"]:
                    raise TypeError(
                        "All provided elements in array must be of type int or float.")
            except ValueError:
                raise ValueError(
                    f"Number of elements must be compatible with provided shape.\n{rows} rows, {cols} columns, expected {rows * cols} elements, got {np.size(elements_to_reshape)}.")
        else:
            raise TypeError(
                "Provided elements must be of type int, float, list, or NumPy array.")

    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return f"Matrix({self.matrix.shape[0]}, {self.matrix.shape[1]}, {self.matrix})"

    def __add__(self, other):
        if isinstance(other, Matrix):
            try:
                new_matrix = self.matrix + other.matrix
            except ValueError:
                raise ValueError(
                    f"Matrices must have same shape (trying to add {self.matrix.shape[0]} and {other.matrix.shape[0]} rows, and {self.matrix.shape[1]} and {other.matrix.shape[1]} columns).")
        elif isinstance(other, (int, float)):
            new_matrix = self.matrix + other

        return Matrix(self.matrix.shape[0], self.matrix.shape[1], new_matrix)

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Matrix):
            try:
                new_matrix = self.matrix - other.matrix
            except ValueError:
                raise ValueError(
                    f"Matrices must have same shape (trying to subtract {self.matrix.shape[0]} and {other.matrix.shape[0]} rows, and {self.matrix.shape[1]} and {other.matrix.shape[1]} columns).")
        elif isinstance(other, (int, float)):
            new_matrix = self.matrix - other

        return Matrix(self.matrix.shape[0], self.matrix.shape[1], new_matrix)

    def __isub__(self, other):
        return self - other

    def __rsub__(self, other):
        return self - other

    def __neg__(self):
        new_matrix = self.matrix * (-1)
        return Matrix(self.matrix.shape[0], self.matrix.shape[1], new_matrix)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            try:
                new_matrix = np.matmul(self.matrix, other.matrix)
                new_matrix_object = Matrix(
                    self.matrix.shape[0], other.matrix.shape[1], new_matrix)
            except ValueError:
                raise ValueError(
                    f"Number of columns in first matrix must be equal to number of rows in second matrix (first matrix has {self.matrix.shape[1]} columns and second matrix has {other.matrix.shape[0]} rows.")
        elif isinstance(other, (int, float)):
            new_matrix = self.matrix * other
            new_matrix_object = Matrix(
                self.matrix.shape[0], self.matrix.shape[1], new_matrix)

        return new_matrix_object

    def __imul__(self, other):
        return self * other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            try:
                new_matrix = self.matrix / other
            except ZeroDivisionError:
                raise ZeroDivisionError("Cannot divide by 0!")

            return Matrix(self.matrix.shape[0], self.matrix.shape[1], new_matrix)
        else:
            raise ValueError("Matrix can only be divided by a scalar.")

    def __idiv__(self, other):
        return self / other

    def to_array(self) -> List[float] | List[int]:
        arr = np.reshape(
            self.matrix, self.no_of_elements(self))
        arr = list(arr)
        return arr

    @staticmethod
    def no_of_elements(matrix):
        return matrix.matrix.shape[0] * matrix.matrix.shape[1]

    @classmethod
    def column_matrix(cls, arr: List[float]):
        return cls(len(arr), 1, arr)

    @classmethod
    def random_matrix(cls, rows: int, cols: int, low: float, high: float):
        elements = [np.random.uniform(low, high) for _ in range(rows*cols)]
        return cls(rows, cols, elements)

    @classmethod
    def crossover(cls, matrix, other_matrix, mutation_rate, low, high):
        matrix_elements = Matrix.to_array(matrix)
        other_matrix_elements = Matrix.to_array(other_matrix)

        elements = []

        for i in range(cls.no_of_elements(matrix)):
            rng = np.random.uniform(0, 1)
            if rng < mutation_rate:
                elements.append(np.random.uniform(low, high))
            elif rng < (0.5 + mutation_rate / 2):
                elements.append(matrix_elements[i])
            else:
                elements.append(other_matrix_elements[i])

        return cls(matrix.matrix.shape[0], matrix.matrix.shape[1], elements)

    @classmethod
    def map(cls, matrix, map_function):
        return cls(matrix.matrix.shape[0], matrix.matrix.shape[1], [map_function(x) for x in matrix.matrix])
