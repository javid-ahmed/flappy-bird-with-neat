import unittest
import numpy as np
from matrix import Matrix


class TestPhaseDif(unittest.TestCase):
    def test_given_dimensions_and_single_value_when_matrix_created_then_check_equal_elements(self):
        expected = np.ones((2, 3))
        actual = Matrix(2, 3, 1).matrix
        np.testing.assert_array_equal(expected, actual)

    def test_given_dimensions_and_array_of_values_when_matrix_created_then_check_matrix_created_successfully(self):
        expected = np.array([[0, 1], [2, 3], [4, 5]])
        actual = Matrix(3, 2, [0, 1, 2, 3, 4, 5]).matrix
        np.testing.assert_array_equal(expected, actual)

    def test_given_dimensions_and_matrix_when_matrix_created_then_check_matrix_created_successfully(self):
        expected = np.array([[0, 1, 2], [3, 4, 5]])
        temp_matrix = Matrix(3, 2, [0, 1, 2, 3, 4, 5])
        actual = Matrix(2, 3, temp_matrix).matrix
        np.testing.assert_array_equal(expected, actual)

    def test_given_invalid_elements_type_when_matrix_created_then_check_TypeError_raised(self):
        with self.assertRaises(TypeError):
            Matrix(2, 2, "0 1 2 3 4 5")

    def test_given_incorrect_number_of_elements_type_when_matrix_created_then_check_ValueError_raised(self):
        with self.assertRaises(ValueError):
            Matrix(2, 2, [1, 2, 3])

    def test_given_invalid_element_within_elements_array_type_when_matrix_created_then_check_TypeError_raised(self):
        with self.assertRaises(TypeError):
            Matrix(2, 2, [1, 2, "3", 4])

    def test_given_two_matrices_when_adding_then_check_resulting_matrix_is_correct(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(2, 2, [5, 6, 7, 8])

        expected = Matrix(2, 2, [6, 8, 10, 12])
        actual = matrix1 + matrix2

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_two_matrices_when_subtracting_then_check_resulting_matrix_is_correct(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(2, 2, [5, 6, 7, 8])

        expected = Matrix(2, 2, [4, 4, 4, 4])
        actual = matrix2 - matrix1

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_two_matrices_with_incompatible_shapes_when_adding_then_check_ValueError_raised(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(2, 3, [5, 6, 7, 8, 9, 10])

        with self.assertRaises(ValueError):
            matrix1 + matrix2

    def test_given_two_matrices_with_incompatible_shapes_when_subtracting_then_check_ValueError_raised(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(2, 3, [5, 6, 7, 8, 9, 10])

        with self.assertRaises(ValueError):
            matrix2 - matrix1

    def test_given_matrix_and_scalar_when_adding_scalar_then_check_resulting_matrix_is_correct(self):
        matrix = Matrix(2, 2, [1, 2, 3, 4])

        expected = Matrix(2, 2, [2, 3, 4, 5])
        actual = matrix + 1

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_matrix_and_scalar_when_subtracting_scalar_then_check_resulting_matrix_is_correct(self):
        matrix = Matrix(2, 2, [1, 2, 3, 4])

        expected = Matrix(2, 2, [0, 1, 2, 3])
        actual = matrix - 1

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_two_matrices_when_using_assignment_operator_to_add_then_check_matrix_modified_correctly(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(2, 2, [5, 6, 7, 8])

        expected = Matrix(2, 2, [6, 8, 10, 12])
        matrix1 += matrix2
        actual = matrix1

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_two_matrices_with_incompatible_shapes_when_using_assignment_operator_to_add_then_check_ValueError_raised(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(2, 3, [5, 6, 7, 8, 9, 10])

        with self.assertRaises(ValueError):
            matrix1 += matrix2

    def test_given_two_matrices_when_using_assignment_operator_to_subtract_then_check_matrix_modified_correctly(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(2, 2, [5, 6, 7, 8])

        expected = Matrix(2, 2, [4, 4, 4, 4])
        matrix2 -= matrix1
        actual = matrix2

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_two_matrices_with_incompatible_shapes_when_using_assignment_operator_to_subtract_then_check_ValueError_raised(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(2, 3, [5, 6, 7, 8, 9, 10])

        with self.assertRaises(ValueError):
            matrix2 -= matrix1

    def test_given_matrix_when_signs_flipped_then_check_resulting_matrix_is_correct(self):
        matrix = Matrix(2, 2, [1, 2, 3, 4])

        expected = Matrix(2, 2, [-1, -2, -3, -4])
        actual = -matrix

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_matrix_and_scalar_when_multiplied_together_then_check_matrix_correctly_calculated(self):
        matrix = Matrix(3, 2, [0, 1, 2, 3, 4, 5])

        expected = Matrix(3, 2, [0, 2, 4, 6, 8, 10])
        actual = matrix * 2

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_two_matrices_when_multiplied_together_then_check_matrix_correctly_calculated(self):
        matrix1 = Matrix(3, 2, [1, 2, 3, 4, 5, 6])
        matrix2 = Matrix(2, 3, [2, 4, 6, 8, 10, 12])

        expected = Matrix(3, 3, [18, 24, 30, 38, 52, 66, 58, 80, 102])
        actual = matrix1 * matrix2

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_matrix_and_scalar_when_using_assignment_operator_to_multiply_then_check_matrix_correctly_calculated(self):
        matrix = Matrix(2, 3, [1, 2, 3, 4, 5, 6])

        expected = Matrix(2, 3, [2, 4, 6, 8, 10, 12])

        matrix *= 2
        actual = matrix

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_two_matrices_with_incompatible_shapes_when_using_assignment_operator_to_multiply_then_check_ValueError_raised(self):
        matrix1 = Matrix(2, 2, [1, 2, 3, 4])
        matrix2 = Matrix(3, 2, [5, 6, 7, 8, 9, 10])

        with self.assertRaises(ValueError):
            matrix1 *= matrix2

    def test_given_matrix_and_scalar_when_dividing_matrix_by_scalar_then_check_matrix_correctly_calculated(self):
        matrix = Matrix(2, 3, [2, 4, 6, 8, 10, 12])

        expected = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        actual = matrix / 2

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_matrix_when_divided_by_incompatible_type_then_check_ValueError_raised(self):
        matrix = Matrix(2, 2, 1)

        with self.assertRaises(ValueError):
            matrix / "a"

    def test_given_matrix_and_scalar_when_using_assignment_operator_to_divide_then_check_matrix_correctly_calculated(self):
        matrix = Matrix(2, 3, [2, 4, 6, 8, 10, 12])

        expected = Matrix(2, 3, [1, 2, 3, 4, 5, 6])

        matrix /= 2
        actual = matrix

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_example_expression_when_calculated_then_check_matrix_calculated_correctly(self):
        a = Matrix(2, 2, [0, 2, 3, 6])
        b = Matrix(2, 2, [1, 2, -3, 5])
        c = Matrix(2, 2, 1)

        actual = 2 * a * b + c
        expected = Matrix(2, 2, [-11, 21, -29, 73])

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_matrix_when_converted_to_array_then_check_array_correct(self):
        matrix = Matrix(2, 3, [0, 2, 3, 6, 9, 10])

        actual = matrix.to_array()
        expected = [0, 2, 3, 6, 9, 10]

        np.testing.assert_array_equal(expected, actual)

    def test_given_array_when_creating_column_matrix_then_check_matrix_correct(self):
        elements = [2, 3, 4, 1]

        actual = Matrix.column_matrix(elements)
        expected = Matrix(4, 1, elements)

        np.testing.assert_array_equal(expected.matrix, actual.matrix)

    def test_given_matrix_when_mapping_elements_then_check_matrix_correct(self):
        matrix = Matrix(2, 3, [0, 1, 2, 3, 4, 5])

        def add_two(x):
            return x + 2

        actual = Matrix.map(matrix, add_two)
        expected = Matrix(2, 3, [2, 3, 4, 5, 6, 7])

        np.testing.assert_array_equal(expected.matrix, actual.matrix)


if __name__ == '__main__':
    unittest.main()
