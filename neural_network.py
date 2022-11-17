from matrix import Matrix
from typing import List


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        """
        M_H = W_IH x M_I + B_IH -> (hiddenNodes, inputNodes) x (inputNodes, 1) + (hiddenNodes, 1) = (hiddenNodes, 1)
        M_O = W_HO x M_H + B_OH -> (outputNodes, hiddenNodes) x (hiddenNodes, 1) + (outputNodes, 1) = (outputNodes, 1) 
        """
        self.weights_IH = Matrix.random_matrix(hiddenNodes, inputNodes, -1, 1)
        self.weights_OH = Matrix.random_matrix(outputNodes, hiddenNodes, -1, 1)
        self.bias_IH = Matrix.random_matrix(hiddenNodes, 1, -0.5, 0.5)
        self.bias_HO = Matrix.random_matrix(outputNodes, 1, -0.5, 0.5)

        self.new_weights_IH = Matrix(hiddenNodes, inputNodes, 0)
        self.new_weights_OH = Matrix(outputNodes, hiddenNodes, 0)
        self.new_bias_IH = Matrix(hiddenNodes, 1, 0)
        self.new_bias_HO = Matrix(outputNodes, 1, 0)

    def feedforward(self, input_array: List[float]) -> List[float]:
        hidden = self.weights_IH * \
            Matrix.column_matrix(input_array) + self.bias_IH
        output = self.weights_HO * hidden + self.bias_HO
        return Matrix.to_array(output)

    # TODO: Write crossover
    # TODO: Write apply
    # TODO: Write activation functions
