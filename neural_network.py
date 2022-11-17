from matrix import Matrix
from typing import List
import math


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, mutation_rate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.mutation_rate = mutation_rate

        self.weights_range = [-1, 1]
        self.bias_range = [-0.5, 0.5]

        self.weights_IH = Matrix.random_matrix(
            self.hiddenNodes, self.inputNodes, self.weights_range[0], self.weights_range[1])
        self.weights_OH = Matrix.random_matrix(
            self.outputNodes, self.hiddenNodes, self.weights_range[0], self.weights_range[1])
        self.bias_IH = Matrix.random_matrix(
            self.hiddenNodes, 1, self.bias_range[0], self.bias_range[1])
        self.bias_HO = Matrix.random_matrix(
            self.outputNodes, 1, self.bias_range[0], self.bias_range[1])

        self.new_weights_IH = Matrix(self.hiddenNodes, self.inputNodes, 0)
        self.new_weights_OH = Matrix(self.outputNodes, self.hiddenNodes, 0)
        self.new_bias_IH = Matrix(self.hiddenNodes, 1, 0)
        self.new_bias_HO = Matrix(self.outputNodes, 1, 0)

    def feedforward(self, input_array: List[float]) -> List[float]:
        """
        M_H = W_IH x M_I + B_IH -> (hiddenNodes, inputNodes) x (inputNodes, 1) + (hiddenNodes, 1) = (hiddenNodes, 1)
        M_O = W_HO x M_H + B_OH -> (outputNodes, hiddenNodes) x (hiddenNodes, 1) + (outputNodes, 1) = (outputNodes, 1) 
        """
        hidden = self.weights_IH * \
            Matrix.column_matrix(input_array) + self.bias_IH
        hidden = Matrix.map(hidden, self.sigmoid)

        output = self.weights_HO * hidden + self.bias_HO
        output = Matrix.map(output, self.sigmoid)

        return Matrix.to_array(output)

    def crossover(self, nn, other_nn):
        self.new_weights_IH = Matrix.crossover(
            nn.weights_IH, other_nn.weights_IH, self.mutation_rate, self.weights_range[0], self.weights_range[1])

        self.new_weights_HO = Matrix.crossover(
            nn.weights_HO, other_nn.weights_HO, self.mutation_rate, self.weights_range[0], self.weights_range[1])

        self.new_bias_IH = Matrix.crossover(
            nn.bias_IH, other_nn.bias_IH, self.mutation_rate, self.bias_range[0], self.bias_range[1])

        self.new_bias_HO = Matrix.crossover(
            nn.bias_HO, other_nn.bias_HO, self.mutation_rate, self.bias_range[0], self.bias_range[1])

    def apply(self):
        self.weights_IH = self.new_weights_IH
        self.weights_HO = self.new_weights_HO
        self.bias_IH = self.new_bias_IH
        self.bias_HO = self.new_bias_HO

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
