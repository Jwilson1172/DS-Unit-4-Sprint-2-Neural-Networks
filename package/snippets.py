# Update our weights 10,000 times - (fingers crossed that this process reduces error)
import pandas as pd
import numpy as np


class Perceptron(object):
    def __init__(self, weights: np.array, epoch: int):

        # set the object's weights/inputs/and epochs to train for
        self.weights = weights
        self.epoch = epoch

    def fit(self, inputs: np.array, y_true: np.array):

        self.inputs = inputs
        self.train(y_true)

        return

    def predict(self, inputs: np.array) -> np.array:
        ws = np.dot(inputs, self.weights)
        activated_output = self.sigmoid(ws)
        return activated_output

    def train(self, y_true):
        for epoch_cur in range(self.epoch):

            # Weighted sum of inputs / weights
            weighted_sum = np.dot(self.inputs, self.weights)
            print("weighted_sumn:", weighted_sum)
            # Activate!
            activated_output = self.sigmoid(weighted_sum)
            print("activated_output", activated_output)
            # Cac error
            error = np.subtract(y_true, activated_output)
            print("error",error)
            adjustments = error * self.sigmoid_derivative(weighted_sum)
            print("adjustments",adjustments)
            # Update the Weights
            self.weights += np.dot(self.inputs.T, adjustments)

            print(f"epoch[{epoch_cur}]\n\ttraining error:\t\n", error)
        return

    # helper functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        dx = self.sigmoid(x)
        return dx * (1 - dx)

    # Getters
    def getWeights(self) -> np.array:
        return self.weights

    # Setters
    def setEpochs(self, epochs: int):
        self.epoch = epochs


if __name__ == "__main__":
    import numpy as np
    from numpy.random import random

    inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    labels = [[0], [1], [1], [0]]
    np.random.seed(42)
    weights = 2 * random((3, 1)) - 1

    p = Perceptron(weights=weights, epoch=10000)
    p.fit(inputs=inputs, y_true=labels)
    y_pred = p.predict(inputs)

    acc = y_pred - labels
    print("training accuracy: ", acc)
