import numpy as np


class Perceptron:
    def __init__(self, epochs=10):
        self.epochs = epochs

    def __sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def __sigmoid_derivative(self, x):
        dx = self.__sigmoid(x)
        return dx * (1 - dx)

    def fit(self, weights, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """

        for i in range(self.epochs):
            # Weighted sum of inputs / weights
            ws = np.dot(X, weights)
            # Activate!
            activated = self.__sigmoid(ws)
            # Cac error
            error = y - activated
            # Update the Weights
            adjustments = error * self.__sigmoid_derivative(ws)
            weights += np.dot(X.T, adjustments)

            print(f"Epoch - {i} ******* error - {error}%\n")
        self.weights = weights
        print(f" ***** ***** *****\nFinal error score of {error}")
        return

    def setWeights(self, weights):
        self.weights = weights

    def predict(self, X):
        """Return class label after unit step"""
        return self.__sigmoid(np.dot(X, self.weights))
