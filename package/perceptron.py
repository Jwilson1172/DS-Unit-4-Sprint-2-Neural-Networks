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
            adjustments = error.T * self.__sigmoid_derivative(ws)
            weights += np.dot(X.T, adjustments)

            print(f"Epoch - {i} ******* error - {error}%\n")
        self.weights = weights
        print(f" ***** ***** *****\nFinal error score of {error}")
        return

    def setWeights(self, weights):
        self.weights = weights

    def setBias(self, x):
        """Sets the learning rate for the model to use while training
        Arguments:
        ------------
        x { } : Array like object or float
        """
        print("setting bias:")
        print("\told bias", self.bias)
        self.bias = x
        print("new bias", self.bias)
        return

    def predict(self, X):
        """Return class label after unit step"""
        return self.__sigmoid(np.dot(X, self.weights))


if __name__ == '__main__':
    import pandas as pd
    data = {'x1': [0, 1, 0, 1],
            'x2': [0, 0, 1, 1],
            'y':  [1, 1, 1, 0]}

    df = pd.DataFrame.from_dict(data).astype('int')
    import numpy as np
    p = Perceptron(100)
    inputs = df.drop('y', axis=1)
    weights = np.random.random((2,))
    bias = np.ones((2,))
    print(bias, weights)
