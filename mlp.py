import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# set numpy random seed
np.random.seed(42)


def f(x):
    return np.sin(2 * np.pi * x[0] + 0.5 * x[1]) + 0.5 * x[1]


def f1(x):
    function1 = (x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2
    function2 = (x[0] - 0.75) ** 2 + (x[1] - 0.75) ** 2
    return function1, function2


def plot(function=f):
    # add 5% noise to the function
    samples = 1000
    X = np.random.rand(samples, 2)
    Y = function(X.T) + 0.05 * np.random.randn(samples)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y, c=Y, cmap='viridis')
    plt.show()


def get_data(function: str = f):
    if function == "f1":
        samples = 1250
        X = np.random.rand(samples, 2)
        Y1, Y2 = f1(X.T)

        func1, func2 = f1(X)
        # get all the values of the function where value of function is < 0.2**2
        indices = np.where(Y1 < 0.2 ** 2)
        indices2 = np.where(Y2 < 0.2 ** 2)
        Y1[indices] = 1
        Y2[indices2] = 2
        Y = np.where(Y1 == 1, 1, np.where(Y2 == 2, 2, 0))
        return train_test_split(X, Y, test_size=0.2)

    elif function == "csv":
        import pandas as pd
        df = pd.read_csv("cens")


    samples = 1000
    X = np.random.rand(samples, 2)
    Y = f1(X.T) + 0.05 * np.random.randn(samples)
    return train_test_split(X, Y, test_size=0.2)


class MultiLayerPerceptron:
    def __init__(self, layers: list):
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.biases = [np.random.randn(layers[i]) for i in range(1, len(layers))]
        self.activations = [np.zeros(layer) for layer in layers]

    def _sig(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.activations[0] = X
        for i in range(len(self.weights) - 1):
            self.activations[i + 1] = self._sig(np.dot(self.activations[i], self.weights[i]) + self.biases[i])
        self.activations[-1] = np.dot(self.activations[-2], self.weights[-1]) + self.biases[-1]
        return self.activations[-1]

    def error(self, X, Y):
        return np.mean((self.forward(X) - Y) ** 2)

    def backprop(self, X, Y):
        """
        Backpropagation algorithm with stochastic gradient descent (incremental learning).
        :param X: Inputs
        :param Y: Outputs
        :return:
        """
        lr = 0.01
        max_iter = 200
        vareps = 1e-6
        error_hist = []
        for _ in (range(max_iter)):
            for i in range(len(X)):
                # draw random sample
                x = X[i]
                y = Y[i]
                self.forward(x)
                # Compute the error
                error = y - self.activations[-1]
                # Compute the gradient
                gradient = [error * self.activations[-1] * (1 - self.activations[-1])]
                # gradient for the last layer with linear activation function
                gradient.append(error)
                for j in range(len(self.weights) - 2, -1, -1):
                    gradient.append(np.dot(self.weights[j + 1], gradient[-1]) * self.activations[j + 1] * (
                            1 - self.activations[j + 1]))
                gradient = gradient[::-1]  # reverse the list
                # Update the weights
                weights_updates = []
                biases_updates = []
                for j in range(len(self.weights)):
                    weights_updates.append(lr * self.activations[j].reshape(-1, 1) * gradient[j])
                    biases_updates.append(lr * gradient[j])
                for j in range(len(self.weights)):
                    self.weights[j] += weights_updates[j]
                    self.biases[j] += biases_updates[j]

            error_hist.append(self.error(x, y))
            if _ % 100 == 0:
                print(f"Iteration: {_}, Error: {self.error(x, y)}")
            if self.error(x, y) < vareps:
                print(f"Converged after {_} iterations with error {self.error(x, y)}")
                break

        # plt.plot(error_hist, label="Error", marker="o", markersize=1)
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title("Iteration vs Error")
        plt.legend()
        plt.grid()
        # plt.show()

    def predict(self, X):
        predictions = dict()
        for x in X:
            prediction = self.forward(x)
            predictions[tuple(x)] = prediction.item()

        return predictions

    def accuracy(self, X, Y):
        return np.mean((self.predict(X) - Y) ** 2)


if __name__ == "__main__":
    # plot()
    data = get_data("f1")
    X_train = data[0]
    X_test = data[1]
    Y_train = data[2]
    Y_test = data[3]

    # layers = [[2, 2, 1], [2, 4, 1], [2, 8, 1], [2, 8, 8, 1], [2, 16, 16, 1], [2, 32, 32, 1], [2, 64, 64, 1], [2, 120, 120, 1]]
    #
    # layer_error_dict = dict()
    # for layer in tqdm(layers):
    #     mlp = MultiLayerPerceptron(layer)
    #     mlp.backprop(X=X_train, Y=Y_train)
    #     error_test = mlp.error(X_test, Y_test)
    #     layer_error_dict[str(layer)] = error_test
    #
    # layer_error_dict_sorted = dict(sorted(layer_error_dict.items(), key=lambda item: item[1]))
    # print(f"Layer Error: {layer_error_dict_sorted}")
    # # plot layer error dict
    # plt.bar(layer_error_dict_sorted.keys(), layer_error_dict_sorted.values())


    # mlp = MultiLayerPerceptron([2, 8, 8, 8, 1])
    # mlp.backprop(X=X_train, Y=Y_train)
    # error_test = mlp.error(X_test, Y_test)
    # prediction = mlp.predict(X_test)
    # print(f"Predictions: {prediction}")
    #
    # # plot predictions vs actual
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_test[:, 0], X_test[:, 1], Y_test, c='r', label='Actual')
    # ax.scatter(X_test[:, 0], X_test[:, 1], list(prediction.values()), c='b', label='Predicted')
    # plt.legend()
    #plt.show()

    # accuracy = mlp.accuracy(X_test, Y_test)
    # print(f"Test Error: {error_test}")
    # print(f"Test Accuracy: {accuracy}")
