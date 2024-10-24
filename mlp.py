import numpy as np


class LCG:
    def __init__(self, seed=42):
        self.a = 1664525
        self.c = 1013904223
        self.m = 2 ** 32
        self.state = seed

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def random(self):
        return self.next() / self.m


class MLP:
    def __init__(self, layers, learning_rate=0.01, seed=42):
        self.layers = layers
        self.learning_rate = learning_rate
        self.lcg = LCG(seed)
        self.weights = []
        self.biases = []
        self.initialize_parameters()

    def initialize_parameters(self):
        for i in range(len(self.layers) - 1):
            weight_matrix = np.zeros((self.layers[i], self.layers[i + 1]))
            bias_vector = np.zeros((1, self.layers[i + 1]))
            limit = np.sqrt(1 / self.layers[i])
            for j in range(self.layers[i]):
                for k in range(self.layers[i + 1]):
                    weight_matrix[j][k] = (self.lcg.random() * 2 - 1) * limit
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def binary_cross_entropy_derivative(self, y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def forward(self, X):
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            a = self.sigmoid(z)
            activations.append(a)
        return activations

    def train(self, X, y, epochs):
        log_interval = epochs // 10 if epochs >= 10 else 1
        for epoch in range(1, epochs + 1):
            activations = self.forward(X)
            loss = np.mean(self.binary_cross_entropy(y, activations[-1]))

            # Backpropagation
            error = self.binary_cross_entropy_derivative(y, activations[-1])
            delta = error * self.sigmoid_derivative(activations[-1])

            for l in range(len(self.layers) - 2, -1, -1):
                a = activations[l]
                dw = np.dot(a.T, delta)
                db = np.sum(delta, axis=0, keepdims=True)
                if l != 0:
                    delta = np.dot(delta, self.weights[l].T) * self.sigmoid_derivative(activations[l])
                self.weights[l] -= self.learning_rate * dw
                self.biases[l] -= self.learning_rate * db

            if epoch % log_interval == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]


# Usage Example
if __name__ == "__main__":
    # Define XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Initialize MLP with [2,2,1], learning_rate=0.01, seed=42
    mlp = MLP(layers=[2, 2, 1], learning_rate=0.01, seed=42)

    # Train MLP
    epochs = 50000
    mlp.train(X, y, epochs)

    # Make predictions
    predictions = mlp.predict(X)
    predictions_binary = (predictions > 0.5).astype(int)

    print("\nPredictions after training:")
    for i, (input_sample, prediction) in enumerate(zip(X, predictions_binary)):
        print(f"Input: {input_sample}, Predicted Output: {prediction[0]}, True Output: {y[i][0]}")
