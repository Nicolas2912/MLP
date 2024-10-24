// mlp.cpp
#include "mlp1.h"
#include <cmath>
#include <iostream>
#include <iomanip>

// Constructor
MLP::MLP(int input_size, int hidden_size, int output_size, double learning_rate, bool use_tanh_activation)
    : input_size(input_size), hidden_size(hidden_size),
      output_size(output_size), learning_rate(learning_rate),
      use_tanh(use_tanh_activation) {}

// Initialize parameters using LCG
void MLP::initialize_parameters(LCG &lcg) {
    // Initialize weights_input_hidden with Xavier initialization
    double limit_input_hidden = std::sqrt(1.0 / input_size);
    weights_input_hidden.resize(input_size, std::vector<double>(hidden_size, 0.0));
    for(int i = 0; i < input_size; ++i) {
        for(int j = 0; j < hidden_size; ++j) {
            double rand_uniform = lcg.random(); // [0,1)
            weights_input_hidden[i][j] = (rand_uniform * 2.0 - 1.0) * limit_input_hidden; // [-limit, limit)
        }
    }

    // Initialize bias_hidden to zeros
    bias_hidden.assign(hidden_size, 0.0);

    // Initialize weights_hidden_output with Xavier initialization
    double limit_hidden_output = std::sqrt(1.0 / hidden_size);
    weights_hidden_output.resize(hidden_size, std::vector<double>(output_size, 0.0));
    for(int i = 0; i < hidden_size; ++i) {
        for(int j = 0; j < output_size; ++j) {
            double rand_uniform = lcg.random(); // [0,1)
            weights_hidden_output[i][j] = (rand_uniform * 2.0 - 1.0) * limit_hidden_output; // [-limit, limit)
        }
    }

    // Initialize bias_output to zeros
    bias_output.assign(output_size, 0.0);
}

// Activation function: sigmoid
double MLP::activate(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of sigmoid
double MLP::activate_derivative(double activated_x) {
    return activated_x * (1.0 - activated_x);
}

// Binary Cross-Entropy Loss
double MLP::binary_cross_entropy(double y_true, double y_pred) {
    // Adding epsilon to prevent log(0)
    const double epsilon = 1e-8;
    y_pred = std::max(std::min(y_pred, 1.0 - epsilon), epsilon);
    return - (y_true * std::log(y_pred) + (1.0 - y_true) * std::log(1.0 - y_pred));
}

// Derivative of Binary Cross-Entropy Loss
double MLP::binary_cross_entropy_derivative(double y_true, double y_pred) {
    // Adding epsilon to prevent division by zero
    const double epsilon = 1e-8;
    y_pred = std::max(std::min(y_pred, 1.0 - epsilon), epsilon);
    return (y_pred - y_true) / (y_pred * (1.0 - y_pred));
}

// Forward Pass
void MLP::forward(const std::vector<double> &input,
                std::vector<double> &hidden_activations,
                std::vector<double> &output_activations) {

    // Calculate hidden layer activations
    hidden_activations.resize(hidden_size);
    for(int i = 0; i < hidden_size; ++i) {
        double activation = bias_hidden[i];
        for(int j = 0; j < input_size; ++j) {
            activation += input[j] * weights_input_hidden[j][i];
        }
        hidden_activations[i] = activate(activation);
    }

    // Calculate output layer activations
    output_activations.resize(output_size);
    for(int i = 0; i < output_size; ++i) {
        double activation = bias_output[i];
        for(int j = 0; j < hidden_size; ++j) {
            activation += hidden_activations[j] * weights_hidden_output[j][i];
        }
        output_activations[i] = activate(activation);
    }
}

// Backward Pass and Weights Update
void MLP::backward(const std::vector<double> &input,
                const std::vector<double> &hidden_activations,
                const std::vector<double> &output_activations,
                const std::vector<double> &y_true) {

    // Calculate output errors
    std::vector<double> output_errors(output_size, 0.0);
    for(int i = 0; i < output_size; ++i) {
        double derivative = activate_derivative(output_activations[i]);
        output_errors[i] = binary_cross_entropy_derivative(y_true[i], output_activations[i]) * derivative;
    }

    // Calculate hidden layer errors
    std::vector<double> hidden_errors(hidden_size, 0.0);
    for(int i = 0; i < hidden_size; ++i) {
        double derivative = activate_derivative(hidden_activations[i]);
        double error = 0.0;
        for(int j = 0; j < output_size; ++j) {
            error += output_errors[j] * weights_hidden_output[i][j];
        }
        hidden_errors[i] = error * derivative;
    }

    // Update weights_hidden_output and bias_output
    for(int i = 0; i < hidden_size; ++i) {
        for(int j = 0; j < output_size; ++j) {
            weights_hidden_output[i][j] -= learning_rate * output_errors[j] * hidden_activations[i];
        }
    }
    for(int i = 0; i < output_size; ++i) {
        bias_output[i] -= learning_rate * output_errors[i];
    }

    // Update weights_input_hidden and bias_hidden
    for(int i = 0; i < input_size; ++i) {
        for(int j = 0; j < hidden_size; ++j) {
            weights_input_hidden[i][j] -= learning_rate * hidden_errors[j] * input[i];
        }
    }
    for(int i = 0; i < hidden_size; ++i) {
        bias_hidden[i] -= learning_rate * hidden_errors[i];
    }
}

// Train the MLP
void MLP::train(const std::vector<std::vector<double>> &X,
               const std::vector<std::vector<double>> &Y,
               int epochs) {
    int log_interval = epochs / 10;
    if (log_interval == 0) log_interval = 1;

    for(int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_loss = 0.0;
        for(int i = 0; i < X.size(); ++i) {
            // Forward pass
            std::vector<double> hidden_activations;
            std::vector<double> output_activations;
            forward(X[i], hidden_activations, output_activations);

            // Calculate loss
            for(int j = 0; j < output_size; ++j) {
                epoch_loss += binary_cross_entropy(Y[i][j], output_activations[j]);
            }

            // Backward pass and update
            backward(X[i], hidden_activations, output_activations, Y[i]);
        }
        epoch_loss /= X.size();

        // Logging
        if(epoch % log_interval == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << "/" << epochs << ", Loss: " 
                      << std::fixed << std::setprecision(6) << epoch_loss << std::endl;
        }
    }
}

// Predict Output
std::vector<double> MLP::predict(const std::vector<double> &input) {
    std::vector<double> hidden_activations;
    std::vector<double> output_activations;
    forward(input, hidden_activations, output_activations);
    return output_activations;
}
