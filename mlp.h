// mlp.h
#ifndef MLP_H
#define MLP_H

#include <vector>
#include "lcg.h"

class MLP {
public:
    // Constructor
    MLP(int input_size, int hidden_size, int output_size, double learning_rate, bool use_tanh = true);

    // Initialize parameters with LCG
    void initialize_parameters(LCG &lcg);

    // Train the MLP
    void train(const std::vector<std::vector<double>> &X,
               const std::vector<std::vector<double>> &Y,
               int epochs);

    // Predict output for a given input
    std::vector<double> predict(const std::vector<double> &input);

    // Accessors for weights (for debugging)
    const std::vector<std::vector<double>>& get_weights_input_hidden() const { return weights_input_hidden; }
    const std::vector<std::vector<double>>& get_weights_hidden_output() const { return weights_hidden_output; }

private:
    // Network architecture parameters
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;
    bool use_tanh;

    // Weights and Biases
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<double> bias_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> bias_output;

    // Activation functions and their derivatives
    double activate(double x);
    double activate_derivative(double activated_x);

    // Forward pass
    void forward(const std::vector<double> &input,
                std::vector<double> &hidden_activations,
                std::vector<double> &output_activations);

    // Backward pass and weights update
    void backward(const std::vector<double> &input,
                const std::vector<double> &hidden_activations,
                const std::vector<double> &output_activations,
                const std::vector<double> &y_true);

    // Utility functions
    double binary_cross_entropy(double y_true, double y_pred);
    double binary_cross_entropy_derivative(double y_true, double y_pred);
};

#endif // MLP_H
