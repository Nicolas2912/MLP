// main.cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include "mlp1.h"
#include "lcg.h"

int main(int argc, char* argv[]) {
    // Check for correct number of arguments
    if(argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <epochs> <learning_rate>" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    int epochs = std::atoi(argv[1]);
    double learning_rate = std::atof(argv[2]);

    // Define the XOR dataset
    std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> Y = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Initialize LCG with fixed seed for reproducibility
    const uint32_t FIXED_SEED = 42;
    LCG lcg(FIXED_SEED);

    // Initialize the MLP
    // 2 input neurons, 2 hidden neurons, 1 output neuron
    MLP mlp(2, 2, 1, learning_rate, false); // Using sigmoid activation

    // Initialize weights using LCG
    mlp.initialize_parameters(lcg);

    // Train the MLP
    std::cout << "Training MLP for XOR problem with " << epochs 
              << " epochs and learning rate " << learning_rate << std::endl;

    mlp.train(X, Y, epochs);

    // Make Predictions
    std::cout << "\nPredictions after training:\n";
    for(int i = 0; i < X.size(); ++i) {
        std::vector<double> output = mlp.predict(X[i]);
        // For sigmoid activation, threshold at 0.5
        int predicted = (output[0] > 0.5) ? 1 : 0;
        std::cout << "Input: [" << X[i][0] << " " << X[i][1] 
                  << "], Predicted Output: " << predicted 
                  << ", True Output: " << static_cast<int>(Y[i][0]) << std::endl;
    }

    return 0;
}
