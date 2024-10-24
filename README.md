# XOR Neural Network Implementation

A simple implementation of a Multi-Layer Perceptron (MLP) neural network solving the XOR problem, with both C++ and Python implementations. The project demonstrates fundamental concepts of neural networks including forward propagation, backpropagation, and weight initialization using a Linear Congruential Generator (LCG) for reproducibility.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
  - [Neural Network Architecture](#neural-network-architecture)
  - [Key Components](#key-components)
- [Usage](#usage)
  - [C++ Implementation](#c-implementation)
  - [Python Implementation](#python-implementation)
  - [Example Output](#example-output)
- [Technical Details](#technical-details)
  - [LCG Implementation](#lcg-implementation)
  - [Neural Network Implementation](#neural-network-implementation)
- [Requirements](#requirements)
  - [C++](#c)
  - [Python](#python)
- [Performance Considerations](#performance-considerations)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Features

- Implementation in both C++ and Python for learning and comparison
- Single hidden layer neural network architecture (2-2-1)
- Sigmoid activation function
- Binary cross-entropy loss function
- Xavier/Glorot weight initialization
- Custom Linear Congruential Generator (LCG) for reproducible random number generation
- Configurable learning rate and number of epochs

## Project Structure

```plaintext
├── C++ Implementation
│   ├── lcg.cpp        # Linear Congruential Generator implementation
│   ├── lcg.h          # LCG header file
│   ├── mlp.cpp        # MLP implementation
│   ├── mlp.h          # MLP header file
│   └── main.cpp       # Main program and XOR example
└── Python Implementation
    └── mlp.py         # Complete Python implementation with LCG and MLP classes
```

## Implementation Details 

### Neural Network Architecture

- **Input Layer**: 2 neurons (for XOR iunputs)
- **Hidden Layer**: 2 neurons with sigmoid activiation.
- **Output Layer**: 1 neuron with sigmoid activiation.

### Key Components

- **Weight Initialization**: Uses Xavier/Glorot initialization with LCG for reproducibility
- **Forward Propagation**: Implements matrix multiplication and sigmoid activation
- **Backward Propagation**: Calculates gradients and updates weights
- **Loss Function**: Binary cross-entropy with numerical stability considerations

## Usage

### C++ Implementation

Compile the project:

```bash
g++ -std=c++11 main.cpp mlp.cpp lcg.cpp -o xor_mlp
```

Run the program with epochs and learning rate parameters:

```bash
./xor_mlp 50000 0.01
```

### Python Implementation

Run the Python script directly:

```bash
python mlp.py
```

### Example Output

The program will train on the XOR dataset and output something like:

```plaintext
Training MLP for XOR problem with 50000 epochs and learning rate 0.01
Epoch 1/50000, Loss: 0.693147
Epoch 5000/50000, Loss: 0.082394
...
Epoch 50000/50000, Loss: 0.000127

Predictions after training:
Input: [0 0], Predicted Output: 0, True Output: 0
Input: [0 1], Predicted Output: 1, True Output: 1
Input: [1 0], Predicted Output: 1, True Output: 1
Input: [1 1], Predicted Output: 0, True Output: 0
```

## Technical Details

### LCG Implementation

- Uses the parameters: `a = 1664525`, `c = 1013904223`
- 32-bit implementation for fast random number generation
- Provides reproducible results with same seed
- Both implementations deliver nearly the same results because of the LCG implementation

### Neuronal Network Implementation

- Learning rate and epochs are configurable
- Uses vectorized operations in Python for efficiency
- Implements numerical stability measures in loss calculations
- Provides consistent results across runs with same seed

## Requierements

### C++

- C++11 or higher
- Standard Template Library (STL)

### Python

- NumPy
- Python 3.x

## Performance Considerations

- The C++ implementation is generally faster due to compiled execution
- The Python implementation uses NumPy for vectorized operations
- Both implementations use the same algorithmic approach for comparison purposes

## Acknowledgments

This implementation serves as an educational resource for understanding:

- Neural network fundamentals
- Backpropagation algorithm
- Weight initialization techniques
- Random number generation in scientific computing





