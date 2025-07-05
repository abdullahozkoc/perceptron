-----

# Perceptron Classifier

This repository contains a basic implementation of the Perceptron algorithm, a foundational machine learning model for binary classification. This code was developed as part of my learning journey in machine learning, providing a clear and understandable example of how a Perceptron works from scratch.

-----

## Features

  * **Simple Implementation:** A clear and concise implementation of the Perceptron algorithm.
  * **Configurable Learning Rate (`eta`):** Adjust the step size for weight updates.
  * **Configurable Iterations (`n_iter`):** Control the number of passes over the training dataset.
  * **Bias Unit Included:** The model incorporates a bias term for improved flexibility.
  * **Error Tracking:** Tracks the number of misclassifications in each epoch, useful for monitoring convergence.

-----

## How it Works

The Perceptron is a single-layer neural network that makes predictions based on a linear predictor function combining a set of weights with the input features. It's designed for binary classification tasks.

The `Perceptron` class works as follows:

1.  **Initialization (`__init__`)**: Sets the learning rate (`eta`), number of iterations (`n_iter`), and a random seed for reproducible weight initialization.
2.  **Fitting (`fit`)**:
      * Initializes weights (`w_`) to small random values and the bias (`b_`) to zero.
      * Iterates through the training data for a specified number of epochs (`n_iter`).
      * For each training example, it calculates the `net_input` and makes a `predict`ion.
      * If the prediction is incorrect, it updates the weights and bias based on the learning rate and the error.
      * It records the number of misclassifications in each epoch in `errors_`.
3.  **Net Input Calculation (`net_input`)**: Computes the weighted sum of inputs and adds the bias term. This is the core linear combination.
4.  **Prediction (`predict`)**: Applies a unit step function to the net input. If the net input is greater than or equal to 0, it predicts `1`; otherwise, it predicts `0`.

-----

## Installation

This implementation only requires `NumPy`.

```bash
pip install numpy
```

-----

## Usage

Here's a quick example of how to use the Perceptron classifier:

```python
import numpy as np
from perceptron_classifier import Perceptron # Assuming your class is in perceptron_classifier.py

# Sample data
X = np.array([
    [2, 3],
    [1, 1],
    [3, 2],
    [1, 3],
    [3, 4]
])
y = np.array([0, 0, 1, 0, 1])

# Initialize Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)

# Train the model
ppn.fit(X, y)

# Make predictions
predictions = ppn.predict(X)
print(f"Predictions: {predictions}")

# Check errors per epoch
print(f"Errors per epoch: {ppn.errors_}")

# Predict for a new data point
new_data = np.array([4, 1])
print(f"Prediction for [4, 1]: {ppn.predict(new_data)}")
```

-----

## Parameters

  * `eta` (float, default=0.01): The learning rate. A value between 0.0 and 1.0. Determines the step size at each iteration while moving toward a minimum of the loss function.
  * `n_iter` (int, default=50): The number of training epochs (passes over the training dataset).
  * `random_state` (int, default=1): The seed for the random number generator used for initializing weights, ensuring reproducibility.

-----

## Attributes

  * `w_` (1d-array): Weights after the training process.
  * `b_` (Scalar): Bias unit after the training process.
  * `errors_` (list): A list containing the number of misclassifications (updates to weights) in each epoch during training.

-----
