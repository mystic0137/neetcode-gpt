import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        
        activation = x
        for layer, (W, b) in enumerate(
            zip(weights, biases)
        ):
            activation = W.T @ activation + b
            if layer < len(weights) - 1:
                activation = np.maximum(0, activation)
            
        return activation