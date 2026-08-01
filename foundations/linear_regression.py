import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        forward_pass = np.dot(X, weights)
        return np.round(forward_pass, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        error = model_prediction - ground_truth
        squared_error = np.square(error)
        mean_squared_error = np.sum(squared_error) / ground_truth.shape[0]
        return np.round(mean_squared_error, 5)