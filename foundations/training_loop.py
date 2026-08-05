import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features) (batch_size, input_dim)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        y = np.atleast_2d(y).T #(batch_size, output_dim)
        
        size = (y.shape[1], X.shape[1]) #(output_dim, input_dim)
        N = y.shape[0]

        w = np.zeros(size)
        b = np.zeros(y.shape[1])
        for i in range(epochs):
            y_hat = X @ w.T + b #(batch_size, output_dim)

            dl_dy = 2 * (y_hat - y) / N #(batch_size, output_dim)
            dl_dw = dl_dy.T @ X

            dl_db = np.sum(dl_dy, axis=0)

            w -= lr * (dl_dw)
            b -= lr *(dl_db)

        return (np.round(w.reshape(-1,), decimals=5), round(b.item(), 5))