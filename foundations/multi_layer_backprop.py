import numpy as np
from typing import List


class Solution:
    def _relu_backward(self, x):
        return np.where(x > 0, 1, 0)

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.atleast_2d(np.array(x))
        y_true = np.atleast_2d(np.array(y_true))
        
        W1, b1 = np.atleast_2d(W1), np.atleast_2d(np.array(b1))
        W2, b2 = np.atleast_2d(W2), np.atleast_2d(np.array(b2))

        z = x @ W1.T + b1
        h = np.maximum(0, z)
        y_pred = h @ W2.T + b2
        squared_error = np.square(y_pred - y_true)
        mse = np.mean(squared_error)

        N = x.shape[0]
        dl_dy = 2 * (y_pred - y_true) / N
        dl_dh = dl_dy @ W2
        relu_mask = (z > 0).astype(float)
        dl_dz = dl_dh * relu_mask
        
        dW1 = dl_dz.T @ x
        db1 = np.sum(dl_dz, axis=0)

        dW2 = dl_dy.T @ h
        db2 = np.sum(dl_dy, axis=0)

        return {
            'loss': round(mse, 4),
            'dW1': np.round(dW1, 4),
            'db1': np.round(db1, 4),
            'dW2': np.round(dW2, 4),
            'db2': np.round(db2, 4)
        }