import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        m = momentum

        if training:
            x_hat = (x - x_mean) / np.sqrt(x_var + eps)

            running_mean = [(1 - m) * z + m * y for z, y in zip(running_mean, x_mean)]
            running_var = [(1 - m) * z + m * y for z, y in zip(running_var, x_var)]
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        
        y = gamma * x_hat + beta
        running_mean = [round(z, 4) for z in running_mean]
        running_var = [round(z, 4) for z in running_var]

        return (np.round(y, decimals=4), running_mean, running_var)