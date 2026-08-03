import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        mean_squared = np.mean(np.square(x))
        rms_norm = np.sqrt(mean_squared + eps)

        scaled_x = x / rms_norm

        return np.round(gamma * scaled_x, 4)