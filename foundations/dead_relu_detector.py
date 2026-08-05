import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        stats = []
        for module in model.children():

            x = module(x)
            if isinstance(module, nn.ReLU):
                if x.dim() >= 2:
                    dead_fraction = round((x <= 0).all(dim=0).float().mean().item(), 4)
                else:
                    dead_fracion = round((x <= 0).float().mean().item(), 4)
                stats.append(dead_fraction)
        
        return stats

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if max(dead_fractions) < 0.1:
            return 'healthy'
        elif any(x > 0.5 for x in dead_fractions):
            return 'use_leaky_relu'
        elif dead_fractions[0] > 0.3:
            return 'reinitialize'
        elif dead_fractions[-1] > 0.1:
            status_reduce_lr = True
            for i in range(1, len(dead_fractions)):
                if not (dead_fractions[i] > dead_fractions[i-1]):
                    status_reduce_lr = False
            if status_reduce_lr:
                return 'reduce_learning_rate'
            else:
                return 'healthy'
        else:
            return 'healthy'