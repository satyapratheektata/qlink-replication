"""Trajectory variance computation."""

import numpy as np
from typing import List

def compute_trajectory_variance(all_grads: List[np.ndarray]) -> float:
    """
    Computes the variance of the mean gradient vector over the SGD optimization trajectory,
    averaged across multiple random seeds.
    """
    if not all_grads:
        return 0.0
        
    min_g = min(len(g) for g in all_grads)
    # Average across seeds
    avg_g = np.mean([g[:min_g] for g in all_grads], axis=0)
    
    # Variance over time (trajectory variance)
    return float(np.var(avg_g))
