# trajectory-variance metric matching authors' compute_avg_loss_gradient_stopiter
from typing import List                                                      # gradient list per seed
import numpy as np                                                           # mean/var primitives


def compute_trajectory_variance(all_grads: List[np.ndarray]) -> float:       # one scalar over (iters, params, seeds)
    if not all_grads:                                                        # caller passed no seeds -> by convention 0
        return 0.0                                                           # avoids min() on empty
    min_g = min(len(g) for g in all_grads)                                   # align trajectories to the shortest run
    avg_g = np.mean([g[:min_g] for g in all_grads], axis=0)                  # mean across seeds at each iter
    return float(np.var(avg_g))                                              # variance over (iter, param) flat array
