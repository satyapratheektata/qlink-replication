# depth schedule from Yi & Bhadani (2026) — depth(n) = ceil(n^2 * log n)
import numpy as np                                                           # ceil + log


def yb_depth(n_data: int) -> int:                                            # used by every (n, model) cell
    if n_data <= 1: return 1                                                 # guard log(1)=0 -> depth 0 collapse
    return int(np.ceil(n_data ** 2 * np.log(n_data)))                        # ceil keeps depth >= 1 for n>=2
