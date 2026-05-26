"""Geometric scaling and circuit depth formulas."""

import numpy as np

def yb_depth(n_data: int) -> int:
    """The Yi-Bhadani finite-depth scaling formula: ceil(n_data^2 * ln(n_data))."""
    if n_data <= 1:
        return 1
    return int(np.ceil(n_data ** 2 * np.log(n_data)))
