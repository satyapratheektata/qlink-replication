"""Random state vector generation module."""

import numpy as np

def generate_random_product_state(n_data: int, n_tot: int, model: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a Haar-random data state and appends the messenger qubit if needed.
    """
    psi_data = rng.standard_normal(2**n_data) + 1j * rng.standard_normal(2**n_data)
    psi_data /= np.linalg.norm(psi_data)
    
    if model.startswith("Q-LINK"):
        psi_init = np.kron(psi_data, np.array([1.0, 0.0], dtype=np.complex128))
        # Tensor permutation to place the messenger qubit at the expected index
        psi_init = psi_init.reshape([2] * n_tot).transpose().flatten()
    else:
        psi_init = psi_data.reshape([2] * n_data).transpose().flatten()
        
    return psi_init
