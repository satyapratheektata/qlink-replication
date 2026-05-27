# random product state psi_data (x) |0>_messenger in TC convention
import numpy as np                                                           # complex sampling + norm


def generate_random_product_state(                                           # called once per SGD seed
    n_data: int, n_tot: int, model: str, rng: np.random.Generator,           # n_tot - n_data = 0 (Vanilla) or 1 (Q-LINK)
) -> np.ndarray:
    psi_data = (                                                             # Gaussian complex amplitudes...
        rng.standard_normal(2 ** n_data)
        + 1j * rng.standard_normal(2 ** n_data)
    )
    psi_data /= np.linalg.norm(psi_data)                                     # ...normalized to unit length
    if model.startswith("Q-LINK"):                                           # both Fixed and Adaptive get a messenger
        psi_init = np.kron(                                                  # kron places data MSB-side, messenger LSB-side
            psi_data, np.array([1.0, 0.0], dtype=np.complex128)
        )
        psi_init = (                                                         # reshape+transpose puts the result in
            psi_init.reshape([2] * n_tot).transpose().flatten()              # TC-ordered indices for downstream gates
        )
    else:                                                                    # Vanilla: no messenger
        psi_init = psi_data.reshape([2] * n_data).transpose().flatten()      # same convention as the Q-LINK branch
    return psi_init                                                          # ready for qulacs.QuantumState.load
