"""Circuit expressibility via KL divergence against the Haar fidelity distribution.

Mirrors the protocol in AARC-lab/DCAS_2026_QLINK ``src/qlink/metrics/compute_expressibility.py``:

* Sample ``num_fidelity * 2`` parameter vectors uniformly in ``[0, 2π)^P``.
* Run the parametric circuit on the fixed input state ``|0...0⟩`` for each parameter set.
* Pair the resulting state vectors and compute fidelities ``|⟨ψ_a|ψ_b⟩|²``.
* Histogram fidelities into ``num_bins`` equal bins on ``[0, 1]``.
* Compare against the integrated Haar PMF over the same bins and take the KL divergence.

The Haar distribution on a ``d``-dimensional Hilbert space has fidelity PDF
``(d-1)(1-F)^(d-2)``; its integrated PMF on the bin ``[a, b]`` is
``(1 - (1-b)^(d-1)) - (1 - (1-a)^(d-1))``.
"""

from typing import Optional

import numpy as np
import qulacs
from scipy import stats


def haar_fidelity_pmf(num_bins: int, dim: int) -> np.ndarray:
    """Integrated Haar fidelity probability per bin on a uniform ``[0, 1]`` grid."""
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    cdf = 1.0 - np.power(1.0 - edges, dim - 1)
    return np.diff(cdf)


def compute_expressibility(
    circuit: qulacs.ParametricQuantumCircuit,
    num_fidelity: int = 500,
    num_bins: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """KL(P_circuit || P_Haar) of the fidelity histogram against the Haar PMF.

    Returns ``0`` when the empirical histogram exactly matches the Haar PMF.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_tot = circuit.get_qubit_count()
    num_params = circuit.get_parameter_count()
    dim = 2 ** n_tot

    psi_init = np.zeros(dim, dtype=np.complex128)
    psi_init[0] = 1.0

    state_vectors = np.empty((2 * num_fidelity, dim), dtype=np.complex128)
    for k in range(2 * num_fidelity):
        params = rng.uniform(0.0, 2.0 * np.pi, size=num_params)
        for p_idx in range(num_params):
            circuit.set_parameter(p_idx, params[p_idx])
        state = qulacs.QuantumState(n_tot)
        state.load(psi_init)
        circuit.update_quantum_state(state)
        state_vectors[k] = state.get_vector()

    fidelities = np.abs(
        np.einsum("ki,ki->k", np.conj(state_vectors[:num_fidelity]), state_vectors[num_fidelity:])
    ) ** 2

    bin_index = np.floor(fidelities * num_bins).clip(0, num_bins - 1).astype(int)
    counts = np.bincount(bin_index, minlength=num_bins).astype(float)
    p_empirical = counts / counts.sum()

    p_haar = haar_fidelity_pmf(num_bins, dim)
    return float(stats.entropy(p_empirical, p_haar))
