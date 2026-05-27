# KL(fidelity histogram || Haar PDF) — Sim/Aspuru-Guzik expressibility
from typing import Optional                                                  # rng is optional
import numpy as np                                                           # arrays, bincount
import qulacs                                                                # state vectors
from scipy import stats                                                     # scipy.stats.entropy = KL


def haar_fidelity_pmf(num_bins: int, dim: int) -> np.ndarray:                # integrated Haar prob per bin
    edges = np.linspace(0.0, 1.0, num_bins + 1)                              # uniform bin edges on [0, 1]
    cdf = 1.0 - np.power(1.0 - edges, dim - 1)                               # CDF of (d-1)(1-F)^(d-2)
    return np.diff(cdf)                                                      # per-bin probability mass


def compute_expressibility(                                                  # mirrors authors' ComputeExpress
    circuit: qulacs.ParametricQuantumCircuit,                                # parametric circuit to probe
    num_fidelity: int = 500, num_bins: int = 20,                             # paper defaults
    rng: Optional[np.random.Generator] = None,                               # seed for parameter sampling
) -> float:
    if rng is None: rng = np.random.default_rng()                            # default to fresh entropy
    n_tot = circuit.get_qubit_count()                                        # total qubits in circuit
    num_params = circuit.get_parameter_count()                               # parametric gate count
    dim = 2 ** n_tot                                                         # Hilbert-space dimension
    psi_init = np.zeros(dim, dtype=np.complex128); psi_init[0] = 1.0         # always start from |0...0>
    state_vectors = np.empty((2 * num_fidelity, dim), dtype=np.complex128)   # pre-allocate output buffer
    for k in range(2 * num_fidelity):                                        # sample 2N parameter sets
        params = rng.uniform(0.0, 2.0 * np.pi, size=num_params)              # uniform in [0, 2pi)^P
        for p_idx in range(num_params):                                      # write into qulacs gates
            circuit.set_parameter(p_idx, params[p_idx])                      # no sign flip needed (Haar is symmetric)
        state = qulacs.QuantumState(n_tot)                                   # fresh state
        state.load(psi_init)                                                 # initialise to computational zero
        circuit.update_quantum_state(state)                                  # evolve
        state_vectors[k] = state.get_vector()                                # store output state
    fidelities = np.abs(                                                     # |<psi_a | psi_b>|^2 for paired draws
        np.einsum("ki,ki->k", np.conj(state_vectors[:num_fidelity]),
                              state_vectors[num_fidelity:])
    ) ** 2
    bin_index = (                                                            # discrete bin index per sample
        np.floor(fidelities * num_bins).clip(0, num_bins - 1).astype(int)
    )
    counts = np.bincount(bin_index, minlength=num_bins).astype(float)        # histogram
    p_emp = counts / counts.sum()                                            # empirical pmf
    p_haar = haar_fidelity_pmf(num_bins, dim)                                # theoretical Haar pmf
    return float(stats.entropy(p_emp, p_haar))                               # KL(p_emp || p_haar)
