# SGD loop in TC convention; negates at qulacs gate-set/grad boundary
from typing import Optional, Sequence                                        # record-mask types
import numpy as np                                                           # array math
import qulacs                                                                # ParametricQuantumCircuit + Observable


def run_sgd_trajectory(                                                      # one seed's worth of SGD
    circuit: qulacs.ParametricQuantumCircuit,                                # parametric circuit, gates already added
    obs: qulacs.Observable,                                                  # the cost operator
    psi_init: np.ndarray,                                                    # state to load before every forward pass
    lr: float, max_iters: int, tol: float, seed: int,                        # SGD hyperparams + per-seed RNG seed
    record_indices: Optional[Sequence[int]] = None,                          # subset of params reported in the trajectory
) -> np.ndarray:
    rng = np.random.default_rng(seed)                                        # deterministic seed-level RNG
    num_params = circuit.get_parameter_count()                               # total parametric gates in the circuit
    params = rng.standard_normal(num_params) * 0.1                           # paper init: N(0, 0.1) in TC convention
    grads_traj: list[np.ndarray] = []                                        # collected gradients per iteration

    for _ in range(max_iters):                                               # SGD loop
        for p_idx in range(num_params):                                      # write current params into qulacs gates
            circuit.set_parameter(p_idx, -params[p_idx])                     # negate: qulacs uses exp(+i*a/2), TC exp(-i*a/2)
        state = qulacs.QuantumState(circuit.get_qubit_count())               # fresh state vector
        state.load(psi_init)                                                 # initialise to the seed's input state
        circuit.update_quantum_state(state)                                  # apply the parametric circuit
        loss = obs.get_expectation_value(state).real                         # scalar loss for early-stop test
        grad_tc = -np.array(circuit.backprop(obs))                           # qulacs gradient -> TC gradient via sign flip
        if record_indices is not None:                                       # subset reporting (e.g. u-rotation only)
            grads_traj.append(grad_tc[list(record_indices)].copy())          # store the slice we care about
        else:                                                                # full-vector reporting
            grads_traj.append(grad_tc.copy())                                # store everything backprop returned
        params -= lr * grad_tc                                               # vanilla SGD step in TC coords
        if loss < tol: break                                                 # early stop at requested tolerance
    return np.array(grads_traj)                                              # shape (iters_run, len(record_indices) or num_params)
