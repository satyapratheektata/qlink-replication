"""SGD optimization loop."""

from typing import Optional, Sequence

import numpy as np
import qulacs


def run_sgd_trajectory(
    circuit: qulacs.ParametricQuantumCircuit,
    obs: qulacs.Observable,
    psi_init: np.ndarray,
    lr: float,
    max_iters: int,
    tol: float,
    seed: int,
    record_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Run SGD and return the recorded gradient trajectory.

    ``record_indices`` selects which entries of qulacs' flat gradient vector to keep
    in the returned trajectory. The optimizer still applies the *full* gradient to
    every parameter — the filter only affects what is reported for downstream variance
    computation. This matches the authors' ``scripts/main.py``, which trains both
    ``u_params`` and ``res_params`` but records only ``u_params.grad`` for variance.
    Defaults to ``None`` (all parameters recorded), preserving previous behaviour.
    """
    rng = np.random.default_rng(seed)
    num_params = circuit.get_parameter_count()
    # Parameters are in TC convention (theta_TC). Qulacs gates use exp(+i*angle/2)
    # rotations while TC uses exp(-i*theta/2), so we negate when writing the qulacs
    # parameter and negate qulacs' returned gradient to recover dL/dtheta_TC.
    params = rng.standard_normal(num_params) * 0.1

    grads_traj = []

    for it in range(max_iters):
        for p_idx in range(num_params):
            circuit.set_parameter(p_idx, -params[p_idx])

        state = qulacs.QuantumState(circuit.get_qubit_count())
        state.load(psi_init)
        circuit.update_quantum_state(state)

        loss = obs.get_expectation_value(state).real
        grad_tc = -np.array(circuit.backprop(obs))   # dL/dtheta_TC = -dL/dtheta_qulacs
        if record_indices is not None:
            grads_traj.append(grad_tc[list(record_indices)].copy())
        else:
            grads_traj.append(grad_tc.copy())

        params -= lr * grad_tc
        if loss < tol:
            break

    return np.array(grads_traj)
