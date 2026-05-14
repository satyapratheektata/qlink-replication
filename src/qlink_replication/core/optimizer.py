"""SGD optimization loop."""

import numpy as np
import qulacs

def run_sgd_trajectory(
    circuit: qulacs.ParametricQuantumCircuit,
    obs: qulacs.Observable,
    psi_init: np.ndarray,
    lr: float,
    max_iters: int,
    tol: float,
    seed: int
) -> np.ndarray:
    """Executes the SGD loop and returns the array of gradients over time."""
    rng = np.random.default_rng(seed)
    num_params = circuit.get_parameter_count()
    params = rng.standard_normal(num_params) * 0.1
    
    grads_traj = []
    
    for it in range(max_iters):
        for p_idx in range(num_params):
            circuit.set_parameter(p_idx, params[p_idx])
            
        state = qulacs.QuantumState(circuit.get_qubit_count())
        state.load(psi_init)
        circuit.update_quantum_state(state)
        
        loss = obs.get_expectation_value(state).real
        grad = np.array(circuit.backprop(obs))
        grads_traj.append(grad.copy())
        
        params -= lr * grad
        if loss < tol:
            break
            
    return np.array(grads_traj)
