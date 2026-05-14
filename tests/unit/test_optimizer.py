"""Smoke tests for the SGD optimization loop."""

import numpy as np
from qlink_replication.core.optimizer import run_sgd_trajectory
from qlink_replication.architectures.vanilla import build_vanilla_circuit
from qlink_replication.core.observables import create_observable_fixed

def test_sgd_flow():
    """Verify the SGD loop runs without errors on a minimal system."""
    n_data = 2
    circuit = build_vanilla_circuit(n_data, depth=1)
    obs = create_observable_fixed(n_data, "Vanilla")
    psi_init = np.array([1.0, 0.0, 0.0, 0.0])
    
    grads = run_sgd_trajectory(circuit, obs, psi_init, lr=0.1, max_iters=5, tol=1e-3, seed=42)
    
    assert len(grads) > 0
    assert grads.shape[1] == circuit.get_parameter_count()
