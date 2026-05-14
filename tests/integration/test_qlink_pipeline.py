"""Integration test: full Q-LINK pipeline from circuit to trajectory variance."""

import numpy as np
from qlink_replication.config.geometry import yb_depth
from qlink_replication.config import hyperparameters as hp
from qlink_replication.architectures.qlink import build_qlink_circuit
from qlink_replication.core.observables import create_observable_fixed
from qlink_replication.core.state import generate_random_product_state
from qlink_replication.core.optimizer import run_sgd_trajectory
from qlink_replication.core.metrics import compute_trajectory_variance


def test_qlink_pipeline_produces_positive_variance():
    """Full Q-LINK pipeline at n_data=2 must yield a finite, positive trajectory variance."""
    n_data = 2
    n_tot = n_data + 1
    depth = yb_depth(n_data)
    circuit = build_qlink_circuit(n_data, depth)
    obs = create_observable_fixed(n_tot, "Q-LINK(Fixed)")
    rng = np.random.default_rng(hp.DEFAULT_RANDOM_SEED)

    all_grads = []
    for seed in range(3):
        psi = generate_random_product_state(n_data, n_tot, "Q-LINK(Fixed)", rng)
        grads = run_sgd_trajectory(circuit, obs, psi, lr=0.1, max_iters=50, tol=1e-3, seed=seed)
        all_grads.append(grads)

    var = compute_trajectory_variance(all_grads)
    assert var > 0, f"Q-LINK trajectory variance must be positive, got {var}"
    assert np.isfinite(var), f"Q-LINK trajectory variance must be finite, got {var}"
