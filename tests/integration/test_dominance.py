"""Integration test: Q-LINK trajectory variance strictly exceeds Vanilla."""

import numpy as np
from qlink_replication.config.geometry import yb_depth
from qlink_replication.config import hyperparameters as hp
from qlink_replication.architectures.vanilla import build_vanilla_circuit
from qlink_replication.architectures.qlink import build_qlink_circuit
from qlink_replication.core.observables import create_observable_fixed
from qlink_replication.core.state import generate_random_product_state
from qlink_replication.core.optimizer import run_sgd_trajectory
from qlink_replication.core.metrics import compute_trajectory_variance


def _run_full(model: str, n_tot: int):
    n_data = n_tot if model == "Vanilla" else n_tot - 1
    depth = yb_depth(n_data)
    circuit = build_vanilla_circuit(n_data, depth) if model == "Vanilla" else build_qlink_circuit(n_data, depth)
    obs = create_observable_fixed(n_tot, model)
    rng = np.random.default_rng(hp.DEFAULT_RANDOM_SEED)

    all_grads = []
    for seed in range(hp.DEFAULT_NUM_SEEDS):
        psi = generate_random_product_state(n_data, n_tot, model, rng)
        grads = run_sgd_trajectory(circuit, obs, psi, lr=0.1, max_iters=200, tol=1e-3, seed=seed)
        all_grads.append(grads)
    return compute_trajectory_variance(all_grads)


def test_qlink_dominates_vanilla_at_n5():
    """The core empirical claim: Q-LINK trajectory variance > Vanilla at n_tot=5."""
    var_v = _run_full("Vanilla", n_tot=5)
    var_q = _run_full("Q-LINK(Fixed)", n_tot=5)
    ratio = var_q / var_v
    assert ratio > 1.0, f"Q-LINK must dominate Vanilla, got ratio={ratio:.2f}x"
