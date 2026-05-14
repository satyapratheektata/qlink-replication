"""Integration test: buggy vs fixed observable produce distinct variances."""

import numpy as np
from qlink_replication.config.geometry import yb_depth
from qlink_replication.config import hyperparameters as hp
from qlink_replication.architectures.qlink import build_qlink_circuit
from qlink_replication.core.observables import create_observable_buggy, create_observable_fixed
from qlink_replication.core.state import generate_random_product_state
from qlink_replication.core.optimizer import run_sgd_trajectory
from qlink_replication.core.metrics import compute_trajectory_variance


def _run_with_obs(obs_fn, n_tot=4):
    n_data = n_tot - 1
    depth = yb_depth(n_data)
    circuit = build_qlink_circuit(n_data, depth)
    obs = obs_fn(n_tot, "Q-LINK(Fixed)")
    rng = np.random.default_rng(hp.DEFAULT_RANDOM_SEED)

    all_grads = []
    for seed in range(3):
        psi = generate_random_product_state(n_data, n_tot, "Q-LINK(Fixed)", rng)
        grads = run_sgd_trajectory(circuit, obs, psi, lr=0.1, max_iters=50, tol=1e-3, seed=seed)
        all_grads.append(grads)
    return compute_trajectory_variance(all_grads)


def test_buggy_and_fixed_differ():
    """The buggy and fixed cost functions must produce measurably different variances."""
    var_buggy = _run_with_obs(create_observable_buggy)
    var_fixed = _run_with_obs(create_observable_fixed)
    assert not np.isclose(var_buggy, var_fixed, rtol=0.05), (
        f"Buggy ({var_buggy:.3e}) and fixed ({var_fixed:.3e}) should differ"
    )
