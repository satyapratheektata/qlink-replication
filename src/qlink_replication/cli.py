"""Command Line Interface for the standalone Q-LINK replication."""

import argparse
import time

from qlink_replication.config import hyperparameters as hp
from qlink_replication.config.geometry import yb_depth
from qlink_replication.core.state import generate_random_product_state
from qlink_replication.core.observables import create_observable_fixed, create_observable_buggy
from qlink_replication.core.optimizer import run_sgd_trajectory
from qlink_replication.core.metrics import compute_trajectory_variance
from qlink_replication.architectures.vanilla import build_vanilla_circuit
from qlink_replication.architectures.qlink import build_qlink_circuit
import numpy as np

def run_experiment(model: str, n_tot: int, fix_cost: bool):
    """Runs the trajectory variance experiment for a given configuration."""
    n_data = n_tot if model == "Vanilla" else n_tot - 1
    depth = yb_depth(n_data)
    
    if model == "Vanilla":
        circuit = build_vanilla_circuit(n_data, depth)
    else:
        circuit = build_qlink_circuit(n_data, depth)
        
    obs_builder = create_observable_fixed if fix_cost else create_observable_buggy
    obs = obs_builder(n_tot, model)
    
    all_grads = []
    rng = np.random.default_rng(hp.DEFAULT_RANDOM_SEED)
    
    for r in range(hp.DEFAULT_NUM_SEEDS):
        psi_init = generate_random_product_state(n_data, n_tot, model, rng)
        grads_traj = run_sgd_trajectory(
            circuit=circuit,
            obs=obs,
            psi_init=psi_init,
            lr=hp.DEFAULT_LEARNING_RATE,
            max_iters=hp.DEFAULT_MAX_ITERS,
            tol=hp.DEFAULT_CONVERGENCE_THRESHOLD,
            seed=hp.DEFAULT_RANDOM_SEED + r
        )
        all_grads.append(grads_traj)
        
    return compute_trajectory_variance(all_grads)

def main():
    parser = argparse.ArgumentParser(description="Q-LINK Empirical Replication")
    parser.add_argument("--qubits", type=int, default=5, help="Total number of qubits (n_tot)")
    parser.add_argument("--fix-cost", action="store_true", help="Use fixed marginal cost function")
    args = parser.parse_args()
    
    n_tot = args.qubits
    print(f"Running n_tot={n_tot}, fix_cost={args.fix_cost}")
    
    t0 = time.time()
    var_v = run_experiment("Vanilla", n_tot, args.fix_cost)
    var_q = run_experiment("Q-LINK(Fixed)", n_tot, args.fix_cost)
    ratio = var_q / var_v if var_v > 0 else float('nan')
    elapsed = time.time() - t0
    
    print(f"Vanilla: {var_v:.3e} | Q-LINK: {var_q:.3e} | Ratio: {ratio:.2f}x | ({elapsed:.1f}s)")

if __name__ == "__main__":
    main()
