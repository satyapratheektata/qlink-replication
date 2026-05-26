"""Generate Fig. 6 — loss landscapes for n in {8, 9, 10}.

Mirrors ``plot_loss_landscape`` from AARC-lab/DCAS_2026_QLINK with the same
defaults the authors use (``num_points=200``, ``alpha, beta in [-3, 3]``):

* Train each (model, n_data) until convergence or 1500 iters.
* Sample two random parameter-space directions ``d_1, d_2``, normalize each to unit
  Frobenius norm.
* For each (alpha, beta) on a 200x200 grid in [-3, 3]^2, set parameters to
  ``theta_trained + alpha * d_1 + beta * d_2`` and evaluate the cost.
* Save a 3-D surface plot per (model, n).

Outputs to ``--out replication_results/figures/landscape/`` as
``n{n}_{Model}.pdf``.

The 200x200 grid implies 40k forward passes per panel — for n=10 with depth
``ceil(81 * log(9)) ~= 178``, that's ~7M gate applications per panel. Pass
``--grid 50`` for a smoke run.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qulacs

from qlink_replication.architectures.qlink import (
    build_qlink_circuit,
    u_rotation_indices as qlink_u_indices,
)
from qlink_replication.architectures.vanilla import (
    build_vanilla_circuit,
    u_rotation_indices as vanilla_u_indices,
)
from qlink_replication.config.geometry import yb_depth
from qlink_replication.core.observables import create_observable_fixed
from qlink_replication.core.state import generate_random_product_state


MODELS = ["Vanilla", "Q-LINK(Fixed)", "Q-LINK(Adaptive)"]


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_circuit(model: str, n_data: int, depth: int) -> qulacs.ParametricQuantumCircuit:
    if model == "Vanilla":
        return build_vanilla_circuit(n_data, depth)
    return build_qlink_circuit(n_data, depth, is_adaptive=(model == "Q-LINK(Adaptive)"))


def _init_params(model: str, n_data: int, depth: int, num_params: int) -> np.ndarray:
    """N(0, 0.1) init in the same flat order the architecture inserts gates."""
    is_adaptive = model == "Q-LINK(Adaptive)"
    params = np.zeros(num_params)
    cursor = 0
    if is_adaptive:
        params[cursor : cursor + n_data] = np.random.randn(n_data) * 0.1
        cursor += n_data
    for j in range(depth):
        params[cursor : cursor + 3 * n_data] = np.random.randn(3 * n_data) * 0.1
        cursor += 3 * n_data
        if is_adaptive and j != depth - 1:
            params[cursor : cursor + n_data] = np.random.randn(n_data) * 0.1
            cursor += n_data
    return params


def _eval_loss(circuit: qulacs.ParametricQuantumCircuit, obs: qulacs.Observable,
               psi_init: np.ndarray, params: np.ndarray) -> float:
    for p in range(len(params)):
        circuit.set_parameter(p, -float(params[p]))   # TC -> Qulacs sign flip
    state = qulacs.QuantumState(circuit.get_qubit_count())
    state.load(psi_init.astype(np.complex128))
    circuit.update_quantum_state(state)
    return float(obs.get_expectation_value(state).real)


def _train(circuit: qulacs.ParametricQuantumCircuit, obs: qulacs.Observable,
           psi_init: np.ndarray, params: np.ndarray, lr: float, max_iters: int,
           tol: float) -> np.ndarray:
    """SGD until loss < tol or max_iters. Returns the final parameter vector."""
    for _ in range(max_iters):
        for p in range(len(params)):
            circuit.set_parameter(p, -float(params[p]))
        state = qulacs.QuantumState(circuit.get_qubit_count())
        state.load(psi_init.astype(np.complex128))
        circuit.update_quantum_state(state)
        loss = float(obs.get_expectation_value(state).real)
        if loss < tol:
            break
        grad_tc = -np.array(circuit.backprop(obs))
        params -= lr * grad_tc
    return params


def _landscape(circuit: qulacs.ParametricQuantumCircuit, obs: qulacs.Observable,
               psi_init: np.ndarray, theta_trained: np.ndarray, grid: int,
               span: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    d1 = rng.standard_normal(len(theta_trained))
    d2 = rng.standard_normal(len(theta_trained))
    # Gram-Schmidt + normalise so the axes are orthonormal in parameter space.
    d1 /= np.linalg.norm(d1)
    d2 -= np.dot(d1, d2) * d1
    d2 /= np.linalg.norm(d2)

    alphas = np.linspace(-span, span, grid)
    betas = np.linspace(-span, span, grid)
    Z = np.empty((grid, grid))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            theta = theta_trained + a * d1 + b * d2
            Z[i, j] = _eval_loss(circuit, obs, psi_init, theta)
    return alphas, betas, Z


def _plot(alphas: np.ndarray, betas: np.ndarray, Z: np.ndarray, title: str,
          out_path: Path) -> None:
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel("Loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qubits", type=str, default="8,9,10")
    p.add_argument("--grid", type=int, default=200,
                   help="grid size; authors use 200, drop to 50 for smoke runs")
    p.add_argument("--span", type=float, default=3.0)
    p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("replication_results"))
    args = p.parse_args()

    qubits = [int(q) for q in args.qubits.split(",")]
    out_dir = args.out / "figures" / "landscape"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in qubits:
        n_data = n - 1
        depth = yb_depth(n_data)
        for model in MODELS:
            n_tot = n if model != "Vanilla" else n - 1
            _set_global_seed(42)
            circuit = _build_circuit(model, n_data, depth)
            obs = create_observable_fixed(n_tot, model)
            psi_init = generate_random_product_state(
                n_data, n_tot, model, np.random.default_rng(0)
            )
            params = _init_params(model, n_data, depth, circuit.get_parameter_count())
            t0 = time.time()
            theta_trained = _train(
                circuit, obs, psi_init, params, args.lr, args.max_iters, args.tol,
            )
            train_dt = time.time() - t0
            t0 = time.time()
            alphas, betas, Z = _landscape(
                circuit, obs, psi_init, theta_trained, args.grid, args.span,
            )
            land_dt = time.time() - t0
            tag = model.replace("(", "_").replace(")", "").replace(" ", "")
            out_path = out_dir / f"n{n}_{tag}.pdf"
            _plot(alphas, betas, Z, f"{model} - {n} qubits", out_path)
            print(
                f"n={n} {model:18s} train={train_dt:5.1f}s "
                f"landscape={land_dt:6.1f}s -> {out_path}"
            )


if __name__ == "__main__":
    main()
