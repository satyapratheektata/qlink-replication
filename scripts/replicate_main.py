"""Drive Table I, Fig. 4 (loss vs. iteration), and Fig. 5 (avg-iter vs. qubits).

Mirrors AARC-lab/DCAS_2026_QLINK ``scripts/main.py`` line-by-line:

* Global seed 42 across ``random``, ``numpy``, ``torch`` (we replace torch with
  numpy here, but match the consumption order).
* ``qubit_list = range(5, 11)`` and ``data_qubits = n - 1`` for **both** models.
* Per repeat: random data state -> ``u_params`` -> ``res_params`` (built even for
  Vanilla so the global RNG consumption order matches the authors').
* SGD with lr=0.1, max 1500 iters, early-stop at ``loss < 1e-3``.
* Records ONLY u_rotation gradients (3*n_data*depth values per iter); res_params
  gradients are dropped, matching ``u_params.grad`` in the authors' protocol.
* Expressibility: 20 bins, 500 fidelity pairs, KL vs Haar PMF.

Outputs (under ``data/`` and ``figures/``):

* ``data/n{n}_loss_comparison.json``  -> for Fig. 4 plotting.
* ``data/avgiter_vs_qubits.json``     -> for Fig. 5 plotting.
* ``data/table1.csv``                 -> the Table I numbers.
* ``figures/n{n}_loss_comparison.pdf`` -> the Fig. 4 panels.
* ``figures/avgiter_vs_qubits.pdf``    -> the Fig. 5 plot.

Run with ``uv run python scripts/replicate_main.py [--qubits 5,6,7] [--seeds 5]``
to override the sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qulacs

from qlink_replication.architectures.qlink import build_qlink_circuit, u_rotation_indices as qlink_u_indices
from qlink_replication.architectures.vanilla import build_vanilla_circuit, u_rotation_indices as vanilla_u_indices
from qlink_replication.config.geometry import yb_depth
from qlink_replication.core.expressibility import compute_expressibility
from qlink_replication.core.observables import create_observable_fixed
from qlink_replication.core.state import generate_random_product_state


MODELS = ["Q-LINK(Fixed)", "Q-LINK(Adaptive)", "Vanilla"]


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_circuit(model: str, n_data: int, depth: int) -> qulacs.ParametricQuantumCircuit:
    if model == "Vanilla":
        return build_vanilla_circuit(n_data, depth)
    return build_qlink_circuit(n_data, depth, is_adaptive=(model == "Q-LINK(Adaptive)"))


def _u_indices(model: str, n_data: int, depth: int) -> list[int]:
    if model == "Vanilla":
        return vanilla_u_indices(n_data, depth)
    return qlink_u_indices(n_data, depth, is_adaptive=(model == "Q-LINK(Adaptive)"))


def _run_single_seed(
    circuit: qulacs.ParametricQuantumCircuit,
    obs: qulacs.Observable,
    psi_init: np.ndarray,
    u_idx: list[int],
    n_data: int,
    depth: int,
    is_adaptive: bool,
    lr: float,
    max_iters: int,
    tol: float,
) -> tuple[list[float], list[np.ndarray], int]:
    """One SGD run. Returns (loss_per_iter, u_grad_per_iter, stop_iter).

    Parameters live in TC convention. Qulacs gates apply exp(+i*angle/2*sigma), so we
    negate at the gate-setter boundary and negate the qulacs gradient to recover the
    TC-frame gradient. See architectures.qlink for the same convention on fixed gates.
    """
    num_params = circuit.get_parameter_count()
    # Authors initialise both u_params and res_params N(0, 0.1). The flat order must
    # match circuit insertion order — u rotations come after the initial collection
    # block (if adaptive) and are interleaved with residual collection blocks between
    # layers.
    params = np.zeros(num_params)
    if is_adaptive:
        # initial collection (n_data res params)
        params[:n_data] = np.random.randn(n_data) * 0.1
        cursor = n_data
    else:
        cursor = 0
    for j in range(depth):
        # u rotations
        params[cursor : cursor + 3 * n_data] = np.random.randn(3 * n_data) * 0.1
        cursor += 3 * n_data
        if is_adaptive and j != depth - 1:
            params[cursor : cursor + n_data] = np.random.randn(n_data) * 0.1
            cursor += n_data

    losses: list[float] = []
    u_grads: list[np.ndarray] = []
    stop_iter = max_iters

    for it in range(max_iters):
        for p_idx in range(num_params):
            circuit.set_parameter(p_idx, -float(params[p_idx]))   # TC -> qulacs sign

        state = qulacs.QuantumState(circuit.get_qubit_count())
        state.load(psi_init.astype(np.complex128))
        circuit.update_quantum_state(state)

        loss = float(obs.get_expectation_value(state).real)
        losses.append(loss)
        grad_tc = -np.array(circuit.backprop(obs))            # qulacs -> TC sign
        u_grads.append(grad_tc[u_idx].copy())

        params -= lr * grad_tc

        if loss < tol:
            stop_iter = it + 1
            break

    return losses, u_grads, stop_iter


def replicate(
    qubit_list: list[int],
    n_seeds: int,
    max_iters: int,
    lr: float,
    tol: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    fig_dir = out_dir / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    table_rows: list[tuple[int, str, float, float]] = []
    avgiter: dict[str, list[float]] = {m: [] for m in MODELS}

    for n in qubit_list:
        print(f"\n=== n = {n} (data_qubits = {n - 1}) ===")
        n_loss_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for model in MODELS:
            n_data = n - 1
            n_tot = n if model != "Vanilla" else n - 1
            depth = yb_depth(n_data)
            circuit = _build_circuit(model, n_data, depth)
            obs = create_observable_fixed(n_tot, model)
            u_idx = _u_indices(model, n_data, depth)
            is_adaptive = model == "Q-LINK(Adaptive)"

            all_losses: list[list[float]] = []
            all_grads: list[list[np.ndarray]] = []
            stop_iters: list[int] = []

            t0 = time.time()
            _set_global_seed(42)
            for r in range(n_seeds):
                psi_init = generate_random_product_state(
                    n_data, n_tot, model, np.random.default_rng(seed=np.random.randint(0, 2**31))
                )
                losses, grads, stop_iter = _run_single_seed(
                    circuit, obs, psi_init, u_idx, n_data, depth, is_adaptive, lr, max_iters, tol,
                )
                all_losses.append(losses)
                all_grads.append(grads)
                stop_iters.append(stop_iter)
                print(f"  {model:18s} seed {r}: stop={stop_iter:4d} final_loss={losses[-1]:.4f}")

            # Variance over the (iters, U_params) array averaged across seeds.
            min_len = min(len(g) for g in all_grads)
            avg_g = np.mean([np.stack(g[:min_len]) for g in all_grads], axis=0)
            grad_variance = float(np.var(avg_g))

            # Expressibility on the same circuit with random Uniform[0, 2pi) params.
            kl = compute_expressibility(
                circuit, num_fidelity=500, num_bins=20, rng=np.random.default_rng(42),
            )

            # Aligned loss curves for Fig. 4: pad short runs with their last value
            # (authors' compute_avg_loss_gradient_stopiter behaviour).
            max_l = max(len(L) for L in all_losses)
            padded = np.array([L + [L[-1]] * (max_l - len(L)) for L in all_losses])
            avg_l = padded.mean(axis=0)
            std_l = padded.std(axis=0)
            n_loss_stats[model] = (avg_l, std_l)
            avgiter[model].append(float(np.mean(stop_iters)))

            table_rows.append((n, model, kl, grad_variance))
            print(
                f"  {model:18s} | KL={kl:.3e} | Var={grad_variance:.3e} | avg_iter={np.mean(stop_iters):.1f}"
                f" | wall {time.time()-t0:.1f}s"
            )

        # Save and plot per-n loss comparison (Fig. 4 panel)
        payload = {
            "n": n,
            "models": {
                model: {
                    "iteration": list(range(len(avg_l))),
                    "avg_loss": avg_l.tolist(),
                    "std_loss": std_l.tolist(),
                }
                for model, (avg_l, std_l) in n_loss_stats.items()
            },
        }
        with open(data_dir / f"n{n}_loss_comparison.json", "w") as f:
            json.dump(payload, f, indent=2)
        _plot_loss_panel(n, n_loss_stats, fig_dir / f"n{n}_loss_comparison.pdf")

    # Write Table I CSV
    with open(data_dir / "table1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Qubits", "Model", "Expressibility (KL)", "Gradient Variance"])
        for n, model, kl, var in table_rows:
            writer.writerow([n, model, f"{kl:.4e}", f"{var:.4e}"])

    # Save and plot Fig. 5
    with open(data_dir / "avgiter_vs_qubits.json", "w") as f:
        json.dump({"qubit_list": qubit_list, "avgiter": avgiter}, f, indent=2)
    _plot_avgiter(qubit_list, avgiter, fig_dir / "avgiter_vs_qubits.pdf")

    print(f"\nDone. Table I -> {data_dir / 'table1.csv'}")


def _plot_loss_panel(n: int, stats: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    palette = {"Q-LINK(Fixed)": "orange", "Q-LINK(Adaptive)": "green", "Vanilla": "blue"}
    fig, ax = plt.subplots(figsize=(4, 3))
    for model, (avg, std) in stats.items():
        iters = np.arange(len(avg))
        ax.plot(iters, avg, color=palette[model], label=model)
        ax.fill_between(iters, avg - std, avg + std, color=palette[model], alpha=0.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"{n} qubits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_avgiter(qubits: list[int], avgiter: dict[str, list[float]], path: Path) -> None:
    palette = {"Q-LINK(Fixed)": "orange", "Q-LINK(Adaptive)": "green", "Vanilla": "blue"}
    fig, ax = plt.subplots(figsize=(4, 3))
    for model, values in avgiter.items():
        ax.scatter(qubits, values, color=palette[model], label=model)
        ax.plot(qubits, values, color=palette[model], alpha=0.5)
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Avg iterations to converge")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qubits", type=str, default="5,6,7,8,9,10",
                   help="comma-separated qubit counts; defaults to authors' 5..10")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("replication_results"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    qubit_list = [int(q) for q in args.qubits.split(",")]
    replicate(
        qubit_list=qubit_list,
        n_seeds=args.seeds,
        max_iters=args.max_iters,
        lr=args.lr,
        tol=args.tol,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
