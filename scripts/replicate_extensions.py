"""Part B — run E2 (interaction-basis ablation) and E3 (mechanism ablation).

Same SGD recipe and metric as Part A (replicate_main.py): seed 42, lr 0.1, max
1500 iters, convergence at 1e-3, ``n_data = n - 1`` data qubits, depth
``yb_depth(n_data)``. Trajectory variance restricted to U-rotation gradients.

For each extension we sweep ``--qubits`` (default 5, 6, 7) and produce a
CSV plus a comparison plot. Bootstrap CIs over the 5 seeds are computed by
resampling the seed-level mean gradient trajectories.

Outputs::

    <out>/data/extensions_E2_basis.csv
    <out>/data/extensions_E3_mechanism.csv
    <out>/figures/extensions_E2_basis.pdf
    <out>/figures/extensions_E3_mechanism.pdf
"""

from __future__ import annotations

import argparse
import csv
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

from qlink_replication.architectures.vanilla import build_vanilla_circuit, u_rotation_indices as vanilla_u_indices
from qlink_replication.architectures.qlink_variants import build_qlink_variant_circuit, variant_u_rotation_indices
from qlink_replication.config.geometry import yb_depth
from qlink_replication.core.observables import create_observable_buggy
from qlink_replication.core.state import generate_random_product_state


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _trajectory_variance(circuit, obs, psi_init, lr, max_iters, tol, u_idx, n_seeds, depth, n_data):
    """Run SGD across ``n_seeds`` seeds, return list of u-only grad trajectories.

    Parameters are in TC convention; qulacs gates use the opposite sign so we
    negate at the boundary and negate the qulacs gradient to recover dL/dtheta_TC.
    """
    all_grads = []
    num_params = circuit.get_parameter_count()
    for seed in range(n_seeds):
        # Identical init protocol to scripts.replicate_main: seed RNG globally, then
        # consume 3*n_data*depth values for u_params. The variant has no Rxx params,
        # so no extra consumption is needed.
        _set_global_seed(42 + seed)
        params = np.random.randn(num_params) * 0.1
        grads_traj = []
        for _ in range(max_iters):
            for p_idx in range(num_params):
                circuit.set_parameter(p_idx, -float(params[p_idx]))
            state = qulacs.QuantumState(circuit.get_qubit_count())
            state.load(psi_init.astype(np.complex128))
            circuit.update_quantum_state(state)
            loss = float(obs.get_expectation_value(state).real)
            grad_tc = -np.array(circuit.backprop(obs))
            grads_traj.append(grad_tc[u_idx].copy())
            params -= lr * grad_tc
            if loss < tol:
                break
        all_grads.append(np.array(grads_traj))
    return all_grads


def _summary_stat(all_grads):
    """Mean trajectory across seeds -> overall np.var (scalar). Bootstrap CI."""
    min_len = min(len(g) for g in all_grads)
    aligned = [g[:min_len] for g in all_grads]
    mean_traj = np.mean(aligned, axis=0)
    point = float(np.var(mean_traj))
    # Bootstrap over seeds
    rng = np.random.default_rng(0)
    B = 200
    boot = np.empty(B)
    n = len(aligned)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        m = np.mean([aligned[i] for i in idx], axis=0)
        boot[b] = float(np.var(m))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def _run_one(circuit, n_tot, n_data, lr, max_iters, tol, n_seeds, rng_seed):
    """Build observable, generate input state, run SGD, return variance + CI."""
    obs = create_observable_buggy(n_tot, "Q-LINK(Fixed)") if n_tot != n_data else create_observable_buggy(n_tot, "Vanilla")
    psi_init = generate_random_product_state(
        n_data, n_tot,
        "Q-LINK(Fixed)" if n_tot != n_data else "Vanilla",
        np.random.default_rng(rng_seed),
    )
    u_idx = variant_u_rotation_indices(n_data, yb_depth(n_data)) if n_tot != n_data else vanilla_u_indices(n_data, yb_depth(n_data))
    grads = _trajectory_variance(circuit, obs, psi_init, lr, max_iters, tol, u_idx, n_seeds, yb_depth(n_data), n_data)
    return _summary_stat(grads)


def _run_e2(qubits, n_seeds, lr, max_iters, tol, out_dir):
    """Interaction-basis ablation: XX (= baseline), YY, ZZ, plus Vanilla."""
    rows = []
    print("\n=== E2 — Interaction-basis ablation ===")
    for n in qubits:
        n_data = n - 1
        depth = yb_depth(n_data)
        # Vanilla baseline
        vc = build_vanilla_circuit(n_data, depth)
        v_point, v_lo, v_hi = _run_one(vc, n_data, n_data, lr, max_iters, tol, n_seeds, rng_seed=n)
        rows.append((n, "Vanilla", v_point, v_lo, v_hi))
        print(f"  n={n} Vanilla       Var={v_point:.3e} [95% CI {v_lo:.3e}, {v_hi:.3e}]")
        for basis in ("XX", "YY", "ZZ"):
            c = build_qlink_variant_circuit(n_data, depth, basis=basis, mechanism="full")
            p, lo, hi = _run_one(c, n_data + 1, n_data, lr, max_iters, tol, n_seeds, rng_seed=n)
            rows.append((n, f"Q-LINK({basis})", p, lo, hi))
            print(f"  n={n} Q-LINK({basis})    Var={p:.3e} [95% CI {lo:.3e}, {hi:.3e}]")
    csv_path = out_dir / "data" / "extensions_E2_basis.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Qubits", "Variant", "Var", "CI_lo", "CI_hi"])
        for r in rows: w.writerow(r)
    print(f"  -> {csv_path}")
    return rows


def _run_e3(qubits, n_seeds, lr, max_iters, tol, out_dir):
    """Mechanism ablation: full, coll_only, dist_only, plus Vanilla."""
    rows = []
    print("\n=== E3 — Mechanism ablation ===")
    for n in qubits:
        n_data = n - 1
        depth = yb_depth(n_data)
        vc = build_vanilla_circuit(n_data, depth)
        v_point, v_lo, v_hi = _run_one(vc, n_data, n_data, lr, max_iters, tol, n_seeds, rng_seed=n)
        rows.append((n, "Vanilla", v_point, v_lo, v_hi))
        print(f"  n={n} Vanilla         Var={v_point:.3e} [95% CI {v_lo:.3e}, {v_hi:.3e}]")
        for mech, label in (("full", "full"), ("coll_only", "coll-only"), ("dist_only", "dist-only")):
            c = build_qlink_variant_circuit(n_data, depth, basis="XX", mechanism=mech)
            p, lo, hi = _run_one(c, n_data + 1, n_data, lr, max_iters, tol, n_seeds, rng_seed=n)
            rows.append((n, f"Q-LINK({label})", p, lo, hi))
            print(f"  n={n} Q-LINK({label}) Var={p:.3e} [95% CI {lo:.3e}, {hi:.3e}]")
    csv_path = out_dir / "data" / "extensions_E3_mechanism.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Qubits", "Variant", "Var", "CI_lo", "CI_hi"])
        for r in rows: w.writerow(r)
    print(f"  -> {csv_path}")
    return rows


def _plot_rows(rows, title, out_path):
    """Group rows by Qubits, plot bars with error bars per variant."""
    qubits = sorted({r[0] for r in rows})
    variants = []
    for r in rows:
        if r[1] not in variants:
            variants.append(r[1])
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.8 / len(variants)
    for vi, var in enumerate(variants):
        xs, ys, errs = [], [], []
        for q in qubits:
            row = next((r for r in rows if r[0] == q and r[1] == var), None)
            if row is None: continue
            xs.append(q + width * (vi - len(variants) / 2 + 0.5))
            ys.append(row[2])
            errs.append([row[2] - row[3], row[4] - row[2]])
        errs = np.array(errs).T
        ax.bar(xs, ys, width=width, yerr=errs, capsize=3, label=var)
    ax.set_yscale("log")
    ax.set_xlabel("Qubits ($n$)")
    ax.set_ylabel("Trajectory gradient variance")
    ax.set_title(title)
    ax.set_xticks(qubits)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qubits", type=str, default="5,6,7")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("replication_results"))
    p.add_argument("--only", choices=["E2", "E3", "both"], default="both")
    args = p.parse_args()

    qubits = [int(q) for q in args.qubits.split(",")]
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.only in {"E2", "both"}:
        rows = _run_e2(qubits, args.seeds, args.lr, args.max_iters, args.tol, args.out)
        _plot_rows(rows, "E2 — Interaction-basis ablation", args.out / "figures" / "extensions_E2_basis.pdf")
    if args.only in {"E3", "both"}:
        rows = _run_e3(qubits, args.seeds, args.lr, args.max_iters, args.tol, args.out)
        _plot_rows(rows, "E3 — Mechanism ablation", args.out / "figures" / "extensions_E3_mechanism.pdf")
    print(f"\nTotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
