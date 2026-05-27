# Part B: E2 (interaction basis) and E3 (mechanism ablation) with bootstrap CIs
import _lib.path                                                             # add ../src to sys.path
import argparse, csv, time                                                   # CLI + outputs
from pathlib import Path                                                     # filesystem
import matplotlib                                                            # backend
matplotlib.use("Agg")                                                        # headless
import matplotlib.pyplot as plt                                              # plotting
import numpy as np                                                           # arrays
import qulacs                                                                # state for forward passes
from _lib.cell import reset_global_seed                                      # set seeds 42 + r
from qlink_replication.architectures.vanilla import build_vanilla_circuit, u_rotation_indices as v_idx
from qlink_replication.architectures.qlink_variants import build_qlink_variant_circuit, variant_u_rotation_indices as q_idx
from qlink_replication.config.geometry import yb_depth                       # depth schedule
from qlink_replication.core.observables import create_observable_buggy       # matches authors' Table I
from qlink_replication.core.state import generate_random_product_state       # input state

def _traj(c, o, psi, lr, mx, tol, ui, n_seeds):                              # SGD across n_seeds; return u-grad trajectories
    nparams = c.get_parameter_count(); out = []                              # collected per-seed grad arrays
    for r in range(n_seeds):                                                 # one seed = one SGD run
        reset_global_seed(42 + r)                                            # spread RNG so seeds differ
        params = np.random.randn(nparams) * 0.1                              # paper init
        grads = []                                                           # this seed's trajectory
        for _ in range(mx):                                                  # SGD loop
            for p in range(nparams): c.set_parameter(p, -float(params[p]))   # TC->qulacs sign flip
            s = qulacs.QuantumState(c.get_qubit_count()); s.load(psi.astype(np.complex128))  # load
            c.update_quantum_state(s); L = float(o.get_expectation_value(s).real)            # forward
            g = -np.array(c.backprop(o)); grads.append(g[ui].copy())         # TC-frame grad, u-only
            params -= lr * g                                                 # SGD step
            if L < tol: break                                                # early stop
        out.append(np.array(grads))                                          # done with this seed
    return out                                                               # caller bootstraps

def _stat(all_grads):                                                        # mean-trajectory variance + 95% bootstrap CI
    n = min(len(g) for g in all_grads); a = [g[:n] for g in all_grads]       # align to shortest run
    point = float(np.var(np.mean(a, axis=0)))                                # the headline scalar
    rng = np.random.default_rng(0); B = 200; boot = np.empty(B)              # bootstrap setup
    for b in range(B):                                                       # B resamples over seeds
        idx = rng.integers(0, len(a), size=len(a))                           # with-replacement indices
        boot[b] = float(np.var(np.mean([a[i] for i in idx], axis=0)))        # one bootstrap statistic
    lo, hi = np.percentile(boot, [2.5, 97.5])                                # 95% CI
    return point, float(lo), float(hi)                                       # (point, low, high)

def _cell(c, n_tot, n_data, model, lr, mx, tol, n_seeds, rng_seed):          # one (variant, n) cell
    obs = create_observable_buggy(n_tot, model)                              # buggy cost matches paper numbers
    psi = generate_random_product_state(n_data, n_tot, model, np.random.default_rng(rng_seed))
    ui = q_idx(n_data, yb_depth(n_data)) if n_tot != n_data else v_idx(n_data, yb_depth(n_data))
    return _stat(_traj(c, obs, psi, lr, mx, tol, ui, n_seeds))               # variance + CI

def _sweep(qubits, n_seeds, lr, mx, tol, title, variants, csv_path):         # E2 or E3 sweep
    rows = []; print(f"\n=== {title} ===")                                   # banner
    for n in qubits:                                                         # for each n
        nd = n - 1; d = yb_depth(nd)                                         # data qubits + depth
        c0 = build_vanilla_circuit(nd, d)                                    # always include Vanilla baseline
        p, lo, hi = _cell(c0, nd, nd, "Vanilla", lr, mx, tol, n_seeds, rng_seed=n)
        rows.append((n, "Vanilla", p, lo, hi)); print(f"  n={n} Vanilla       Var={p:.3e} [{lo:.3e}, {hi:.3e}]")
        for lbl, build in variants:                                          # one row per variant per n
            c = build(nd, d); p, lo, hi = _cell(c, nd + 1, nd, "Q-LINK(Fixed)", lr, mx, tol, n_seeds, rng_seed=n)
            rows.append((n, lbl, p, lo, hi)); print(f"  n={n} {lbl:18s} Var={p:.3e} [{lo:.3e}, {hi:.3e}]")
    w = csv.writer(open(csv_path, "w", newline=""))                          # write CSV
    w.writerow(["Qubits","Variant","Var","CI_lo","CI_hi"]); [w.writerow(r) for r in rows]
    return rows                                                              # caller plots

def main(qubits, n_seeds, lr, mx, tol, out, only):                           # entry point
    (out / "data").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    t0 = time.time()                                                         # wall-clock start
    if only in {"E2", "both"}:                                               # E2 sweep
        e2 = [(f"Q-LINK({b})", lambda nd, d, b=b: build_qlink_variant_circuit(nd, d, basis=b, mechanism="full")) for b in ("XX", "YY", "ZZ")]
        _sweep(qubits, n_seeds, lr, mx, tol, "E2 - Interaction-basis ablation", e2, out / "data" / "extensions_E2_basis.csv")
    if only in {"E3", "both"}:                                               # E3 sweep
        e3 = [(f"Q-LINK({lbl})", lambda nd, d, mech=mech: build_qlink_variant_circuit(nd, d, basis="XX", mechanism=mech)) for mech, lbl in (("full","full"), ("coll_only","coll-only"), ("dist_only","dist-only"))]
        _sweep(qubits, n_seeds, lr, mx, tol, "E3 - Mechanism ablation", e3, out / "data" / "extensions_E3_mechanism.csv")
    print(f"\nTotal wall: {time.time() - t0:.1f}s")                          # final timing

if __name__ == "__main__":                                                   # CLI
    p = argparse.ArgumentParser()                                            # standard argparse
    p.add_argument("--qubits", type=str, default="5,6,7"); p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=1500); p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-3); p.add_argument("--out", type=Path, default=Path("replication_results"))
    p.add_argument("--only", choices=["E2", "E3", "both"], default="both"); a = p.parse_args()
    main([int(q) for q in a.qubits.split(",")], a.seeds, a.lr, a.max_iters, a.tol, a.out, a.only)
