# Part B E5: m-local cost — R_m(n) = (n+1)/(n+1-m) empirical validation
import _lib.path                                                             # add ../src to sys.path
import argparse, csv, itertools, time                                        # CLI + outputs + combinations
from pathlib import Path                                                     # filesystem
import matplotlib                                                            # backend
matplotlib.use("Agg")                                                        # headless
import matplotlib.pyplot as plt                                              # plotting
import numpy as np                                                           # arrays
import qulacs                                                                # observable + state
from _lib.cell import reset_global_seed                                      # seed = 42 + r
from qlink_replication.architectures.vanilla import build_vanilla_circuit    # Vanilla baseline
from qlink_replication.architectures.qlink import build_qlink_circuit        # Q-LINK Fixed
from qlink_replication.config.geometry import yb_depth                       # depth schedule
from qlink_replication.core.state import generate_random_product_state       # input state


def m_local_obs(n_tot, model, m):                                            # O_m = 1/2 I - 1/(2 C(n,m)) sum_{|S|=m} prod Z
    n_data = n_tot if model == "Vanilla" else n_tot - 1                      # data qubit count
    obs = qulacs.Observable(n_tot)                                           # operator container
    def q(i): return n_tot - 1 - i                                           # TC -> qulacs mapping
    m = min(m, n_data)                                                       # clamp
    subsets = list(itertools.combinations(range(n_data), m))                 # all m-subsets of {0..n_data-1}
    C = len(subsets)                                                         # binomial coefficient
    obs.add_operator(0.5, "")                                                # constant term
    for sub in subsets: obs.add_operator(-0.5 / C, " ".join(f"Z {q(i)}" for i in sub))  # one Z_S per subset
    return obs                                                               # done


def traj_var(c, o, psi, lr, mx, tol, n_seeds, num_params):                   # SGD across seeds, return variance
    grads_all = []                                                           # per-seed gradient trajectories
    for r in range(n_seeds):                                                 # seed loop
        reset_global_seed(42 + r)                                            # deterministic but distinct
        params = np.random.randn(num_params) * 0.1                           # N(0, 0.1) init
        traj = []                                                            # this-seed trajectory
        for _ in range(mx):                                                  # SGD loop
            for p in range(num_params): c.set_parameter(p, -float(params[p]))  # TC->qulacs sign flip
            s = qulacs.QuantumState(c.get_qubit_count()); s.load(psi.astype(np.complex128))  # load
            c.update_quantum_state(s)                                        # forward pass
            L = float(o.get_expectation_value(s).real)                       # cost
            g = -np.array(c.backprop(o)); traj.append(g.copy())              # TC-frame gradient
            params -= lr * g                                                 # SGD step
            if L < tol: break                                                # early stop
        grads_all.append(np.array(traj))                                     # done with this seed
    n = min(len(g) for g in grads_all)                                       # align to shortest run
    return float(np.var(np.mean([g[:n] for g in grads_all], axis=0)))        # variance of mean trajectory


def sweep(qubits, ms, n_seeds, lr, mx, tol, out):                            # full E5 grid
    rows = []                                                                # (n, m, model, Var)
    for n in qubits:                                                         # outer: n_data
        d = yb_depth(n)                                                      # depth
        for m in ms:                                                         # inner: locality
            if m > n: continue                                               # m must be <= n_data
            for model in ("Vanilla", "Q-LINK(Fixed)"):                       # two models per cell
                n_tot = n if model == "Vanilla" else n + 1                   # add messenger for Q-LINK
                c = build_vanilla_circuit(n, d) if model == "Vanilla" else build_qlink_circuit(n, d, is_adaptive=False)
                o = m_local_obs(n_tot, model, m)                             # m-local observable
                psi = generate_random_product_state(n, n_tot, model, np.random.default_rng(n * 11 + m))
                v = traj_var(c, o, psi, lr, mx, tol, n_seeds, c.get_parameter_count())  # measure
                rows.append((n, m, model, v))                                # collect
                print(f"  n={n} m={m} {model:14s} Var={v:.3e}")
    csv.writer(open(out / "data" / "extensions_E5_mlocal.csv", "w", newline="")).writerows([["Qubits","m","Model","Var"], *rows])
    return rows                                                              # plot uses this


def plot_ratio(rows, out):                                                   # empirical R_m(n) vs theory
    qubits = sorted({r[0] for r in rows}); ms = sorted({r[1] for r in rows}) # axes
    fig, ax = plt.subplots(figsize=(5, 3.5))                                 # canvas
    for m in ms:                                                             # one line per m
        xs, ys, thy = [], [], []                                             # n, empirical, theoretical
        for n in qubits:                                                     # accumulate
            try:                                                             # may be missing if m > n
                vv = next(r[3] for r in rows if r[0] == n and r[1] == m and r[2] == "Vanilla")
                vq = next(r[3] for r in rows if r[0] == n and r[1] == m and r[2] == "Q-LINK(Fixed)")
            except StopIteration: continue                                   # skip missing
            xs.append(n); ys.append(vq / vv if vv > 0 else float('nan'))     # empirical ratio
            thy.append((n + 1) / max(n + 1 - m, 1))                          # theory R_m(n)
        ax.plot(xs, ys, marker="o", label=f"emp m={m}")                      # empirical
        ax.plot(xs, thy, linestyle="--", alpha=0.6, label=f"theory m={m}")   # theory
    ax.set_xlabel("$n_{data}$"); ax.set_ylabel("$\\mathcal{R}_m(n) = Var_{QLINK}/Var_{Vanilla}$")
    ax.set_yscale("log"); ax.set_title("E5: m-local trajectory ratio")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(out / "figures" / "e5_variance_ratio.pdf", dpi=150); plt.close(fig)


def main(qubits, ms, n_seeds, lr, mx, tol, out):                             # entry point
    (out / "data").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    t0 = time.time(); rows = sweep(qubits, ms, n_seeds, lr, mx, tol, out)    # sweep
    plot_ratio(rows, out); print(f"Total wall: {time.time() - t0:.1f}s")     # plot + timing

if __name__ == "__main__":                                                   # CLI
    p = argparse.ArgumentParser()                                            # standard argparse
    p.add_argument("--qubits", type=str, default="3,4,5,6,7"); p.add_argument("--m", type=str, default="1,2,3,4,5")
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.1); p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("replication_results")); a = p.parse_args()
    main([int(x) for x in a.qubits.split(",")], [int(x) for x in a.m.split(",")], a.seeds, a.lr, a.max_iters, a.tol, a.out)
