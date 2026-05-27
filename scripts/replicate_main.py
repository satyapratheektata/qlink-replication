# Part A driver: Table I + Fig. 4 + Fig. 5 across n=5..10 with 5 seeds
import _lib.path                                                             # add ../src to sys.path
import argparse, csv, json, time                                             # CLI + outputs
from pathlib import Path                                                     # filesystem
import numpy as np                                                           # arrays
from _lib.init import init_params                                            # paper's N(0, 0.1) layout
from _lib.sgd import run_single_seed                                         # single SGD run
from _lib.cell import reset_global_seed, aggregate_seeds                     # seed sweep + summary
from _lib.plots import plot_loss_panel, plot_avgiter                         # Fig. 4 + Fig. 5
from qlink_replication.config.geometry import yb_depth                       # depth = ceil(n^2 log n)
from qlink_replication.core.expressibility import compute_expressibility     # Table I expressibility column
from qlink_replication.core.observables import create_observable_buggy, create_observable_fixed
from qlink_replication.core.state import generate_random_product_state       # per-seed input state
from qlink_replication.architectures.qlink import build_qlink_circuit, u_rotation_indices as q_idx
from qlink_replication.architectures.vanilla import build_vanilla_circuit, u_rotation_indices as v_idx

MODELS = ["Q-LINK(Fixed)", "Q-LINK(Adaptive)", "Vanilla"]                    # authors' order in main.py

def _circuit(model, n_data, depth):                                          # dispatch to the right architecture
    if model == "Vanilla": return build_vanilla_circuit(n_data, depth)       # n_tot = n_data
    return build_qlink_circuit(n_data, depth, is_adaptive=(model == "Q-LINK(Adaptive)"))

def _uidx(model, n_data, depth):                                             # u-rotation indices for variance mask
    if model == "Vanilla": return v_idx(n_data, depth)                       # everything is u in Vanilla
    return q_idx(n_data, depth, is_adaptive=(model == "Q-LINK(Adaptive)"))   # skips parametric Rxx in Adaptive

def main(qubit_list, n_seeds, max_iters, lr, tol, out, use_buggy):           # one entry point
    out.mkdir(parents=True, exist_ok=True); data = out / "data"; fig = out / "figures"  # output layout
    data.mkdir(exist_ok=True); fig.mkdir(exist_ok=True)                      # ensure both subdirs
    obs_factory = create_observable_buggy if use_buggy else create_observable_fixed     # paper vs reference-code cost
    rows = []; avgiter = {m: [] for m in MODELS}                             # collected outputs
    for n in qubit_list:                                                     # for each row of Table I
        stats = {}                                                           # per-model loss stats for this n
        for m in MODELS:                                                     # 3 models per row
            n_data = n - 1; n_tot = n if m != "Vanilla" else n - 1           # authors: data_qubits = n - 1 for both
            d = yb_depth(n_data); c = _circuit(m, n_data, d)                 # build circuit
            obs = obs_factory(n_tot, m); ui = _uidx(m, n_data, d)            # observable + u-indices
            losses, grads, stops = [], [], []                                # per-seed buffers
            reset_global_seed(42); t0 = time.time()                          # match authors' global seed
            for r in range(n_seeds):                                         # 5 seeds per cell
                psi = generate_random_product_state(n_data, n_tot, m, np.random.default_rng(np.random.randint(0, 2**31)))
                p = init_params(m, n_data, d, c.get_parameter_count())       # RNG-consuming init
                L, G, S = run_single_seed(c, obs, psi, p, ui, lr, max_iters, tol)  # SGD
                losses.append(L); grads.append(G); stops.append(S)           # accumulate
                print(f"  {m:18s} seed {r}: stop={S:4d} final_loss={L[-1]:.4f}")
            avg_l, std_l, var, avg_i = aggregate_seeds(losses, grads, stops) # summarize
            kl = compute_expressibility(c, num_fidelity=500, num_bins=20, rng=np.random.default_rng(42))
            rows.append((n, m, kl, var)); avgiter[m].append(avg_i); stats[m] = (avg_l, std_l)
            print(f"  {m:18s} | KL={kl:.3e} | Var={var:.3e} | avg_iter={avg_i:.1f} | wall {time.time()-t0:.1f}s")
        json.dump({"n": n, "models": {m: {"iteration": list(range(len(a))), "avg_loss": a.tolist(), "std_loss": s.tolist()} for m, (a, s) in stats.items()}}, open(data / f"n{n}_loss_comparison.json", "w"), indent=2)
        plot_loss_panel(n, stats, fig / f"n{n}_loss_comparison.pdf")         # Fig. 4 panel for this n
    csv.writer(open(data / "table1.csv", "w", newline="")).writerows([["Qubits","Model","Expressibility (KL)","Gradient Variance"], *((n, m, f"{kl:.4e}", f"{v:.4e}") for n, m, kl, v in rows)])
    json.dump({"qubit_list": qubit_list, "avgiter": avgiter}, open(data / "avgiter_vs_qubits.json", "w"), indent=2)
    plot_avgiter(qubit_list, avgiter, fig / "avgiter_vs_qubits.pdf")         # Fig. 5
    print(f"\nDone. Table I -> {data / 'table1.csv'}")                       # final breadcrumb

if __name__ == "__main__":                                                   # CLI entry point
    p = argparse.ArgumentParser()                                            # standard argparse
    p.add_argument("--qubits", type=str, default="5,6,7,8,9,10")             # authors' range
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--max-iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.1); p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("replication_results"))  # default sibling of scripts/
    p.add_argument("--cost", choices=["buggy", "fixed"], default="buggy")    # buggy matches paper Table I
    a = p.parse_args()                                                       # parse
    print(f"Using cost: {'BUGGY (matches authors Table I)' if a.cost == 'buggy' else 'FIXED (corrected)'}")
    main([int(q) for q in a.qubits.split(",")], a.seeds, a.max_iters, a.lr, a.tol, a.out, a.cost == "buggy")
