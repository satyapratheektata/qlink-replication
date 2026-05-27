# Part A Fig. 6: loss landscapes for n in {8, 9, 10}, 3 models each
import _lib.path                                                             # add ../src to sys.path
import argparse, time                                                        # CLI + timing
from pathlib import Path                                                     # filesystem
import matplotlib                                                            # backend
matplotlib.use("Agg")                                                        # headless
import matplotlib.pyplot as plt                                              # plotting
import numpy as np                                                           # arrays
import qulacs                                                                # state for landscape sampling
from _lib.init import init_params                                            # paper's N(0, 0.1) layout
from _lib.cell import reset_global_seed                                      # set seeds 42
from qlink_replication.architectures.qlink import build_qlink_circuit        # Fixed and Adaptive
from qlink_replication.architectures.vanilla import build_vanilla_circuit    # Vanilla baseline
from qlink_replication.config.geometry import yb_depth                       # depth schedule
from qlink_replication.core.observables import create_observable_fixed       # cost operator
from qlink_replication.core.state import generate_random_product_state       # input state

MODELS = ["Vanilla", "Q-LINK(Fixed)", "Q-LINK(Adaptive)"]                    # plot order

def _circuit(m, nd, d):                                                      # dispatch
    if m == "Vanilla": return build_vanilla_circuit(nd, d)                   # n_tot == n_data
    return build_qlink_circuit(nd, d, is_adaptive=(m == "Q-LINK(Adaptive)")) # n_tot == n_data + 1

def _loss(c, o, psi, params):                                                # one forward pass
    for p in range(len(params)): c.set_parameter(p, -float(params[p]))       # TC->qulacs sign flip
    s = qulacs.QuantumState(c.get_qubit_count()); s.load(psi.astype(np.complex128))  # load input
    c.update_quantum_state(s); return float(o.get_expectation_value(s).real) # measure cost

def _train(c, o, psi, params, lr, mx, tol):                                  # SGD to theta_trained
    for _ in range(mx):                                                      # max_iters cap
        L = _loss(c, o, psi, params)                                         # forward + loss
        if L < tol: break                                                    # early stop
        params -= lr * (-np.array(c.backprop(o)))                            # TC-frame step
    return params                                                            # return optimum

def _landscape(c, o, psi, theta, grid, span):                                # 2D surface around theta
    rng = np.random.default_rng(42)                                          # deterministic directions
    d1 = rng.standard_normal(len(theta)); d1 /= np.linalg.norm(d1)           # first axis
    d2 = rng.standard_normal(len(theta))                                     # second axis
    d2 -= np.dot(d1, d2) * d1; d2 /= np.linalg.norm(d2)                      # Gram-Schmidt + normalize
    axs = np.linspace(-span, span, grid); Z = np.empty((grid, grid))         # grid + buffer
    for i, a in enumerate(axs):                                              # outer (alpha)
        for j, b in enumerate(axs): Z[i, j] = _loss(c, o, psi, theta + a*d1 + b*d2)  # inner (beta)
    return axs, Z                                                            # symmetric on both axes

def _plot(axs, Z, title, out):                                               # 3D surface PDF
    A, B = np.meshgrid(axs, axs, indexing="ij"); fig = plt.figure(figsize=(4, 3))  # canvas
    ax = fig.add_subplot(111, projection="3d"); ax.plot_surface(A, B, Z, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$"); ax.set_zlabel("Loss"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)            # write + free

def main(qubits, grid, span, mx, lr, tol, out):                              # entry point
    d_out = out / "figures" / "landscape"; d_out.mkdir(parents=True, exist_ok=True)
    for n in qubits:                                                         # per qubit count
        nd = n - 1; d = yb_depth(nd)                                         # data qubits + depth
        for m in MODELS:                                                     # 3 models per row
            nt = n if m != "Vanilla" else n - 1; reset_global_seed(42)       # n_tot + global seed
            c = _circuit(m, nd, d); o = create_observable_fixed(nt, m)       # circuit + observable
            psi = generate_random_product_state(nd, nt, m, np.random.default_rng(0))
            params = init_params(m, nd, d, c.get_parameter_count())          # init
            t0 = time.time(); theta = _train(c, o, psi, params, lr, mx, tol) # train
            t1 = time.time(); axs, Z = _landscape(c, o, psi, theta, grid, span)  # sweep
            tag = m.replace("(", "_").replace(")", "").replace(" ", "")      # safe filename
            _plot(axs, Z, f"{m} - {n} qubits", d_out / f"n{n}_{tag}.pdf")    # write panel
            print(f"n={n} {m:18s} train={t1-t0:5.1f}s landscape={time.time()-t1:6.1f}s")

if __name__ == "__main__":                                                   # CLI
    p = argparse.ArgumentParser(); p.add_argument("--qubits", type=str, default="8,9,10")
    p.add_argument("--grid", type=int, default=200); p.add_argument("--span", type=float, default=3.0)
    p.add_argument("--max-iters", type=int, default=1500); p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-3); p.add_argument("--out", type=Path, default=Path("replication_results"))
    a = p.parse_args(); main([int(q) for q in a.qubits.split(",")], a.grid, a.span, a.max_iters, a.lr, a.tol, a.out)
