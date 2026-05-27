# one SGD run; returns (loss_per_iter, u_grads_per_iter, stop_iter)
import numpy as np                                                           # arrays
import qulacs                                                                # state + circuit


def run_single_seed(circuit, obs, psi_init, params, u_idx, lr, max_iters, tol):
    num_params = circuit.get_parameter_count()                               # gates to write each iter
    losses: list[float] = []                                                 # per-iter loss values
    u_grads: list[np.ndarray] = []                                           # per-iter u-only gradients
    stop = max_iters                                                         # default: didn't early-stop
    for it in range(max_iters):                                              # SGD loop
        for p_idx in range(num_params):                                      # write params into qulacs
            circuit.set_parameter(p_idx, -float(params[p_idx]))              # TC->qulacs sign flip
        s = qulacs.QuantumState(circuit.get_qubit_count())                   # fresh state
        s.load(psi_init.astype(np.complex128))                               # load initial state
        circuit.update_quantum_state(s)                                      # apply circuit
        loss = float(obs.get_expectation_value(s).real)                      # measure cost
        losses.append(loss)                                                  # record
        grad_tc = -np.array(circuit.backprop(obs))                           # qulacs grad -> TC grad
        u_grads.append(grad_tc[u_idx].copy())                                # record only u-rotation grads
        params -= lr * grad_tc                                               # vanilla SGD step
        if loss < tol:                                                       # early-stop test
            stop = it + 1; break                                             # +1 because iteration count is 1-indexed
    return losses, u_grads, stop                                             # caller aggregates over seeds
