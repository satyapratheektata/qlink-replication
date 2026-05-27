# one (n, model) cell: build circuit/observable/state, sweep seeds, return metrics
import random                                                                # seed RNG inside loop
import numpy as np                                                           # array stats


def reset_global_seed(seed: int) -> None:                                    # mirrors authors' main.py
    random.seed(seed); np.random.seed(seed)                                  # both stdlib and numpy RNGs


def aggregate_seeds(all_losses, all_grads, stop_iters):                      # padded mean+std + variance
    max_l = max(len(L) for L in all_losses)                                  # longest run determines pad length
    padded = np.array([L + [L[-1]] * (max_l - len(L)) for L in all_losses])  # repeat-last padding
    avg_l = padded.mean(axis=0); std_l = padded.std(axis=0)                  # per-iter mean and std
    min_g = min(len(g) for g in all_grads)                                   # shortest grad trajectory
    avg_g = np.mean([np.stack(g[:min_g]) for g in all_grads], axis=0)        # mean across seeds
    grad_var = float(np.var(avg_g))                                          # scalar variance over (iter, param)
    avg_iter = float(np.mean(stop_iters))                                    # convergence-efficiency input
    return avg_l, std_l, grad_var, avg_iter                                  # everything the loop slide needs
