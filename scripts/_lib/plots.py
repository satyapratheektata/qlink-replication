# matplotlib helpers for Fig. 4 (loss vs iter) and Fig. 5 (avgiter vs qubits)
from pathlib import Path                                                     # path type hint
import matplotlib                                                            # backend selection
matplotlib.use("Agg")                                                        # headless figures
import matplotlib.pyplot as plt                                              # plotting API
import numpy as np                                                           # arrays for fill_between

PALETTE = {"Q-LINK(Fixed)": "orange", "Q-LINK(Adaptive)": "green", "Vanilla": "blue"}  # paper colors


def plot_loss_panel(n: int, stats: dict, path: Path) -> None:                # one Fig. 4 panel
    fig, ax = plt.subplots(figsize=(4, 3))                                   # square-ish for slide
    for model, (avg, std) in stats.items():                                  # mean +/- std per model
        it = np.arange(len(avg))                                             # x axis
        ax.plot(it, avg, color=PALETTE[model], label=model)                  # mean line
        ax.fill_between(it, avg - std, avg + std,                            # std band
                        color=PALETTE[model], alpha=0.2)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")                        # labels
    ax.set_title(f"{n} qubits"); ax.legend(fontsize=8)                       # legend on slide
    fig.tight_layout(); fig.savefig(path); plt.close(fig)                    # write + free


def plot_avgiter(qubits: list, avgiter: dict, path: Path) -> None:           # Fig. 5 single plot
    fig, ax = plt.subplots(figsize=(4, 3))                                   # match Fig. 4 size
    for model, values in avgiter.items():                                    # per-model scatter+line
        ax.scatter(qubits, values, color=PALETTE[model], label=model)        # markers
        ax.plot(qubits, values, color=PALETTE[model], alpha=0.5)             # connecting line
    ax.set_xlabel("Qubits"); ax.set_ylabel("Avg iterations to converge")     # labels
    ax.legend(fontsize=8); fig.tight_layout()                                # cleanup
    fig.savefig(path); plt.close(fig)                                        # write + free
