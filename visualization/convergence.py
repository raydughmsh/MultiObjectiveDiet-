"""
Member 5 — Hypervolume Convergence Curves
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_convergence(
    history: list,
    title: str = "Hypervolume Convergence",
    filename: str = None,
    color: str = "steelblue",
    label: str = None,
):
    """
    Plot hypervolume vs generation for a single algorithm run.
    history: list of dicts with keys 'gen' and 'hypervolume'
    """
    gens = [e["gen"] for e in history]
    hvs  = [e["hypervolume"] if e["hypervolume"] is not None else float("nan")
            for e in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, hvs, color=color, linewidth=1.8, label=label or title)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Hypervolume", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    if label:
        ax.legend(fontsize=10)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"[plot] saved -> {filename}")
    else:
        plt.show()
    plt.close(fig)


def plot_convergence_overlay(
    histories: list,
    labels: list,
    title: str = "Hypervolume Convergence Comparison",
    filename: str = None,
):
    """
    Overlay HV convergence curves from multiple runs.
    histories: list of history lists
    labels: list of str
    """
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (history, label) in enumerate(zip(histories, labels)):
        gens = [e["gen"] for e in history]
        hvs  = [e["hypervolume"] if e["hypervolume"] is not None else float("nan")
                for e in history]
        ax.plot(gens, hvs, color=colors[i % len(colors)], linewidth=1.8, label=label)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Hypervolume", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"[plot] saved -> {filename}")
    else:
        plt.show()
    plt.close(fig)
