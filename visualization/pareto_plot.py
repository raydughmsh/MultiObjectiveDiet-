"""
Member 5 — Pareto Front Scatter Plots
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pareto_2d(
    F: np.ndarray,
    title: str = "Pareto Front",
    filename: str = None,
    x_col: int = 1,
    y_col: int = 0,
    xlabel: str = "Cost",
    ylabel: str = "Preference (higher = better)",
):
    """
    2-D scatter: cost (x) vs preference (y) for one Pareto front.
    F shape: (n, 3) — [neg_preference, cost, time]
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    x = F[:, x_col]
    y = -F[:, y_col]
    ax.scatter(x, y, c="steelblue", edgecolors="white", s=60, zorder=3)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"[plot] saved -> {filename}")
    else:
        plt.show()
    plt.close(fig)


def plot_pareto_3d(
    F: np.ndarray,
    title: str = "Pareto Front (3D)",
    filename: str = None,
):
    """
    3-D scatter of all 3 objectives.
    F shape: (n, 3) — [neg_preference, cost, time]
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(-F[:, 0], F[:, 1], F[:, 2], c="steelblue", edgecolors="white", s=50)
    ax.set_xlabel("Preference", fontsize=9)
    ax.set_ylabel("Cost", fontsize=9)
    ax.set_zlabel("Time", fontsize=9)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"[plot] saved -> {filename}")
    else:
        plt.show()
    plt.close(fig)


def plot_pareto_overlay(
    F_list: list,
    labels: list,
    title: str = "Pareto Front Comparison",
    filename: str = None,
    x_col: int = 1,
    y_col: int = 0,
    xlabel: str = "Cost",
    ylabel: str = "Preference (higher = better)",
):
    """
    Overlay multiple Pareto fronts on one 2-D scatter plot.
    F_list: list of np.ndarray each shape (n_i, 3)
    labels: list of str
    """
    colors  = ["steelblue", "tomato", "seagreen", "darkorange"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (F, label) in enumerate(zip(F_list, labels)):
        ax.scatter(F[:, x_col], -F[:, y_col],
                   label=label,
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   edgecolors="white", s=60, alpha=0.85, zorder=3)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
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
