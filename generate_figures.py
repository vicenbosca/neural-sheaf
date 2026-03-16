#!/usr/bin/env python3
"""
Generate all figures for §6 (Experiments) of
"Neural Networks as Local-to-Global Computations".

Each panel is saved as a separate PDF for LaTeX composition.
Requires neural_sheaf installed as editable package.

Usage:
    cd neural-sheaf
    python generate_figures.py              # all figures
    python generate_figures.py --fig 1      # convergence panels
    python generate_figures.py --fig 2      # training panels
    python generate_figures.py --fig 3      # beta scaling
    python generate_figures.py --fig 4      # diagnostics panels
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Library imports ──────────────────────────────────────────────────────────
from neural_sheaf.sheaf import NeuralSheaf
from neural_sheaf.dynamics import SheafDynamics
from neural_sheaf.trainer import SheafTrainer
from neural_sheaf.baseline import TraditionalNN
from neural_sheaf.datasets import (
    paraboloid, generate_regression_data, generate_circular_data,
)
from neural_sheaf.discord import compute_discord, extract_edge_data
from neural_sheaf.spectral import spectral_analysis_per_sample

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

C_BLUE   = "#2066a8"
C_RED    = "#c03830"
C_ORANGE = "#e87d28"
C_GREEN  = "#3a9a5b"
C_PURPLE = "#7b4ea3"
C_GREY   = "#555555"
C_GOLD   = "#d4a017"

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def _block_labels(block_ranges):
    """Convert block range labels like 'z1' to LaTeX '$z^{(1)}$'."""
    labels = []
    for lbl, _, _ in block_ranges:
        kind = lbl[0]
        idx = lbl[1:]
        labels.append(f"${kind}^{{({idx})}}$")
    return labels


def _bar_palette(n):
    """Return n colours cycling through the palette."""
    base = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED, C_GREY, C_GOLD]
    return [base[i % len(base)] for i in range(n)]


def _plot_eigvec_energy(energy, block_labels, ylabel, fname):
    """Bar chart of per-block eigenvector energy (mean ± std over samples)."""
    energy_mean = energy.mean(axis=1)
    energy_std  = energy.std(axis=1)
    n = len(block_labels)

    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * n + 1.5), 3.5))
    x_pos = np.arange(n)
    ax.bar(x_pos, energy_mean, yerr=energy_std,
           color=_bar_palette(n), edgecolor="white", linewidth=0.6,
           capsize=4.5, error_kw=dict(linewidth=1.2, color="black"))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(block_labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, min(1.05, energy_mean.max() + energy_std.max() + 0.1))
    _save(fig, fname)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Convergence: total discord + output stalk
# ═══════════════════════════════════════════════════════════════════════════════

def figure1():
    print("--- Figure 1: Convergence ---")

    torch.manual_seed(7)
    arch = [2, 4, 1]
    sheaf = NeuralSheaf(arch, output_activation="identity", seed=7)

    x = torch.tensor([[0.5], [-0.3]], dtype=torch.float64)
    fwd_out, _ = sheaf.forward(x)
    target_val = fwd_out.item()

    dynamics = SheafDynamics(sheaf, alpha=1.0, dt=0.01)
    torch.manual_seed(42)
    state = sheaf.init_stalks(x, method="random")

    max_iter = 10_000
    tol = 1e-15

    iters, total_disc, out_stalk = [], [], []
    crossing_iters = []

    def _record(st, it):
        disc = compute_discord(sheaf, st, x)
        iters.append(it)
        total_disc.append(disc["total"])
        out_stalk.append(st["z"][-1].item())

    _record(state, 0)

    for it in range(1, max_iter + 1):
        old_z = [z.clone() for z in state["z"]]
        new_state = dynamics.step(state, x)

        for ell in range(sheaf.k):
            if ((old_z[ell][:, 0] >= 0) != (new_state["z"][ell][:, 0] >= 0)).any():
                crossing_iters.append(it)
                break

        state = new_state
        do_rec = (it <= 200) or (it <= 2000 and it % 5 == 0) or (it % 20 == 0)
        if do_rec:
            _record(state, it)

        mc = max((state["z"][i] - old_z[i]).abs().max().item()
                 for i in range(len(old_z)))
        if mc < tol:
            if iters[-1] != it:
                _record(state, it)
            print(f"  Converged at iteration {it}")
            break

    iters = np.array(iters)
    total_disc = np.maximum(np.array(total_disc), 1e-22)
    out_stalk = np.array(out_stalk)
    crossing_iters = np.array(crossing_iters)

    half = iters[-1] // 2
    mask = iters <= half

    # ── Total discord ──
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.semilogy(iters[mask], total_disc[mask], color=C_BLUE, linewidth=2.0)
    for ci in crossing_iters[crossing_iters <= half]:
        ax.axvline(ci, color=C_RED, alpha=0.25, linewidth=1.2)
    ax.plot([], [], color=C_RED, alpha=0.6, linewidth=3.0,
            label="ReLU boundary crossing")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total discord")
    ymin_data = total_disc[mask].min()
    ax.set_ylim(bottom=max(ymin_data * 0.1, 1e-14))
    ax.legend(loc="upper right", frameon=False)
    _save(fig, "convergence_total_discord.pdf")

    # ── Output stalk trajectory ──
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(iters[mask], out_stalk[mask], color=C_BLUE, linewidth=2.0,
            label="Output stalk")
    ax.axhline(target_val, color="black", linestyle="--", linewidth=1.3,
               label="Forward-pass value")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Output stalk value")
    ax.legend(loc="center right", frameon=False)
    _save(fig, "convergence_output_stalk.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Training: separate PDFs, shared y-limits per task
# ═══════════════════════════════════════════════════════════════════════════════

def figure2():
    print("--- Figure 2: Training ---")

    # ── Paraboloid regression ──
    X_tr, Y_tr, X_te, Y_te = generate_regression_data(
        paraboloid, n_train=300, n_test=100, seed=42)

    print("  Paraboloid -- sheaf training...")
    sheaf_p = NeuralSheaf([2, 30, 1], output_activation="identity", seed=0)
    trainer_p = SheafTrainer(sheaf_p, alpha=1.0, beta=1.0/300, dt=0.005)
    hist_sp = trainer_p.train(X_tr, Y_tr, X_te, Y_te,
                              epochs=100, steps_per_epoch=1000,
                              warm_start=True, init_method="random", seed=42)

    print("  Paraboloid -- SGD training...")
    sgd_p = TraditionalNN([2, 30, 1], learning_rate=0.005,
                          output_activation="identity", seed=0)
    hist_gp = sgd_p.train(X_tr, Y_tr, X_te, Y_te,
                          epochs=10000, track_freq=100)

    all_p = (list(hist_sp["train_loss"]) + list(hist_sp["test_loss"]) +
             list(hist_gp["train_loss"]) + list(hist_gp["test_loss"]))
    p_ymin = max(min(all_p) * 0.5, 1e-6)
    p_ymax = max(all_p) * 2.0

    # ── Circular classification ──
    Xc_tr, Yc_tr, Xc_te, Yc_te = generate_circular_data(
        n_train=300, n_test=100, seed=42)

    print("  Circular -- sheaf training...")
    sheaf_c = NeuralSheaf([2, 25, 1], output_activation="sigmoid", seed=0)
    trainer_c = SheafTrainer(sheaf_c, alpha=1.0, beta=1.0/300, dt=0.005)
    hist_sc = trainer_c.train(Xc_tr, Yc_tr, Xc_te, Yc_te,
                              epochs=100, steps_per_epoch=1000,
                              warm_start=False, init_method="forward_pass",
                              seed=42)

    print("  Circular -- SGD training...")
    sgd_c = TraditionalNN([2, 25, 1], learning_rate=0.01,
                          output_activation="sigmoid", seed=0)
    hist_gc = sgd_c.train(Xc_tr, Yc_tr, Xc_te, Yc_te,
                          epochs=10000, track_freq=100)

    all_c = (list(hist_sc["train_loss"]) + list(hist_sc["test_loss"]) +
             list(hist_gc["train_loss"]) + list(hist_gc["test_loss"]))
    c_ymin = max(min(all_c) * 0.5, 1e-6)
    c_ymax = max(all_c) * 2.0

    # ── Paraboloid: sheaf curve ──
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    sx = np.array(hist_sp["epoch_steps"])
    ax.semilogy(sx, hist_sp["train_loss"], color=C_BLUE, linewidth=2.0, label="Train")
    ax.semilogy(sx, hist_sp["test_loss"], color=C_BLUE, linewidth=2.0, linestyle="--", label="Test")
    ax.set_xlabel("Euler step"); ax.set_ylabel("Loss (MSE)")
    ax.set_title("Sheaf — Paraboloid"); ax.set_ylim(p_ymin, p_ymax)
    ax.legend(frameon=False); _save(fig, "train_paraboloid_sheaf_curve.pdf")

    # ── Paraboloid: SGD curve ──
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    gx = np.array(hist_gp["epoch"])
    ax.semilogy(gx, hist_gp["train_loss"], color=C_RED, linewidth=2.0, label="Train")
    ax.semilogy(gx, hist_gp["test_loss"], color=C_RED, linewidth=2.0, linestyle="--", label="Test")
    ax.set_xlabel("SGD epoch"); ax.set_ylabel("Loss (MSE)")
    ax.set_title("SGD — Paraboloid"); ax.set_ylim(p_ymin, p_ymax)
    ax.legend(frameon=False); _save(fig, "train_paraboloid_sgd_curve.pdf")

    # ── Paraboloid surfaces ──
    gn = 60
    xx = np.linspace(-2, 2, gn); yy = np.linspace(-2, 2, gn)
    XX, YY = np.meshgrid(xx, yy)
    Xg = torch.tensor(np.stack([XX.ravel(), YY.ravel()]), dtype=torch.float64)
    Zt = paraboloid(Xg).numpy().reshape(gn, gn)
    with torch.no_grad():
        Zs = trainer_p.predict(Xg).numpy().reshape(gn, gn)
        Zg = sgd_p.predict(Xg).numpy().reshape(gn, gn)
    vmin = min(Zt.min(), Zs.min(), Zg.min())
    vmax = max(Zt.max(), Zs.max(), Zg.max())
    levels = np.linspace(vmin, vmax, 25)

    for Z, label, fname in [(Zt, "Target", "surface_paraboloid_target.pdf"),
                             (Zs, "Sheaf",  "surface_paraboloid_sheaf.pdf"),
                             (Zg, "SGD",    "surface_paraboloid_sgd.pdf")]:
        fig, ax = plt.subplots(figsize=(4.3, 3.8))
        cf = ax.contourf(XX, YY, Z, levels=levels, cmap="RdYlBu_r", extend="both")
        cb = fig.colorbar(cf, ax=ax, pad=0.03, fraction=0.046)
        cb.set_label(r"$f(x_1, x_2)$", fontsize=10); cb.ax.tick_params(labelsize=9)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_title(label); ax.set_aspect("equal"); _save(fig, fname)

    # ── Circular: sheaf curve ──
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    scx = np.array(hist_sc["epoch_steps"])
    ax.semilogy(scx, hist_sc["train_loss"], color=C_BLUE, linewidth=2.0, label="Train")
    ax.semilogy(scx, hist_sc["test_loss"], color=C_BLUE, linewidth=2.0, linestyle="--", label="Test")
    ax.set_xlabel("Euler step"); ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title("Sheaf — Circular"); ax.set_ylim(c_ymin, c_ymax)
    ax.legend(frameon=False); _save(fig, "train_circular_sheaf_curve.pdf")

    # ── Circular: SGD curve ──
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    gcx = np.array(hist_gc["epoch"])
    ax.semilogy(gcx, hist_gc["train_loss"], color=C_RED, linewidth=2.0, label="Train")
    ax.semilogy(gcx, hist_gc["test_loss"], color=C_RED, linewidth=2.0, linestyle="--", label="Test")
    ax.set_xlabel("SGD epoch"); ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title("SGD — Circular"); ax.set_ylim(c_ymin, c_ymax)
    ax.legend(frameon=False); _save(fig, "train_circular_sgd_curve.pdf")

    # ── Circular boundaries ──
    gc = 80; rmax = 3.5
    xxc = np.linspace(-rmax, rmax, gc); yyc = np.linspace(-rmax, rmax, gc)
    XXc, YYc = np.meshgrid(xxc, yyc)
    Xcg = torch.tensor(np.stack([XXc.ravel(), YYc.ravel()]), dtype=torch.float64)
    with torch.no_grad():
        Zcs = trainer_c.predict(Xcg).numpy().reshape(gc, gc)
        Zcg = sgd_c.predict(Xcg).numpy().reshape(gc, gc)
    Yn = Yc_tr.numpy().ravel(); Xn = Xc_tr.numpy()

    for Zc, bc, lab, fn in [(Zcs, C_BLUE, "Sheaf", "boundary_circular_sheaf.pdf"),
                             (Zcg, C_RED,  "SGD",   "boundary_circular_sgd.pdf")]:
        fig, ax = plt.subplots(figsize=(4.3, 3.8))
        ax.scatter(Xn[0, Yn==0], Xn[1, Yn==0], c=C_BLUE, alpha=0.45, s=14, edgecolors="none", label="Class 0")
        ax.scatter(Xn[0, Yn==1], Xn[1, Yn==1], c=C_ORANGE, alpha=0.45, s=14, edgecolors="none", label="Class 1")
        ax.contour(XXc, YYc, Zc, levels=[0.5], colors=[bc], linewidths=2.5)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_title(lab)
        ax.set_aspect("equal"); ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax)
        ax.legend(loc="upper right", fontsize=9, frameon=False, markerscale=1.5)
        _save(fig, fn)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Beta Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def figure3():
    print("--- Figure 3: Beta Scaling ---")

    n_train = 300
    beta_values = np.array([
        1e-5, 3e-5, 1e-4, 3e-4,
        1e-3, 2e-3, 3.33e-3,
        5e-3, 1e-2, 3e-2, 1e-1,
    ])

    X_tr, Y_tr, X_te, Y_te = generate_regression_data(
        paraboloid, n_train=n_train, n_test=100, seed=42)

    n_seeds = 3
    final_losses = np.zeros((len(beta_values), n_seeds))

    for bi, beta in enumerate(beta_values):
        for si in range(n_seeds):
            sheaf = NeuralSheaf([2, 30, 1], output_activation="identity",
                                seed=si * 100)
            trainer = SheafTrainer(sheaf, alpha=1.0, beta=beta, dt=0.005)
            hist = trainer.train(
                X_tr, Y_tr, X_te, Y_te,
                epochs=100, steps_per_epoch=1000,
                warm_start=True, init_method="random", seed=42 + si)
            final_losses[bi, si] = hist["test_loss"][-1]
            print(f"    beta={beta:.1e}  seed={si}  "
                  f"test_loss={final_losses[bi, si]:.4f}")

    betaN = beta_values * n_train
    mean_loss = final_losses.mean(axis=1)
    std_loss = final_losses.std(axis=1)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(betaN, mean_loss, yerr=std_loss,
                fmt="o-", color=C_BLUE, markersize=6.5, capsize=4,
                linewidth=2.2, markeredgecolor="white", markeredgewidth=0.7,
                capthick=1.3, elinewidth=1.3,
                label=r"$n_{\mathrm{train}} = " + str(n_train) + "$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\beta \, n_{\mathrm{train}}$", fontsize=13)
    ax.set_ylabel("Final test loss (MSE)", fontsize=12)

    xmin, xmax = betaN.min() * 0.5, betaN.max() * 2.0
    bnd_l, bnd_r = 0.3, 10.0
    ax.set_xlim(xmin, xmax)

    ax.axvspan(xmin, bnd_l, alpha=0.07, color=C_RED, zorder=0)
    ax.axvspan(bnd_l, bnd_r, alpha=0.07, color=C_GREEN, zorder=0)
    ax.axvspan(bnd_r, xmax, alpha=0.07, color=C_ORANGE, zorder=0)
    ax.axvline(1.0, color=C_GREY, linestyle="--", linewidth=1.3, zorder=1)

    ylo, yhi = ax.get_ylim()
    label_y = yhi * 0.55

    # Regime labels — tuned positions
    xc_stag = np.sqrt(xmin * bnd_l) * 1.6      # rightward
    xc_opt  = np.sqrt(bnd_l * bnd_r) * 1.15     # slightly left of previous
    xc_div  = np.sqrt(bnd_r * xmax) * 0.7       # leftward

    ax.text(xc_stag, label_y, "Stagnation", fontsize=10.5,
            color=C_RED, ha="center", va="center", weight="bold", alpha=0.9)
    ax.text(xc_opt, label_y, "Optimal", fontsize=10.5,
            color=C_GREEN, ha="center", va="center", weight="bold", alpha=0.9)
    ax.text(xc_div, label_y, "Divergence", fontsize=10.5,
            color=C_ORANGE, ha="center", va="center", weight="bold", alpha=0.9)

    ax.text(1.15, label_y * 0.55, r"$\beta = 1/n_{\mathrm{train}}$",
            fontsize=10.5, color=C_GREY, ha="left", va="center")

    ax.legend(loc="lower left", frameon=False, fontsize=10)
    _save(fig, "beta_scaling.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Diagnostics: spectral + discord for 1H and 3H networks
# ═══════════════════════════════════════════════════════════════════════════════

def _train_and_snapshot(arch, dt, n_epochs, tag, X_tr, Y_tr, X_te, Y_te):
    """Train a sheaf network and return (sheaf, final_state, hist)."""
    print(f"  Training {tag} sheaf {arch}, {n_epochs}k steps...")
    sheaf = NeuralSheaf(arch, output_activation="identity", seed=0)
    trainer = SheafTrainer(sheaf, alpha=1.0, beta=1.0/300, dt=dt)
    hist, state = trainer.train(
        X_tr, Y_tr, X_te, Y_te,
        epochs=n_epochs, steps_per_epoch=1000,
        warm_start=True, init_method="random", seed=42,
        return_state=True)
    print(f"    Final test loss: {hist['test_loss'][-1]:.6f}")
    return sheaf, state, hist


def _plot_discord_edges(sheaf, state, tag):
    """Plot weight discord and ReLU discord per edge for a given sheaf."""
    weight_edges, relu_edges = extract_edge_data(sheaf, state)
    k = sheaf.k

    # Weight edges: discord = (Wa+b - z)_j  vs  z_j
    for ell, ed in enumerate(weight_edges):
        li = ell + 1
        actual = ed["actual"]
        discord = ed["discord"]       # pred - actual = (Wa+b - z)
        active = (actual >= 0)

        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        ax.scatter(actual[~active], discord[~active],
                   c=C_PURPLE, alpha=0.55, s=12, edgecolors="none",
                   label=r"Inactive ($z_j < 0$)", zorder=2, rasterized=True)
        ax.scatter(actual[active], discord[active],
                   c=C_ORANGE, alpha=0.55, s=12, edgecolors="none",
                   label=r"Active ($z_j \geq 0$)", zorder=2, rasterized=True)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, zorder=1)
        ax.set_xlabel(r"$z_j^{(" + str(li) + r")}$")
        ax.set_ylabel(
            r"$(W^{(" + str(li) + r")} a^{(" + str(li-1)
            + r")} + b^{(" + str(li) + r")})_j - z_j^{("
            + str(li) + r")}$"
        )
        ax.set_title(f"Weight edge {li}")
        ax.legend(loc="best", frameon=False, markerscale=2)
        _save(fig, f"diagnostics_{tag}_weight_discord_edge{li}.pdf")

    # ReLU edges: ReLU(z) - a  vs  z_j
    for ell, ed in enumerate(relu_edges):
        li = ell + 1
        z_vals = ed["z"]
        disc = ed["discord"]
        active = ed["active"]

        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        ax.scatter(z_vals[~active], disc[~active],
                   c=C_PURPLE, alpha=0.55, s=12, edgecolors="none",
                   label=r"Inactive ($z_j < 0$)", zorder=2, rasterized=True)
        ax.scatter(z_vals[active], disc[active],
                   c=C_ORANGE, alpha=0.55, s=12, edgecolors="none",
                   label=r"Active ($z_j \geq 0$)", zorder=2, rasterized=True)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, zorder=1)
        ax.set_xlabel(r"$z_j^{(" + str(li) + r")}$")
        ax.set_ylabel(
            r"$\mathrm{ReLU}(z_j^{(" + str(li)
            + r")}) - a_j^{(" + str(li) + r")}$"
        )
        ax.set_title(f"ReLU edge {li}")
        ax.legend(loc="best", frameon=False, markerscale=2)
        _save(fig, f"diagnostics_{tag}_relu_discord_edge{li}.pdf")


def figure4():
    print("--- Figure 4: Diagnostics ---")

    X_tr, Y_tr, X_te, Y_te = generate_regression_data(
        paraboloid, n_train=300, n_test=100, seed=42)

    # Diagnostic inputs for spectral analysis
    torch.manual_seed(99)
    n_diag = 50
    X_diag = torch.rand(2, n_diag, dtype=torch.float64) * 4 - 2

    # ── 1H network: [2, 30, 1] ──
    sheaf_1h, state_1h, _ = _train_and_snapshot(
        [2, 30, 1], dt=0.005, n_epochs=100, tag="1H",
        X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te)

    print("  1H: spectral analysis...")
    spec_1h = spectral_analysis_per_sample(sheaf_1h, X_diag)
    labels_1h = _block_labels(spec_1h["block_ranges"])

    _plot_eigvec_energy(spec_1h["fiedler_layer_energy"], labels_1h,
                        "Fraction of Fiedler energy",
                        "diagnostics_1H_fiedler_energy.pdf")
    _plot_eigvec_energy(spec_1h["max_layer_energy"], labels_1h,
                        r"Fraction of $\lambda_{\max}$ energy",
                        "diagnostics_1H_lambdamax_energy.pdf")

    print("  1H: discord plots...")
    _plot_discord_edges(sheaf_1h, state_1h, "1H")

    # ── 3H network: [2, 12, 8, 6, 1] ──
    sheaf_3h, state_3h, _ = _train_and_snapshot(
        [2, 12, 8, 6, 1], dt=0.05, n_epochs=200, tag="3H",
        X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te)

    print("  3H: spectral analysis...")
    spec_3h = spectral_analysis_per_sample(sheaf_3h, X_diag)
    labels_3h = _block_labels(spec_3h["block_ranges"])

    _plot_eigvec_energy(spec_3h["fiedler_layer_energy"], labels_3h,
                        "Fraction of Fiedler energy",
                        "diagnostics_3H_fiedler_energy.pdf")
    _plot_eigvec_energy(spec_3h["max_layer_energy"], labels_3h,
                        r"Fraction of $\lambda_{\max}$ energy",
                        "diagnostics_3H_lambdamax_energy.pdf")

    print("  3H: discord plots...")
    _plot_discord_edges(sheaf_3h, state_3h, "3H")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate section 6 figures for the paper.")
    parser.add_argument("--fig", type=int, default=None,
                        help="Generate only this figure group (1-4).")
    args = parser.parse_args()

    dispatch = {1: figure1, 2: figure2, 3: figure3, 4: figure4}

    if args.fig is not None:
        if args.fig not in dispatch:
            print(f"Error: --fig must be 1, 2, 3, or 4 (got {args.fig})")
            sys.exit(1)
        dispatch[args.fig]()
    else:
        for k in sorted(dispatch):
            dispatch[k]()

    print("\nDone.")