#!/usr/bin/env python3
"""
generate_appendix_figures.py — Appendix E figures and tables.
Requires neural_sheaf installed as editable package.

Place at the repo root and run:

    python generate_appendix_figures.py

Outputs:  figures/appendix/*.pdf   tables/*.tex
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from neural_sheaf.sheaf import NeuralSheaf
from neural_sheaf.dynamics import SheafDynamics, track_trajectory, detect_mask_changes
from neural_sheaf.trainer import SheafTrainer
from neural_sheaf.baseline import TraditionalNN
from neural_sheaf.pinning import HardPin, SoftPin
from neural_sheaf.discord import (
    compute_discord, compute_training_discord,
    extract_edge_data, compute_pinned_discord,
)
from neural_sheaf.spectral import spectral_analysis_per_sample
from neural_sheaf.tasks import (
    TASK_CONFIGS, snapshot_sheaf, nn_to_sheaf,
    train_sheaf_full, train_sgd_full,
)
from neural_sheaf.datasets import (
    paraboloid, saddle,
    generate_regression_data,
    generate_circular_data,
    generate_blob_data,
)

torch.set_default_dtype(torch.float64)
warnings.filterwarnings('ignore', category=UserWarning)

FIG_DIR = Path('figures/appendix')
TABLE_DIR = Path('tables')
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Style
# =====================================================================

def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': 1.8,
        'axes.grid': False,
        'text.usetex': False,
        'mathtext.fontset': 'cm',
    })

# Palette: exact match with generate_figures.py (main text)
C_BLUE   = '#2066a8'   # sheaf / primary
C_RED    = '#c03830'    # SGD / secondary
C_ORANGE = '#e87d28'    # active neurons / layer accent
C_GREEN  = '#3a9a5b'
C_PURPLE = '#7b4ea3'    # inactive neurons
C_GREY   = '#555555'
C_GOLD   = '#d4a017'

# Semantic aliases
C_SHEAF  = C_BLUE
C_SGD    = C_RED
C_TOTAL  = C_BLUE
C_WEIGHT = '#e377c2'
C_ACTIV  = C_GREEN
C_OUTPUT = C_RED
C_FWD    = 'black'      # forward-pass reference lines = black dashed (main text)
C_ACTIVE   = C_ORANGE
C_INACTIVE = C_PURPLE
LAYER_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED, C_GREY, C_GOLD]


def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    print(f"    -> {path}")
    plt.close(fig)


def _save_table(lines, name):
    p = TABLE_DIR / name
    p.write_text('\n'.join(lines))
    print(f"    -> {p}")

def _max_change(old, new):
    """Max absolute stalk change between two states (for convergence check)."""
    mc = 0.0
    for zo, zn in zip(old['z'], new['z']):
        mc = max(mc, (zn - zo).abs().max().item())
    for ao, an in zip(old['a'][1:], new['a'][1:]):
        mc = max(mc, (an - ao).abs().max().item())
    if 'a_output' in old:
        mc = max(mc, (new['a_output'] - old['a_output']).abs().max().item())
    return mc

# =====================================================================
# Shared helpers
# =====================================================================

def _run_full_trajectory(sheaf, x, *, alpha=1.0, dt=0.01, tol=1e-12,
                         max_iter=200_000, seed=42):
    dynamics = SheafDynamics(sheaf, alpha=alpha, dt=dt)
    torch.manual_seed(seed)
    state = sheaf.init_stalks(x, method='random')
    x2 = x.to(dtype=sheaf.dtype)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)

    total, by_type, by_layer, out_vals = [], \
        {k: [] for k in ('weight', 'activation', 'output')}, \
        {ell: [] for ell in range(1, sheaf.k + 2)}, []

    def _rec(st):
        d = compute_discord(sheaf, st, x2)
        total.append(d['total'])
        by_type['weight'].append(sum(v[0] for v in d['weight_edges'].values()))
        by_type['activation'].append(sum(v[0] for v in d['activation_edges'].values()))
        by_type['output'].append(d['output_edge'][0] if d['output_edge'] else 0.0)
        for ell in by_layer:
            by_layer[ell].append(d['by_layer'].get(ell, 0.0))
        o = st['a_output'] if 'a_output' in st else st['z'][-1]
        out_vals.append(o[:, 0].detach().clone().numpy())

    _rec(state)
    for it in range(1, max_iter + 1):
        ns = dynamics.step(state, x2)
        mc = _max_change(state, ns)
        state = ns
        _rec(state)
        if mc < tol:
            break

    return dict(
        total=np.array(total),
        by_type={k: np.array(v) for k, v in by_type.items()},
        by_layer={k: np.array(v) for k, v in by_layer.items()},
        output_vals=np.array(out_vals),
        iterations=it, final_state=state,
    )


def _train_pair(arch, task_key, n_steps, dt, seed=42):
    cfg = TASK_CONFIGS[task_key]
    X_tr, Y_tr, X_te, Y_te = cfg['data_fn']()
    n_train = X_tr.shape[1]
    beta = 1.0 / n_train

    sheaf = NeuralSheaf(arch, output_activation=cfg['output_activation'], seed=seed)
    trainer = SheafTrainer(sheaf, alpha=1.0, beta=beta, dt=dt)
    epochs = max(n_steps // 1000, 1)
    spe = min(n_steps, 1000)
    h_s = trainer.train(
        X_tr, Y_tr, X_te, Y_te,
        epochs=epochs, steps_per_epoch=spe,
        warm_start=(cfg['task_type'] == 'regression'),
        init_method=cfg['init_method'], seed=seed,
    )

    oa = None if cfg['output_activation'] == 'identity' else cfg['output_activation']
    nn = TraditionalNN(arch, learning_rate=cfg['sgd_lr'], output_activation=oa, seed=seed)
    h_g = nn.train(X_tr, Y_tr, X_te, Y_te, epochs=n_steps,
                   track_freq=max(n_steps // 100, 1))

    return dict(trainer=trainer, nn=nn, sheaf=sheaf, h_s=h_s, h_g=h_g,
                X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te, cfg=cfg)


# =====================================================================
# E.1  Convergence dashboard — deep sigmoid [2,6,4,4,1]
# =====================================================================

def figure_E1():
    print("\n[Fig E.1] 3-hidden-layer sigmoid convergence dashboard ...")
    arch = [2, 6, 4, 4, 1]
    sheaf = NeuralSheaf(arch, output_activation='sigmoid', seed=0)
    x = torch.randn(2, 1)
    fwd_out, _ = sheaf.forward(x)
    fwd_val = fwd_out[0, 0].item()

    r = _run_full_trajectory(sheaf, x, dt=0.01, tol=1e-12,
                             max_iter=62_500, seed=42)
    it = np.arange(len(r['total']))

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))

    ax = axes[0, 0]
    ax.semilogy(it, r['total'], color=C_TOTAL)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Total discord')
    ax.set_title('Total discord (log scale)')
    _clean(ax)

    ax = axes[0, 1]
    ax.semilogy(it, r['by_type']['weight'],     label='Weight',     color=C_WEIGHT)
    ax.semilogy(it, r['by_type']['activation'], label='Activation', color=C_ACTIV)
    ax.semilogy(it, r['by_type']['output'],     label='Output',     color=C_OUTPUT)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Discord')
    ax.set_title('Discord by edge type')
    ax.legend(frameon=False); _clean(ax)

    ax = axes[1, 0]
    for ell in range(1, sheaf.k + 2):
        lbl = f'Layer {ell}' if ell <= sheaf.k else 'Output layer'
        ax.semilogy(it, r['by_layer'][ell], label=lbl,
                     color=LAYER_COLORS[(ell - 1) % len(LAYER_COLORS)])
    ax.set_xlabel('Iteration'); ax.set_ylabel('Discord')
    ax.set_title('Discord by layer')
    ax.legend(frameon=False, fontsize=7); _clean(ax)

    ax = axes[1, 1]
    ax.plot(it, r['output_vals'][:, 0], color=C_SHEAF, label='Sheaf output')
    ax.axhline(fwd_val, color=C_FWD, ls='--', lw=1.3, label='Forward-pass value')
    ax.set_xlabel('Iteration'); ax.set_ylabel('$\\sigma(z^{(4)})$')
    ax.set_title('Output stalk convergence')
    ax.legend(frameon=False); _clean(ax)

    fig.suptitle(
        f'Convergence: {arch}, sigmoid output  '
        f'({r["iterations"]:,} iterations)',
        fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, 'convergence_deep_sigmoid.pdf')
    print(f"      converged in {r['iterations']:,} iters")


# =====================================================================
# E.2  Phase-plane dynamics
# =====================================================================

def figure_E2():
    print("\n[Fig E.2] Phase-plane dynamics ...")
    archs = [[2, 4, 1], [2, 6, 4, 1], [2, 6, 4, 4, 1]]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, arch in enumerate(archs):
        ax = axes[idx]

        # Try several random inputs; pick one with >4 crossings
        # but not too many (which looks messy).
        best_data = None
        best_count = 0
        TARGET_MIN = 5
        for trial_seed in range(40):
            sheaf = NeuralSheaf(arch, seed=0)
            torch.manual_seed(trial_seed * 7 + 3)
            x = torch.randn(2, 1)
            _, ints = sheaf.forward(x)
            fwd_z1 = ints['z'][0][:, 0].detach().numpy()

            dynamics = SheafDynamics(sheaf, alpha=1.0, dt=0.01)
            states, dh = track_trajectory(
                dynamics, x, max_iter=100_000, tol=1e-12, freq=1,
                init_method='random', seed=trial_seed,
            )
            ci, cj = 0, min(1, arch[1] - 1)
            zi = np.array([s['z'][0][ci, 0].item() for s in states])
            zj = np.array([s['z'][0][cj, 0].item() for s in states])
            cross_i = np.where(np.diff(np.sign(zi)) != 0)[0]
            cross_j = np.where(np.diff(np.sign(zj)) != 0)[0]
            crosses = np.unique(np.concatenate([cross_i, cross_j]))
            nc = len(crosses)

            # Accept first input with enough crossings
            if nc >= TARGET_MIN:
                best_data = (zi, zj, crosses, fwd_z1, ci, cj, len(states))
                best_count = nc
                break
            if nc > best_count:
                best_count = nc
                best_data = (zi, zj, crosses, fwd_z1, ci, cj, len(states))

        zi, zj, crosses, fwd_z1, ci, cj, n = best_data
        print(f"    {arch}: {best_count} crossings, {n} iters")

        # Plot trajectory with fade
        for i in range(n - 1):
            a = 0.15 + 0.85 * (i / n)
            ax.plot(zi[i:i + 2], zj[i:i + 2], color=C_SHEAF,
                    lw=1.2, alpha=a)

        # ReLU boundaries
        ax.axvline(0, color=C_GREY, ls='--', lw=1.0, alpha=0.6,
                   label='ReLU boundary')
        ax.axhline(0, color=C_GREY, ls='--', lw=1.0, alpha=0.6)

        # Crossings
        if len(crosses):
            ax.scatter(zi[crosses], zj[crosses], color=C_RED, marker='x',
                       s=30, linewidths=1.2, zorder=5,
                       label=f'Crossings ({len(crosses)})')

        # Fixed point
        ax.scatter([fwd_z1[ci]], [fwd_z1[cj]], color=C_GOLD, marker='*',
                   s=100, zorder=6, edgecolors='k', linewidths=0.4,
                   label='Fixed point')

        # Start point
        ax.scatter([zi[0]], [zj[0]], color=C_GREY, marker='o',
                   s=35, zorder=6, edgecolors='k', linewidths=0.4,
                   label='Start')

        ax.set_xlabel(f'$z_{{{ci + 1}}}^{{(1)}}$')
        ax.set_ylabel(f'$z_{{{cj + 1}}}^{{(1)}}$')
        ax.set_title(str(arch).replace(' ', ''), fontsize=10)
        ax.legend(frameon=False, fontsize=6, loc='best')
        _clean(ax)

    fig.tight_layout()
    _save(fig, 'phase_planes.pdf')


# =====================================================================
# Table E.1  Full training results  +  Figure E.3  Learned outputs (1H)
# =====================================================================

_1H = [
    ('Paraboloid', [2, 30, 1], 'paraboloid', 100_000, 0.005),
    ('Saddle',     [2, 30, 1], 'saddle',     100_000, 0.005),
    ('Circular',   [2, 25, 1], 'circular',   100_000, 0.005),
    ('Blobs',      [2, 25, 4], 'blobs',      100_000, 0.005),
]
_2H = [
    ('Paraboloid', [2, 10, 8, 1], 'paraboloid', 200_000, 0.01),
    ('Saddle',     [2, 10, 8, 1], 'saddle',     200_000, 0.01),
    ('Circular',   [2, 10, 8, 1], 'circular',   200_000, 0.01),
    ('Blobs',      [2, 10, 8, 4], 'blobs',      200_000, 0.01),
]


def table_E1_and_figure_E3():
    print("\n[Table E.1 + Fig E.3] Training all 8 configs ...")
    rows = []
    models_1h = {}

    for tag, configs in [('1H', _1H), ('2H', _2H)]:
        for label, arch, tkey, nsteps, dt in configs:
            print(f"    {label} {tag} {arch} ...")
            m = _train_pair(arch, tkey, nsteps, dt, seed=42)
            st = m['h_s']['train_loss'][-1]
            gt = m['h_g']['train_loss'][-1]
            se = m['h_s']['test_loss'][-1]
            ge = m['h_g']['test_loss'][-1]
            ratio = se / ge if ge > 1e-10 else float('inf')
            rows.append((label, tag, st, gt, se, ge, ratio))
            if tag == '1H':
                models_1h[tkey] = m
            print(f"      sheaf test={se:.4f}  SGD test={ge:.4f}  "
                  f"ratio={ratio:.1f}x")

    # LaTeX table
    lines = [
        r'\begin{table}[ht]', r'\centering',
        r'\caption{Training results across all task/depth configurations.}',
        r'\label{tab:full-results}',
        r'\begin{tabular}{llccccc}', r'\toprule',
        r'Task & Depth & Sheaf Train & SGD Train & Sheaf Test & SGD Test & Ratio \\',
        r'\midrule',
    ]
    for tn, d, st, gt, se, ge, ra in rows:
        lines.append(
            f'  {tn} & {d} & {st:.4f} & {gt:.4f} & '
            f'{se:.4f} & {ge:.4f} & {ra:.1f}$\\times$ \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _save_table(lines, 'table_E1_full_results.tex')

    # ----- Figure E.3: learned outputs, 1H -----
    print("    Plotting learned surfaces / boundaries (1H) ...")
    tasks_ordered = ['paraboloid', 'saddle', 'circular', 'blobs']
    labels_ordered = ['Paraboloid', 'Saddle', 'Circular', 'Blobs']

    fig, axes = plt.subplots(4, 3, figsize=(7.5, 10.0))

    for row, (tkey, lab) in enumerate(zip(tasks_ordered, labels_ordered)):
        m = models_1h[tkey]
        _fill_output_row(axes[row], m['trainer'], m['nn'], m['cfg'],
                         m['X_tr'], m['Y_tr'], lab)

    axes[0, 0].set_title('Target', fontsize=10)
    axes[0, 1].set_title('Sheaf', fontsize=10)
    axes[0, 2].set_title('SGD', fontsize=10)
    fig.tight_layout(h_pad=1.0, w_pad=0.5)
    _save(fig, 'all_training_outputs_1H.pdf')

    return models_1h


def _fill_output_row(axes, trainer, nn, cfg, X_tr, Y_tr, label):
    if cfg['task_type'] == 'regression':
        _reg_row(axes, trainer, nn, cfg, X_tr, label)
    else:
        _cls_row(axes, trainer, nn, cfg, X_tr, Y_tr, label)


def _reg_row(axes, trainer, nn, cfg, X_tr, label):
    if 'paraboloid' in label.lower():
        fn, lo, hi = paraboloid, -2.0, 2.0
    else:
        fn, lo, hi = saddle, 0.0, 2.0

    g = torch.linspace(lo, hi, 60)
    XX, YY = torch.meshgrid(g, g, indexing='ij')
    Xg = torch.stack([XX.flatten(), YY.flatten()])
    Zt = fn(Xg).reshape(60, 60).numpy()
    with torch.no_grad():
        Zs = trainer.predict(Xg).reshape(60, 60).numpy()
        Zg = nn.predict(Xg).reshape(60, 60).numpy()

    vmin = min(Zt.min(), Zs.min(), Zg.min())
    vmax = max(Zt.max(), Zs.max(), Zg.max())
    for ax, Z in zip(axes, [Zt, Zs, Zg]):
        im = ax.contourf(XX.numpy(), YY.numpy(), Z, levels=20,
                          cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        _clean(ax)
    axes[0].set_ylabel(label, fontsize=10, fontweight='bold')
    # Colorbar on last panel
    cb = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)


def _cls_row(axes, trainer, nn, cfg, X_tr, Y_tr, label):
    binary = cfg['task_type'] == 'binary'
    xl = X_tr[0].min().item() - 0.5
    xh = X_tr[0].max().item() + 0.5
    yl = X_tr[1].min().item() - 0.5
    yh = X_tr[1].max().item() + 0.5
    g_x = torch.linspace(xl, xh, 120)
    g_y = torch.linspace(yl, yh, 120)
    XX, YY = torch.meshgrid(g_x, g_y, indexing='ij')
    Xg = torch.stack([XX.flatten(), YY.flatten()])

    labs = Y_tr[0].numpy() if binary else Y_tr.argmax(dim=0).numpy()
    n_cls = 2 if binary else int(labs.max()) + 1

    if binary:
        cmap = plt.cm.coolwarm
        scatter_kw = dict(c=labs, cmap=cmap, s=5, alpha=0.6, edgecolors='none')
    else:
        colors_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        cmap = plt.cm.colors.ListedColormap(colors_list[:n_cls])
        scatter_kw = dict(c=labs, cmap=cmap, s=5, alpha=0.6, edgecolors='none',
                          vmin=-0.5, vmax=n_cls - 0.5)

    # Data-only (target)
    axes[0].scatter(X_tr[0].numpy(), X_tr[1].numpy(), **scatter_kw)
    axes[0].set_xlim(xl, xh); axes[0].set_ylim(yl, yh)
    axes[0].set_aspect('equal')
    axes[0].set_ylabel(label, fontsize=10, fontweight='bold')
    _clean(axes[0])

    # Add legend for classification
    if binary:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.0), ms=5, label='Class 0'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1.0), ms=5, label='Class 1'),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors_list[i], ms=5, label=f'Class {i}')
            for i in range(n_cls)
        ]
    axes[0].legend(handles=legend_handles, frameon=False, fontsize=5,
                   loc='upper right')

    for ax, model in zip(axes[1:], [trainer, nn]):
        with torch.no_grad():
            pr = model.predict_classes(Xg)
            Z = (pr[0] if binary else pr).float().reshape(120, 120).numpy()
        if binary:
            lvls = np.linspace(-0.5, 1.5, 10)
        else:
            lvls = np.arange(-0.5, n_cls + 0.5)
        ax.contourf(XX.numpy(), YY.numpy(), Z, levels=lvls,
                     cmap=cmap, alpha=0.3)
        ax.scatter(X_tr[0].numpy(), X_tr[1].numpy(), **scatter_kw)
        ax.set_xlim(xl, xh); ax.set_ylim(yl, yh)
        ax.set_aspect('equal')
        _clean(ax)


# =====================================================================
# E.4  Depth x T sweep + Table: spectral gap vs depth
# =====================================================================

def figure_E4():
    print("\n[Fig E.4 + Table] Depth x T sweep ...")
    depth_cfgs = [
        ('1H', [2, 30, 1],       0.005),
        ('2H', [2, 10, 8, 1],    0.01),
        ('3H', [2, 8, 6, 4, 1],  0.01),
    ]
    T_vals = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0]

    X_tr, Y_tr, X_te, Y_te = generate_regression_data(
        paraboloid, n_train=300, n_test=100)
    n_train = X_tr.shape[1]
    beta = 1.0 / n_train

    results = {}
    spec_gaps = {}

    for dlabel, arch, dt_base in depth_cfgs:
        results[dlabel] = []
        si = NeuralSheaf(arch, seed=42)
        sa = spectral_analysis_per_sample(si, X_tr[:, :50])
        spec_gaps[dlabel] = (arch, float(np.median(sa['gaps'])),
                              float(np.std(sa['gaps'])))

        for T in T_vals:
            nsteps = max(int(T / dt_base), 100)
            ep = max(nsteps // 1000, 1)
            spe = min(nsteps, 1000)
            sheaf = NeuralSheaf(arch, output_activation='identity', seed=42)
            tr = SheafTrainer(sheaf, alpha=1.0, beta=beta, dt=dt_base)
            h = tr.train(X_tr, Y_tr, X_te, Y_te,
                         epochs=ep, steps_per_epoch=spe,
                         warm_start=True, init_method='random', seed=42)
            tl = h['test_loss'][-1]
            results[dlabel].append(tl)
            print(f"    {dlabel}  T={T:5.1f}  test_loss={tl:.4f}")

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    markers = ['o', 's', 'D']
    for i, (dl, arch, _) in enumerate(depth_cfgs):
        ax.loglog(T_vals, results[dl], f'{markers[i]}-',
                   color=LAYER_COLORS[i], markersize=6, lw=2.0,
                   markeredgecolor='white', markeredgewidth=0.6,
                   label=f'{dl} {arch}')
    ax.set_xlabel('Total integration time $T = \\Delta t \\cdot n_{\\mathrm{steps}}$')
    ax.set_ylabel('Test loss')
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)
    fig.tight_layout()
    _save(fig, 'depth_T_sweep.pdf')

    # Table
    lines = [
        r'\begin{table}[ht]', r'\centering',
        r'\caption{Spectral gap $\lambda_1$ vs.\ depth at initialization '
        r'(median over 50 inputs).}',
        r'\label{tab:spectral-gap-depth}',
        r'\begin{tabular}{lcc}', r'\toprule',
        r'Architecture & $\lambda_1$ (median) & $\lambda_1$ (std) \\',
        r'\midrule',
    ]
    for dl, _, _ in depth_cfgs:
        arch, med, std = spec_gaps[dl]
        astr = str(arch).replace(' ', '')
        lines.append(f'  \\texttt{{{astr}}} & {med:.4f} & {std:.4f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _save_table(lines, 'table_spectral_gap_depth.tex')


# =====================================================================
# E.5  Spectral evolution — individual plots
# =====================================================================

def figure_E5():
    print("\n[Fig E.5] Spectral evolution plots ...")
    arch = [2, 30, 1]
    X_tr, Y_tr, X_te, Y_te = generate_regression_data(
        paraboloid, n_train=300, n_test=100)
    n_train = X_tr.shape[1]
    n_steps = 100_000
    X_sub = X_tr[:, :50]
    sf = 5000
    snap_st = [1000, 20_000, 100_000]

    print("    sheaf training ...")
    hs, ss = train_sheaf_full(
        arch, X_tr, Y_tr, X_te, Y_te,
        n_steps=n_steps, beta=1.0 / n_train, dt=0.005,
        output_activation='identity',
        snapshot_steps=snap_st, spectral_freq=sf,
        X_sub=X_sub, seed_w=42, seed_s=42,
    )
    print("    SGD training ...")
    hg, sg = train_sgd_full(
        arch, X_tr, Y_tr, X_te, Y_te,
        n_steps=n_steps, output_activation='identity',
        lr=0.005, snapshot_steps=snap_st,
        spectral_freq=sf, X_sub=X_sub, seed=42,
    )

    step_s = np.array(hs['step']) / 1000  # in k-steps for cleaner axes
    step_g = np.array(hg['step']) / 1000

    # Pre-compute shared y-limits across sheaf and SGD
    all_gap_q25 = hs['spectral']['q25'] + hg['spectral']['q25']
    all_gap_q75 = hs['spectral']['q75'] + hg['spectral']['q75']
    gap_ymin = min(all_gap_q25) * 0.85
    gap_ymax = max(all_gap_q75) * 1.15

    all_loss = hs['train_loss'] + hg['train_loss']
    loss_ymin = max(min(all_loss) * 0.5, 1e-6)
    loss_ymax = max(all_loss) * 2.0

    all_lmax_q75 = hs['lambda_max']['q75'] + hg['lambda_max']['q75']
    all_lmax_q25 = hs['lambda_max']['q25'] + hg['lambda_max']['q25']
    lmax_ymin = min(all_lmax_q25) * 0.85
    lmax_ymax = max(all_lmax_q75) * 1.15

    all_kappa_q75 = hs['condition']['q75'] + hg['condition']['q75']
    all_kappa_q25 = hs['condition']['q25'] + hg['condition']['q25']
    kappa_ymin = max(min(all_kappa_q25) * 0.85, 1.0)
    kappa_ymax = max(all_kappa_q75) * 1.15

    FIGSIZE = (5.2, 3.5)

    # --- (1) lambda_1 sheaf with loss ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(step_s, hs['spectral']['median'], color=C_SHEAF)
    ax.fill_between(step_s, hs['spectral']['q25'], hs['spectral']['q75'],
                     color=C_SHEAF, alpha=0.15)
    ax2 = ax.twinx()
    ax2.semilogy(step_s, hs['train_loss'], color=C_GREY, ls='--', alpha=0.7)
    ax2.set_ylabel('Train loss', color=C_GREY)
    ax2.tick_params(axis='y', labelcolor=C_GREY)
    ax2.set_ylim(loss_ymin, loss_ymax)
    ax.set_yscale('log')
    ax.set_ylim(gap_ymin, gap_ymax)
    ax.set_xlabel('Step ($\\times 10^3$)')
    ax.set_ylabel('$\\lambda_1$ (median $\\pm$ IQR)')
    ax.set_title('Spectral gap — sheaf')
    _clean(ax)
    fig.tight_layout()
    _save(fig, 'spectral_gap_sheaf.pdf')

    # --- (2) lambda_1 SGD with loss ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(step_g, hg['spectral']['median'], color=C_SGD)
    ax.fill_between(step_g, hg['spectral']['q25'], hg['spectral']['q75'],
                     color=C_SGD, alpha=0.15)
    ax2 = ax.twinx()
    ax2.semilogy(step_g, hg['train_loss'], color=C_GREY, ls='--', alpha=0.7)
    ax2.set_ylabel('Train loss', color=C_GREY)
    ax2.tick_params(axis='y', labelcolor=C_GREY)
    ax2.set_ylim(loss_ymin, loss_ymax)
    ax.set_yscale('log')
    ax.set_ylim(gap_ymin, gap_ymax)
    ax.set_xlabel('Step ($\\times 10^3$)')
    ax.set_ylabel('$\\lambda_1$ (median $\\pm$ IQR)')
    ax.set_title('Spectral gap — SGD')
    _clean(ax)
    fig.tight_layout()
    _save(fig, 'spectral_gap_sgd.pdf')

    # --- (3) lambda_1 comparison: sheaf vs SGD ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(step_s, hs['spectral']['median'], color=C_SHEAF, label='Sheaf')
    ax.fill_between(step_s, hs['spectral']['q25'], hs['spectral']['q75'],
                     color=C_SHEAF, alpha=0.12)
    ax.plot(step_g, hg['spectral']['median'], color=C_SGD, label='SGD')
    ax.fill_between(step_g, hg['spectral']['q25'], hg['spectral']['q75'],
                     color=C_SGD, alpha=0.12)
    ax.set_yscale('log')
    ax.set_ylim(gap_ymin, gap_ymax)
    ax.set_xlabel('Step ($\\times 10^3$)')
    ax.set_ylabel('$\\lambda_1$ (median $\\pm$ IQR)')
    ax.set_title('Spectral gap — sheaf vs SGD')
    ax.legend(frameon=False); _clean(ax)
    fig.tight_layout()
    _save(fig, 'spectral_gap_comparison.pdf')

    # --- (4) lambda_max comparison ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(step_s, hs['lambda_max']['median'], color=C_SHEAF, label='Sheaf')
    ax.fill_between(step_s, hs['lambda_max']['q25'], hs['lambda_max']['q75'],
                     color=C_SHEAF, alpha=0.12)
    ax.plot(step_g, hg['lambda_max']['median'], color=C_SGD, label='SGD')
    ax.fill_between(step_g, hg['lambda_max']['q25'], hg['lambda_max']['q75'],
                     color=C_SGD, alpha=0.12)
    ax.set_ylim(lmax_ymin, lmax_ymax)
    ax.set_xlabel('Step ($\\times 10^3$)')
    ax.set_ylabel('$\\lambda_{\\max}$ (median $\\pm$ IQR)')
    ax.set_title('Largest eigenvalue — sheaf vs SGD')
    ax.legend(frameon=False); _clean(ax)
    fig.tight_layout()
    _save(fig, 'spectral_lambda_max_comparison.pdf')

    # --- (5) kappa comparison ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(step_s, hs['condition']['median'], color=C_SHEAF, label='Sheaf')
    ax.fill_between(step_s, hs['condition']['q25'], hs['condition']['q75'],
                     color=C_SHEAF, alpha=0.12)
    ax.plot(step_g, hg['condition']['median'], color=C_SGD, label='SGD')
    ax.fill_between(step_g, hg['condition']['q25'], hg['condition']['q75'],
                     color=C_SGD, alpha=0.12)
    ax.set_ylim(kappa_ymin, kappa_ymax)
    ax.set_xlabel('Step ($\\times 10^3$)')
    ax.set_ylabel('$\\kappa$ (median $\\pm$ IQR)')
    ax.set_title('Condition number — sheaf vs SGD')
    ax.legend(frameon=False); _clean(ax)
    fig.tight_layout()
    _save(fig, 'spectral_kappa_comparison.pdf')

    # --- (6-7) Spectral gap histograms: all 3 levels overlaid ---
    # Following Notebook 3 pattern: log-spaced bins, shared x-range
    stage_labels = ['Poorly trained (1k)', 'Intermediate (20k)', 'Well trained (100k)']
    stage_colors = [C_RED, C_GOLD, C_BLUE]
    N_DIST = 100  # more samples for smoother distributions
    X_dist = X_tr[:, :N_DIST]
    n_bins = 35

    # Compute spectral analysis at each snapshot for both methods
    sheaf_gaps = {}
    sgd_gaps = {}
    for step_val, slabel in zip(snap_st, stage_labels):
        sa = spectral_analysis_per_sample(ss[step_val][0], X_dist)
        sheaf_gaps[step_val] = sa['gaps']
        sa = spectral_analysis_per_sample(sg[step_val][0], X_dist)
        sgd_gaps[step_val] = sa['gaps']

    # Shared x-range from ALL data (both methods, all levels)
    all_gaps = np.concatenate(
        [sheaf_gaps[s] for s in snap_st] + [sgd_gaps[s] for s in snap_st])
    gap_lo = all_gaps[all_gaps > 0].min() * 0.85
    gap_hi = all_gaps.max() * 1.15
    bins = np.logspace(np.log10(gap_lo), np.log10(gap_hi), n_bins + 1)

    # Helper: draw bars
    def _bar(ax, data, **kw):
        counts, _ = np.histogram(data, bins=bins)
        ax.bar(bins[:-1], counts / len(data), width=np.diff(bins),
               align='edge', **kw)

    # Sheaf: all three levels overlaid
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for step_val, slabel, scolor in zip(snap_st, stage_labels, stage_colors):
        _bar(ax, sheaf_gaps[step_val], alpha=0.5, color=scolor, label=slabel,
             edgecolor='white', linewidth=0.3)
    ax.set_xscale('log'); ax.set_xlim(gap_lo, gap_hi)
    ax.set_xlabel('$\\lambda_1$')
    ax.set_ylabel('Fraction of input samples')
    ax.set_title('Spectral gap distribution — sheaf')
    ax.legend(frameon=False); _clean(ax)
    fig.tight_layout()
    _save(fig, 'spectral_histograms_sheaf.pdf')

    # SGD: all three levels overlaid, same x-range
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for step_val, slabel, scolor in zip(snap_st, stage_labels, stage_colors):
        _bar(ax, sgd_gaps[step_val], alpha=0.5, color=scolor, label=slabel,
             edgecolor='white', linewidth=0.3)
    ax.set_xscale('log'); ax.set_xlim(gap_lo, gap_hi)
    ax.set_xlabel('$\\lambda_1$')
    ax.set_ylabel('Fraction of input samples')
    ax.set_title('Spectral gap distribution — SGD')
    ax.legend(frameon=False); _clean(ax)
    fig.tight_layout()
    _save(fig, 'spectral_histograms_sgd.pdf')

    # --- (8) Well-trained: sheaf vs SGD on same axes ---
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    _bar(ax, sheaf_gaps[100_000], alpha=0.5, color=C_SHEAF, label='Sheaf',
         edgecolor='white', linewidth=0.3)
    _bar(ax, sgd_gaps[100_000], alpha=0.5, color=C_SGD, label='SGD',
         edgecolor='white', linewidth=0.3)
    ax.set_xscale('log'); ax.set_xlim(gap_lo, gap_hi)
    ax.set_xlabel('$\\lambda_1$')
    ax.set_ylabel('Fraction of input samples')
    ax.set_title('$\\lambda_1$ distribution — well-trained')
    ax.legend(frameon=False); _clean(ax)
    fig.tight_layout()
    _save(fig, 'spectral_gap_well_trained_comparison.pdf')


# =====================================================================
# E.6  Discord residuals at all training stages (3x2) — sheaf and SGD
# =====================================================================

def figure_E6():
    print("\n[Fig E.6] Discord residuals at 3 training stages (sheaf + SGD) ...")
    arch = [2, 30, 1]
    X_tr, Y_tr, X_te, Y_te = generate_regression_data(
        paraboloid, n_train=300, n_test=100)
    n_train = X_tr.shape[1]

    lvls = [1_000, 20_000, 100_000]
    lvl_labels = ['Poorly trained (1k)', 'Intermediate (20k)',
                  'Well trained (100k)']
    INTERMEDIATE_IDX = 1  # index into lvls for setting y-limits

    # ── Train sheaf with snapshots ──
    print("    sheaf training ...")
    sheaf = NeuralSheaf(arch, output_activation='identity', seed=42)
    trainer = SheafTrainer(sheaf, alpha=1.0, beta=1.0 / n_train, dt=0.005)
    torch.manual_seed(42)
    state = sheaf.init_stalks(X_tr, method='random')

    sheaf_snaps = {}
    for s in range(1, max(lvls) + 1):
        state = trainer.train_step(state, X_tr, Y_tr)
        if s in lvls:
            sheaf_snaps[s] = snapshot_sheaf(sheaf)
            print(f"      sheaf snapshot @ step {s:,}")

    # ── Train SGD with snapshots ──
    print("    SGD training ...")
    nn = TraditionalNN(arch, learning_rate=0.005, seed=42)
    # Step-by-step so we can snapshot at exact steps
    sgd_snaps = {}
    snap_set = set(lvls)
    for epoch in range(1, max(lvls) + 1):
        nn._zero_grad()
        y_pred = nn.forward(X_tr)
        loss = nn._compute_loss(y_pred, Y_tr)
        loss.backward()
        with torch.no_grad():
            for W in nn.weights:
                W -= nn.learning_rate * W.grad
            for b in nn.biases:
                b -= nn.learning_rate * b.grad
        if epoch in snap_set:
            sgd_snaps[epoch] = nn_to_sheaf(nn, arch, 'identity')
            print(f"      SGD snapshot @ step {epoch:,}")

    # ── Compute pinned discord data for all snapshots ──
    def _get_edge_data(snap_model):
        """Run pinned diffusion and extract edge data."""
        disc, pstate, info = compute_pinned_discord(snap_model, X_te, Y_te)
        we, re = extract_edge_data(snap_model, pstate)
        return we, re

    sheaf_data = {s: _get_edge_data(sheaf_snaps[s]) for s in lvls}
    sgd_data = {s: _get_edge_data(sgd_snaps[s]) for s in lvls}

    # ── Determine y-limits from the intermediate stage (both methods) ──
    inter_step = lvls[INTERMEDIATE_IDX]

    # Weight edge y-limits
    w_disc_all = np.concatenate([
        sheaf_data[inter_step][0][0]['discord'],
        sgd_data[inter_step][0][0]['discord'],
    ])
    w_ylim = (w_disc_all.min() * 1.1, w_disc_all.max() * 1.1)

    # ReLU edge y-limits
    r_disc_all = np.concatenate([
        sheaf_data[inter_step][1][0]['discord'],
        sgd_data[inter_step][1][0]['discord'],
    ])
    r_ylim = (r_disc_all.min() * 1.1, r_disc_all.max() * 1.1)

    # ── Plot function (shared by sheaf and SGD) ──
    def _plot_discord_grid(data_dict, method_label, fname):
        fig, axes = plt.subplots(3, 2, figsize=(6.5, 7.5))

        for row, (step, label) in enumerate(zip(lvls, lvl_labels)):
            we, re = data_dict[step]

            # LEFT: Weight edge discord residual
            ax = axes[row, 0]
            w = we[0]
            z_vals = w['actual']
            discord_vals = w['discord']
            active = z_vals >= 0

            ax.scatter(z_vals[~active], discord_vals[~active],
                       c=C_INACTIVE, alpha=0.55, s=6, edgecolors='none',
                       label=r'Inactive ($z_j < 0$)', zorder=2, rasterized=True)
            ax.scatter(z_vals[active], discord_vals[active],
                       c=C_ACTIVE, alpha=0.55, s=6, edgecolors='none',
                       label=r'Active ($z_j \geq 0$)', zorder=2, rasterized=True)
            ax.axhline(0, color='black', linestyle='--', linewidth=1.2,
                       alpha=0.7, zorder=1)
            ax.set_xlabel(r'$z_j^{(1)}$')
            ax.set_ylabel(r'$(W^{(1)} a^{(0)} + b^{(1)})_j - z_j^{(1)}$')
            ax.set_title(f'Weight edge — {label}', fontsize=8)
            ax.set_ylim(*w_ylim)
            if row == 0:
                ax.legend(frameon=False, fontsize=6, markerscale=2, loc='best')
            _clean(ax)

            # RIGHT: ReLU edge — z vs ReLU discord
            ax = axes[row, 1]
            r = re[0]
            act = r['active']
            ax.scatter(r['z'][~act], r['discord'][~act],
                       c=C_INACTIVE, alpha=0.55, s=6, edgecolors='none',
                       label=r'Inactive ($z_j < 0$)', zorder=2, rasterized=True)
            ax.scatter(r['z'][act], r['discord'][act],
                       c=C_ACTIVE, alpha=0.55, s=6, edgecolors='none',
                       label=r'Active ($z_j \geq 0$)', zorder=2, rasterized=True)
            ax.axhline(0, color='black', linestyle='--', linewidth=1.2,
                       alpha=0.7, zorder=1)
            ax.set_xlabel(r'$z_j^{(1)}$')
            ax.set_ylabel(r'$\mathrm{ReLU}(z_j^{(1)}) - a_j^{(1)}$')
            ax.set_title(f'ReLU edge — {label}', fontsize=8)
            ax.set_ylim(*r_ylim)
            if row == 0:
                ax.legend(frameon=False, fontsize=6, markerscale=2, loc='best')
            _clean(ax)

        fig.suptitle(f'{method_label}', fontsize=11, y=1.01)
        fig.tight_layout(h_pad=0.8)
        _save(fig, fname)

    _plot_discord_grid(sheaf_data, 'Sheaf-trained',
                       'discord_residuals_all_stages_sheaf.pdf')
    _plot_discord_grid(sgd_data, 'SGD-trained',
                       'discord_residuals_all_stages_sgd.pdf')


# =====================================================================
# E.8  Pinned discord — table instead of barplot
# =====================================================================

def table_E2_pinned_discord():
    print("\n[Table E.2] Pinned discord profiles (sheaf vs SGD) ...")
    arch = [2, 30, 1]
    X_tr, Y_tr, X_te, Y_te = generate_regression_data(
        paraboloid, n_train=300, n_test=100)
    n_train = X_tr.shape[1]

    print("    sheaf training ...")
    sheaf = NeuralSheaf(arch, output_activation='identity', seed=42)
    tr = SheafTrainer(sheaf, alpha=1.0, beta=1.0 / n_train, dt=0.005)
    tr.train(X_tr, Y_tr, epochs=100, steps_per_epoch=1000,
             warm_start=True, init_method='random', seed=42)
    sheaf_model = snapshot_sheaf(sheaf)

    print("    SGD training ...")
    nn = TraditionalNN(arch, learning_rate=0.005, seed=42)
    nn.train(X_tr, Y_tr, epochs=100_000)
    sgd_model = nn_to_sheaf(nn, arch, 'identity')

    Xs = X_te[:, :50]
    Ys = Y_te[:, :50]

    lines = [
        r'\begin{table}[ht]', r'\centering',
        r'\caption{Per-edge pinned discord (mean $\pm$ std over 50 test inputs). '
        r'Output stalk hard-pinned to true labels.}',
        r'\label{tab:pinned-discord}',
        r'\begin{tabular}{lcc}', r'\toprule',
        r'Edge & Sheaf-trained & SGD-trained \\',
        r'\midrule',
    ]

    for model, lab in [(sheaf_model, 'sheaf'), (sgd_model, 'sgd')]:
        disc, ps, info = compute_pinned_discord(model, Xs, Ys)
        k = model.k
        results = {}
        for ell in range(1, k + 2):
            _, vec = disc['weight_edges'][ell]
            ps_d = (vec ** 2).sum(dim=0)
            results[f'W{ell}'] = (ps_d.mean().item(), ps_d.std().item())
        for ell in range(1, k + 1):
            _, vec = disc['activation_edges'][ell]
            ps_d = (vec ** 2).sum(dim=0)
            results[f'R{ell}'] = (ps_d.mean().item(), ps_d.std().item())
        results['Total'] = (disc['total'], 0.0)
        if lab == 'sheaf':
            sheaf_res = results
        else:
            sgd_res = results

    for edge_name in sheaf_res:
        sm, ss_val = sheaf_res[edge_name]
        gm, gs = sgd_res[edge_name]
        if edge_name == 'Total':
            lines.append(r'\midrule')
            lines.append(
                f'  {edge_name} & {sm:.4f} & {gm:.4f} \\\\')
        else:
            lines.append(
                f'  {edge_name} & {sm:.4f} $\\pm$ {ss_val:.4f} & '
                f'{gm:.4f} $\\pm$ {gs:.4f} \\\\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _save_table(lines, 'table_pinned_discord.tex')


# =====================================================================
# Main
# =====================================================================

def main():
    setup_style()
    t0 = time.time()

    print("=" * 60)
    print("  Appendix E — Figure & Table Generation")
    print("=" * 60)

    figure_E1()                          # convergence dashboard
    figure_E2()                          # phase planes
    models = table_E1_and_figure_E3()    # training table + learned outputs
    figure_E4()                          # depth x T sweep + spectral table
    figure_E5()                          # spectral (individual plots)
    figure_E6()                          # discord residuals (3 stages)
    table_E2_pinned_discord()            # pinned discord table

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  Done — {elapsed / 60:.1f} min")
    print(f"  Figures -> {FIG_DIR}/")
    print(f"  Tables  -> {TABLE_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()