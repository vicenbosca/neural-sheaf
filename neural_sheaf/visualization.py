"""
Visualization utilities for the neural sheaf library.

Matplotlib-based plotting for all analysis tasks:

**Inference plots** -- discord evolution, stalk trajectories, phase planes,
convergence comparisons, and dynamics dashboards.

**Training plots** -- training curves, regression surfaces, classification
decision boundaries.

**Discord plots** -- predicted-vs-actual scatter, discord residuals,
and summary bar charts comparing sheaf vs SGD.

**Spectral plots** -- eigenvalue spectra, spectral gap distributions,
condition number distributions, spectral tracking during training,
loss--spectral dual axes, eigenvector layer energy, eigenvector stability,
neuron contributions, and sheaf-vs-SGD comparisons.

Consistent color palette, clean styling, composable (most accept optional
``ax`` argument).
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Dict, List, Optional, Tuple

from .sheaf import NeuralSheaf


# ============================================================================
# Color palette (consistent across plots)
# ============================================================================

PALETTE = {
    'total': '#1f77b4',
    'weight': '#e377c2',
    'activation': '#2ca02c',
    'output': '#d62728',
    'forward': '#ff7f0e',
    'converged': '#1f77b4',
    'layers': plt.cm.tab10.colors,
}


def _clean_spines(ax: plt.Axes) -> None:
    """Remove top and right spines for a cleaner modern look."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _compute_accuracy(trainer, X: torch.Tensor, Y: torch.Tensor,
                      task: str) -> float:
    """
    Compute classification accuracy for any trainer-like object.

    Works with both ``SheafTrainer`` and ``TraditionalNN`` -- any object
    that provides a ``predict_classes(X)`` method.

    Parameters
    ----------
    trainer : SheafTrainer or TraditionalNN
    X : torch.Tensor, shape ``(input_dim, n_samples)``
    Y : torch.Tensor
    task : str
        ``'binary'`` or ``'multiclass'``.

    Returns
    -------
    float
        Fraction of correctly classified samples.
    """
    with torch.no_grad():
        preds = trainer.predict_classes(X)
        if task == 'binary':
            return (preds.squeeze(0) == Y.squeeze(0)).float().mean().item()
        else:
            return (preds == Y.argmax(dim=0)).float().mean().item()


# ============================================================================
# Discord evolution
# ============================================================================

def plot_discord_evolution(
    discord_history: List[Dict],
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    mask_changes: Optional[List[Dict]] = None,
    title: str = r'Total Discord vs Iteration',
) -> plt.Axes:
    """
    Plot total discord decreasing over iterations.

    Parameters
    ----------
    discord_history : list of dict
        Output from ``track_trajectory``. Each dict must have
        'iteration' and 'total' keys.
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    log_scale : bool
        Use log scale for y-axis. Default True.
    mask_changes : list of dict, optional
        Output from ``detect_mask_changes``. If provided, vertical lines
        mark iterations where a ReLU mask changed.
    title : str
        Plot title.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    iters = [d['iteration'] for d in discord_history]
    totals = [d['total'] for d in discord_history]

    ax.plot(iters, totals, '-', color=PALETTE['total'], linewidth=1.5,
            label='Total discord')

    if mask_changes:
        added = False
        for mc in mask_changes:
            ax.axvline(mc['iteration'], color='red', linestyle=':',
                       alpha=0.5, linewidth=0.8,
                       label='ReLU boundary' if not added else None)
            added = True

    if log_scale and min(t for t in totals if t > 0) > 0:
        ax.set_yscale('log')

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Discord')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _clean_spines(ax)
    return ax


def plot_discord_by_layer(
    discord_history: List[Dict],
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    title: str = 'Discord by Layer',
) -> plt.Axes:
    """
    Plot discord grouped by layer over iterations.

    Parameters
    ----------
    discord_history : list of dict
        Output from ``track_trajectory``.
    ax : matplotlib Axes, optional
    log_scale : bool
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    iters = [d['iteration'] for d in discord_history]
    layers = sorted(discord_history[0]['by_layer'].keys())

    for i, layer in enumerate(layers):
        vals = [d['by_layer'].get(layer, 0.0) for d in discord_history]
        color = PALETTE['layers'][i % len(PALETTE['layers'])]
        ax.plot(iters, vals, '-', color=color, linewidth=1.2,
                label=f'Layer {layer}')

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Discord')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _clean_spines(ax)
    return ax


def plot_discord_by_edge_type(
    discord_history: List[Dict],
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    title: str = 'Discord by Edge Type',
) -> plt.Axes:
    """
    Plot total weight-edge vs activation-edge discord over iterations.

    Parameters
    ----------
    discord_history : list of dict
    ax : matplotlib Axes, optional
    log_scale : bool
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    iters = [d['iteration'] for d in discord_history]

    weight_totals = []
    act_totals = []
    out_totals = []

    for d in discord_history:
        w_sum = sum(norm for norm, _ in d['weight_edges'].values())
        a_sum = sum(norm for norm, _ in d['activation_edges'].values())
        weight_totals.append(w_sum)
        act_totals.append(a_sum)
        if d['output_edge'] is not None:
            out_totals.append(d['output_edge'][0])

    ax.plot(iters, weight_totals, '-', color=PALETTE['weight'],
            linewidth=1.2, label='Weight edges')
    ax.plot(iters, act_totals, '-', color=PALETTE['activation'],
            linewidth=1.2, label='Activation edges')
    if out_totals:
        ax.plot(iters, out_totals, '-', color=PALETTE['output'],
                linewidth=1.2, label='Output edge')

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Discord')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _clean_spines(ax)
    return ax


# ============================================================================
# Stalk trajectories
# ============================================================================

def plot_stalk_trajectories(
    states: List[Dict],
    discord_history: List[Dict],
    sheaf: NeuralSheaf,
    forward_state: Optional[Dict] = None,
    stalk_type: str = 'z',
    layer_idx: int = -1,
    max_components: int = 4,
    mask_changes: Optional[List[Dict]] = None,
    log_error: bool = False,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how individual stalk components evolve during dynamics.

    Parameters
    ----------
    states : list of dict
        State snapshots from ``track_trajectory``.
    discord_history : list of dict
        Discord history (used for iteration numbers).
    sheaf : NeuralSheaf
        The neural sheaf (for labeling).
    forward_state : dict, optional
        Forward pass state for reference lines.
    stalk_type : str
        'z' for pre-activations, 'a' for post-activations, 'a_output'.
    layer_idx : int
        Index into the stalk list. -1 means last (output z or a_output).
    max_components : int
        Maximum number of components to plot.
    mask_changes : list of dict, optional
        Output from ``detect_mask_changes``. If provided, vertical lines
        mark iterations where a ReLU mask changed.
    log_error : bool
        If True (requires ``forward_state``), plot
        :math:`|v(t) - v^*|` on a log scale instead of raw values.
        Reveals exponential convergence rates.
    title : str, optional
        Plot title. Auto-generated if None.

    Returns
    -------
    matplotlib Figure
    """
    iters = [d['iteration'] for d in discord_history]

    # Extract trajectories for each component
    if stalk_type == 'a_output':
        traj = torch.stack([s['a_output'][:, 0] for s in states])
        ref = forward_state['a_output'][:, 0] if forward_state else None
        label_prefix = r'$a_{\mathrm{out}}$'
        label_raw = 'a_output'
    elif stalk_type == 'z':
        traj = torch.stack([s['z'][layer_idx][:, 0] for s in states])
        ref = forward_state['z'][layer_idx][:, 0] if forward_state else None
        ell = layer_idx + 1 if layer_idx >= 0 else len(states[0]['z'])
        label_prefix = rf'$z_{{{ell}}}$'
        label_raw = f'z_{ell}'
    else:
        traj = torch.stack([s['a'][layer_idx][:, 0] for s in states])
        ref = forward_state['a'][layer_idx][:, 0] if forward_state else None
        label_prefix = rf'$a_{{{layer_idx}}}$'
        label_raw = f'a_{layer_idx}'

    dim = traj.shape[1]
    n_plot = min(dim, max_components)

    fig, axes = plt.subplots(n_plot, 1, figsize=(8, 2.5 * n_plot),
                             sharex=True, squeeze=False)

    colors = plt.cm.tab10.colors

    for j in range(n_plot):
        ax = axes[j, 0]

        if log_error and ref is not None:
            # Plot |v(t) - v*| on log scale
            err = (traj[:, j] - ref[j]).abs().numpy()
            err = np.maximum(err, 1e-20)  # floor for log
            ax.plot(iters, err, '-', color=colors[j % 10], linewidth=1.5,
                    label=rf'{label_prefix}$[{j}]$')
            ax.set_yscale('log')
            ax.set_ylabel(rf'$|v_{j} - v_{j}^*|$')
        else:
            vals = traj[:, j].numpy()
            ax.plot(iters, vals, '-', color=colors[j % 10], linewidth=1.5,
                    label=rf'{label_prefix}$[{j}]$')
            if ref is not None:
                ax.axhline(ref[j].item(), color=colors[j % 10],
                           linestyle='--', alpha=0.6, linewidth=1,
                           label='Forward pass')
            ax.set_ylabel(rf'Component {j}')

        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        _clean_spines(ax)

    # Overlay mask-change markers
    if mask_changes:
        _added_label = False
        for mc in mask_changes:
            for ax_row in axes[:, 0]:
                ax_row.axvline(
                    mc['iteration'], color='red', linestyle=':',
                    alpha=0.5, linewidth=0.8,
                    label='ReLU boundary' if not _added_label else None,
                )
            _added_label = True
        axes[0, 0].legend(fontsize=7, loc='upper right')

    axes[-1, 0].set_xlabel('Iteration')

    if title is None:
        suffix = '(log error)' if log_error else '(first sample)'
        title = f'Stalk Trajectory: {label_raw} {suffix}'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_stalk_phase_plane(
    states: List[Dict],
    discord_history: List[Dict],
    sheaf: NeuralSheaf,
    forward_state: Optional[Dict] = None,
    stalk_type: str = 'z',
    layer_idx: int = 0,
    dims: Tuple[int, int] = (0, 1),
    mask_changes: Optional[List[Dict]] = None,
    sample_idx: int = 0,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a 2D phase-plane trajectory of two coordinates of a stalk.

    Shows how two selected components of a stalk co-evolve during dynamics.
    The trajectory fades from light to dark as iterations progress.
    Optionally marks points where a ReLU mask changes.

    Parameters
    ----------
    states : list of dict
        State snapshots from ``track_trajectory``.
    discord_history : list of dict
        Discord history (for iteration numbers).
    sheaf : NeuralSheaf
        The neural sheaf.
    forward_state : dict, optional
        Forward pass state for the target point.
    stalk_type : str
        'z' for pre-activations, 'a' for post-activations, 'a_output'.
    layer_idx : int
        Index into the stalk list (ignored when stalk_type='a_output').
    dims : tuple of int
        Which two components to plot as (x, y).
    mask_changes : list of dict, optional
        Output from ``detect_mask_changes``. If provided, 'x' markers
        indicate trajectory points nearest to each mask change.
    sample_idx : int
        Which sample in the batch to plot. Default 0.
    title : str, optional
        Plot title. Auto-generated if None.

    Returns
    -------
    matplotlib Axes
    """
    from matplotlib.collections import LineCollection

    # Extract coordinates and build LaTeX label
    if stalk_type == 'a_output':
        coords = torch.stack([s['a_output'][:, sample_idx] for s in states]).numpy()
        target = forward_state['a_output'][:, sample_idx].numpy() if forward_state else None
        stalk_tex = r'a_{\mathrm{out}}'
    elif stalk_type == 'z':
        coords = torch.stack([s['z'][layer_idx][:, sample_idx] for s in states]).numpy()
        target = forward_state['z'][layer_idx][:, sample_idx].numpy() if forward_state else None
        ell = layer_idx + 1 if layer_idx >= 0 else len(states[0]['z'])
        stalk_tex = rf'z_{{{ell}}}'
    elif stalk_type == 'a':
        coords = torch.stack([s['a'][layer_idx][:, sample_idx] for s in states]).numpy()
        target = forward_state['a'][layer_idx][:, sample_idx].numpy() if forward_state else None
        stalk_tex = rf'a_{{{layer_idx}}}'
    else:
        raise ValueError(f"stalk_type must be 'z', 'a', or 'a_output', got '{stalk_type}'")

    d0, d1 = dims
    iters = np.array([d['iteration'] for d in discord_history])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw trajectory as a LineCollection with fading alpha
    pts = np.column_stack([coords[:, d0], coords[:, d1]])
    segments = np.array([pts[i:i+2] for i in range(len(pts) - 1)])
    n_seg = len(segments)
    # Color: single hue, fading from light to full saturation
    base_color = np.array(mcolors.to_rgba(PALETTE['total']))
    seg_colors = np.zeros((n_seg, 4))
    for i in range(n_seg):
        t = i / max(n_seg - 1, 1)
        seg_colors[i] = base_color
        seg_colors[i, 3] = 0.15 + 0.85 * t  # alpha ramps up
    lc = LineCollection(segments, colors=seg_colors, linewidths=1.8)
    ax.add_collection(lc)

    # Dashed lines at x=0 and y=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

    # Mask-change markers (x in red)
    if mask_changes:
        mc_iters = set(mc['iteration'] for mc in mask_changes)
        mc_indices = [i for i, it in enumerate(iters) if it in mc_iters]
        if mc_indices:
            ax.scatter(coords[mc_indices, d0], coords[mc_indices, d1],
                       marker='x', s=70, c='red', linewidths=2.0,
                       zorder=4, label='ReLU boundary')

    # Start marker (hollow circle)
    ax.plot(coords[0, d0], coords[0, d1], 'o', color=PALETTE['total'],
            markersize=9, markerfacecolor='white', markeredgewidth=2,
            zorder=5, label='Start')

    # Converged marker (filled circle)
    ax.plot(coords[-1, d0], coords[-1, d1], 'o', color=PALETTE['total'],
            markersize=9, zorder=5, label='Converged')

    # Forward pass target (star)
    if target is not None:
        ax.plot(target[d0], target[d1], '*', color=PALETTE['forward'],
                markersize=14, markeredgecolor='k', markeredgewidth=0.5,
                zorder=5, label='Forward pass')

    ax.set_xlabel(rf'${stalk_tex}[{d0}]$')
    ax.set_ylabel(rf'${stalk_tex}[{d1}]$')
    if title is None:
        title = rf'Phase Plane: ${stalk_tex}$ (dims {d0}, {d1})'
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale_view()
    _clean_spines(ax)
    return ax


# ============================================================================
# Convergence comparison
# ============================================================================

def plot_convergence_comparison(
    converged_state: Dict,
    forward_state: Dict,
    sheaf: NeuralSheaf,
    sample_idx: int = 0,
    title: str = 'Converged vs Forward Pass',
) -> plt.Figure:
    """
    Side-by-side comparison of converged dynamics output vs forward pass.

    Parameters
    ----------
    converged_state : dict
        Final state from dynamics.
    forward_state : dict
        State from ``init_stalks(method='forward_pass')``.
    sheaf : NeuralSheaf
    sample_idx : int
        Which sample in the batch to visualize.
    title : str

    Returns
    -------
    matplotlib Figure
    """
    k = sheaf.k

    # Collect all stalk values for comparison
    labels = []
    conv_vals = []
    fwd_vals = []

    for ell in range(k + 1):
        z_c = converged_state['z'][ell][:, sample_idx].detach().numpy()
        z_f = forward_state['z'][ell][:, sample_idx].detach().numpy()
        for j in range(len(z_c)):
            labels.append(f'z{ell+1}[{j}]')
            conv_vals.append(z_c[j])
            fwd_vals.append(z_f[j])

    for ell in range(1, k + 1):
        a_c = converged_state['a'][ell][:, sample_idx].detach().numpy()
        a_f = forward_state['a'][ell][:, sample_idx].detach().numpy()
        for j in range(len(a_c)):
            labels.append(f'a{ell}[{j}]')
            conv_vals.append(a_c[j])
            fwd_vals.append(a_f[j])

    if 'a_output' in converged_state:
        a_c = converged_state['a_output'][:, sample_idx].detach().numpy()
        a_f = forward_state['a_output'][:, sample_idx].detach().numpy()
        for j in range(len(a_c)):
            labels.append(f'aout[{j}]')
            conv_vals.append(a_c[j])
            fwd_vals.append(a_f[j])

    n = len(labels)
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 0.5), 7),
                                    gridspec_kw={'height_ratios': [3, 1]})

    width = 0.35
    ax1.bar(x - width/2, fwd_vals, width, color=PALETTE['forward'],
            label='Forward pass', alpha=0.8)
    ax1.bar(x + width/2, conv_vals, width, color=PALETTE['converged'],
            label='Converged dynamics', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8,
                         family='monospace')
    ax1.set_ylabel('Value')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    _clean_spines(ax1)

    # Error plot
    errors = np.array(conv_vals) - np.array(fwd_vals)
    ax2.bar(x, errors, color='gray', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8,
                         family='monospace')
    ax2.set_ylabel('Error')
    ax2.set_title(r'Difference (converged $-$ forward pass)')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    _clean_spines(ax2)

    fig.tight_layout()
    return fig


# ============================================================================
# Summary dashboard
# ============================================================================

def plot_dynamics_dashboard(
    states: List[Dict],
    discord_history: List[Dict],
    sheaf: NeuralSheaf,
    forward_state: Optional[Dict] = None,
    title: str = 'Sheaf Dynamics Dashboard',
) -> plt.Figure:
    """
    A 2x2 dashboard summarizing a dynamics run.

    Panels: total discord, discord by edge type, discord by layer,
    and output stalk trajectory (first 2 components or component evolution).

    Parameters
    ----------
    states : list of dict
    discord_history : list of dict
    sheaf : NeuralSheaf
    forward_state : dict, optional
    title : str

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    plot_discord_evolution(discord_history, ax=axes[0, 0],
                           title='Total Discord')
    plot_discord_by_edge_type(discord_history, ax=axes[0, 1],
                              title='Discord by Edge Type')
    plot_discord_by_layer(discord_history, ax=axes[1, 0],
                           title='Discord by Layer')

    # Output trajectory: plot component evolution for output z
    ax = axes[1, 1]
    iters = [d['iteration'] for d in discord_history]
    z_out = torch.stack([s['z'][-1][:, 0] for s in states]).numpy()
    n_comp = min(z_out.shape[1], 4)

    for j in range(n_comp):
        ax.plot(iters, z_out[:, j], '-', linewidth=1.2,
                label=rf'$z_{{\mathrm{{out}}}}[{j}]$')
        if forward_state is not None:
            ax.axhline(forward_state['z'][-1][j, 0].item(),
                       linestyle='--', alpha=0.4, linewidth=1)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title(r'Output $z$ Components')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    _clean_spines(ax)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ============================================================================
# Training visualization (Phase 3)
# ============================================================================

def plot_training_curves(
    histories: List[Dict],
    labels: List[str],
    title: str = 'Training Curves',
    ax_train: Optional[plt.Axes] = None,
    ax_test: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot train and test loss over epochs for one or more training runs.

    Parameters
    ----------
    histories : list of dict
        Each dict must have ``'train_loss'`` (list of floats); may have
        ``'test_loss'``.
    labels : list of str
        Legend label for each history.
    title : str
    ax_train, ax_test : Axes, optional
        If both provided, plots into them. Otherwise creates a new figure.

    Returns
    -------
    matplotlib Figure
    """
    if ax_train is None or ax_test is None:
        fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig = ax_train.figure

    colors = plt.cm.tab10.colors
    for i, (h, lab) in enumerate(zip(histories, labels)):
        epochs = range(len(h['train_loss']))
        ax_train.plot(epochs, h['train_loss'], '-', color=colors[i],
                      lw=1.5, label=lab)
        if h.get('test_loss'):
            ax_test.plot(epochs, h['test_loss'], '-', color=colors[i],
                         lw=1.5, label=lab)

    for ax, name in [(ax_train, 'Train'), (ax_test, 'Test')]:
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name} Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_regression_surfaces(
    trainers: list,
    labels: List[str],
    target_func,
    input_range: Tuple[float, float] = (-2.0, 2.0),
    n_pts: int = 60,
    suptitle: str = 'Regression Results',
) -> plt.Figure:
    """
    Compare learned regression surfaces against a target.

    Row 1: 3D surface (target + one per trainer).
    Row 2: absolute error heatmaps + predicted-vs-actual scatter.

    Parameters
    ----------
    trainers : list of SheafTrainer
    labels : list of str
    target_func : callable
        Maps (2, n) -> (1, n).
    input_range : (float, float)
    n_pts : int
        Grid resolution per axis.
    suptitle : str

    Returns
    -------
    matplotlib Figure
    """
    lo, hi = input_range
    grid = torch.linspace(lo, hi, n_pts, dtype=torch.float64)
    xx, yy = torch.meshgrid(grid, grid, indexing='ij')
    X_g = torch.stack([xx.flatten(), yy.flatten()])
    Z_true = target_func(X_g).squeeze(0).numpy().reshape(n_pts, n_pts)
    xx_np, yy_np = xx.numpy(), yy.numpy()

    n_models = len(trainers)
    fig = plt.figure(figsize=(5 * (n_models + 1), 10))

    # Row 1: 3D surfaces
    ax = fig.add_subplot(2, n_models + 1, 1, projection='3d')
    ax.plot_surface(xx_np, yy_np, Z_true, cmap='viridis',
                    alpha=0.85, edgecolor='none')
    ax.set_title('Target', fontsize=10)
    ax.set_xlabel('x\u2081'); ax.set_ylabel('x\u2082')

    for i, (tr, lab) in enumerate(zip(trainers, labels)):
        with torch.no_grad():
            Z_pred = tr.predict(X_g).squeeze(0).numpy().reshape(n_pts, n_pts)
        ax = fig.add_subplot(2, n_models + 1, i + 2, projection='3d')
        ax.plot_surface(xx_np, yy_np, Z_pred, cmap='plasma',
                        alpha=0.85, edgecolor='none')
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel('x\u2081'); ax.set_ylabel('x\u2082')

    # Row 2: error heatmaps + scatter
    for i, (tr, lab) in enumerate(zip(trainers, labels)):
        with torch.no_grad():
            Z_pred = tr.predict(X_g).squeeze(0).numpy().reshape(n_pts, n_pts)
        error = np.abs(Z_pred - Z_true)
        ax = fig.add_subplot(2, n_models + 1, n_models + 2 + i)
        im = ax.contourf(xx_np, yy_np, error, levels=30, cmap='hot')
        plt.colorbar(im, ax=ax, label='|error|')
        ax.set_title(f'{lab} \u2014 abs. error', fontsize=9)
        ax.set_xlabel('x\u2081'); ax.set_ylabel('x\u2082')
        ax.set_aspect('equal')

    # Predicted vs actual scatter (last panel)
    ax = fig.add_subplot(2, n_models + 1, 2 * (n_models + 1))
    colors_sc = plt.cm.tab10.colors
    for i, (tr, lab) in enumerate(zip(trainers, labels)):
        with torch.no_grad():
            Z_pred = tr.predict(X_g).squeeze(0).numpy().flatten()
        Z_flat = Z_true.flatten()
        sub = np.random.choice(len(Z_flat), min(500, len(Z_flat)), replace=False)
        ax.scatter(Z_flat[sub], Z_pred[sub], s=4, alpha=0.4,
                   color=colors_sc[i], label=lab)
    lims = [Z_true.min(), Z_true.max()]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.6, label='y = x')
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual', fontsize=9)
    ax.legend(fontsize=7); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_binary_boundaries(
    trainers: list,
    labels: List[str],
    X: torch.Tensor,
    Y: torch.Tensor,
    suptitle: str = 'Binary Classification',
    grid_res: int = 200,
) -> plt.Figure:
    """
    Plot decision boundaries for binary classifiers.

    Parameters
    ----------
    trainers : list of SheafTrainer
    labels : list of str
    X : Tensor of shape (2, n)
    Y : Tensor of shape (1, n)
    suptitle : str
    grid_res : int

    Returns
    -------
    matplotlib Figure
    """
    n = len(trainers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    margin = 0.5
    x_min = X[0].min().item() - margin
    x_max = X[0].max().item() + margin
    y_min = X[1].min().item() - margin
    y_max = X[1].max().item() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                          np.linspace(y_min, y_max, grid_res))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()].T, dtype=torch.float64)
    Y_np = Y.squeeze(0).numpy()

    for i, (tr, lab) in enumerate(zip(trainers, labels)):
        ax = axes[0, i]
        with torch.no_grad():
            probs = tr.predict(grid).squeeze(0).numpy().reshape(xx.shape)
        ax.contourf(xx, yy, probs, levels=50, cmap='RdBu_r', alpha=0.6)
        ax.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
        ax.scatter(X[0, Y_np < 0.5].numpy(), X[1, Y_np < 0.5].numpy(),
                   c='blue', s=10, alpha=0.5, edgecolors='k',
                   linewidths=0.2, label='Class 0')
        ax.scatter(X[0, Y_np >= 0.5].numpy(), X[1, Y_np >= 0.5].numpy(),
                   c='red', s=10, alpha=0.5, edgecolors='k',
                   linewidths=0.2, label='Class 1')
        # Reference circles for circular data
        for r, ls in [(1.0, '--'), (1.5, ':'), (2.5, ':')]:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color='gray',
                                    linestyle=ls, linewidth=1.2, alpha=0.5))
        acc = _compute_accuracy(tr, X, Y, 'binary')
        ax.set_title(f'{lab}  (acc = {acc:.1%})', fontsize=10)
        ax.set_xlabel('x\u2081'); ax.set_ylabel('x\u2082')
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_multiclass_boundaries(
    trainers: list,
    labels: List[str],
    X: torch.Tensor,
    Y: torch.Tensor,
    n_classes: int = 4,
    suptitle: str = 'Multiclass Classification',
    grid_res: int = 200,
) -> plt.Figure:
    """
    Plot decision boundaries for multiclass classifiers.

    Parameters
    ----------
    trainers : list of SheafTrainer
    labels : list of str
    X : Tensor of shape (2, n)
    Y : Tensor of shape (n_classes, n) one-hot
    n_classes : int
    suptitle : str
    grid_res : int

    Returns
    -------
    matplotlib Figure
    """
    n = len(trainers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
              '#9467bd', '#8c564b']
    cmap = ListedColormap(colors[:n_classes])
    margin = 1.0
    x_min = X[0].min().item() - margin
    x_max = X[0].max().item() + margin
    y_min = X[1].min().item() - margin
    y_max = X[1].max().item() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                          np.linspace(y_min, y_max, grid_res))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()].T, dtype=torch.float64)
    true_labels = Y.argmax(dim=0).numpy()

    for i, (tr, lab) in enumerate(zip(trainers, labels)):
        ax = axes[0, i]
        with torch.no_grad():
            cls = tr.predict_classes(grid).numpy().reshape(xx.shape)
        ax.contourf(xx, yy, cls, levels=np.arange(-0.5, n_classes, 1),
                    cmap=cmap, alpha=0.3)
        ax.contour(xx, yy, cls,
                   levels=np.arange(0.5, n_classes - 0.5, 1),
                   colors='black', linewidths=1.5)
        for c in range(n_classes):
            mask = true_labels == c
            ax.scatter(X[0, mask].numpy(), X[1, mask].numpy(),
                       c=colors[c], s=12, alpha=0.7, edgecolors='k',
                       lw=0.2, label=f'Class {c}')
        acc = _compute_accuracy(tr, X, Y, 'multiclass')
        ax.set_title(f'{lab}  (acc = {acc:.1%})', fontsize=10)
        ax.set_xlabel('x\u2081'); ax.set_ylabel('x\u2082')
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ============================================================================
# Discord scatter plots (predicted-vs-actual and residuals)
# ============================================================================

def plot_discord_pva(
    sheaf,
    state: Dict,
    title: str = '',
    figsize_per_panel: Tuple[float, float] = (5, 4.5),
) -> plt.Figure:
    """
    Predicted-vs-actual scatter for every edge in the sheaf graph.

    Weight edges plot (W a + b)_j on x vs z_j on y with an identity
    reference line.  ReLU edges plot z_j on x vs a_j on y with the
    ReLU curve overlaid.  Points are coloured by neuron index.

    Parameters
    ----------
    sheaf : NeuralSheaf
    state : dict
        Training stalk state with 'z' and 'a' lists.
    title : str
        Prefix for subplot titles.
    figsize_per_panel : tuple
        (width, height) per subplot panel.
    """
    from .discord import extract_edge_data

    weight_edges, relu_edges = extract_edge_data(sheaf, state)
    k = sheaf.k
    ncols = max(k + 1, max(k, 1))
    nrows = 2 if k > 0 else 1
    fw, fh = figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for ell, ed in enumerate(weight_edges):
        ax = axes[0, ell]
        sc = ax.scatter(ed['predicted'], ed['actual'], alpha=0.3, s=8,
                        c=ed['neuron_idx'], cmap='viridis', rasterized=True)
        lo = min(ed['predicted'].min(), ed['actual'].min())
        hi = max(ed['predicted'].max(), ed['actual'].max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2, label='$y = x$')
        ax.set_xlabel(f'$(W^{{({ell+1})}} a^{{({ell})}} + b^{{({ell+1})}})_j$')
        ax.set_ylabel(f'$z_j^{{({ell+1})}}$')
        ax.set_title(f'{title}Weight edge {ell+1}')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
        if ed['neuron_idx'].max() > 0:
            fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02).set_label(
                'Neuron', fontsize=8)
    for j in range(len(weight_edges), ncols):
        axes[0, j].set_visible(False)

    for ell, ed in enumerate(relu_edges):
        ax = axes[1, ell]
        sc = ax.scatter(ed['z'], ed['a'], alpha=0.3, s=8,
                        c=ed['neuron_idx'], cmap='plasma', rasterized=True)
        zr = np.linspace(ed['z'].min() - 0.1, ed['z'].max() + 0.1, 200)
        ax.plot(zr, np.maximum(zr, 0), 'r--', lw=1.5,
                label='$\\mathrm{ReLU}(z)$')
        ax.set_xlabel(f'$z_j^{{({ell+1})}}$')
        ax.set_ylabel(f'$a_j^{{({ell+1})}}$')
        ax.set_title(f'{title}ReLU edge {ell+1}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
        if ed['neuron_idx'].max() > 0:
            fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02).set_label(
                'Neuron', fontsize=8)
    for j in range(len(relu_edges), ncols):
        if nrows > 1:
            axes[1, j].set_visible(False)

    fig.suptitle(f'{title}Stalk Consistency (Predicted vs Actual)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_discord_residuals(
    sheaf,
    state: Dict,
    title: str = '',
    figsize_per_panel: Tuple[float, float] = (5, 4),
) -> plt.Figure:
    """
    Discord residual scatter coloured by active/inactive status.

    Weight edges: x = mapped value, y = (predicted - actual), coloured
    by sign of the mapped value.  ReLU edges: x = z_j, y = ReLU(z_j) - a_j,
    coloured by active (z >= 0) vs inactive (z < 0).

    Parameters
    ----------
    sheaf : NeuralSheaf
    state : dict
        Training stalk state.
    title : str
        Prefix for subplot titles.
    figsize_per_panel : tuple
        (width, height) per subplot panel.
    """
    from .discord import extract_edge_data

    weight_edges, relu_edges = extract_edge_data(sheaf, state)
    k = sheaf.k
    ncols = max(k + 1, max(k, 1))
    nrows = 2 if k > 0 else 1
    fw, fh = figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for ell, ed in enumerate(weight_edges):
        ax = axes[0, ell]
        pos = ed['predicted'] >= 0
        ax.scatter(ed['predicted'][pos], ed['discord'][pos],
                   alpha=0.3, s=6, c='steelblue', label='Predicted \u2265 0',
                   rasterized=True)
        ax.scatter(ed['predicted'][~pos], ed['discord'][~pos],
                   alpha=0.3, s=6, c='lightcoral', label='Predicted < 0',
                   rasterized=True)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel(f'$(W^{{({ell+1})}} a^{{({ell})}} + b^{{({ell+1})}})_j$')
        ax.set_ylabel('Discord')
        ax.set_title(f'{title}Weight edge {ell+1} residuals')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    for j in range(len(weight_edges), ncols):
        axes[0, j].set_visible(False)

    for ell, ed in enumerate(relu_edges):
        ax = axes[1, ell]
        act = ed['active']
        ax.scatter(ed['z'][act], ed['discord'][act],
                   alpha=0.3, s=6, c='darkorange',
                   label='Active ($z \\geq 0$)', rasterized=True)
        ax.scatter(ed['z'][~act], ed['discord'][~act],
                   alpha=0.3, s=6, c='mediumpurple',
                   label='Inactive ($z < 0$)', rasterized=True)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel(f'$z_j^{{({ell+1})}}$')
        ax.set_ylabel('Discord: $\\mathrm{ReLU}(z_j) - a_j$')
        ax.set_title(f'{title}ReLU edge {ell+1} residuals')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    for j in range(len(relu_edges), ncols):
        if nrows > 1:
            axes[1, j].set_visible(False)

    fig.suptitle(f'{title}Discord Residuals', fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_discord_summary(
    sheaf_discords: Dict[str, Dict],
    sgd_discords: Dict[str, Dict],
    level_labels: List[str],
    level_colors: List[str],
    title: str = 'Discord Summary: Sheaf vs SGD',
) -> plt.Figure:
    """
    Four-panel summary comparing discord across training levels and methods.

    Panel 1 — **Loss (output discord)**: prediction error for both methods.
    Panel 2 — **Internal edge discord**: sheaf training discord vs SGD
              inference discord (after heat diffusion from forward pass).
    Panel 3 — **Sheaf per-edge breakdown**: sheaf training discord
              decomposed by edge type.
    Panel 4 — **SGD per-edge breakdown**: SGD pinned discord decomposed
              by edge type.

    Parameters
    ----------
    sheaf_discords : dict  tag → dict with keys 'total', 'weight_*',
        'activation_*', 'loss'.
    sgd_discords : dict  tag → dict with keys 'total', 'weight_*',
        'activation_*', 'loss', 'inference_total'.
    level_labels : list of str
    level_colors : list of str
    title : str

    Returns
    -------
    plt.Figure
    """
    tags = list(sheaf_discords.keys())
    n_levels = len(tags)

    fig, axes = plt.subplots(
        1, 4, figsize=(5 + 3 * n_levels, 5))
    ax1, ax2, ax3, ax4 = axes

    x = np.arange(n_levels)
    w = 0.35

    def _annotate(ax, bars, fmt='.2e'):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f'{h:{fmt}}', ha='center', va='bottom', fontsize=7)

    # ── Panel 1: Loss (output discord) ──
    sheaf_loss = [sheaf_discords[t].get('loss', 0) for t in tags]
    sgd_loss = [sgd_discords[t].get('loss', 0) for t in tags]

    bars_s = ax1.bar(x - w / 2, sheaf_loss, w, label='Sheaf',
                     color='steelblue', alpha=0.8)
    bars_g = ax1.bar(x + w / 2, sgd_loss, w, label='SGD',
                     color='darkorange', alpha=0.8)
    _annotate(ax1, bars_s, '.4f')
    _annotate(ax1, bars_g, '.4f')

    ax1.set_xticks(x)
    ax1.set_xticklabels(level_labels, fontsize=8, rotation=15, ha='right')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss (Output Discord)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis='y')

    # ── Panel 2: Internal edge discord ──
    sheaf_totals = [sheaf_discords[t]['total'] for t in tags]
    sgd_infer = [sgd_discords[t].get('inference_total',
                                      sgd_discords[t]['total'])
                 for t in tags]

    bars_s = ax2.bar(x - w / 2, sheaf_totals, w,
                     label='Sheaf (training)', color='steelblue', alpha=0.8)
    bars_g = ax2.bar(x + w / 2, sgd_infer, w,
                     label='SGD (pinned)', color='darkorange', alpha=0.8)
    _annotate(ax2, bars_s)
    _annotate(ax2, bars_g)

    ax2.set_xticks(x)
    ax2.set_xticklabels(level_labels, fontsize=8, rotation=15, ha='right')
    ax2.set_ylabel('Internal Edge Discord')
    ax2.set_title('Internal Discord')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')
    all_vals = sheaf_totals + sgd_infer
    positives = [v for v in all_vals if v > 0]
    if positives and max(positives) / min(positives) > 100:
        ax2.set_yscale('log')

    # ── Panel 3: Per-edge-type breakdown (sheaf) ──
    def _plot_breakdown(ax, discords, label_prefix):
        weight_keys = sorted(k for k in discords[tags[0]]
                             if k.startswith('weight_'))
        act_keys = sorted(k for k in discords[tags[0]]
                          if k.startswith('activation_'))

        weight_vals = [[discords[t].get(k, 0) for k in weight_keys]
                       for t in tags]
        act_vals = [[discords[t].get(k, 0) for k in act_keys]
                    for t in tags]

        bottom = np.zeros(n_levels)
        for i, k in enumerate(weight_keys):
            vals = [weight_vals[j][i] for j in range(n_levels)]
            ax.bar(x, vals, 0.6, bottom=bottom, label=k.replace('_', ' '),
                    alpha=0.8)
            bottom += np.array(vals)
        for i, k in enumerate(act_keys):
            vals = [act_vals[j][i] for j in range(n_levels)]
            ax.bar(x, vals, 0.6, bottom=bottom, label=k.replace('_', ' '),
                    alpha=0.8)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(level_labels, fontsize=8, rotation=15, ha='right')
        ax.set_ylabel('Discord')
        ax.set_title(f'{label_prefix} Per-Edge Breakdown')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')
        if bottom.min() > 0:
            ax.set_yscale('log')

    _plot_breakdown(ax3, sheaf_discords, 'Sheaf')

    # ── Panel 4: Per-edge-type breakdown (SGD) ──
    _plot_breakdown(ax4, sgd_discords, 'SGD')

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ============================================================================
# Spectral analysis plots
# ============================================================================

def plot_eigenvalue_spectra(
    eigvals_list: List[List],
    labels: List[str],
    colors: List[str],
    title: str = '',
) -> plt.Figure:
    """
    Eigenvalue spectra in separate panels with shared log y-axis.

    Each panel shows faint per-sample lines and a bold median curve.

    Parameters
    ----------
    eigvals_list : list of list of ndarray
        eigvals_list[i][j] is the sorted eigenvalue array for model i,
        sample j.
    labels : list of str
        One label per model.
    colors : list of str
        One colour per model.
    title : str
        Suptitle prefix.
    """
    n = len(eigvals_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, eigvals, label, color in zip(axes, eigvals_list, labels, colors):
        for ev in eigvals:
            pos = np.sort(ev[ev > 1e-10])
            ax.plot(range(len(pos)), pos, alpha=0.2, color=color, lw=0.8)
        all_sorted = [np.sort(ev[ev > 1e-10]) for ev in eigvals]
        max_len = max(len(s) for s in all_sorted)
        padded = np.full((len(all_sorted), max_len), np.nan)
        for i, s in enumerate(all_sorted):
            padded[i, :len(s)] = s
        ax.plot(range(max_len), np.nanmedian(padded, axis=0),
                color=color, lw=2.5, alpha=0.9, label='Median')
        ax.set_yscale('log')
        ax.set_xlabel('Non-zero eigenvalue index')
        ax.set_title(label)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel('Eigenvalue')
    fig.suptitle(
        f'{title}$L_F[\\Omega,\\Omega]$ Non-zero Eigenvalue Spectra',
        fontsize=13)
    fig.tight_layout()
    return fig


def plot_spectral_gap_dist(
    gaps_list: List,
    labels: List[str],
    colors: List[str],
    n_bins: int = 35,
    title: str = '',
    xlabel: str = 'Smallest non-zero eigenvalue of $L_F[\\Omega,\\Omega]$',
    dist_label: str = 'Spectral Gap',
    data_range: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, List[plt.Figure]]:
    """
    Spectral gap distributions with consistent log-spaced bins.

    Returns a combined overlay figure plus individual figures.
    y-axis = fraction of input samples per bin.

    Parameters
    ----------
    gaps_list : list of ndarray
    labels, colors : matching lists of str.
    n_bins : int
    title : str
    xlabel : str
    dist_label : str
    data_range : (lo, hi) or None
        If provided, use these bounds for the log-spaced bins instead of
        computing from the data.  Pass the same range to sheaf and SGD
        calls to get identical x-axes for visual comparison.

    Returns
    -------
    fig_combined : Figure
    figs_individual : list of Figure
    """
    all_data = np.concatenate(gaps_list)
    if data_range is not None:
        lo, hi = data_range
    else:
        lo = all_data[all_data > 0].min() * 0.85
        hi = all_data.max() * 1.15
    bins = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    x_lo, x_hi = bins[0], bins[-1]

    y_max = 0
    for g in gaps_list:
        c, _ = np.histogram(g, bins=bins)
        y_max = max(y_max, (c / len(g)).max())
    y_max *= 1.15

    def _bar(ax, data, **kw):
        counts, _ = np.histogram(data, bins=bins)
        ax.bar(bins[:-1], counts / len(data), width=np.diff(bins),
               align='edge', **kw)

    fig_c, ax = plt.subplots(figsize=(7, 4.5))
    for g, label, color in zip(gaps_list, labels, colors):
        _bar(ax, g, alpha=0.5, color=color, label=label,
             edgecolor='white', linewidth=0.3)
    ax.set_xscale('log'); ax.set_xlim(x_lo, x_hi); ax.set_ylim(0, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Fraction of input samples')
    ax.set_title(f'{title}{dist_label} Distribution \u2014 All Levels')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig_c.tight_layout()

    figs_i = []
    for g, label, color in zip(gaps_list, labels, colors):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _bar(ax, g, alpha=0.7, color=color,
             edgecolor='white', linewidth=0.3)
        med, mean, std = np.median(g), np.mean(g), np.std(g)
        ax.axvline(med, color='black', ls='--', lw=1.5,
                    label=f'Median: {med:.4f}')
        ax.axvline(mean, color='gray', ls=':', lw=1.5,
                    label=f'Mean: {mean:.4f}')
        ax.set_xscale('log'); ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Fraction of input samples')
        ax.set_title(f'{title}{dist_label} \u2014 {label}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        ax.text(0.97, 0.92, f'Std: {std:.4f}\nN = {len(g)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8))
        fig.tight_layout()
        figs_i.append(fig)

    return fig_c, figs_i


def plot_spectral_gap_training(
    history: Dict,
    title: str = '',
) -> plt.Figure:
    """
    Spectral gap median +/- IQR over training steps (log y-axis).

    Parameters
    ----------
    history : dict
        Keys: ``'step'`` and ``'spectral'`` sub-dict with
        ``'median'``, ``'q25'``, ``'q75'``.
    title : str
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    s = history['step']
    spec = history['spectral']
    ax.plot(s, spec['median'], '-o', ms=4, color='steelblue',
            label='Median')
    ax.fill_between(s, spec['q25'], spec['q75'],
                    alpha=0.2, color='steelblue', label='IQR')
    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel(
        'Smallest non-zero eigenvalue of $L_F[\\Omega,\\Omega]$')
    ax.set_title(f'{title}Spectral Gap During Training')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_loss_spectral_dual(
    history: Dict,
    title: str = '',
) -> plt.Figure:
    """
    Dual-axis: loss (left, blue, log) and spectral gap (right, red, log).

    Parameters
    ----------
    history : dict
        Keys: ``'step'``, ``'train_loss'``, ``'test_loss'``, and
        ``'spectral'`` sub-dict with ``'median'``.
    title : str
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    s = history['step']
    ax1.plot(s, history['train_loss'], 'b-o', ms=4, label='Train loss')
    ax1.plot(s, history['test_loss'], 'b--s', ms=4, label='Test loss')
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(s, history['spectral']['median'], 'r-^', ms=4,
             label='Spectral gap (median)')
    ax2.set_ylabel('Spectral gap', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yscale('log')

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper right')
    ax1.set_title(f'{title}Loss and Spectral Gap')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================================
# Extended spectral analysis plots
# ============================================================================

def plot_condition_number_dist(
    kappa_list: List,
    labels: List[str],
    colors: List[str],
    n_bins: int = 35,
    title: str = '',
    data_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Condition number (κ = λ_max/λ₁) distributions across training levels.

    Handles infinite condition numbers by filtering them before binning.

    Parameters
    ----------
    kappa_list : list of ndarray
        Condition numbers per sample, one array per training level.
    labels, colors : matching lists of str.
    n_bins : int
    title : str
    data_range : (lo, hi) or None
        If provided, use these bounds for the log-spaced bins.  Pass the
        same range to sheaf and SGD calls for identical x-axes.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_data = np.concatenate(kappa_list)
    finite = all_data[np.isfinite(all_data)]
    if len(finite) == 0:
        ax.text(0.5, 0.5, 'No finite condition numbers',
                transform=ax.transAxes, ha='center')
        return fig
    if data_range is not None:
        lo, hi = data_range
    else:
        lo = finite.min() * 0.85
        hi = finite.max() * 1.15
    bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), n_bins + 1)
    for g, label, color in zip(kappa_list, labels, colors):
        g_f = g[np.isfinite(g)]
        counts, _ = np.histogram(g_f, bins=bins)
        ax.bar(bins[:-1], counts / max(len(g_f), 1), width=np.diff(bins),
               align='edge', alpha=0.5, color=color, label=label,
               edgecolor='white', linewidth=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'Condition number $\kappa = \lambda_{\max} / \lambda_1$')
    ax.set_ylabel('Fraction of input samples')
    ax.set_title(f'{title}Condition Number Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_spectral_tracking_extended(
    history: Dict,
    title: str = '',
) -> plt.Figure:
    """
    Three-panel spectral tracking: λ₁, λ_max, and κ over training.

    Each panel shows median ± IQR bands (log y-axis).

    Parameters
    ----------
    history : dict
        Must have keys: 'step', and for each of 'spectral', 'lambda_max',
        'condition': sub-dicts with 'median', 'q25', 'q75'.
    title : str
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    s = history['step']

    panels = [
        ('spectral', r'$\lambda_1$ (spectral gap)', 'steelblue'),
        ('lambda_max', r'$\lambda_{\max}$', 'firebrick'),
        ('condition', r'$\kappa = \lambda_{\max}/\lambda_1$', 'darkorange'),
    ]
    for ax, (key, ylabel, color) in zip(axes, panels):
        sub = history[key]
        ax.plot(s, sub['median'], '-o', ms=3, color=color, label='Median')
        ax.fill_between(s, sub['q25'], sub['q75'],
                        alpha=0.2, color=color, label='IQR')
        ax.set_yscale('log')
        ax.set_xlabel('Training step')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _clean_spines(ax)

    fig.suptitle(f'{title}Spectral Quantities During Training', fontsize=13)
    fig.tight_layout()
    return fig


def plot_eigenvector_layer_energy_row(
    energy_list: List['np.ndarray'],
    block_ranges: List[Tuple],
    col_labels: List[str],
    colors: Optional[List[str]] = None,
    suptitle: str = '',
) -> plt.Figure:
    """
    Side-by-side bar charts of eigenvector layer energy across training levels.

    Produces one row of subplots sharing the same y-axis for direct
    comparison across columns (e.g. poorly trained, intermediate,
    well trained).

    Parameters
    ----------
    energy_list : list of ndarray, each (n_blocks, n_samples)
        Layer energy fractions, one per column.
    block_ranges : list of (label, start, end)
        Block labels (shared across columns).
    col_labels : list of str
        Title for each column.
    colors : list of str or None
        Bar colour for each column.  Defaults to ``'steelblue'`` for all.
    suptitle : str
        Figure suptitle.

    Returns
    -------
    plt.Figure
    """
    n_cols = len(energy_list)
    n_blocks = energy_list[0].shape[0]
    blk_labels = [br[0] for br in block_ranges]
    if colors is None:
        colors = ['steelblue'] * n_cols

    fig, axes = plt.subplots(
        1, n_cols, figsize=(max(5, n_blocks * 0.7) * n_cols, 4.5),
        sharey=True)
    if n_cols == 1:
        axes = [axes]

    x = np.arange(n_blocks)
    for ax, energy, col_label, color in zip(axes, energy_list,
                                            col_labels, colors):
        means = energy.mean(axis=1)
        stds = energy.std(axis=1)
        ax.bar(x, means, yerr=stds, capsize=3, color=color,
               edgecolor='white', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(blk_labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(col_label, fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')
        _clean_spines(ax)
    axes[0].set_ylabel(r'$\|v_{\mathrm{block}}\|^2 / \|v\|^2$')

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_stability_overlay(
    cosines_list: List['np.ndarray'],
    labels: List[str],
    colors: List[str],
    eigvec_label: str = 'Fiedler',
    title: str = '',
    n_bins: int = 30,
) -> plt.Figure:
    """
    Overlay eigenvector stability histograms for multiple conditions.

    All histograms share the same bins on [0, 1] so that bar heights
    are directly comparable.

    Parameters
    ----------
    cosines_list : list of ndarray
        Pairwise |cos θ| arrays, one per condition.
    labels : list of str
        Legend label for each condition.
    colors : list of str
    eigvec_label : str
        Eigenvector name (e.g. ``'Fiedler'``, ``r'$\\lambda_{\\max}$'``).
    title : str
        Prepended to the eigvec_label in the plot title.
    n_bins : int

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    edges = np.linspace(0, 1, n_bins + 1)
    widths = np.diff(edges)

    for cosines, label, color in zip(cosines_list, labels, colors):
        counts, _ = np.histogram(cosines, bins=edges)
        fracs = counts / max(len(cosines), 1)
        mu = cosines.mean()
        ax.bar(edges[:-1], fracs, width=widths, align='edge',
               alpha=0.45, color=color, edgecolor='white', linewidth=0.3,
               label=f'{label} (mean={mu:.3f})')

    ax.set_xlabel(r'$|\cos\theta|$ (pairwise)')
    ax.set_ylabel('Fraction of input sample pairs')
    ax.set_xlim(0, 1.05)
    ax.set_title(f'{title} — {eigvec_label}' if title else eigvec_label)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    _clean_spines(ax)
    fig.tight_layout()
    return fig


def plot_neuron_contributions(
    neuron_data: Dict,
    eigvec_label: str = 'Fiedler',
    color: str = 'steelblue',
    max_neurons: int = 20,
    title: str = '',
) -> Optional[plt.Figure]:
    """
    Bar chart showing per-neuron energy within the dominant stalk block.

    If the dominant block has only one neuron, prints the value instead
    of producing a single-bar chart and returns ``None``.

    Parameters
    ----------
    neuron_data : dict
        Output of ``_top_block_neuron_contributions`` with keys:
        'block_label', 'neuron_energy_mean', 'neuron_energy_std',
        'top_neurons'.
    eigvec_label : str
        Label for the eigenvector type.
    color : str
    max_neurons : int
        Show at most this many neurons (sorted by descending energy).
    title : str

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` if the block has only one neuron (value is printed
        instead of plotted).
    """
    block_label = neuron_data['block_label']
    means = neuron_data['neuron_energy_mean']
    stds = neuron_data['neuron_energy_std']
    order = neuron_data['top_neurons']

    # Single-neuron block: print instead of plotting
    if len(means) == 1:
        print(f"  {title}{eigvec_label} — {block_label}: "
              f"|v|² = {means[0]:.4f} ± {stds[0]:.4f} (single neuron)")
        return None

    n_show = min(max_neurons, len(means))
    idx = order[:n_show]

    fig, ax = plt.subplots(figsize=(max(6, n_show * 0.4), 4.5))
    x = np.arange(n_show)
    ax.bar(x, means[idx], yerr=stds[idx], capsize=2, color=color,
           edgecolor='white', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(j) for j in idx], fontsize=7,
                       rotation=45 if n_show > 10 else 0)
    ax.set_xlabel(f'Neuron index within {block_label}')
    ax.set_ylabel(r'$|v_j|^2$ (mean over samples)')
    ax.set_title(f'{title}{eigvec_label} — Neuron Contributions in {block_label}')
    ax.grid(True, alpha=0.2, axis='y')
    _clean_spines(ax)
    fig.tight_layout()
    return fig


def plot_loss_spectral_dual_extended(
    history: Dict,
    title: str = '',
) -> plt.Figure:
    """
    Dual-axis: loss (left) and λ₁ + λ_max + κ (right, log).

    Parameters
    ----------
    history : dict
        Keys: 'step', 'train_loss', 'test_loss', and sub-dicts
        'spectral', 'lambda_max', 'condition' each with 'median'.
    title : str
    """
    fig, ax1 = plt.subplots(figsize=(11, 5))
    s = history['step']
    ax1.plot(s, history['train_loss'], 'b-o', ms=3, label='Train loss')
    ax1.plot(s, history['test_loss'], 'b--s', ms=3, label='Test loss')
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(s, history['spectral']['median'], 'g-^', ms=3,
             label=r'$\lambda_1$ (median)')
    ax2.plot(s, history['lambda_max']['median'], 'r-v', ms=3,
             label=r'$\lambda_{\max}$ (median)')
    ax2.plot(s, history['condition']['median'], color='darkorange',
             marker='D', ms=3, ls='-', label=r'$\kappa$ (median)')
    ax2.set_ylabel('Spectral quantities', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yscale('log')

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper right', fontsize=8)
    ax1.set_title(f'{title}Loss and Spectral Quantities')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================================
# Training comparison plots
# ============================================================================

def plot_restriction_map_norms(
    models: list,
    labels: List[str],
    suptitle: str = 'Restriction Map Norms',
) -> plt.Figure:
    """
    Compare Frobenius norms of weights and biases across models.

    Shows grouped bar charts: one group per layer, one bar per model.
    Works with any object that has ``.weights`` and ``.biases`` lists
    (``NeuralSheaf``, ``SheafTrainer``, ``TraditionalNN``).

    Parameters
    ----------
    models : list
        Objects with ``.weights`` and ``.biases`` attributes.
        For ``SheafTrainer``, accesses ``.sheaf.weights``.
    labels : list of str
        Legend label for each model.
    suptitle : str

    Returns
    -------
    matplotlib Figure
    """
    # Unwrap SheafTrainer -> sheaf if needed
    def _get(m):
        return m.sheaf if hasattr(m, 'sheaf') else m
    objs = [_get(m) for m in models]

    n_models = len(objs)
    n_layers = len(objs[0].weights)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_pos = np.arange(n_layers)
    width = 0.8 / n_models

    for ax, attr, ylabel, title in [
        (axes[0], 'weights', r'$\|W_\ell\|_F$', 'Weight Frobenius Norms'),
        (axes[1], 'biases', r'$\|b_\ell\|$', 'Bias Norms'),
    ]:
        for i, (obj, lab) in enumerate(zip(objs, labels)):
            vals = [getattr(obj, attr)[j].detach().norm().item()
                    for j in range(n_layers)]
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(x_pos + offset, vals, width * 0.9,
                   label=lab, color=colors[i], alpha=0.8)
        ax.set_xlabel('Layer')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([fr'$\ell={l + 1}$' for l in range(n_layers)])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        _clean_spines(ax)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_training_discord_evolution(
    discord_data: List[Dict],
    k: int,
    suptitle: str = 'Training Discord Evolution',
) -> plt.Figure:
    """
    Plot per-edge discord evolution during sheaf training.

    Two panels: (1) log-scale discord by edge over training steps,
    (2) stacked composition showing each edge's fraction of total.

    Parameters
    ----------
    discord_data : list of dict
        From ``SheafTrainer.train`` with ``discord_freq`` set.
        Each dict has ``'step'``, ``'total'``, ``'weight_1'``, ...,
        ``'weight_{k+1}'``, ``'activation_1'``, ..., ``'activation_k'``,
        and optionally ``'output'`` (for non-identity output activations).
    k : int
        Number of hidden layers (determines edge count).
    suptitle : str

    Returns
    -------
    matplotlib Figure
    """
    steps = [d['step'] for d in discord_data]

    # Collect series (keys are 1-indexed: weight_1 .. weight_{k+1},
    # activation_1 .. activation_k, and optionally 'output')
    w_keys = [f'weight_{ell}' for ell in range(1, k + 2)]
    a_keys = [f'activation_{ell}' for ell in range(1, k + 1)]
    has_output = 'output' in discord_data[0]

    w_series = {key: [d.get(key, 0) for d in discord_data] for key in w_keys}
    a_series = {key: [d.get(key, 0) for d in discord_data] for key in a_keys}
    o_series = [d.get('output', 0) for d in discord_data] if has_output else None
    totals = [d['total'] for d in discord_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Panel 1: Log-scale per-edge
    cw = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(w_keys), 1)))
    ca = plt.cm.Oranges(np.linspace(0.4, 0.9, max(len(a_keys), 1)))
    for i, key in enumerate(w_keys):
        ax1.plot(steps, w_series[key], '-', color=cw[i], lw=1.2,
                 label=f'$W_{{{i + 1}}}$')
    for i, key in enumerate(a_keys):
        ax1.plot(steps, a_series[key], '--', color=ca[i], lw=1.2,
                 label=f'$R_{{{i + 1}}}$')
    if has_output:
        ax1.plot(steps, o_series, '-', color=PALETTE['output'], lw=1.5,
                 label=r'Output $\varphi$')
    ax1.plot(steps, totals, 'k-', lw=2, alpha=0.6, label='Total')
    ax1.set_yscale('log')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel(r'Discord')
    ax1.set_title('Per-Edge Discord (log scale)')
    ax1.legend(fontsize=7, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.3)
    _clean_spines(ax1)

    # Panel 2: Stacked composition
    all_vals = [w_series[k] for k in w_keys] + [a_series[k] for k in a_keys]
    all_labels = ([f'$W_{{{i + 1}}}$' for i in range(len(w_keys))]
                  + [f'$R_{{{i + 1}}}$' for i in range(len(a_keys))])
    all_colors = (list(plt.cm.Blues(np.linspace(0.3, 0.8, max(len(w_keys), 1))))
                  + list(plt.cm.Oranges(np.linspace(0.3, 0.8, max(len(a_keys), 1)))))

    if has_output:
        all_vals.append(o_series)
        all_labels.append(r'Output $\varphi$')
        all_colors.append(PALETTE['output'])

    stacked = np.array(all_vals)
    total_arr = stacked.sum(axis=0)
    safe = np.where(total_arr > 0, total_arr, 1.0)
    fracs = stacked / safe

    ax2.stackplot(steps, *fracs, labels=all_labels, colors=all_colors, alpha=0.8)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Fraction of Total Discord')
    ax2.set_title('Discord Composition')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7, ncol=2, loc='center right')
    ax2.grid(True, alpha=0.3, axis='y')
    _clean_spines(ax2)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig