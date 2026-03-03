"""
Discord analysis for the neural sheaf library.

Discord measures edge-by-edge discrepancy: how far a 0-cochain is from
being consistent on each edge of the sheaf graph.  At the harmonic
extension (converged dynamics), all discrepancies are zero.

Provides:
    - ``compute_discord``: per-edge discord during inference (needs a_output)
    - ``compute_training_discord``: per-edge discord during training
    - ``compute_mean_deviation``: stalk deviation from forward pass
    - ``extract_edge_data``: per-coordinate arrays for scatter plots
    - ``compute_pinned_discord``: run diffusion with pinned output, then
      measure discord (used for analyzing SGD-trained networks)
    - ``plotly_pva``: interactive predicted-vs-actual scatter (plotly)
    - ``plotly_residuals``: interactive discord residual scatter (plotly)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from .sheaf import NeuralSheaf
from .activations import relu_mask, get_activation
from .losses import cross_entropy_loss


# ============================================================================
# Private helpers
# ============================================================================

def _squared_norm_mean(v: torch.Tensor) -> float:
    """
    Mean over batch of squared L2 norm.

    Parameters
    ----------
    v : torch.Tensor
        Shape ``(dim, batch_size)``.

    Returns
    -------
    float
        ``mean_b ||v[:, b]||^2``.
    """
    return (v ** 2).sum(dim=0).mean().item()


# ============================================================================
# Discord computation
# ============================================================================

def compute_discord(
    sheaf: NeuralSheaf,
    state: Dict[str, list],
    x: torch.Tensor,
) -> Dict:
    """
    Compute edge-by-edge discrepancy for a sheaf state.

    Each edge in the sheaf graph has a discrepancy measuring how far the
    stalks at its endpoints are from being consistent. At convergence
    (the harmonic extension), all discrepancies are zero.

    Parameters
    ----------
    sheaf : NeuralSheaf
        The neural sheaf encoding the network.
    state : dict
        Current stalk values:
        ``'z': [z_1, ..., z_{k+1}]``, ``'a': [a_0, a_1, ..., a_k]``.
        Optionally ``'a_output'`` if output activation is non-identity.
    x : torch.Tensor
        Input data, shape ``(n_0, batch_size)`` or ``(n_0,)``. Used to
        ensure ``a_0 = x`` in the computation.

    Returns
    -------
    dict
        ``'weight_edges'``: ``{ell: (norm, vector), ...}`` for ell = 1..k+1,
            where vector = ``W_ell @ a_{ell-1} + b_ell - z_ell``
            and norm = mean over batch of ``||vector||^2``.
        ``'activation_edges'``: ``{ell: (norm, vector), ...}`` for ell = 1..k,
            where vector = ``mask_ell * z_ell - a_ell``.
        ``'output_edge'``: ``(norm, vector)`` or ``None``,
            where vector = ``phi(z_{k+1}) - a_output`` (non-identity only).
        ``'total'``: float, sum of all edge norms.
        ``'by_layer'``: ``{ell: float, ...}``, discord grouped by layer.
    """
    x = x.to(device=sheaf.device, dtype=sheaf.dtype)
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    k = sheaf.k
    z = state['z']
    a = state['a']

    # Ensure a[0] is x for the computation
    a_with_x = [x] + list(a[1:])

    weight_edges = {}
    activation_edges = {}
    total = 0.0
    by_layer = {}

    # Weight edges: ell = 1, ..., k+1 (0-indexed in lists: ell = 0..k)
    for ell in range(k + 1):
        W = sheaf.weights[ell]
        b = sheaf.biases[ell]
        a_prev = a_with_x[ell]
        vec = W @ a_prev + b - z[ell]
        norm = _squared_norm_mean(vec)
        weight_edges[ell + 1] = (norm, vec)
        total += norm
        by_layer[ell + 1] = norm

    # Activation edges: ell = 1, ..., k (0-indexed: ell = 0..k-1)
    for ell in range(k):
        mask_f = relu_mask(z[ell]).to(sheaf.dtype)
        vec = mask_f * z[ell] - a_with_x[ell + 1]
        norm = _squared_norm_mean(vec)
        activation_edges[ell + 1] = (norm, vec)
        total += norm
        by_layer[ell + 1] += norm

    # Output edge (non-identity output activation)
    output_edge = None
    if sheaf.output_activation != 'identity':
        phi_z = sheaf._output_activation_fn(z[k])
        a_output = state['a_output']
        vec = phi_z - a_output
        norm = _squared_norm_mean(vec)
        output_edge = (norm, vec)
        total += norm
        by_layer[k + 1] = by_layer.get(k + 1, 0.0) + norm

    return {
        'weight_edges': weight_edges,
        'activation_edges': activation_edges,
        'output_edge': output_edge,
        'total': total,
        'by_layer': by_layer,
    }


def compute_training_discord(
    sheaf: NeuralSheaf,
    state: Dict[str, list],
    y: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Compute per-edge squared discrepancy from a training stalk state.

    During training, stalks evolve via joint dynamics and are generally
    not at the forward-pass equilibrium. The discord on each edge measures
    how far the adjacent stalk values are from consistency -- a per-edge
    diagnostic that reveals which parts of the network are farthest from
    local consistency.

    If ``y`` is provided, the output edge discrepancy is also computed:
    L2 loss ``mean ||z_{k+1} - y||^2`` for identity output, or
    cross-entropy loss for sigmoid/softmax output.

    .. todo::
       Add L1 output edge discord for regression. This would also
       require changing the output edge force in ``SheafTrainer``
       from ``z - y`` to ``sign(z - y)`` for consistency.

    Parameters
    ----------
    sheaf : NeuralSheaf
        The neural sheaf encoding the network.
    state : dict
        Training stalk state: ``'z': [z_1, ..., z_{k+1}]``,
        ``'a': [a_0, a_1, ..., a_k]``.
    y : torch.Tensor, optional
        Training targets, shape ``(n_{k+1}, batch_size)``.
        If provided, the output edge discord is computed and
        included in the result under the key ``'output'``.

    Returns
    -------
    dict
        ``'weight_ell'``: float for ell = 1..k+1 -- weight edge discord.
        ``'activation_ell'``: float for ell = 1..k -- activation edge discord.
        ``'output'``: float (only if ``y`` is provided) -- output edge discord.
        ``'total'``: float -- sum of all edge discords.
    """
    k = sheaf.k
    z, a = state['z'], state['a']

    result = {}
    total = 0.0

    # Weight edges: W_ell @ a_{ell-1} + b_ell - z_ell
    for ell in range(k + 1):
        vec = sheaf.weights[ell] @ a[ell] + sheaf.biases[ell] - z[ell]
        d = _squared_norm_mean(vec)
        result[f'weight_{ell + 1}'] = d
        total += d

    # Activation edges: R_ell z_ell - a_ell
    for ell in range(k):
        mask_f = relu_mask(z[ell]).to(sheaf.dtype)
        vec = mask_f * z[ell] - a[ell + 1]
        d = _squared_norm_mean(vec)
        result[f'activation_{ell + 1}'] = d
        total += d

    # Output edge: discrepancy between network output and targets
    if y is not None:
        z_out = z[k]  # z_{k+1} (0-indexed: z[k] is the last z)
        act = sheaf.output_activation
        if act == 'identity':
            # L2 discord: mean ||z_{k+1} - y||^2
            d = _squared_norm_mean(z_out - y)
        else:
            # Cross-entropy discord
            phi, _ = get_activation(act)
            d = cross_entropy_loss(phi(z_out), y).item()
        result['output'] = d
        total += d
        
    result['total'] = total
    return result


def compute_mean_deviation(
    sheaf: NeuralSheaf,
    state: Dict[str, list],
    x: torch.Tensor,
) -> Dict:
    """
    Compute stalk deviation between training equilibrium and forward pass.

    For each stalk (z and a), computes the per-sample difference between
    the training stalk values and the standard forward pass values.
    Returns both per-sample deviations and their batch mean.

    The mean deviations can be passed to ``NeuralSheaf.corrected_forward``
    for tension-corrected inference. The per-sample statistics (especially
    the std-to-mean ratio) diagnose how input-dependent the tension is:
    a ratio >> 1 means the mean correction is a poor approximation and
    a kernel-weighted or GP-based approach would be more appropriate.

    Parameters
    ----------
    sheaf : NeuralSheaf
        The neural sheaf encoding the network.
    state : dict
        Final training stalk state:
        ``'z': [z_1, ..., z_{k+1}]``, ``'a': [a_0, a_1, ..., a_k]``.
    x : torch.Tensor
        Training inputs, shape ``(n_0, n_train)``.

    Returns
    -------
    dict
        ``'delta_z'``: list of tensors ``[delta_z_1, ..., delta_z_{k+1}]``,
            each shape ``(n_ell, 1)``. Mean deviation per layer.
        ``'delta_a'``: list of tensors ``[delta_a_1, ..., delta_a_k]``,
            each shape ``(n_ell, 1)``. Mean deviation per layer.
        ``'delta_z_per_sample'``: list of tensors, each ``(n_ell, n_train)``.
        ``'delta_a_per_sample'``: list of tensors, each ``(n_ell, n_train)``.
        ``'delta_z_norms'``: list of float. L2 norm of each mean ``delta_z``.
        ``'delta_a_norms'``: list of float. L2 norm of each mean ``delta_a``.
        ``'delta_z_std'``: list of float. Mean per-dimension std of ``delta_z``
            across the batch.
        ``'delta_a_std'``: list of float. Same for ``delta_a``.
    """
    x = x.to(device=sheaf.device, dtype=sheaf.dtype)
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    _, intermediates = sheaf.forward(x)
    z_fwd = intermediates['z']
    a_fwd = intermediates['a']

    z_state = state['z']
    a_state = state['a']

    # Per-sample deviations
    delta_z_samples = [
        z_state[ell] - z_fwd[ell] for ell in range(len(z_fwd))
    ]
    delta_a_samples = [
        a_state[ell + 1] - a_fwd[ell + 1] for ell in range(sheaf.k)
    ]

    # Mean over batch (keepdim for broadcasting during inference)
    delta_z = [d.mean(dim=1, keepdim=True) for d in delta_z_samples]
    delta_a = [d.mean(dim=1, keepdim=True) for d in delta_a_samples]

    # Diagnostics
    delta_z_norms = [d.norm().item() for d in delta_z]
    delta_a_norms = [d.norm().item() for d in delta_a]
    delta_z_std = [d.std(dim=1).mean().item() for d in delta_z_samples]
    delta_a_std = [d.std(dim=1).mean().item() for d in delta_a_samples]

    return {
        'delta_z': delta_z,
        'delta_a': delta_a,
        'delta_z_per_sample': delta_z_samples,
        'delta_a_per_sample': delta_a_samples,
        'delta_z_norms': delta_z_norms,
        'delta_a_norms': delta_a_norms,
        'delta_z_std': delta_z_std,
        'delta_a_std': delta_a_std,
    }


# ============================================================================
# Per-coordinate edge data (for scatter-plot diagnostics)
# ============================================================================

def extract_edge_data(
    sheaf: NeuralSheaf,
    state: Dict[str, list],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract per-coordinate predicted/actual/discord for every edge.

    Unlike ``compute_discord`` (which returns one scalar per edge), this
    returns the raw coordinate-level arrays needed for scatter-plot
    diagnostics such as predicted-vs-actual and residual plots.

    Parameters
    ----------
    sheaf : NeuralSheaf
        The neural sheaf encoding the network.
    state : dict
        Training stalk state: ``'z'`` and ``'a'`` lists.

    Returns
    -------
    weight_edges : list of dict
        One per weight edge (ell = 0 ... k). Keys: ``'predicted'``,
        ``'actual'``, ``'discord'``, ``'neuron_idx'``, ``'sample_idx'``
        -- all 1-D numpy arrays.
    relu_edges : list of dict
        One per ReLU edge (ell = 0 ... k-1). Keys: ``'z'``, ``'a'``,
        ``'relu_z'``, ``'discord'``, ``'neuron_idx'``, ``'sample_idx'``,
        ``'active'``.
    """
    z_list, a_list = state['z'], state['a']
    k = sheaf.k

    weight_edges = []
    for ell in range(k + 1):
        W, b = sheaf.weights[ell], sheaf.biases[ell]
        predicted = (W @ a_list[ell] + b).detach().numpy()
        actual = z_list[ell].detach().numpy()
        n_neurons, batch = predicted.shape
        weight_edges.append(dict(
            predicted=predicted.flatten(),
            actual=actual.flatten(),
            discord=(predicted - actual).flatten(),
            neuron_idx=np.repeat(np.arange(n_neurons), batch),
            sample_idx=np.tile(np.arange(batch), n_neurons),
        ))

    relu_edges = []
    for ell in range(k):
        z = z_list[ell].detach().numpy()
        a = a_list[ell + 1].detach().numpy()
        relu_z = np.maximum(z, 0)
        n_neurons, batch = z.shape
        relu_edges.append(dict(
            z=z.flatten(),
            a=a.flatten(),
            relu_z=relu_z.flatten(),
            discord=(relu_z - a).flatten(),
            neuron_idx=np.repeat(np.arange(n_neurons), batch),
            sample_idx=np.tile(np.arange(batch), n_neurons),
            active=(z >= 0).flatten(),
        ))

    return weight_edges, relu_edges


# ============================================================================
# Pinned discord (diffusion with output pinned to true labels)
# ============================================================================

def compute_pinned_discord(
    sheaf: NeuralSheaf,
    X: torch.Tensor,
    Y: torch.Tensor,
    dt: float = 0.01,
    max_iter: int = 100_000,
    tol: float = 1e-10,
) -> Tuple[Dict, Dict, Dict]:
    """
    Run heat diffusion with output pinned to true labels, then measure discord.

    Initializes stalks from the forward pass (zero internal discord),
    hard-pins the output to Y, and runs diffusion. The converged state
    is the harmonic extension interpolating between the network's input
    and the true output -- its edge discords reveal where in the network
    the prediction error lives.

    For classification (sigmoid/softmax), the output activation edge uses
    identity mode (cross-entropy-like force) to avoid saturation.

    Parameters
    ----------
    sheaf : NeuralSheaf
    X : torch.Tensor
        Inputs, shape ``(input_dim, batch_size)``.
    Y : torch.Tensor
        True labels, shape ``(output_dim, batch_size)``.
    dt : float
        Euler step size for diffusion.
    max_iter : int
        Maximum diffusion steps.
    tol : float
        Convergence tolerance.

    Returns
    -------
    discord_dict : dict
        Per-edge discord from ``compute_discord``.
    state : dict
        Converged stalk state.
    info : dict
        Convergence info from ``SheafDynamics.run()``.
    """
    from .dynamics import SheafDynamics
    from .pinning import HardPin

    out_act = sheaf.output_activation
    edge_mode = 'identity' if out_act != 'identity' else 'jacobian'

    dynamics = SheafDynamics(sheaf, alpha=1.0, dt=dt,
                             output_edge_mode=edge_mode)

    state = sheaf.init_stalks(X, method='forward_pass')

    if out_act == 'identity':
        output_pin = HardPin('z', layer=sheaf.k, values=Y)
    else:
        output_pin = HardPin('a_output', layer=None, values=Y)

    _, state, info = dynamics.run(
        X, state=state, pins=[output_pin],
        max_iter=max_iter, tol=tol,
    )

    disc = compute_discord(sheaf, state, X)
    return disc, state, info


# ============================================================================
# Interactive plotly visualization
# ============================================================================

def plotly_pva(
    sheaf: NeuralSheaf,
    state: Dict[str, list],
    title: str = "",
):
    """
    Predicted-vs-actual scatter for every edge (interactive plotly).

    Weight edges plot ``W a + b`` vs ``z`` (should lie on identity line).
    ReLU edges plot ``z`` vs ``a`` (should lie on ReLU curve).

    Requires ``plotly`` (optional dependency).

    Parameters
    ----------
    sheaf : NeuralSheaf
    state : dict
        Stalk state with ``'z'`` and ``'a'`` lists.
    title : str
        Prefix for the figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    weight_edges, relu_edges = extract_edge_data(sheaf, state)
    k = sheaf.k
    ncols = max(k + 1, max(k, 1))
    nrows = 2 if k > 0 else 1
    subtitles = (
        [f'Weight edge {i+1}' for i in range(k + 1)]
        + [''] * (ncols - k - 1)
        + [f'ReLU edge {i+1}' for i in range(k)]
        + [''] * (ncols - k)
    )
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=subtitles[:nrows * ncols])

    for ell, ed in enumerate(weight_edges):
        custom = np.stack([ed['neuron_idx'], ed['sample_idx'],
                           ed['discord']], axis=1)
        fig.add_trace(go.Scattergl(
            x=ed['predicted'], y=ed['actual'], mode='markers',
            marker=dict(size=3, color=ed['neuron_idx'],
                        colorscale='Viridis', opacity=0.5),
            customdata=custom, name=f'W{ell + 1}',
            hovertemplate=(
                'Pred: %{x:.4f}<br>Actual: %{y:.4f}<br>'
                'Neuron: %{customdata[0]:.0f}<br>'
                'Discord: %{customdata[2]:.4f}<extra></extra>'),
        ), row=1, col=ell + 1)
        lo = min(ed['predicted'].min(), ed['actual'].min())
        hi = max(ed['predicted'].max(), ed['actual'].max())
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode='lines',
            line=dict(color='red', dash='dash'), showlegend=False,
        ), row=1, col=ell + 1)

    for ell, ed in enumerate(relu_edges):
        custom = np.stack([ed['neuron_idx'], ed['sample_idx'],
                           ed['discord']], axis=1)
        fig.add_trace(go.Scattergl(
            x=ed['z'], y=ed['a'], mode='markers',
            marker=dict(size=3, color=ed['neuron_idx'],
                        colorscale='Plasma', opacity=0.5),
            customdata=custom, name=f'ReLU{ell + 1}',
            hovertemplate=(
                'z: %{x:.4f}<br>a: %{y:.4f}<br>'
                'Neuron: %{customdata[0]:.0f}<br>'
                'Discord: %{customdata[2]:.4f}<extra></extra>'),
        ), row=2, col=ell + 1)
        zr = np.linspace(ed['z'].min() - 0.1, ed['z'].max() + 0.1, 200)
        fig.add_trace(go.Scatter(
            x=zr, y=np.maximum(zr, 0), mode='lines',
            line=dict(color='red', dash='dash'), showlegend=False,
        ), row=2, col=ell + 1)

    fig.update_layout(
        title=f'{title}Stalk Consistency',
        height=450 * nrows, width=500 * ncols,
        showlegend=False,
    )
    return fig


def plotly_residuals(
    sheaf: NeuralSheaf,
    state: Dict[str, list],
    title: str = "",
):
    """
    Discord residual scatter for every edge (interactive plotly).

    Weight edges plot predicted value vs discord residual.
    ReLU edges plot z vs ``ReLU(z) - a``, colored by active/inactive.

    Requires ``plotly`` (optional dependency).

    Parameters
    ----------
    sheaf : NeuralSheaf
    state : dict
        Stalk state with ``'z'`` and ``'a'`` lists.
    title : str
        Prefix for the figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    weight_edges, relu_edges = extract_edge_data(sheaf, state)
    k = sheaf.k
    ncols = max(k + 1, max(k, 1))
    nrows = 2 if k > 0 else 1
    subtitles = (
        [f'Weight edge {i+1}' for i in range(k + 1)]
        + [''] * (ncols - k - 1)
        + [f'ReLU edge {i+1}' for i in range(k)]
        + [''] * (ncols - k)
    )
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=subtitles[:nrows * ncols])

    for ell, ed in enumerate(weight_edges):
        pos = ed['predicted'] >= 0
        for mask, label, color in [(pos, 'Predicted >= 0', 'steelblue'),
                                    (~pos, 'Predicted < 0', 'lightcoral')]:
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scattergl(
                x=ed['predicted'][mask], y=ed['discord'][mask],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.4),
                name=label, showlegend=(ell == 0),
            ), row=1, col=ell + 1)
        fig.add_hline(y=0, line_color='black', line_width=0.8,
                      row=1, col=ell + 1)

    for ell, ed in enumerate(relu_edges):
        act = ed['active']
        for mask, label, color in [(act, 'Active (z>=0)', 'darkorange'),
                                    (~act, 'Inactive (z<0)', 'mediumpurple')]:
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scattergl(
                x=ed['z'][mask], y=ed['discord'][mask], mode='markers',
                marker=dict(size=3, color=color, opacity=0.4),
                name=label, showlegend=(ell == 0),
            ), row=2, col=ell + 1)
        fig.add_hline(y=0, line_color='black', line_width=0.8,
                      row=2, col=ell + 1)

    fig.update_layout(
        title=f'{title}Discord Residuals',
        height=400 * nrows, width=500 * ncols,
    )
    return fig
