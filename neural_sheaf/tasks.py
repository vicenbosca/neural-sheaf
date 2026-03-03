"""
Shared task configurations, model snapshot helpers, and analysis training loops.

Used by the analysis notebooks and standalone scripts (``spectral.py``,
``discord.py``).  Defines the four benchmark tasks with their default
architectures, data generators, and hyperparameters (matching Notebook 2).

Training loops
--------------
``train_sheaf_full`` and ``train_sgd_full`` are specialised training loops
that record spectral snapshots and model checkpoints at configurable
intervals.  These differ from ``SheafTrainer.train()`` in that they
operate step-by-step to allow fine-grained snapshotting.
"""

import torch
from typing import Any, Dict, List, Tuple

from .sheaf import NeuralSheaf
from .trainer import SheafTrainer
from .baseline import TraditionalNN
from .spectral import make_tracking_history, record_spectral_snapshot
from .datasets import (
    paraboloid, saddle,
    generate_regression_data,
    generate_circular_data,
    generate_blob_data,
)


# ============================================================================
# Task configurations
# ============================================================================

# Hyperparameters match those validated in Notebook 2:
#   - beta = 1/n_train for all tasks (unified scaling)
#   - dt = 0.005 for all 1H tasks
#   - SGD lr: 0.005 for regression, 0.01 for classification
TASK_CONFIGS = {
    'paraboloid': dict(
        default_arch=[2, 30, 1],
        data_fn=lambda: generate_regression_data(paraboloid, n_train=300, n_test=100),
        output_activation='identity',
        n_train=300,
        default_dt=0.005,
        default_n_steps=100_000,
        sgd_lr=0.005,
        task_type='regression',
        init_method='random',
    ),
    'saddle': dict(
        default_arch=[2, 30, 1],
        data_fn=lambda: generate_regression_data(
            saddle, n_train=300, n_test=100, input_range=(0.0, 2.0)),
        output_activation='identity',
        n_train=300,
        default_dt=0.005,
        default_n_steps=100_000,
        sgd_lr=0.005,
        task_type='regression',
        init_method='random',
    ),
    'circular': dict(
        default_arch=[2, 25, 1],
        data_fn=lambda: generate_circular_data(n_train=300, n_test=100),
        output_activation='sigmoid',
        n_train=300,
        default_dt=0.005,
        default_n_steps=100_000,
        sgd_lr=0.01,
        task_type='binary',
        init_method='forward_pass',
    ),
    'blobs': dict(
        default_arch=[2, 25, 4],
        data_fn=lambda: generate_blob_data(n_train=300, n_test=100),
        output_activation='softmax',
        n_train=300,
        default_dt=0.005,
        default_n_steps=100_000,
        sgd_lr=0.01,
        task_type='multiclass',
        init_method='forward_pass',
    ),
}


# ============================================================================
# Training level helpers
# ============================================================================

def compute_levels(n_steps: int) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    """
    Return deduplicated (label, steps, color) triples and short tags.

    Computes three training levels (poorly trained, intermediate,
    well trained) as fractions of ``n_steps``, deduplicating if any
    coincide.

    Parameters
    ----------
    n_steps : int
        Total training steps.

    Returns
    -------
    levels : list of (label, step_count, color)
    tags : list of str
        Short identifiers like ``'1k'``, ``'20k'``.
    """
    raw = [
        ('Poorly trained',  max(n_steps // 100, 100), 'tomato'),
        ('Intermediate',    max(n_steps //   5, 500), 'goldenrod'),
        ('Well trained',    n_steps,                  'steelblue'),
    ]
    raw = [(l, min(s, n_steps), c) for l, s, c in raw]
    seen = set()
    levels = []
    for l, s, c in raw:
        if s not in seen:
            seen.add(s)
            levels.append((l, s, c))

    def _tag(ns):
        return f"{ns // 1000}k" if ns >= 1000 else str(ns)

    tags = [_tag(s) for _, s, _ in levels]
    return levels, tags


# ============================================================================
# Model snapshot helpers
# ============================================================================

def snapshot_sheaf(sheaf: NeuralSheaf) -> NeuralSheaf:
    """
    Deep-copy a NeuralSheaf's weights into a new instance.

    Parameters
    ----------
    sheaf : NeuralSheaf

    Returns
    -------
    NeuralSheaf
        Independent copy with cloned weights and biases.
    """
    w = [wi.detach().clone() for wi in sheaf.weights]
    b = [bi.detach().clone() for bi in sheaf.biases]
    return NeuralSheaf(sheaf.layer_dims, weights=w, biases=b,
                       output_activation=sheaf.output_activation)


def nn_to_sheaf(nn, arch: List[int], output_activation: str) -> NeuralSheaf:
    """
    Copy a TraditionalNN's weights into a NeuralSheaf for analysis.

    Parameters
    ----------
    nn : TraditionalNN
        A trained baseline network.
    arch : list of int
        Layer dimensions ``[n_0, n_1, ..., n_{k+1}]``.
    output_activation : str
        ``'identity'``, ``'sigmoid'``, or ``'softmax'``.

    Returns
    -------
    NeuralSheaf
        A sheaf with the same weights as the neural network.
    """
    w = [wi.detach().clone() for wi in nn.weights]
    b = [bi.detach().clone() for bi in nn.biases]
    return NeuralSheaf(arch, weights=w, biases=b,
                       output_activation=output_activation)


# ============================================================================
# Analysis training loops (with spectral snapshotting)
# ============================================================================

def _copy_state(state: Dict[str, list]) -> Dict[str, list]:
    """
    Deep-copy a stalk state dict.

    Parameters
    ----------
    state : dict
        Stalk state with ``'z'``, ``'a'``, and optionally ``'a_output'``.

    Returns
    -------
    dict
        Independent copy with cloned tensors.
    """
    new = {
        'z': [t.detach().clone() for t in state['z']],
        'a': [t.detach().clone() for t in state['a']],
    }
    if 'a_output' in state:
        new['a_output'] = state['a_output'].detach().clone()
    return new


def train_sheaf_full(
    arch: List[int],
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    n_steps: int,
    beta: float,
    dt: float,
    output_activation: str,
    snapshot_steps: List[int],
    spectral_freq: int,
    X_sub: torch.Tensor,
    seed_w: int = 0,
    seed_s: int = 42,
    init_method: str = 'random',
) -> Tuple[Dict[str, Any], Dict[int, tuple]]:
    """
    Train sheaf once, recording spectral snapshots and model checkpoints.

    Runs step-by-step training (not ``SheafTrainer.train()``) so that
    spectral snapshots and stalk states can be captured at arbitrary step
    counts for post-hoc analysis.

    Parameters
    ----------
    arch : list of int
        Layer dimensions ``[n_0, ..., n_{k+1}]``.
    X_tr, Y_tr : torch.Tensor
        Training data.
    X_te, Y_te : torch.Tensor
        Test data (for loss tracking).
    n_steps : int
        Total training steps.
    beta : float
        Weight learning rate.
    dt : float
        Stalk dynamics step size.
    output_activation : str
        ``'identity'``, ``'sigmoid'``, or ``'softmax'``.
    snapshot_steps : list of int
        Steps at which to save model + stalk state.
    spectral_freq : int
        Record spectral statistics every ``spectral_freq`` steps.
    X_sub : torch.Tensor
        Input subset for spectral evaluation.
    seed_w, seed_s : int
        Seeds for weight initialisation and stalk initialisation.

    Returns
    -------
    history : dict
        Spectral tracking history (from ``make_tracking_history``).
    snapshots : dict
        Maps step (int) -> ``(NeuralSheaf, stalk_state, train_loss)``.
    """
    sheaf = NeuralSheaf(arch, seed=seed_w,
                        output_activation=output_activation)
    trainer = SheafTrainer(sheaf, alpha=1.0, beta=beta, dt=dt)

    history = make_tracking_history()
    snapshot_set = set(snapshot_steps)
    snapshots: Dict[int, tuple] = {}

    torch.manual_seed(seed_s)
    state = sheaf.init_stalks(X_tr, method=init_method)

    # Record initial state
    record_spectral_snapshot(
        history, 0,
        trainer._compute_loss(X_tr, Y_tr),
        trainer._compute_loss(X_te, Y_te),
        sheaf, X_sub)
    print(f"  Step     0 | loss={history['train_loss'][-1]:.6f}"
          f"  λ₁={history['spectral']['median'][-1]:.4f}"
          f"  λ_max={history['lambda_max']['median'][-1]:.2f}"
          f"  κ={history['condition']['median'][-1]:.1f}")

    for s in range(1, n_steps + 1):
        state = trainer.train_step(state, X_tr, Y_tr)

        if s in snapshot_set:
            loss = trainer._compute_loss(X_tr, Y_tr)
            snapshots[s] = (snapshot_sheaf(sheaf), _copy_state(state), loss)

        if s % spectral_freq == 0 or s == n_steps:
            snap = (snapshots[s][0] if s in snapshots
                    else snapshot_sheaf(sheaf))
            record_spectral_snapshot(
                history, s,
                trainer._compute_loss(X_tr, Y_tr),
                trainer._compute_loss(X_te, Y_te),
                snap, X_sub)
            print(f"  Step {s:6d} | loss={history['train_loss'][-1]:.6f}"
                  f"  λ₁={history['spectral']['median'][-1]:.4f}"
                  f"  λ_max={history['lambda_max']['median'][-1]:.2f}"
                  f"  κ={history['condition']['median'][-1]:.1f}")

    if n_steps not in snapshots:
        loss = trainer._compute_loss(X_tr, Y_tr)
        snapshots[n_steps] = (
            snapshot_sheaf(sheaf), _copy_state(state), loss)

    return history, snapshots


def train_sgd_full(
    arch: List[int],
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    n_steps: int,
    output_activation: str,
    lr: float,
    snapshot_steps: List[int],
    spectral_freq: int,
    X_sub: torch.Tensor,
    seed: int = 0,
) -> Tuple[Dict[str, Any], Dict[int, tuple]]:
    """
    Train SGD once, recording spectral snapshots and model checkpoints.

    At each snapshot, SGD weights are copied into a ``NeuralSheaf`` and
    stalks initialised from the forward pass (zero internal discord).

    Parameters
    ----------
    arch : list of int
        Layer dimensions ``[n_0, ..., n_{k+1}]``.
    X_tr, Y_tr : torch.Tensor
        Training data.
    X_te, Y_te : torch.Tensor
        Test data.
    n_steps : int
        Total training epochs.
    output_activation : str
    lr : float
        SGD learning rate.
    snapshot_steps : list of int
        Steps at which to save model snapshots.
    spectral_freq : int
        Record spectral statistics every ``spectral_freq`` steps.
    X_sub : torch.Tensor
        Input subset for spectral evaluation.
    seed : int
        Seed for weight initialisation.

    Returns
    -------
    history : dict
        Spectral tracking history.
    snapshots : dict
        Maps step (int) -> ``(NeuralSheaf, stalk_state, train_loss)``.
    """
    oa = None if output_activation == 'identity' else output_activation
    nn = TraditionalNN(arch, learning_rate=lr, output_activation=oa, seed=seed)

    history = make_tracking_history()
    snapshot_set = set(snapshot_steps)
    snapshots: Dict[int, tuple] = {}

    def _losses():
        with torch.no_grad():
            trl = nn._compute_loss(nn.forward(X_tr), Y_tr).item()
            tel = nn._compute_loss(nn.forward(X_te), Y_te).item()
        return trl, tel

    # Record initial state
    snap0 = nn_to_sheaf(nn, arch, output_activation)
    trl, tel = _losses()
    record_spectral_snapshot(history, 0, trl, tel, snap0, X_sub)
    print(f"  Step     0 | loss={history['train_loss'][-1]:.6f}"
          f"  λ₁={history['spectral']['median'][-1]:.4f}"
          f"  λ_max={history['lambda_max']['median'][-1]:.2f}"
          f"  κ={history['condition']['median'][-1]:.1f}")

    for epoch in range(1, n_steps + 1):
        nn._zero_grad()
        y_pred = nn.forward(X_tr)
        loss = nn._compute_loss(y_pred, Y_tr)
        loss.backward()
        with torch.no_grad():
            for W in nn.weights:
                W -= nn.learning_rate * W.grad
            for b in nn.biases:
                b -= nn.learning_rate * b.grad

        if epoch in snapshot_set:
            with torch.no_grad():
                trl_snap = nn._compute_loss(nn.forward(X_tr), Y_tr).item()
            s = nn_to_sheaf(nn, arch, output_activation)
            state = s.init_stalks(X_tr, method='forward_pass')
            snapshots[epoch] = (s, state, trl_snap)

        if epoch % spectral_freq == 0 or epoch == n_steps:
            snap = (snapshots[epoch][0] if epoch in snapshots
                    else nn_to_sheaf(nn, arch, output_activation))
            trl, tel = _losses()
            record_spectral_snapshot(
                history, epoch, trl, tel, snap, X_sub)
            print(f"  Step {epoch:6d} | loss={history['train_loss'][-1]:.6f}"
                  f"  λ₁={history['spectral']['median'][-1]:.4f}"
                  f"  λ_max={history['lambda_max']['median'][-1]:.2f}"
                  f"  κ={history['condition']['median'][-1]:.1f}")

    if n_steps not in snapshots:
        with torch.no_grad():
            trl = nn._compute_loss(nn.forward(X_tr), Y_tr).item()
        s = nn_to_sheaf(nn, arch, output_activation)
        state = s.init_stalks(X_tr, method='forward_pass')
        snapshots[n_steps] = (s, state, trl)

    return history, snapshots
