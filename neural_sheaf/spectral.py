"""
Spectral analysis of the sheaf Laplacian L_F[Omega, Omega].

The free-free block of the sheaf Laplacian depends on the input x
through the ReLU activation masks.  This module analyses its spectral
properties -- gap (lambda_1), largest eigenvalue (lambda_max), condition
number (kappa = lambda_max / lambda_1), and eigenvector structure --
across the input-dependent Laplacian ensemble.

Functions
---------
spectral_analysis_per_sample
    Full spectral decomposition per input sample, including eigenvector
    stability and per-block energy.
make_tracking_history
    Create an empty history dict for spectral tracking during training.
record_spectral_snapshot
    Record one spectral snapshot (losses + spectral statistics) into a
    tracking history dict.
"""

import numpy as np
import torch
from typing import Any, Dict, List, Tuple

from .sheaf import NeuralSheaf


# ============================================================================
# Per-sample spectral analysis
# ============================================================================

def spectral_analysis_per_sample(
    sheaf: NeuralSheaf,
    X: torch.Tensor,
    threshold: float = 1e-8,
) -> Dict[str, Any]:
    """
    Full spectral analysis of L_F[Omega, Omega] per input sample.

    For each column of X, builds the free-free Laplacian block from the
    input-dependent ReLU masks, then computes the full eigendecomposition.
    Aggregates spectral gap, largest eigenvalue, condition number,
    eigenvector stability, and per-block energy across the ensemble.

    Parameters
    ----------
    sheaf : NeuralSheaf
        The neural sheaf whose Laplacian is analysed.
    X : torch.Tensor
        Input data, shape ``(n_0, n_samples)``.
    threshold : float
        Eigenvalues below this are treated as zero when identifying the
        spectral gap.

    Returns
    -------
    dict with keys:
        ``'gaps'`` : ndarray ``(n_samples,)``
            Smallest non-zero eigenvalue (spectral gap, lambda_1).
        ``'lambda_max'`` : ndarray ``(n_samples,)``
            Largest eigenvalue.
        ``'condition_numbers'`` : ndarray ``(n_samples,)``
            Condition number kappa = lambda_max / lambda_1.
        ``'eigenvalues'`` : list of ndarray
            Full sorted spectrum per sample.
        ``'fiedler_vectors'`` : list of ndarray
            Eigenvector for lambda_1 per sample.
        ``'max_eigenvectors'`` : list of ndarray
            Eigenvector for lambda_max per sample.
        ``'fiedler_stability'`` : dict
            Cosine similarity statistics for Fiedler vectors across samples.
        ``'max_stability'`` : dict
            Cosine similarity statistics for lambda_max vectors.
        ``'fiedler_layer_energy'`` : ndarray ``(n_blocks, n_samples)``
            Per-block fraction of squared norm in each Fiedler vector.
        ``'max_layer_energy'`` : ndarray ``(n_blocks, n_samples)``
            Per-block fraction of squared norm in each lambda_max vector.
        ``'fiedler_neuron_contributions'`` : dict
            Per-neuron breakdown in the block with highest mean Fiedler energy.
        ``'max_neuron_contributions'`` : dict
            Per-neuron breakdown in the block with highest mean lambda_max energy.
        ``'block_ranges'`` : list of (label, start, end)
            Free coordinate block ranges for interpreting eigenvectors.
    """
    n = X.shape[1]
    block_ranges = _compute_free_block_ranges(sheaf)
    n_blocks = len(block_ranges)

    gaps = np.zeros(n)
    lambda_max = np.zeros(n)
    condition_numbers = np.zeros(n)
    eigenvalues: List[np.ndarray] = []
    fiedler_vectors: List[np.ndarray] = []
    max_eigenvectors: List[np.ndarray] = []
    fiedler_energy = np.zeros((n_blocks, n))
    max_energy = np.zeros((n_blocks, n))

    for i in range(n):
        x_i = X[:, i:i+1]
        _, intermediates = sheaf.forward(x_i)
        masks = sheaf.compute_masks(intermediates['z'])
        L = sheaf.build_laplacian_block(masks, block='free').to_dense()

        # Symmetric matrix -> real eigenvalues, sorted ascending
        ev, V = torch.linalg.eigh(L)
        ev_np = ev.numpy()
        V_np = V.numpy()

        eigenvalues.append(ev_np)

        nz_mask = ev_np > threshold
        if nz_mask.any():
            nz_idx = np.where(nz_mask)[0]
            idx_min = nz_idx[0]  # smallest non-zero (eigh returns sorted)

            gaps[i] = ev_np[idx_min]
            lambda_max[i] = ev_np[-1]
            condition_numbers[i] = ev_np[-1] / ev_np[idx_min]

            fiedler_vectors.append(V_np[:, idx_min])
            max_eigenvectors.append(V_np[:, -1])

            # Per-block energy decomposition
            for b, (_, start, end) in enumerate(block_ranges):
                seg_f = V_np[start:end, idx_min]
                seg_m = V_np[start:end, -1]
                fiedler_energy[b, i] = np.sum(seg_f ** 2)
                max_energy[b, i] = np.sum(seg_m ** 2)
        else:
            gaps[i] = 0.0
            lambda_max[i] = ev_np[-1]
            condition_numbers[i] = np.inf
            dim = V_np.shape[0]
            fiedler_vectors.append(np.zeros(dim))
            max_eigenvectors.append(V_np[:, -1])

    fiedler_stability = _eigenvector_stability(fiedler_vectors)
    max_stability = _eigenvector_stability(max_eigenvectors)

    fiedler_neuron = _top_block_neuron_contributions(
        fiedler_vectors, fiedler_energy, block_ranges)
    max_neuron = _top_block_neuron_contributions(
        max_eigenvectors, max_energy, block_ranges)

    return {
        'gaps': gaps,
        'lambda_max': lambda_max,
        'condition_numbers': condition_numbers,
        'eigenvalues': eigenvalues,
        'fiedler_vectors': fiedler_vectors,
        'max_eigenvectors': max_eigenvectors,
        'fiedler_stability': fiedler_stability,
        'max_stability': max_stability,
        'fiedler_layer_energy': fiedler_energy,
        'max_layer_energy': max_energy,
        'fiedler_neuron_contributions': fiedler_neuron,
        'max_neuron_contributions': max_neuron,
        'block_ranges': block_ranges,
    }


# ============================================================================
# Spectral tracking during training
# ============================================================================

def make_tracking_history() -> Dict[str, Any]:
    """
    Create an empty history dict for spectral tracking during training.

    The returned dict has the following structure::

        step, train_loss, test_loss : list of scalar
        spectral   : dict with median, q25, q75 lists  (lambda_1)
        lambda_max : dict with median, q25, q75 lists
        condition  : dict with median, q25, q75 lists  (kappa)

    Use :func:`record_spectral_snapshot` to append entries.

    Returns
    -------
    dict
        Empty tracking history ready for recording.
    """
    return dict(
        step=[], train_loss=[], test_loss=[],
        spectral=dict(median=[], q25=[], q75=[]),
        lambda_max=dict(median=[], q25=[], q75=[]),
        condition=dict(median=[], q25=[], q75=[]),
    )


def record_spectral_snapshot(
    history: Dict[str, Any],
    step: int,
    train_loss: float,
    test_loss: float,
    sheaf: NeuralSheaf,
    X_sub: torch.Tensor,
) -> None:
    """
    Record one spectral snapshot into a tracking history.

    Runs :func:`spectral_analysis_per_sample` on ``X_sub`` and appends
    median and interquartile statistics for lambda_1 (spectral gap),
    lambda_max, and kappa (condition number) to the history dict.

    Parameters
    ----------
    history : dict
        Created by :func:`make_tracking_history`.  Modified in place.
    step : int
        Current training step.
    train_loss : float
        Current training loss.
    test_loss : float
        Current test loss.
    sheaf : NeuralSheaf
        The model at the current training state.
    X_sub : torch.Tensor
        Subset of training data for spectral evaluation, shape
        ``(n_0, n_spectral_samples)``.
    """
    history['step'].append(step)
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)

    result = spectral_analysis_per_sample(sheaf, X_sub)
    gaps = result['gaps']
    lmax = result['lambda_max']
    kappa = result['condition_numbers']

    # Filter infinite condition numbers (from degenerate samples)
    kf = kappa[np.isfinite(kappa)]
    if len(kf) == 0:
        kf = np.array([0.0])

    for key, data in [('spectral', gaps), ('lambda_max', lmax),
                      ('condition', kf)]:
        history[key]['median'].append(float(np.median(data)))
        history[key]['q25'].append(float(np.percentile(data, 25)))
        history[key]['q75'].append(float(np.percentile(data, 75)))


# ============================================================================
# Private helpers
# ============================================================================

def _compute_free_block_ranges(
    sheaf: NeuralSheaf,
) -> List[Tuple[str, int, int]]:
    """
    Compute named ranges for each stalk block in the free coordinates.

    The free coordinates of L_F[Omega, Omega] are ordered as::

        z^(1), a^(1), z^(2), a^(2), ..., z^(k), a^(k), z^(k+1)

    Parameters
    ----------
    sheaf : NeuralSheaf

    Returns
    -------
    list of (label, start, end)
        Each tuple gives a human-readable label (e.g. ``'z1'``, ``'a1'``)
        and the ``[start, end)`` range of indices in the eigenvector.
    """
    n = sheaf.layer_dims
    k = sheaf.k
    ranges: List[Tuple[str, int, int]] = []
    pos = 0
    for ell in range(1, k + 1):
        ranges.append((f'z{ell}', pos, pos + n[ell]))
        pos += n[ell]
        ranges.append((f'a{ell}', pos, pos + n[ell]))
        pos += n[ell]
    ranges.append((f'z{k+1}', pos, pos + n[k + 1]))
    return ranges


def _eigenvector_stability(
    vectors: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Assess directional stability of eigenvectors across samples.

    Eigenvectors are defined up to sign.  We align them to a reference
    direction (first non-zero vector) before computing statistics.

    Parameters
    ----------
    vectors : list of ndarray
        One eigenvector per sample.

    Returns
    -------
    dict with keys:
        ``'pairwise_cosines'`` : ndarray
            Upper-triangle pairwise |cos theta|.
        ``'mean_abs_cosine'`` : float
            Mean of |cos theta| over all pairs.
        ``'std_abs_cosine'`` : float
            Standard deviation of |cos theta|.
        ``'aligned_vectors'`` : ndarray ``(n_samples, dim)``
            Sign-aligned eigenvectors.
        ``'mean_direction'`` : ndarray ``(dim,)``
            Mean of aligned vectors, re-normalised to unit length.
        ``'top_components'`` : ndarray (at most 10)
            Indices of largest components in mean direction.
    """
    vecs = np.array(vectors)
    n, dim = vecs.shape

    if n == 0 or dim == 0:
        return {
            'pairwise_cosines': np.array([]),
            'mean_abs_cosine': 0.0,
            'std_abs_cosine': 0.0,
            'aligned_vectors': vecs,
            'mean_direction': np.zeros(dim),
            'top_components': np.array([], dtype=int),
        }

    # Align signs to first non-zero vector
    ref_idx = 0
    for j in range(n):
        if np.linalg.norm(vecs[j]) > 1e-12:
            ref_idx = j
            break
    ref = vecs[ref_idx]
    aligned = vecs.copy()
    signs = np.sign(aligned @ ref)
    signs[signs == 0] = 1.0
    aligned *= signs[:, np.newaxis]

    # Pairwise absolute cosine similarities (upper triangle)
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    normed = aligned / norms
    gram = np.abs(normed @ normed.T)
    ii, jj = np.triu_indices(n, k=1)
    cosines = gram[ii, jj]
    if cosines.size == 0:
        cosines = np.array([1.0])

    # Mean direction
    mean_dir = aligned.mean(axis=0)
    norm_mean = np.linalg.norm(mean_dir)
    if norm_mean > 1e-15:
        mean_dir /= norm_mean

    n_top = min(10, dim)
    top_idx = np.argsort(np.abs(mean_dir))[::-1][:n_top]

    return {
        'pairwise_cosines': cosines,
        'mean_abs_cosine': float(cosines.mean()),
        'std_abs_cosine': float(cosines.std()),
        'aligned_vectors': aligned,
        'mean_direction': mean_dir,
        'top_components': top_idx,
    }


def _top_block_neuron_contributions(
    vectors: List[np.ndarray],
    block_energy: np.ndarray,
    block_ranges: List[Tuple[str, int, int]],
) -> Dict[str, Any]:
    """
    Per-neuron breakdown within the block carrying the most energy.

    For the block with the highest mean energy across samples, extracts
    the squared eigenvector component for each neuron (coordinate) in
    that block, averaged over samples.

    Parameters
    ----------
    vectors : list of ndarray
        Eigenvectors, one per sample (length = free_dim).
    block_energy : ndarray, shape ``(n_blocks, n_samples)``
        Per-block energy fractions.
    block_ranges : list of (label, start, end)

    Returns
    -------
    dict with keys:
        ``'block_label'`` : str
            Label of the dominant block.
        ``'block_idx'`` : int
            Index of the dominant block.
        ``'neuron_energy_mean'`` : ndarray ``(block_dim,)``
            Mean |v_j|^2 per neuron across samples.
        ``'neuron_energy_std'`` : ndarray ``(block_dim,)``
            Std |v_j|^2 per neuron across samples.
        ``'top_neurons'`` : ndarray
            Neuron indices sorted by descending mean energy.
    """
    vecs = np.array(vectors)  # (n_samples, free_dim)

    # Identify block with highest mean energy
    mean_energy = block_energy.mean(axis=1)
    top_block = int(np.argmax(mean_energy))
    label, start, end = block_ranges[top_block]

    # Per-neuron squared components within that block
    block_vecs = vecs[:, start:end]
    neuron_sq = block_vecs ** 2

    neuron_mean = neuron_sq.mean(axis=0)
    neuron_std = neuron_sq.std(axis=0)
    top_neurons = np.argsort(neuron_mean)[::-1]

    return {
        'block_label': label,
        'block_idx': top_block,
        'neuron_energy_mean': neuron_mean,
        'neuron_energy_std': neuron_std,
        'top_neurons': top_neurons,
    }
