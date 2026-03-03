"""
Benchmark datasets for the neural sheaf library.

Provides target functions and data generators for the standard experiments
described in the paper: regression (paraboloid, saddle) and classification
(circular binary, overlapping Gaussian blobs).

All generators return tensors in the library's (dim, batch_size) layout
with float64 precision.
"""

import math
from typing import Callable, Tuple

import torch


DTYPE = torch.float64


# ============================================================================
# Target functions (regression)
# ============================================================================

def paraboloid(X: torch.Tensor) -> torch.Tensor:
    """
    Paraboloid target: f(x1, x2) = x1^2 + x2^2 - 2/3.

    Parameters
    ----------
    X : Tensor of shape (2, batch_size)

    Returns
    -------
    Tensor of shape (1, batch_size)
    """
    return X[0:1] ** 2 + X[1:2] ** 2 - 2.0 / 3.0


def saddle(X: torch.Tensor) -> torch.Tensor:
    """
    Saddle target: f(x1, x2) = x1^2 - x2^2 + 0.5 sin(2 x1).

    Parameters
    ----------
    X : Tensor of shape (2, batch_size)

    Returns
    -------
    Tensor of shape (1, batch_size)
    """
    return X[0:1] ** 2 - X[1:2] ** 2 + 0.5 * torch.sin(2 * X[0:1])


# ============================================================================
# Data generators
# ============================================================================

def generate_regression_data(
    func: Callable[[torch.Tensor], torch.Tensor],
    n_train: int = 300,
    n_test: int = 100,
    input_range: Tuple[float, float] = (-2.0, 2.0),
    noise_std: float = 0.02,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate 2D regression data from a target function.

    Inputs are sampled uniformly on [lo, hi]^2.  Targets have additive
    Gaussian noise with standard deviation ``noise_std``.

    Parameters
    ----------
    func : callable
        Target function mapping (2, n) -> (1, n).
    n_train, n_test : int
        Number of training / test samples.
    input_range : (float, float)
        Lower and upper bounds for each input coordinate.
    noise_std : float
        Standard deviation of additive Gaussian noise on targets.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, Y_train, X_test, Y_test : Tensors
        Shapes (2, n_train), (1, n_train), (2, n_test), (1, n_test).
    """
    torch.manual_seed(seed)
    n = n_train + n_test
    lo, hi = input_range
    X = torch.rand(2, n, dtype=DTYPE) * (hi - lo) + lo
    Y = func(X) + torch.randn(1, n, dtype=DTYPE) * noise_std
    return X[:, :n_train], Y[:, :n_train], X[:, n_train:], Y[:, n_train:]


def generate_circular_data(
    n_train: int = 300,
    n_test: int = 100,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate binary classification data with circular decision boundary.

    Class 0: points inside a unit disk (r < 1).
    Class 1: points in an annulus (1.5 < r < 2.5).

    Parameters
    ----------
    n_train, n_test : int
    noise : float
        Gaussian radial noise.
    seed : int

    Returns
    -------
    X_train, Y_train, X_test, Y_test : Tensors
        X shapes (2, n), Y shapes (1, n) with values in {0, 1}.
    """
    torch.manual_seed(seed)
    n2 = (n_train + n_test) // 2          # samples per class
    total = 2 * n2                         # exact even total (drops 1 if n_train+n_test odd)
    theta = torch.rand(total, dtype=DTYPE) * (2.0 * math.pi)

    # Class 0: inside unit disk
    r0 = torch.rand(n2, dtype=DTYPE).sqrt() + torch.randn(n2, dtype=DTYPE) * noise
    # Class 1: annulus 1.5 < r < 2.5
    r1 = (1.5 ** 2 + torch.rand(n2, dtype=DTYPE) * (2.5 ** 2 - 1.5 ** 2)).sqrt() \
         + torch.randn(n2, dtype=DTYPE) * noise

    X = torch.stack([
        torch.cat([r0 * theta[:n2].cos(), r1 * theta[n2:].cos()]),
        torch.cat([r0 * theta[:n2].sin(), r1 * theta[n2:].sin()]),
    ])
    Y = torch.cat([torch.zeros(n2), torch.ones(n2)]).unsqueeze(0).to(DTYPE)
    perm = torch.randperm(total)
    X, Y = X[:, perm], Y[:, perm]
    return X[:, :n_train], Y[:, :n_train], X[:, n_train:total], Y[:, n_train:total]


# ============================================================================
# Blob benchmark configuration
# ============================================================================

# Blob metadata: asymmetric centers with anisotropic covariances that create
# deliberate overlap between classes.  Exposed at module level for use by
# visualization code (e.g. drawing covariance ellipses).

BLOB_CENTERS = torch.tensor([
    [-1.5, -1.5],
    [ 1.8, -1.0],
    [ 1.0,  2.0],
    [-2.0,  1.5],
], dtype=DTYPE)

BLOB_COVARIANCES = torch.stack([
    torch.tensor([[ 0.80,  0.30], [ 0.30,  0.40]], dtype=DTYPE),  # elongated, tilted
    torch.tensor([[ 0.30, -0.20], [-0.20,  1.00]], dtype=DTYPE),  # tall, tilted other way
    torch.tensor([[ 1.20,  0.50], [ 0.50,  0.50]], dtype=DTYPE),  # large, tilted
    torch.tensor([[ 0.50,  0.00], [ 0.00,  0.80]], dtype=DTYPE),  # axis-aligned, medium
])  # shape: (4, 2, 2)


def generate_blob_data(
    n_train: int = 300,
    n_test: int = 100,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multiclass data from overlapping Gaussian blobs.

    Four blobs with different covariance matrices (varying size and
    orientation) placed at asymmetric centers with deliberate overlap.
    Centers and covariances are defined by ``BLOB_CENTERS`` and
    ``BLOB_COVARIANCES``.

    Parameters
    ----------
    n_train, n_test : int
    seed : int

    Returns
    -------
    X_train, Y_train, X_test, Y_test : Tensors
        X shapes (2, n), Y shapes (4, n) one-hot encoded.
    """
    torch.manual_seed(seed)
    n_classes = BLOB_CENTERS.shape[0]
    n = n_train + n_test
    n_per = n // n_classes

    X_list, Y_list = [], []
    for i in range(n_classes):
        dist = torch.distributions.MultivariateNormal(
            loc=BLOB_CENTERS[i],
            covariance_matrix=BLOB_COVARIANCES[i],
        )
        pts = dist.sample((n_per,)).T  # (2, n_per), matches library layout
        X_list.append(pts.to(DTYPE))
        y = torch.zeros(n_classes, n_per, dtype=DTYPE)
        y[i] = 1.0
        Y_list.append(y)

    X = torch.cat(X_list, dim=1)
    Y = torch.cat(Y_list, dim=1)
    perm = torch.randperm(X.shape[1])
    X, Y = X[:, perm], Y[:, perm]
    return X[:, :n_train], Y[:, :n_train], X[:, n_train:], Y[:, n_train:]
