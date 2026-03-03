"""
Activation functions and their Jacobians for the neural sheaf library.

All functions operate on tensors with shape (dim, batch_size) and return
tensors on the same device as the input.

ReLU is handled via boolean masks (the "fast selection rule" from the paper):
    mask[j] = (z[j] >= 0)
This means z=0 is treated as active. The full diagonal matrix is available
via relu_matrix() for analysis, but the dynamics loop uses element-wise masks.

All activations (including ReLU) follow the same interface and are accessible
via get_activation(), which returns (function, jacobian_function) pairs.
For element-wise activations (ReLU, sigmoid), the Jacobian function returns
the diagonal values (same shape as input). For softmax, it returns the full
(dim, dim, batch) matrix since outputs are coupled.
"""

import torch
from typing import Callable, Tuple


# ============================================================================
# ReLU and masks
# ============================================================================

def relu_mask(z: torch.Tensor) -> torch.Tensor:
    """
    Boolean mask implementing the fast selection rule.

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values, shape (dim, batch_size) or (dim,).

    Returns
    -------
    torch.Tensor
        Boolean tensor where True means the neuron is active (z >= 0).
    """
    return z >= 0


def relu(z: torch.Tensor) -> torch.Tensor:
    """
    Element-wise ReLU using the mask.

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values, shape (dim, batch_size) or (dim,).

    Returns
    -------
    torch.Tensor
        ReLU(z), same shape and device as input.
    """
    return relu_mask(z).to(z.dtype) * z


def relu_jacobian(z: torch.Tensor) -> torch.Tensor:
    """
    Diagonal of the ReLU Jacobian (the activation mask as floats).

    Since ReLU is element-wise, the Jacobian is diagonal. We return only
    the diagonal values (same shape as input). This is equivalent to
    relu_mask(z) cast to the input dtype.

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values.

    Returns
    -------
    torch.Tensor
        Diagonal Jacobian values, same shape as z.
    """
    return relu_mask(z).to(z.dtype)


def relu_matrix(z: torch.Tensor) -> torch.Tensor:
    """
    Full diagonal ReLU matrix from the mask (for analysis only).

    For a single sample z of shape (dim,), returns shape (dim, dim).
    For batched z of shape (dim, batch_size), returns shape (dim, dim, batch_size).

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values.

    Returns
    -------
    torch.Tensor
        Diagonal matrix R such that R @ z == relu(z).
    """
    mask = relu_mask(z).to(z.dtype)
    if z.dim() == 1:
        return torch.diag(mask)
    else:
        # (dim, batch) -> (batch, dim) -> (batch, dim, dim) -> (dim, dim, batch)
        return torch.diag_embed(mask.T).permute(1, 2, 0)


# ============================================================================
# Sigmoid
# ============================================================================

def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid activation.

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values, shape (dim, batch_size) or (dim,).

    Returns
    -------
    torch.Tensor
        sigmoid(z), same shape and device as input.
    """
    return torch.sigmoid(z)


def sigmoid_jacobian(z: torch.Tensor) -> torch.Tensor:
    """
    Diagonal of the sigmoid Jacobian: sigma(z) * (1 - sigma(z)).

    Since sigmoid is element-wise, the Jacobian is diagonal. We return only
    the diagonal values (same shape as input) rather than a full matrix.

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values.

    Returns
    -------
    torch.Tensor
        Diagonal Jacobian values, same shape as z.
    """
    s = torch.sigmoid(z)
    return s * (1.0 - s)


# ============================================================================
# Softmax
# ============================================================================

def softmax(z: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Numerically stable softmax activation.

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values, shape (dim, batch_size) or (dim,).
    axis : int
        Axis along which to compute softmax. Default 0 (the dimension axis,
        consistent with data layout (dim, batch_size)).

    Returns
    -------
    torch.Tensor
        Softmax probabilities, same shape and device as input. Sums to 1
        along the specified axis.
    """
    return torch.softmax(z, dim=axis)


def softmax_jacobian(z: torch.Tensor) -> torch.Tensor:
    """
    Full Jacobian of softmax with respect to z (along dim=0).

    For softmax output s, the Jacobian is: J[i,j] = s[i](delta[i,j] - s[j]).

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values, shape (dim, batch_size) or (dim,).

    Returns
    -------
    torch.Tensor
        If z is (dim,): returns (dim, dim).
        If z is (dim, batch_size): returns (dim, dim, batch_size).
    """
    s = torch.softmax(z, dim=0)
    if z.dim() == 1:
        return torch.diag(s) - torch.outer(s, s)
    else:
        # s shape: (dim, batch)
        # J[i,j,b] = s[i,b] * (delta[i,j] - s[j,b])
        dim = s.shape[0]
        eye = torch.eye(dim, dtype=z.dtype, device=z.device).unsqueeze(-1)  # (dim, dim, 1)
        diag_s = s.unsqueeze(1) * eye  # (dim, dim, batch)
        outer = s.unsqueeze(1) * s.unsqueeze(0)  # (dim, dim, batch)
        return diag_s - outer


# ============================================================================
# Identity (for output activation)
# ============================================================================

def identity(z: torch.Tensor) -> torch.Tensor:
    """Identity activation (pass-through)."""
    return z


def identity_jacobian(z: torch.Tensor) -> torch.Tensor:
    """
    Jacobian of identity: the identity matrix.

    Returns
    -------
    torch.Tensor
        If z is (dim,): returns (dim, dim) identity.
        If z is (dim, batch_size): returns (dim, dim, batch_size).
    """
    dim = z.shape[0]
    eye = torch.eye(dim, dtype=z.dtype, device=z.device)
    if z.dim() == 1:
        return eye
    else:
        return eye.unsqueeze(-1).expand(dim, dim, z.shape[1])


# ============================================================================
# Activation registry
# ============================================================================

def get_activation(name: str) -> Tuple[Callable, Callable]:
    """
    Look up an activation by name.

    Parameters
    ----------
    name : str
        One of 'relu', 'identity', 'sigmoid', 'softmax'.

    Returns
    -------
    (function, jacobian_function)
        The activation function and its Jacobian function.
        For element-wise activations (relu, sigmoid), the Jacobian function
        returns the diagonal values (same shape as input).
        For softmax, it returns the full (dim, dim, [batch]) matrix.

    Raises
    ------
    ValueError
        If the name is not recognized.
    """
    registry = {
        'relu': (relu, relu_jacobian),
        'identity': (identity, identity_jacobian),
        'sigmoid': (sigmoid, sigmoid_jacobian),
        'softmax': (softmax, softmax_jacobian),
    }
    if name not in registry:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(registry.keys())}"
        )
    return registry[name]
