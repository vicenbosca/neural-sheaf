"""
Loss functions and their gradients for the neural sheaf library.

All functions operate on tensors with shape (dim, batch_size).
Loss functions return a scalar. Gradient functions return tensors with the
same shape as the predictions.
"""

import torch


# ============================================================================
# MSE Loss
# ============================================================================

def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predictions, shape (dim, batch_size).
    y_true : torch.Tensor
        Targets, shape (dim, batch_size).

    Returns
    -------
    torch.Tensor
        Scalar loss: mean of squared differences over all elements.
    """
    return torch.mean((y_pred - y_true) ** 2)


# ============================================================================
# Cross-Entropy Loss
# ============================================================================

def cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss, works for both binary and multiclass.

    For binary (dim=1): -mean[ y*log(p) + (1-y)*log(1-p) ]
    For multiclass (dim>1): -mean_over_batch[ sum_over_classes y*log(p) ]

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted probabilities (after sigmoid or softmax), shape (dim, batch_size).
    y_true : torch.Tensor
        True labels. For binary: values in {0,1}. For multiclass: one-hot encoded.
        Shape (dim, batch_size).

    Returns
    -------
    torch.Tensor
        Scalar loss averaged over the batch.
    """
    eps = 1e-12
    p = torch.clamp(y_pred, eps, 1.0 - eps)

    dim = y_pred.shape[0]
    if dim == 1:
        # Binary cross-entropy
        loss_per_sample = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
    else:
        # Multiclass cross-entropy
        loss_per_sample = -torch.sum(y_true * torch.log(p), dim=0, keepdim=True)

    return torch.mean(loss_per_sample)


def cross_entropy_gradient(
    z: torch.Tensor,
    y_true: torch.Tensor,
    activation: str
) -> torch.Tensor:
    """
    Gradient of cross-entropy loss w.r.t. pre-activation z.

    Exploits the well-known simplification:
        sigmoid + BCE:  grad = sigma(z) - y
        softmax + CE:   grad = softmax(z) - y

    This is the *per-sample* gradient (not averaged), shape (dim, batch_size).

    Parameters
    ----------
    z : torch.Tensor
        Pre-activation values, shape (dim, batch_size).
    y_true : torch.Tensor
        True labels, shape (dim, batch_size).
    activation : str
        Either 'sigmoid' or 'softmax'.

    Returns
    -------
    torch.Tensor
        Gradient of CE w.r.t. z, shape (dim, batch_size).

    Raises
    ------
    ValueError
        If activation is not 'sigmoid' or 'softmax'.
    """
    if activation == 'sigmoid':
        return torch.sigmoid(z) - y_true
    elif activation == 'softmax':
        return torch.softmax(z, dim=0) - y_true
    else:
        raise ValueError(
            f"cross_entropy_gradient requires 'sigmoid' or 'softmax', got '{activation}'"
        )
