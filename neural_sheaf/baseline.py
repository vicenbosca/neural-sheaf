"""
Traditional neural network baseline for comparison with sheaf methods.

Provides a standard feedforward ReLU network trained with SGD via
backpropagation (PyTorch autograd). Uses the same architecture specification,
data layout, and loss functions as the sheaf library for fair comparison.
"""

import torch
from typing import List, Optional, Dict

from .losses import mse_loss, cross_entropy_loss

DTYPE = torch.float64


class TraditionalNN:
    """
    Feedforward ReLU network trained with SGD.

    Architecture matches NeuralSheaf: ReLU on hidden layers, configurable
    output activation (identity / sigmoid / softmax).

    Parameters
    ----------
    layer_dims : list of int
        Layer dimensions [input_dim, hidden1, ..., output_dim].
    learning_rate : float
        SGD step size.
    output_activation : str
        'identity' (default, linear output), 'sigmoid', or 'softmax'.
        None is accepted as a synonym for 'identity'.
    device : str
        Torch device.
    seed : int or None
        Random seed for weight initialization.
    """

    def __init__(
        self,
        layer_dims: List[int],
        learning_rate: float = 0.001,
        output_activation: Optional[str] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.layer_dims = list(layer_dims)
        self.learning_rate = learning_rate
        self.output_activation = output_activation or "identity"
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # He initialization for weights, zero biases
        self.weights: List[torch.Tensor] = []
        self.biases: List[torch.Tensor] = []
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            W = torch.randn(layer_dims[i + 1], fan_in, dtype=DTYPE, device=device) \
                * (2.0 / fan_in) ** 0.5
            b = torch.zeros(layer_dims[i + 1], 1, dtype=DTYPE, device=device)
            W.requires_grad_(True)
            b.requires_grad_(True)
            self.weights.append(W)
            self.biases.append(b)

    def __repr__(self) -> str:
        arch = " -> ".join(str(d) for d in self.layer_dims)
        return (
            f"TraditionalNN({arch}, output={self.output_activation}, "
            f"lr={self.learning_rate}, device={self.device})"
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : Tensor of shape (input_dim, batch_size)

        Returns
        -------
        Tensor of shape (output_dim, batch_size)
        """
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b
            if i < len(self.weights) - 1:
                a = torch.relu(z)
            else:
                # Output layer
                if self.output_activation == "sigmoid":
                    a = torch.sigmoid(z)
                elif self.output_activation == "softmax":
                    a = torch.softmax(z, dim=0)
                else:
                    a = z
        return a

    def _compute_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss appropriate to the output activation.

        Uses MSE for identity output, cross-entropy for sigmoid/softmax.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions (post-activation), shape (output_dim, batch_size).
        y_true : torch.Tensor
            Targets, shape (output_dim, batch_size).

        Returns
        -------
        torch.Tensor
            Scalar loss.

        Raises
        ------
        ValueError
            If output_activation is not recognized.
        """
        if self.output_activation == "identity":
            return mse_loss(y_pred, y_true)
        elif self.output_activation in ("sigmoid", "softmax"):
            return cross_entropy_loss(y_pred, y_true)
        else:
            raise ValueError(
                f"Unknown output activation: '{self.output_activation}'"
            )

    def _zero_grad(self):
        """Zero out gradients on all parameters."""
        for W in self.weights:
            if W.grad is not None:
                W.grad.zero_()
        for b in self.biases:
            if b.grad is not None:
                b.grad.zero_()

    def train(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_test: Optional[torch.Tensor] = None,
        Y_test: Optional[torch.Tensor] = None,
        epochs: int = 10000,
        track_freq: int = 100,
    ) -> Dict[str, list]:
        """
        Train with full-batch SGD.

        Parameters
        ----------
        X_train : Tensor, shape (input_dim, n_train)
        Y_train : Tensor, shape (output_dim, n_train)
        X_test : Tensor or None, shape (input_dim, n_test)
        Y_test : Tensor or None, shape (output_dim, n_test)
        epochs : int
            Number of gradient steps.
        track_freq : int
            Record losses every this many epochs.

        Returns
        -------
        dict with keys 'train_loss', 'test_loss', 'epoch'.
        """
        history = {"epoch": [], "train_loss": [], "test_loss": []}

        # Record true initial loss (before any gradient step)
        with torch.no_grad():
            train_loss = self._compute_loss(
                self.forward(X_train), Y_train
            ).item()
            history["epoch"].append(0)
            history["train_loss"].append(train_loss)
            if X_test is not None and Y_test is not None:
                test_loss = self._compute_loss(
                    self.forward(X_test), Y_test
                ).item()
                history["test_loss"].append(test_loss)

        for epoch in range(1, epochs + 1):
            # Forward + backward + SGD step
            self._zero_grad()
            y_pred = self.forward(X_train)
            loss = self._compute_loss(y_pred, Y_train)
            loss.backward()

            with torch.no_grad():
                for W in self.weights:
                    W -= self.learning_rate * W.grad
                for b in self.biases:
                    b -= self.learning_rate * b.grad

            # Track
            if epoch % track_freq == 0:
                with torch.no_grad():
                    train_loss = self._compute_loss(
                        self.forward(X_train), Y_train
                    ).item()
                    history["epoch"].append(epoch)
                    history["train_loss"].append(train_loss)
                    if X_test is not None and Y_test is not None:
                        test_loss = self._compute_loss(
                            self.forward(X_test), Y_test
                        ).item()
                        history["test_loss"].append(test_loss)

        return history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass (no grad)."""
        with torch.no_grad():
            return self.forward(X)

    def predict_classes(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Returns
        -------
        Tensor of int64 class labels.
            Binary (sigmoid): shape (1, batch_size), values in {0, 1}.
            Multiclass (softmax): shape (batch_size,), argmax indices.
        """
        preds = self.predict(X)
        if self.output_activation == "sigmoid":
            return (preds > 0.5).to(torch.int64)
        elif self.output_activation == "softmax":
            return torch.argmax(preds, dim=0)
        return preds
