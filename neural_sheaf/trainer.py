"""
SheafTrainer -- joint stalk and weight dynamics for training.

Implements the training framework where both stalk values and restriction
maps (weights, biases) evolve via gradient flow on the sheaf energy.
The input stalk a^(0) is hard-pinned to x and the output stalk a^(k+1)
is hard-pinned to y (target labels). These are boundary data in the
sheaf sense. The force on z^(k+1) from the output edge is the
discrepancy phi(z) - y, which for cross-entropy losses equals the
gradient of the loss with respect to z.
"""

import torch
import logging
from typing import Dict, List, Optional

from .sheaf import NeuralSheaf
from .pinning import (
    HardPin, SoftPin, apply_pins, compute_soft_pin_forces,
    build_all_pins, resolve_pinned_set,
)
from .losses import mse_loss, cross_entropy_loss, cross_entropy_gradient

logger = logging.getLogger(__name__)


class SheafTrainer:
    """
    Joint stalk and weight dynamics for training a NeuralSheaf.

    Training minimizes the total sheaf energy (sum of squared edge
    discrepancies) with boundary conditions: the input stalk a^(0) is
    hard-pinned to x and the output stalk a^(k+1) is hard-pinned to y.
    Forward Euler integration updates both stalks and weights/biases
    simultaneously.

    The stalk dynamics match SheafDynamics.step (weight edge pull and
    activation edge pull on hidden vertices), with an added output
    edge force on z^(k+1) from the discrepancy phi(z) - y. For
    identity output this is z - y; for sigmoid/softmax it is
    phi(z) - y (the cross-entropy gradient w.r.t. pre-activation).

    Note: the output stalk a^(k+1) = y is boundary data and never
    updated. The trainer does not create a separate a_output vertex
    in the state -- instead, the pinning target y is referenced
    directly when computing the output edge force.

    Parameters
    ----------
    sheaf : NeuralSheaf
        The neural sheaf to train. Weights and biases are modified
        in-place during training.
    alpha : float
        Stalk dynamics rate (scales Laplacian forces). Default 1.0.
    beta : float
        Weight/bias dynamics rate. Default 1.0.
    dt : float
        Euler step size. Default 0.001.
    """

    def __init__(
        self,
        sheaf: NeuralSheaf,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 0.001,
    ):
        self.sheaf = sheaf
        self.alpha = alpha
        self.beta = beta
        self.dt = dt

    def __repr__(self) -> str:
        return (
            f"SheafTrainer(sheaf={self.sheaf!r}, "
            f"alpha={self.alpha}, beta={self.beta}, dt={self.dt})"
        )

    # ------------------------------------------------------------------ #
    #  Single training step                                                #
    # ------------------------------------------------------------------ #

    def train_step(
        self,
        state: Dict[str, list],
        x: torch.Tensor,
        y: torch.Tensor,
        pins: Optional[List[HardPin]] = None,
        soft_pins: Optional[List[SoftPin]] = None,
    ) -> Dict[str, list]:
        """
        Perform one forward Euler step of the joint dynamics.

        All forces are computed from the current (old) state and current
        weights before any updates are applied. Then stalks are updated
        (returned as a new dict) and weights/biases are updated in-place
        on self.sheaf.

        Boundary stalks a^(0) = x and a^(k+1) = y are hard-pinned and
        never updated by the dynamics. The input stalk a_0 is kept at x
        and the target y is used to compute the output edge force on
        z^(k+1).

        Stalk dynamics: same edge forces as SheafDynamics.step, with an
        added output edge force on z^(k+1): the discrepancy phi(z) - y.

        Weight dynamics::

            dW_l = -beta * (W_l a_{l-1} + b_l - z_l) @ a_{l-1}^T
            db_l = -beta * sum_over_batch(W_l a_{l-1} + b_l - z_l)

        Both weight and bias updates use ``sum`` over the batch (not
        mean). The matmul in ``dW`` naturally sums, and the bias update
        uses the same convention. The learning rate beta absorbs any
        normalization preference.

        Parameters
        ----------
        state : dict
            Current stalk values:
            'z': [z_1, ..., z_{k+1}], 'a': [a_0, a_1, ..., a_k].
        x : torch.Tensor
            Input data, shape (n_0, batch_size).
        y : torch.Tensor
            Target data, shape (n_{k+1}, batch_size).
        pins : list of HardPin, optional
            Additional hard pins beyond the defaults (a[0]=x).
        soft_pins : list of SoftPin, optional
            Soft pins that add restoring forces toward target values.

        Returns
        -------
        dict
            Updated stalk state (new dict; does not mutate input).
        """
        sheaf = self.sheaf
        k = sheaf.k
        dt = self.dt

        z_old = state['z']
        a_old = state['a']

        # Determine which stalks are fully hard-pinned (skip force computation)
        pinned = resolve_pinned_set(pins)

        # Compute soft pin forces
        sp_forces = {}
        if soft_pins:
            sp_forces = compute_soft_pin_forces(state, soft_pins, pinned)

        # Recompute masks from current pre-activations
        masks = sheaf.compute_masks(z_old)
        masks_f = [m.to(sheaf.dtype) for m in masks]

        # Compute weight edge residuals: r_l = W_l @ a_{l-1} + b_l - z_l
        residuals = []
        for ell in range(k + 1):
            r = sheaf.weights[ell] @ a_old[ell] + sheaf.biases[ell] - z_old[ell]
            residuals.append(r)

        # ==============================================================
        # 1. Stalk dynamics
        # ==============================================================

        z_new = [None] * len(z_old)
        a_new = [None] * (k + 1)

        # --- Pre-activations z_l ---
        for ell in range(len(z_old)):
            if ('z', ell) in pinned:
                z_new[ell] = z_old[ell]
                continue

            # Weight edge pull (toward W@a + b): +r = +(Wa + b - z)
            weight_pull = residuals[ell]

            # Activation edge pull (hidden layers only)
            if ell < k:
                act_pull = -(masks_f[ell] * (z_old[ell] - a_old[ell + 1]))
            else:
                act_pull = 0.0

            # Output edge force on z_{k+1}: discrepancy phi(z) - y
            output_force = 0.0
            if ell == k:
                output_force = self._output_edge_force(z_old[ell], y)

            # Soft pin force on z[ell]
            sp_force = sp_forces.get(('z', ell), 0.0)

            z_new[ell] = z_old[ell] + self.alpha * dt * (
                weight_pull + act_pull - output_force + sp_force
            )

        # --- Post-activations a_l (l = 0..k) ---
        for ell in range(k + 1):
            if ('a', ell) in pinned:
                a_new[ell] = a_old[ell]  # will be overwritten by apply_pins
                continue

            if ell == 0:
                # a_0: would be pinned by default; compute force anyway
                # in case user has overridden the default pin
                W1 = sheaf.weights[0]
                b1 = sheaf.biases[0]
                weight_disc = W1 @ a_old[0] + b1 - z_old[0]
                weight_pull = -(W1.T @ weight_disc)

                sp_force = sp_forces.get(('a', 0), 0.0)
                a_new[0] = a_old[0] + self.alpha * dt * (
                    weight_pull + sp_force
                )
            else:
                # a_l, l = 1..k
                # Activation edge pull
                act_pull = -(a_old[ell] - masks_f[ell - 1] * z_old[ell - 1])

                # Weight edge pull from next layer: -W^T @ r
                weight_pull = -(sheaf.weights[ell].T @ residuals[ell])

                sp_force = sp_forces.get(('a', ell), 0.0)
                a_new[ell] = a_old[ell] + self.alpha * dt * (
                    act_pull + weight_pull + sp_force
                )

        new_state = {'z': z_new, 'a': a_new}

        # ==============================================================
        # 2. Weight and bias dynamics (update sheaf in-place)
        # ==============================================================

        for ell in range(k + 1):
            a_prev = a_old[ell]
            r = residuals[ell]

            # dW = -beta * r @ a_prev^T  (sums over batch via matmul)
            dW = -self.beta * (r @ a_prev.T)

            # db = -beta * sum(r, dim=batch)  (keepdim for (n,1) shape)
            db = -self.beta * r.sum(dim=1, keepdim=True)

            sheaf.weights[ell] = sheaf.weights[ell] + dt * dW
            sheaf.biases[ell] = sheaf.biases[ell] + dt * db

        # ==============================================================
        # 3. Apply hard pins
        # ==============================================================

        all_pins = build_all_pins(x, pins)
        apply_pins(new_state, all_pins)
        new_state['pins'] = all_pins
        if soft_pins:
            new_state['soft_pins'] = list(soft_pins)

        return new_state

    # ------------------------------------------------------------------ #
    #  Full training loop                                                  #
    # ------------------------------------------------------------------ #

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        epochs: int = 100,
        steps_per_epoch: int = 1000,
        warm_start: bool = True,
        init_method: str = 'random',
        log_freq: Optional[int] = None,
        discord_freq: Optional[int] = None,
        seed: Optional[int] = None,
        pins: Optional[List[HardPin]] = None,
        soft_pins: Optional[List[SoftPin]] = None,
        return_state: bool = False,
    ) -> Dict:
        """
        Run the full training loop.

        Each epoch runs ``steps_per_epoch`` joint dynamics steps. Between
        epochs, stalks are either kept (warm start) or reinitialized.
        Metrics are recorded after each epoch.

        Parameters
        ----------
        X_train : torch.Tensor
            Training inputs, shape (n_0, n_train) or (n_0,).
        y_train : torch.Tensor
            Training targets, shape (n_{k+1}, n_train) or (n_{k+1},).
        X_test : torch.Tensor, optional
            Test inputs for evaluation.
        y_test : torch.Tensor, optional
            Test targets for evaluation.
        epochs : int
            Number of training epochs. Default 100.
        steps_per_epoch : int
            Dynamics steps per epoch. Default 1000.
        warm_start : bool
            If True, keep stalk state across epochs. If False, reinitialize
            stalks at the start of each epoch. Default True.
        init_method : str
            Stalk initialization: 'random', 'zeros', 'forward_pass'.
            Default 'random'.
        log_freq : int, optional
            If set, record intra-epoch metrics every ``log_freq`` steps.
            Metrics are always recorded at epoch boundaries.
        discord_freq : int, optional
            If set, record per-edge discord from the training stalk state
            every ``discord_freq`` steps. Results stored in
            history['discord'] as a list of dicts from
            ``compute_training_discord``, each augmented with a 'step' key.
        seed : int, optional
            Random seed for stalk initialization.
        pins : list of HardPin, optional
            Additional hard pins beyond the defaults (a[0]=x).
        soft_pins : list of SoftPin, optional
            Soft pins that add restoring forces toward target values.
        return_state : bool
            If True, return ``(history, final_state)`` where final_state
            is the stalk state dict at the end of training. Useful for
            post-training analysis (e.g. tension deviation computation).
            Default False for backward compatibility.

        Returns
        -------
        dict or tuple
            If ``return_state`` is False (default): training history dict.
            If ``return_state`` is True: ``(history, final_state)`` tuple.

            Training history has keys:

            - 'train_loss': list of floats (forward-pass loss per epoch,
              including epoch 0 = initial loss).
            - 'test_loss': list of floats (if test data provided).
            - 'epoch_steps': list of ints (cumulative step count).
            - 'intra_loss': list of (step, loss) tuples (if log_freq set).
            - 'discord': list of dicts (if discord_freq set). Each dict
              has 'step' plus per-edge discord keys.
        """
        sheaf = self.sheaf

        # Prepare data
        X_train = self._prepare_data(X_train)
        y_train = self._prepare_data(y_train)

        has_test = X_test is not None and y_test is not None
        if has_test:
            X_test = self._prepare_data(X_test)
            y_test = self._prepare_data(y_test)

        # Build pin list (warns if user overrides a[0])
        init_pins = build_all_pins(X_train, pins)

        # History
        history: Dict[str, list] = {
            'train_loss': [],
            'test_loss': [],
            'epoch_steps': [],
        }
        if log_freq is not None:
            history['intra_loss'] = []
        if discord_freq is not None:
            from .discord import compute_training_discord
            history['discord'] = []

        # Record initial loss (epoch 0)
        train_loss = self._compute_loss(X_train, y_train)
        history['train_loss'].append(train_loss)
        history['epoch_steps'].append(0)
        if has_test:
            history['test_loss'].append(self._compute_loss(X_test, y_test))

        logger.info("Epoch %3d | train_loss=%.6f", 0, train_loss)

        # Initialize stalks
        if seed is not None:
            torch.manual_seed(seed)
        state = sheaf.init_stalks(X_train, method=init_method)
        apply_pins(state, init_pins)

        # Record initial discord (step 0)
        if discord_freq is not None:
            disc = compute_training_discord(sheaf, state, y_train)
            disc['step'] = 0
            history['discord'].append(disc)

        total_steps = 0

        for epoch in range(1, epochs + 1):
            # Optionally reinitialize stalks
            if not warm_start and epoch > 1:
                state = sheaf.init_stalks(X_train, method=init_method)
                apply_pins(state, init_pins)

            # Inner dynamics loop
            for step in range(steps_per_epoch):
                state = self.train_step(
                    state, X_train, y_train,
                    pins=pins, soft_pins=soft_pins,
                )
                total_steps += 1

                # Intra-epoch logging
                if log_freq is not None and total_steps % log_freq == 0:
                    loss = self._compute_loss(X_train, y_train)
                    history['intra_loss'].append((total_steps, loss))

                # Discord tracking
                if discord_freq is not None and total_steps % discord_freq == 0:
                    disc = compute_training_discord(sheaf, state, y_train)
                    disc['step'] = total_steps
                    history['discord'].append(disc)

            # End-of-epoch metrics (using forward pass, not stalk values)
            train_loss = self._compute_loss(X_train, y_train)
            history['train_loss'].append(train_loss)
            history['epoch_steps'].append(total_steps)

            if has_test:
                test_loss = self._compute_loss(X_test, y_test)
                history['test_loss'].append(test_loss)
                logger.info(
                    "Epoch %3d | train_loss=%.6f | test_loss=%.6f",
                    epoch, train_loss, test_loss,
                )
            else:
                logger.info("Epoch %3d | train_loss=%.6f", epoch, train_loss)

        if return_state:
            return history, state
        return history

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using the standard forward pass (not stalk values).

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_0, batch_size) or (n_0,).

        Returns
        -------
        torch.Tensor
            Network output, shape (n_{k+1}, batch_size).
        """
        output, _ = self.sheaf.forward(X)
        return output

    def predict_classes(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for classification tasks.

        For sigmoid (binary): threshold probabilities at 0.5, return 0/1.
        For softmax (multiclass): return argmax class index.
        For identity: returns raw output (not a classification task).

        Parameters
        ----------
        X : torch.Tensor
            Input data, shape (n_0, batch_size) or (n_0,).

        Returns
        -------
        torch.Tensor
            For sigmoid: shape (1, batch_size), dtype int64.
            For softmax: shape (batch_size,), dtype int64.
            For identity: shape (n_{k+1}, batch_size), raw output.
        """
        with torch.no_grad():
            output = self.predict(X)
            act = self.sheaf.output_activation

            if act == 'sigmoid':
                return (output > 0.5).to(torch.int64)
            elif act == 'softmax':
                return torch.argmax(output, dim=0)
            else:
                return output

    def compute_accuracy(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        task: str = 'binary',
    ) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``(input_dim, n_samples)``.
        Y : torch.Tensor
            For ``task='binary'``: shape ``(1, n_samples)`` with values
            in {0, 1}.
            For ``task='multiclass'``: shape ``(n_classes, n_samples)``
            one-hot encoded.
        task : str
            ``'binary'`` or ``'multiclass'``.

        Returns
        -------
        float
            Fraction of correctly classified samples.
        """
        with torch.no_grad():
            preds = self.predict_classes(X)
            if task == 'binary':
                return (preds.squeeze(0) == Y.squeeze(0)).float().mean().item()
            else:
                return (preds == Y.argmax(dim=0)).float().mean().item()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prepare_data(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Cast a data tensor to the sheaf's device and dtype, unsqueezing
        1-D inputs to column vectors.

        Parameters
        ----------
        tensor : torch.Tensor
            Input or target tensor.

        Returns
        -------
        torch.Tensor
            Tensor on the correct device/dtype with at least 2 dimensions.
        """
        tensor = tensor.to(device=self.sheaf.device, dtype=self.sheaf.dtype)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        return tensor

    def _output_edge_force(
        self, z: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the output edge force on z_{k+1}.

        This is the discrepancy phi(z) - y where phi is the output
        activation. For identity: z - y. For sigmoid: sigma(z) - y.
        For softmax: softmax(z) - y.

        These coincide with the gradient of the appropriate loss
        (MSE for identity, cross-entropy for sigmoid/softmax) with
        respect to the pre-activation z.

        Parameters
        ----------
        z : torch.Tensor
            Output pre-activation, shape (n_{k+1}, batch_size).
        y : torch.Tensor
            Target, shape (n_{k+1}, batch_size).

        Returns
        -------
        torch.Tensor
            Discrepancy phi(z) - y, same shape as z.
        """
        act = self.sheaf.output_activation

        if act == 'identity':
            return z - y

        if act in ('sigmoid', 'softmax'):
            return cross_entropy_gradient(z, y, act)

        raise ValueError(f"Unknown output activation: '{act}'")

    def _compute_loss(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> float:
        """
        Compute forward-pass loss (for logging, not dynamics).

        Uses MSE for identity output activation, cross-entropy for
        sigmoid/softmax.

        Parameters
        ----------
        X : torch.Tensor
            Inputs.
        y : torch.Tensor
            Targets.

        Returns
        -------
        float
            Scalar loss value.
        """
        with torch.no_grad():
            y_pred = self.predict(X)
            act = self.sheaf.output_activation
            if act == 'identity':
                return mse_loss(y_pred, y).item()
            elif act in ('sigmoid', 'softmax'):
                return cross_entropy_loss(y_pred, y).item()
            else:
                raise ValueError(f"Unknown output activation: '{act}'")
