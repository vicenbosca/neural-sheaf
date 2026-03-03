"""
SheafDynamics -- heat equation on a fixed NeuralSheaf.

Implements the restricted heat equation from the paper's Section 4.
Forward Euler integration drives an arbitrary 0-cochain toward the
harmonic extension of the boundary data, which encodes the standard
forward pass. Masks are recomputed each step (state-dependent sheaf).

Also provides ``track_trajectory`` and ``detect_mask_changes`` for
recording dynamics snapshots and diagnosing ReLU boundary crossings.
"""

import logging
import math
import torch
from typing import Dict, List, Optional, Tuple

from .sheaf import NeuralSheaf
from .pinning import (
    HardPin, SoftPin, apply_pins, compute_soft_pin_forces,
    build_all_pins, resolve_pinned_set,
)
from .activations import relu_mask, get_activation

logger = logging.getLogger(__name__)


class SheafDynamics:
    """
    Heat equation dynamics on a fixed NeuralSheaf.

    Runs the restricted sheaf heat equation with forward Euler
    integration. The input vertex is hard-pinned to x; all other
    vertices are free. Convergence recovers the standard forward pass.

    Parameters
    ----------
    sheaf : NeuralSheaf
        The neural sheaf encoding the network.
    alpha : float
        Diffusion rate (scales the Laplacian). Default 1.0.
    dt : float
        Euler step size. Default 0.01.
    output_edge_mode : str
        How to compute the force on z_{k+1} from the output activation
        edge. 'jacobian' (default) uses the Jacobian transpose:
        -J_phi(z)^T (phi(z) - a_output), corresponding to L2 edge energy.
        'identity' uses -(phi(z) - a_output) directly, corresponding to
        cross-entropy-like error propagation (no saturation for
        confident predictions). Use 'identity' when running diffusion
        on SGD-trained classification networks with pinned output.
    """

    def __init__(
        self,
        sheaf: NeuralSheaf,
        alpha: float = 1.0,
        dt: float = 0.01,
        output_edge_mode: str = 'jacobian',
    ):
        if output_edge_mode not in ('jacobian', 'identity'):
            raise ValueError(
                f"output_edge_mode must be 'jacobian' or 'identity', "
                f"got '{output_edge_mode}'."
            )
        self.sheaf = sheaf
        self.alpha = alpha
        self.dt = dt
        self.output_edge_mode = output_edge_mode

        # Output activation: function for evaluation, Jacobian for force
        self._out_act_name = sheaf.output_activation
        _, self._out_jac_fn = get_activation(self._out_act_name)
        self._out_act_fn = sheaf._output_activation_fn

    # ------------------------------------------------------------------ #
    #  Single Euler step                                                   #
    # ------------------------------------------------------------------ #

    def step(
        self,
        state: Dict[str, list],
        x: torch.Tensor,
        pins: Optional[List[HardPin]] = None,
        soft_pins: Optional[List[SoftPin]] = None,
    ) -> Dict[str, list]:
        """
        Perform one forward Euler step of the restricted heat equation.

        Parameters
        ----------
        state : dict
            Current stalk values:
            'z': [z_1, ..., z_{k+1}], 'a': [a_0, a_1, ..., a_k].
            Optionally 'a_output' if output activation is non-identity.
        x : torch.Tensor
            Input data, shape (n_0, batch_size). Used in default input pin.
        pins : list of HardPin, optional
            Additional hard pins beyond the default a_0 = x. If a pin
            targets a[0], it overrides the default input pin.
        soft_pins : list of SoftPin, optional
            Soft pins that add restoring forces toward target values.

        Returns
        -------
        dict
            Updated state (new dict with new tensors; does not mutate input).
        """
        sheaf = self.sheaf
        k = sheaf.k
        scale = self.alpha * self.dt

        z_old = state['z']
        a_old = state['a']

        # Determine which stalks are hard-pinned (skip force computation)
        pinned = resolve_pinned_set(pins)

        # Compute soft pin forces
        sp_forces = {}
        if soft_pins:
            sp_forces = compute_soft_pin_forces(state, soft_pins, pinned)

        # Recompute masks from current pre-activations (state-dependent sheaf)
        masks = sheaf.compute_masks(z_old)
        masks_f = [m.to(sheaf.dtype) for m in masks]

        # Allocate new state
        z_new = [None] * len(z_old)
        a_new = [None] * (k + 1)

        # --- Update pre-activations z_ell ---
        # z_old[ell] = z_{ell+1} in math notation, ell = 0 ... k
        for ell in range(len(z_old)):
            if ('z', ell) in pinned:
                z_new[ell] = z_old[ell]  # will be overwritten by apply_pins
                continue

            W = sheaf.weights[ell]
            b = sheaf.biases[ell]
            a_prev = a_old[ell]  # a_{ell} in math

            # Weight edge: -(z - (W a_prev + b))
            weight_pull = -(z_old[ell] - (W @ a_prev + b))

            # Activation edge (hidden only): -mask * (z - a)
            if ell < k:
                act_pull = -(masks_f[ell] * (z_old[ell] - a_old[ell + 1]))
            else:
                act_pull = 0.0

            # Output activation edge (z_{k+1} with non-identity)
            out_act_pull = 0.0
            if ell == k and self._out_act_name != 'identity':
                a_output = state['a_output']
                phi_z = self._out_act_fn(z_old[ell])
                if self.output_edge_mode == 'identity':
                    # Cross-entropy-like: force = -(phi(z) - a_output)
                    # No Jacobian factor, avoids saturation for sigmoid/softmax
                    out_act_pull = -(phi_z - a_output)
                else:
                    # L2-like: force = -J_phi^T (phi(z) - a_output)
                    Jt_v = self._jacobian_transpose_times_vec(
                        z_old[ell], phi_z - a_output
                    )
                    out_act_pull = -Jt_v

            # Soft pin force on z[ell]
            sp_force = sp_forces.get(('z', ell), 0.0)

            z_new[ell] = z_old[ell] + scale * (
                weight_pull + act_pull + out_act_pull + sp_force
            )

        # --- Update post-activations a_ell ---
        for ell in range(k + 1):
            if ('a', ell) in pinned:
                a_new[ell] = a_old[ell]  # will be overwritten by apply_pins
                continue

            if ell == 0:
                # a_0 force computation: normally unreachable because a_0 is
                # always hard-pinned (input boundary). Kept for completeness
                # in case custom pin sets omit the default input pin.
                W1 = sheaf.weights[0]
                b1 = sheaf.biases[0]
                weight_disc = W1 @ a_old[0] + b1 - z_old[0]
                weight_pull = -(W1.T @ weight_disc)

                sp_force = sp_forces.get(('a', 0), 0.0)
                a_new[0] = a_old[0] + scale * (weight_pull + sp_force)
            else:
                # a_ell, ell = 1 ... k
                # Activation edge: -(a - mask * z)
                act_pull = -(a_old[ell] - masks_f[ell - 1] * z_old[ell - 1])

                # Weight edge from next layer: -W^T (W a + b - z_next)
                W_next = sheaf.weights[ell]
                b_next = sheaf.biases[ell]
                weight_disc = W_next @ a_old[ell] + b_next - z_old[ell]
                weight_pull = -(W_next.T @ weight_disc)

                sp_force = sp_forces.get(('a', ell), 0.0)
                a_new[ell] = a_old[ell] + scale * (
                    act_pull + weight_pull + sp_force
                )

        # --- Update output post-activation (if non-identity and not pinned) ---
        new_state = {'z': z_new, 'a': a_new}
        if self._out_act_name != 'identity':
            if ('a_output', None) in pinned:
                new_state['a_output'] = state['a_output']
            else:
                a_output = state['a_output']
                phi_z = self._out_act_fn(z_old[k])
                out_force = -(a_output - phi_z)

                sp_force = sp_forces.get(('a_output', None), 0.0)
                new_state['a_output'] = a_output + scale * (
                    out_force + sp_force
                )

        # --- Apply all hard pins (default + user) ---
        all_pins = build_all_pins(x, pins)
        apply_pins(new_state, all_pins)

        # Carry pins in state for inspection
        new_state['pins'] = all_pins
        if soft_pins:
            new_state['soft_pins'] = list(soft_pins)

        return new_state

    # ------------------------------------------------------------------ #
    #  Run until convergence                                               #
    # ------------------------------------------------------------------ #

    def run(
        self,
        x: torch.Tensor,
        max_iter: int = 100_000,
        tol: float = 1e-9,
        min_iter: int = 5,
        init_method: str = 'random',
        state: Optional[Dict] = None,
        seed: Optional[int] = None,
        pins: Optional[List[HardPin]] = None,
        soft_pins: Optional[List[SoftPin]] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Run heat equation dynamics until convergence.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (n_0, batch_size) or (n_0,).
        max_iter : int
            Maximum number of Euler steps.
        tol : float
            Convergence tolerance on the max absolute change per step.
        min_iter : int
            Minimum number of steps before checking convergence.
        init_method : str
            Stalk initialization: 'random', 'zeros', 'forward_pass'.
        state : dict, optional
            If provided, use as initial state instead of init_method.
        seed : int, optional
            Random seed for stalk initialization.
        pins : list of HardPin, optional
            Additional hard pins beyond the default a_0 = x. For example,
            pin a_output to target labels for diagnostic analysis.
        soft_pins : list of SoftPin, optional
            Soft pins that add restoring forces toward target values.

        Returns
        -------
        output : torch.Tensor
            Converged output, shape (n_{k+1}, batch_size).
        state : dict
            Final stalk values.
        info : dict
            'iterations': int, 'converged': bool, 'final_change': float.
        """
        sheaf = self.sheaf
        x = x.to(device=sheaf.device, dtype=sheaf.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Initialize stalks
        if state is None:
            if seed is not None:
                torch.manual_seed(seed)
            state = sheaf.init_stalks(x, method=init_method)

        # Apply pins to initial state
        init_pins = build_all_pins(x, pins)
        apply_pins(state, init_pins)

        # Main loop
        converged = False
        final_change = float('inf')

        for iteration in range(max_iter):
            new_state = self.step(state, x, pins=pins, soft_pins=soft_pins)

            # Check convergence after min_iter
            if iteration >= min_iter - 1:
                final_change = self._max_stalk_change(state, new_state)
                if final_change < tol:
                    state = new_state
                    converged = True
                    break

            state = new_state

        # Extract output
        if self._out_act_name != 'identity':
            output = state['a_output']
        else:
            output = state['z'][-1]

        info = {
            'iterations': iteration + 1,
            'converged': converged,
            'final_change': final_change,
        }
        return output, state, info

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _max_stalk_change(
        old_state: Dict[str, list],
        new_state: Dict[str, list],
    ) -> float:
        """
        Compute the maximum absolute change across all free stalks.

        Compares z lists, a lists (skipping a[0] which is pinned), and
        a_output if present.

        Parameters
        ----------
        old_state : dict
            Previous state.
        new_state : dict
            Current state.

        Returns
        -------
        float
            Maximum absolute element-wise change.
        """
        max_change = 0.0
        for z_o, z_n in zip(old_state['z'], new_state['z']):
            val = (z_n - z_o).abs().max().item()
            if not math.isfinite(val):
                raise RuntimeError(
                    "Dynamics diverged (NaN/Inf in z stalks). "
                    "Try reducing dt or checking parameters."
                )
            max_change = max(max_change, val)
        for a_o, a_n in zip(old_state['a'][1:], new_state['a'][1:]):
            val = (a_n - a_o).abs().max().item()
            if not math.isfinite(val):
                raise RuntimeError(
                    "Dynamics diverged (NaN/Inf in a stalks). "
                    "Try reducing dt or checking parameters."
                )
            max_change = max(max_change, val)
        if 'a_output' in old_state and 'a_output' in new_state:
            val = (new_state['a_output'] - old_state['a_output']).abs().max().item()
            if not math.isfinite(val):
                raise RuntimeError(
                    "Dynamics diverged (NaN/Inf in output stalks). "
                    "Try reducing dt or checking parameters."
                )
            max_change = max(max_change, val)
        return max_change

    def _jacobian_transpose_times_vec(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute J_phi(z)^T @ v for the output activation.

        Handles three cases:
        - sigmoid: J is diagonal, so J^T v = diag(sigma'(z)) * v.
        - softmax: J is full (dim, dim, batch), computed via einsum.
        - identity: J^T v = v (trivially).

        Parameters
        ----------
        z : torch.Tensor
            Pre-activation, shape (dim, batch_size).
        v : torch.Tensor
            Vector to multiply, shape (dim, batch_size).

        Returns
        -------
        torch.Tensor
            J^T @ v, shape (dim, batch_size).
        """
        name = self._out_act_name

        if name == 'identity':
            return v

        if name == 'sigmoid':
            J_diag = self._out_jac_fn(z)  # (dim, batch)
            return J_diag * v

        if name == 'softmax':
            J = self._out_jac_fn(z)
            if z.dim() == 1:
                return J.T @ v
            # J[i,j,b], v[i,b] -> (J^T v)[j,b] = sum_i J[i,j,b]*v[i,b]
            return torch.einsum('ijb,ib->jb', J, v)

        raise ValueError(f"Unsupported activation: {name}")


# ====================================================================== #
#  Trajectory tracking and mask-change detection                          #
# ====================================================================== #

def _copy_state(state: Dict) -> Dict:
    """Deep-copy a state dict (detach and clone all tensors)."""
    new = {
        'z': [t.detach().clone() for t in state['z']],
        'a': [t.detach().clone() for t in state['a']],
    }
    if 'a_output' in state:
        new['a_output'] = state['a_output'].detach().clone()
    return new


def track_trajectory(
    dynamics: SheafDynamics,
    x: torch.Tensor,
    max_iter: int = 100_000,
    tol: float = 1e-9,
    freq: int = 10,
    init_method: str = 'random',
    state: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run dynamics while recording state snapshots and discord history.

    Parameters
    ----------
    dynamics : SheafDynamics
        The dynamics runner.
    x : torch.Tensor
        Input data, shape ``(n_0, batch_size)`` or ``(n_0,)``.
    max_iter : int
        Maximum number of Euler steps.
    tol : float
        Convergence tolerance on max absolute change per step.
    freq : int
        Record a snapshot every ``freq`` steps. Also records the initial
        state (step 0) and the final state.
    init_method : str
        Stalk initialization method.
    state : dict, optional
        If provided, use as initial state instead of calling ``init_stalks``.
    seed : int, optional
        Random seed for initialization.

    Returns
    -------
    states : list of dict
        State snapshots at recorded steps.
    discord_history : list of dict
        Discord dicts at the same steps. Each dict also has an
        ``'iteration'`` key recording which step it was taken at.
    """
    from .discord import compute_discord

    sheaf = dynamics.sheaf
    x = x.to(device=sheaf.device, dtype=sheaf.dtype)
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    # Initialize
    if state is None:
        if seed is not None:
            torch.manual_seed(seed)
        state = sheaf.init_stalks(x, method=init_method)

    states = []
    discord_history = []

    def _record(st, iteration):
        """Record a deep copy of the state and its discord."""
        states.append(_copy_state(st))
        disc = compute_discord(sheaf, st, x)
        disc['iteration'] = iteration
        discord_history.append(disc)

    _record(state, 0)

    for iteration in range(1, max_iter + 1):
        new_state = dynamics.step(state, x)

        max_change = SheafDynamics._max_stalk_change(state, new_state)
        state = new_state

        if iteration % freq == 0:
            _record(state, iteration)

        if max_change < tol:
            if discord_history[-1]['iteration'] != iteration:
                _record(state, iteration)
            break
    else:
        # Loop completed without convergence -- record final if needed
        if discord_history[-1]['iteration'] != max_iter:
            _record(state, max_iter)

    return states, discord_history


def detect_mask_changes(
    states: List[Dict],
    discord_history: List[Dict],
    sheaf: NeuralSheaf,
    sample_idx: int = 0,
) -> List[Dict]:
    """
    Detect iterations where ReLU activation masks change between snapshots.

    Compares the sign pattern of each hidden-layer pre-activation between
    consecutive recorded snapshots. A "mask change" means at least one
    neuron crossed the ``z=0`` boundary (switched between active and
    inactive).

    Parameters
    ----------
    states : list of dict
        State snapshots from ``track_trajectory``.
    discord_history : list of dict
        Matching discord history (used for iteration numbers).
    sheaf : NeuralSheaf
        The neural sheaf (for ``k``, the number of hidden layers).
    sample_idx : int
        Which sample in the batch to check.

    Returns
    -------
    list of dict
        Each dict has keys:
        ``'snapshot_idx'``: int -- index into ``states`` of the new snapshot.
        ``'iteration'``: int -- iteration number of the new snapshot.
        ``'layer'``: int -- hidden layer (1-indexed).
        ``'components'``: list of int -- which neurons changed sign.
    """
    changes = []
    for i in range(1, len(states)):
        for ell in range(sheaf.k):
            z_old = states[i - 1]['z'][ell][:, sample_idx]
            z_new = states[i]['z'][ell][:, sample_idx]
            mask_old = z_old >= 0
            mask_new = z_new >= 0
            diff = mask_old != mask_new
            if diff.any():
                comps = diff.nonzero(as_tuple=True)[0].tolist()
                changes.append({
                    'snapshot_idx': i,
                    'iteration': discord_history[i]['iteration'],
                    'layer': ell + 1,
                    'components': comps,
                })
    return changes
