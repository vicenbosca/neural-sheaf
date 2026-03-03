"""
Pinning mechanisms for sheaf dynamics.

Hard pins overwrite stalk coordinates with fixed values after each dynamics
step (boundary conditions).  Soft pins add a restoring force that pulls
coordinates toward target values without clamping them.

Both pin types use the same broadcasting rules for their ``values`` field:
    - scalar  -> fills (n_coords, batch_size)
    - 1-D tensor of length n_coords -> broadcast across batch
    - 2-D tensor (n_coords, batch_size) -> used directly
"""

import logging
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Value broadcasting (shared by hard and soft pins)                       #
# ====================================================================== #

def _broadcast_pin_values(
    vals: Union[float, int, torch.Tensor],
    n_coords: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Broadcast pin values to shape (n_coords, batch_size).

    Parameters
    ----------
    vals : float, int, or torch.Tensor
        Raw pin values (scalar, 1-D, or 2-D).
    n_coords : int
        Number of coordinates being pinned.
    batch_size : int
        Batch dimension of the target stalk.
    dtype : torch.dtype
        Target dtype.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Values broadcast to (n_coords, batch_size).

    Raises
    ------
    TypeError
        If vals is not a float, int, or Tensor.
    ValueError
        If tensor shape cannot be broadcast to (n_coords, batch_size).
    """
    if isinstance(vals, (int, float)):
        vals = torch.tensor(vals, dtype=dtype, device=device)

    if not isinstance(vals, torch.Tensor):
        raise TypeError(
            f"Pin values must be float or torch.Tensor, got {type(vals)}."
        )

    vals = vals.to(dtype=dtype, device=device)

    if vals.dim() == 0:
        return vals.expand(n_coords, batch_size)

    if vals.dim() == 1:
        if vals.shape[0] == n_coords:
            return vals.unsqueeze(1).expand(n_coords, batch_size)
        raise ValueError(
            f"1-D values length {vals.shape[0]} does not match "
            f"number of pinned coords {n_coords}."
        )

    if vals.dim() == 2:
        if vals.shape != (n_coords, batch_size):
            raise ValueError(
                f"2-D values shape {tuple(vals.shape)} does not match "
                f"expected ({n_coords}, {batch_size})."
            )
        return vals

    raise ValueError(
        f"Values must be 0-D, 1-D, or 2-D tensor, got {vals.dim()}-D."
    )


# ====================================================================== #
#  Stalk resolution helpers                                                #
# ====================================================================== #

_VALID_STALKS = ('z', 'a', 'a_output')


def _resolve_stalk_tensor(
    state: Dict[str, list],
    stalk: str,
    layer: Optional[int],
) -> torch.Tensor:
    """
    Look up the tensor for a given (stalk, layer) in a state dict.

    Parameters
    ----------
    state : dict
        Stalk state with keys 'z', 'a', and optionally 'a_output'.
    stalk : str
        One of 'z', 'a', 'a_output'.
    layer : int or None
        Index into state[stalk].  Required for 'z' and 'a'.

    Returns
    -------
    torch.Tensor
        The referenced stalk tensor.
    """
    if stalk not in _VALID_STALKS:
        raise ValueError(
            f"Stalk must be one of {_VALID_STALKS}, got '{stalk}'."
        )

    if stalk == 'a_output':
        if 'a_output' not in state:
            raise ValueError(
                "Cannot access 'a_output': key not present in state. "
                "This vertex only exists for non-identity output activations."
            )
        return state['a_output']

    # stalk is 'z' or 'a'
    if layer is None:
        raise ValueError(f"Pin on '{stalk}' requires a layer index.")
    lst = state[stalk]
    if layer < 0 or layer >= len(lst):
        raise ValueError(
            f"Layer {layer} out of range for state['{stalk}'] "
            f"(length {len(lst)})."
        )
    return lst[layer]


# ====================================================================== #
#  Hard pinning                                                            #
# ====================================================================== #

@dataclass
class HardPin:
    """
    Specification for hard-pinning coordinates in a stalk.

    A hard pin fixes selected coordinates of a stalk variable to given
    values.  After each dynamics step, pinned coordinates are overwritten
    with their target values.  This is the mechanism used for boundary
    data: the input stalk a_0 is pinned to x, and during training the
    output is pinned to y.

    Parameters
    ----------
    stalk : str
        Which stalk family: 'z', 'a', or 'a_output'.
    layer : int or None
        Index into state[stalk].  Required for 'z' and 'a' (e.g.,
        layer=0 means z_1 or a_0).  Ignored for 'a_output'.
    coords : list of int or None
        Which coordinates (rows) to pin.  None pins the entire stalk.
    values : float or torch.Tensor
        Target values.  See module docstring for broadcasting rules.
    """
    stalk: str
    layer: Optional[int] = None
    coords: Optional[List[int]] = None
    values: Union[float, torch.Tensor] = 0.0

    def __repr__(self) -> str:
        parts = [f"stalk='{self.stalk}'"]
        if self.stalk != 'a_output':
            parts.append(f"layer={self.layer}")
        if self.coords is not None:
            parts.append(f"coords={self.coords}")
        if isinstance(self.values, torch.Tensor):
            parts.append(f"values=Tensor{tuple(self.values.shape)}")
        else:
            parts.append(f"values={self.values}")
        return f"HardPin({', '.join(parts)})"


def apply_pins(
    state: Dict[str, list],
    pins: List[HardPin],
) -> Dict[str, list]:
    """
    Apply hard pins to a stalk state, overwriting pinned coordinates.

    Parameters
    ----------
    state : dict
        Stalk state with keys 'z', 'a', and optionally 'a_output'.
    pins : list of HardPin
        Pins to apply.

    Returns
    -------
    dict
        The same state dict (modified in-place for efficiency).
    """
    for pin in pins:
        target = _resolve_stalk_tensor(state, pin.stalk, pin.layer)
        batch_size = target.shape[1]

        if pin.coords is None:
            n_coords = target.shape[0]
        else:
            n_coords = len(pin.coords)

        vals = _broadcast_pin_values(
            pin.values, n_coords, batch_size, target.dtype, target.device,
        )

        # Overwrite the pinned coordinates
        if pin.coords is None:
            replacement = vals.clone()
            if pin.stalk == 'a_output':
                state['a_output'] = replacement
            else:
                state[pin.stalk][pin.layer] = replacement
        else:
            target[pin.coords] = vals

    return state


# ====================================================================== #
#  Pin assembly helpers                                                    #
# ====================================================================== #

def build_all_pins(
    x: torch.Tensor,
    user_pins: Optional[List[HardPin]] = None,
) -> List[HardPin]:
    """
    Assemble the complete pin list: default input pin + user pins.

    Both SheafDynamics and SheafTrainer hard-pin a[0] to the input x
    by default. This function creates that default pin and merges it
    with any user-supplied pins. If a user pin targets a[0], it
    replaces the default and a warning is logged.

    Parameters
    ----------
    x : torch.Tensor
        Input data (used as default a[0] pin value).
    user_pins : list of HardPin or None
        User-supplied pins.

    Returns
    -------
    list of HardPin
        Complete pin list for ``apply_pins``.
    """
    default_pin = HardPin('a', layer=0, values=x)

    if not user_pins:
        return [default_pin]

    overrides_input = any(
        p.stalk == 'a' and p.layer == 0 for p in user_pins
    )
    if overrides_input:
        logger.warning(
            "User pin overrides default input pin (a[0] = x). "
            "The default input pin will be replaced."
        )
        return list(user_pins)

    return [default_pin] + list(user_pins)


def resolve_pinned_set(
    pins: Optional[List[HardPin]] = None,
) -> set:
    """
    Determine which stalks are fully hard-pinned.

    A stalk is considered fully pinned if a HardPin targets it with
    ``coords=None`` (the entire stalk). Partial-coordinate pins are
    excluded because the unpinned coordinates still need forces
    computed; the overwrite happens later in ``apply_pins``.

    The input stalk a[0] is always included, since both SheafDynamics
    and SheafTrainer hard-pin it by default.

    Parameters
    ----------
    pins : list of HardPin or None
        User-supplied hard pins.

    Returns
    -------
    set of (str, int or None)
        Keys for stalks whose forces can be skipped entirely.
    """
    pinned = {('a', 0)}
    if pins:
        for p in pins:
            if p.coords is None:
                key = ('a_output', None) if p.stalk == 'a_output' else (p.stalk, p.layer)
                pinned.add(key)
    return pinned


# ====================================================================== #
#  Soft pinning                                                            #
# ====================================================================== #

@dataclass
class SoftPin:
    """
    Specification for soft-pinning coordinates in a stalk.

    A soft pin adds a restoring force that pulls selected coordinates
    toward target values.  Unlike hard pins, the variable is free to
    deviate -- the force is ``-gamma * (v - target)``, where gamma
    controls the coupling strength.  This force is added to the total
    sheaf force before the Euler step, inside the ``alpha * dt`` scaling.

    Parameters
    ----------
    stalk : str
        Which stalk family: 'z', 'a', or 'a_output'.
    layer : int or None
        Index into state[stalk].  Required for 'z' and 'a'.
    coords : list of int or None
        Which coordinates (rows) to pin.  None pins the entire stalk.
    values : float or torch.Tensor
        Target values.  See module docstring for broadcasting rules.
    gamma : float
        Coupling strength.  Default 1.0.
    """
    stalk: str
    layer: Optional[int] = None
    coords: Optional[List[int]] = None
    values: Union[float, torch.Tensor] = 0.0
    gamma: float = 1.0

    def __repr__(self) -> str:
        parts = [f"stalk='{self.stalk}'"]
        if self.stalk != 'a_output':
            parts.append(f"layer={self.layer}")
        if self.coords is not None:
            parts.append(f"coords={self.coords}")
        if isinstance(self.values, torch.Tensor):
            parts.append(f"values=Tensor{tuple(self.values.shape)}")
        else:
            parts.append(f"values={self.values}")
        parts.append(f"gamma={self.gamma}")
        return f"SoftPin({', '.join(parts)})"


def compute_soft_pin_forces(
    state: Dict[str, list],
    soft_pins: List[SoftPin],
    hard_pinned: set,
) -> Dict[tuple, torch.Tensor]:
    """
    Compute restoring forces from all soft pins.

    Parameters
    ----------
    state : dict
        Current stalk state.
    soft_pins : list of SoftPin
        Soft pins to evaluate.
    hard_pinned : set
        Set of (stalk, layer) keys that are hard-pinned.  Soft pins on
        hard-pinned variables are skipped with a warning.

    Returns
    -------
    dict
        Maps (stalk, layer) -> force tensor.  The force is
        ``-gamma * (current - target)``, ready to be added to the
        total force in the Euler step.
    """
    forces: Dict[tuple, torch.Tensor] = {}

    for sp in soft_pins:
        key = ('a_output', None) if sp.stalk == 'a_output' else (sp.stalk, sp.layer)

        # Skip if hard-pinned
        if key in hard_pinned:
            logger.warning(
                "SoftPin on %s conflicts with hard pin; "
                "hard pin takes precedence.", key,
            )
            continue

        current = _resolve_stalk_tensor(state, sp.stalk, sp.layer)
        batch_size = current.shape[1]

        # Resolve and broadcast target values
        if sp.coords is None:
            n_coords = current.shape[0]
        else:
            n_coords = len(sp.coords)

        target = _broadcast_pin_values(
            sp.values, n_coords, batch_size, current.dtype, current.device,
        )

        # Compute force: -gamma * (v - target)
        if sp.coords is None:
            force = -sp.gamma * (current - target)
        else:
            force = torch.zeros_like(current)
            force[sp.coords] = -sp.gamma * (current[sp.coords] - target)

        # Accumulate (multiple soft pins can target the same stalk)
        if key in forces:
            forces[key] = forces[key] + force
        else:
            forces[key] = force

    return forces
