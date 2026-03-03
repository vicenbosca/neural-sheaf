"""
NeuralSheaf -- a feedforward ReLU network encoded as a cellular sheaf.

The NeuralSheaf class stores the network parameters (weights, biases) and
provides a standard forward pass, activation mask computation, stalk
initialization, and Laplacian reconstruction for analysis.

Data layout: all stalk data has shape (stalk_dim, batch_size), consistent
with weight matrices of shape (n_out, n_in).
"""

import logging
import torch
from typing import Dict, List, Optional, Tuple

from .activations import relu, relu_mask, get_activation

logger = logging.getLogger(__name__)


class NeuralSheaf:
    """
    A feedforward ReLU network stored as a cellular sheaf.

    The network has architecture ``[n_0, n_1, ..., n_{k+1}]`` where n_0 is
    the input dimension, n_1 through n_k are hidden layer dimensions, and
    n_{k+1} is the output dimension.  Hidden layers use ReLU activation;
    the output activation is configurable (identity, sigmoid, or softmax).

    The sheaf graph is a path:

        a_0 -- z_1 -- a_1 -- z_2 -- ... -- a_k -- z_{k+1}

    with weight edges (connecting a to z via W, b) and activation edges
    (connecting z to a via the ReLU mask).  When the output activation is
    non-identity, an additional vertex ``a_output`` is appended.

    Parameters
    ----------
    layer_dims : list of int
        Dimensions ``[n_0, n_1, ..., n_{k+1}]``.  Must have >= 2 elements.
    weights : list of torch.Tensor, optional
        Pre-specified weight matrices.  ``weights[ell]`` has shape
        ``(n_{ell+1}, n_ell)``.  If None, uses He initialization.
    biases : list of torch.Tensor, optional
        Pre-specified bias vectors.  ``biases[ell]`` has shape
        ``(n_{ell+1}, 1)``.  If None, initializes to zeros.
    output_activation : str
        Output activation: ``'identity'``, ``'sigmoid'``, or ``'softmax'``.
    seed : int, optional
        Random seed for reproducible initialization.
    device : str
        Torch device.  Default ``'cpu'``.
    dtype : torch.dtype
        Tensor dtype.  Default ``torch.float64``.
    """

    def __init__(
        self,
        layer_dims: List[int],
        weights: Optional[List[torch.Tensor]] = None,
        biases: Optional[List[torch.Tensor]] = None,
        output_activation: str = 'identity',
        seed: Optional[int] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64,
    ):
        if len(layer_dims) < 2:
            raise ValueError(
                "layer_dims must have at least 2 elements (input and output)."
            )

        self.layer_dims = list(layer_dims)
        self.k = len(layer_dims) - 2          # number of hidden layers
        self.device = device
        self.dtype = dtype

        # --- Initialize weights and biases ---
        num_edges = len(layer_dims) - 1       # k+1 weight matrices

        if weights is not None:
            if len(weights) != num_edges:
                raise ValueError(
                    f"Expected {num_edges} weight matrices, got {len(weights)}."
                )
            self.weights = [
                w.to(device=device, dtype=dtype).clone() for w in weights
            ]
        else:
            if seed is not None:
                torch.manual_seed(seed)
            self.weights = [
                torch.randn(layer_dims[ell + 1], layer_dims[ell],
                             device=device, dtype=dtype)
                * (2.0 / layer_dims[ell]) ** 0.5
                for ell in range(num_edges)
            ]

        if biases is not None:
            if len(biases) != num_edges:
                raise ValueError(
                    f"Expected {num_edges} bias vectors, got {len(biases)}."
                )
            self.biases = [
                b.to(device=device, dtype=dtype).clone() for b in biases
            ]
        else:
            self.biases = [
                torch.zeros(layer_dims[ell + 1], 1, device=device, dtype=dtype)
                for ell in range(num_edges)
            ]

        # --- Validate shapes ---
        for ell, (w, b) in enumerate(zip(self.weights, self.biases)):
            expected_w = (layer_dims[ell + 1], layer_dims[ell])
            if w.shape != expected_w:
                raise ValueError(
                    f"weights[{ell}] shape {tuple(w.shape)}, "
                    f"expected {expected_w}."
                )
            expected_b = (layer_dims[ell + 1], 1)
            if b.shape != expected_b:
                raise ValueError(
                    f"biases[{ell}] shape {tuple(b.shape)}, "
                    f"expected {expected_b}."
                )

        # --- Output activation ---
        self._output_activation_name: Optional[str] = None
        self._output_activation_fn = None
        self._set_output_activation(output_activation)

    # ------------------------------------------------------------------ #
    #  Representation                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"NeuralSheaf(layer_dims={self.layer_dims}, "
            f"output_activation='{self.output_activation}', "
            f"params={self.num_parameters}, device='{self.device}')"
        )

    # ------------------------------------------------------------------ #
    #  Output activation                                                   #
    # ------------------------------------------------------------------ #

    def _set_output_activation(self, name: str) -> None:
        """
        Set the output activation function.

        Parameters
        ----------
        name : str
            One of ``'identity'``, ``'sigmoid'``, ``'softmax'``.
        """
        valid = ('identity', 'sigmoid', 'softmax')
        if name not in valid:
            raise ValueError(
                f"Output activation must be one of {valid}, got '{name}'."
            )
        self._output_activation_name = name
        self._output_activation_fn, _ = get_activation(name)

    @property
    def output_activation(self) -> str:
        """Name of the current output activation."""
        return self._output_activation_name

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters (weights + biases)."""
        return sum(w.numel() + b.numel()
                   for w, b in zip(self.weights, self.biases))

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Standard forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape ``(n_0, batch_size)`` or ``(n_0,)``.

        Returns
        -------
        output : torch.Tensor
            Network output after the output activation,
            shape ``(n_{k+1}, batch_size)``.
        intermediates : dict
            ``'z'``: pre-activations ``[z_1, ..., z_{k+1}]``.
            ``'a'``: post-activations ``[a_0=x, a_1, ..., a_k]``.
            The output is NOT included in ``'a'``; it is returned
            separately.
        """
        x = self._prepare_input(x)

        z_list = []   # z_1, ..., z_{k+1}
        a_list = [x]  # a_0, a_1, ..., a_k

        a = x
        for ell in range(len(self.weights)):
            z = self.weights[ell] @ a + self.biases[ell]
            z_list.append(z)

            if ell < self.k:
                a = relu(z)
                a_list.append(a)
            else:
                output = self._output_activation_fn(z)

        return output, {'z': z_list, 'a': a_list}

    def corrected_forward(
        self, x: torch.Tensor, deviation: Dict[str, list],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with tension correction from training stalk deviations.

        Applies additive corrections at each layer to account for the
        distributed tension that arises during sheaf training.  The
        standard forward pass composes restriction maps exactly, but
        the trained network was co-adapted with tensioned stalk
        configurations where each layer tolerates small discrepancies.
        This method re-introduces that tension as bias corrections::

            z_tilde[ell] = W[ell] @ a_tilde[ell-1] + b[ell] + delta_z[ell]
            a_tilde[ell] = ReLU(z_tilde[ell]) + delta_a[ell]

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape ``(n_0, batch_size)`` or ``(n_0,)``.
        deviation : dict
            Must contain:
            ``'delta_z'``: ``[delta_z_1, ..., delta_z_{k+1}]``, each
            broadcastable to ``(n_ell, batch_size)``.
            ``'delta_a'``: ``[delta_a_1, ..., delta_a_k]``, each
            broadcastable to ``(n_ell, batch_size)``.

        Returns
        -------
        output : torch.Tensor
            Corrected network output, shape ``(n_{k+1}, batch_size)``.
        intermediates : dict
            ``'z'``: corrected pre-activations.
            ``'a'``: corrected post-activations.
        """
        x = self._prepare_input(x)

        delta_z = deviation['delta_z']
        delta_a = deviation['delta_a']

        z_list = []
        a_list = [x]

        a = x
        for ell in range(len(self.weights)):
            z = self.weights[ell] @ a + self.biases[ell] + delta_z[ell]
            z_list.append(z)

            if ell < self.k:
                a = relu(z) + delta_a[ell]
                a_list.append(a)
            else:
                output = self._output_activation_fn(z)

        return output, {'z': z_list, 'a': a_list}

    # ------------------------------------------------------------------ #
    #  Activation masks                                                    #
    # ------------------------------------------------------------------ #

    def compute_masks(self, z_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute boolean activation masks for each hidden layer.

        Parameters
        ----------
        z_list : list of torch.Tensor
            Pre-activations ``[z_1, ..., z_{k+1}]``.  Only the first k
            entries are used (hidden layers); z_{k+1} is ignored.

        Returns
        -------
        list of torch.Tensor
            Boolean masks ``[mask_1, ..., mask_k]`` where ``mask[j]`` is
            True when ``z[j] >= 0`` (fast selection rule).
        """
        return [relu_mask(z_list[ell]) for ell in range(self.k)]

    # ------------------------------------------------------------------ #
    #  Stalk initialization                                                #
    # ------------------------------------------------------------------ #

    def init_stalks(
        self,
        x: torch.Tensor,
        method: str = 'random',
    ) -> Dict[str, list]:
        """
        Initialize stalk vectors for the sheaf dynamics.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape ``(n_0, batch_size)`` or ``(n_0,)``.
        method : str
            Initialization method:
            ``'forward_pass'``: exact forward pass (the harmonic extension).
            ``'random'``: independent Gaussian samples.
            ``'zeros'``: all zeros (except a_0 = x).

        Returns
        -------
        dict
            ``'z'``: ``[z_1, ..., z_{k+1}]``, each ``(n_ell, batch_size)``.
            ``'a'``: ``[a_0=x, a_1, ..., a_k]``, each ``(n_ell, batch_size)``.
            ``'a_output'``: ``(n_{k+1}, batch_size)`` -- present only when
            the output activation is non-identity.
        """
        x = self._prepare_input(x)
        bs = x.shape[1]

        if method == 'forward_pass':
            output, intermediates = self.forward(x)
            state: Dict[str, list] = {
                'z': intermediates['z'],
                'a': intermediates['a'],
            }
            if self._output_activation_name != 'identity':
                state['a_output'] = output
            return state

        if method == 'random':
            def _make(dim: int) -> torch.Tensor:
                return torch.randn(dim, bs, device=self.device, dtype=self.dtype)
        elif method == 'zeros':
            def _make(dim: int) -> torch.Tensor:
                return torch.zeros(dim, bs, device=self.device, dtype=self.dtype)
        else:
            raise ValueError(
                f"Unknown init method '{method}'. "
                "Use 'forward_pass', 'random', or 'zeros'."
            )

        z_list = [_make(self.layer_dims[ell + 1])
                  for ell in range(len(self.weights))]
        a_list = [x] + [_make(self.layer_dims[ell + 1])
                        for ell in range(self.k)]

        state = {'z': z_list, 'a': a_list}
        if self._output_activation_name != 'identity':
            state['a_output'] = _make(self.layer_dims[-1])
        return state

    # ------------------------------------------------------------------ #
    #  Laplacian reconstruction (analysis only)                            #
    # ------------------------------------------------------------------ #

    def build_laplacian_block(
        self,
        masks: List[torch.Tensor],
        block: str = 'free',
    ) -> torch.Tensor:
        """
        Reconstruct a block of the sheaf Laplacian (sparse, for analysis).

        The Laplacian covers all weight edges and activation edges, up to
        and including z_{k+1}.  The output edge (involving the final
        activation phi) is never included, regardless of output activation.

        Parameters
        ----------
        masks : list of torch.Tensor
            Boolean masks ``[mask_1, ..., mask_k]`` for hidden layers.
            If 2-D (batched), column 0 is used.
        block : str
            ``'free'``: ``L_F[Omega, Omega]`` on free coordinates.
            ``'boundary'``: ``L_F[Omega, U]`` coupling free to boundary.
            ``'full'``: complete sheaf Laplacian on extended cochain space.

        Returns
        -------
        torch.Tensor
            Sparse COO matrix of the requested block.
        """
        # Flatten batched masks to 1-D (use column 0)
        masks_1d = [m[:, 0] if m.dim() > 1 else m for m in masks]

        builders = {
            'free': self._build_L_free_sparse,
            'boundary': self._build_L_boundary_sparse,
            'full': self._build_L_full_sparse,
        }
        if block not in builders:
            raise ValueError(
                f"block must be one of {list(builders)}, got '{block}'."
            )
        return builders[block](masks_1d)

    # ------------------------------------------------------------------ #
    #  Private: input preparation                                          #
    # ------------------------------------------------------------------ #

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Cast to correct device/dtype and ensure 2-D shape."""
        x = x.to(device=self.device, dtype=self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x

    # ------------------------------------------------------------------ #
    #  Private: sparse block accumulation                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _accumulate_block(
        mat: torch.Tensor,
        r_off: int,
        c_off: int,
        rows: list,
        cols: list,
        vals: list,
    ) -> None:
        """Append nonzero entries of ``mat`` (offset by r_off, c_off) to COO lists."""
        nz = mat.nonzero(as_tuple=True)
        if nz[0].numel() == 0:
            return
        rows.append(nz[0] + r_off)
        cols.append(nz[1] + c_off)
        vals.append(mat[nz])

    @staticmethod
    def _assemble_sparse(
        rows: list, cols: list, vals: list, size: tuple,
    ) -> torch.Tensor:
        """Create a sparse COO tensor from accumulated block entries."""
        if len(rows) == 0:
            device = 'cpu'
            dtype = torch.float64
            indices = torch.empty(2, 0, dtype=torch.long, device=device)
            values = torch.empty(0, dtype=dtype, device=device)
        else:
            indices = torch.stack([torch.cat(rows), torch.cat(cols)])
            values = torch.cat(vals)
            device = values.device
            dtype = values.dtype
        return torch.sparse_coo_tensor(
            indices, values, size, device=device, dtype=dtype,
        ).coalesce()

    # ------------------------------------------------------------------ #
    #  Private: L_F[Omega, Omega] via block-tridiagonal formula            #
    # ------------------------------------------------------------------ #

    def _build_L_free_sparse(
        self, masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build L_F[Omega, Omega] from the block-tridiagonal formula (sparse).

        Free coordinates: omega = (z_1, a_1, ..., a_k, z_{k+1}).
        Block structure from Appendix A of the paper (eq. A.5).
        """
        k = self.k
        n = self.layer_dims
        _add = self._accumulate_block

        # Block offsets: z_1, a_1, z_2, a_2, ..., a_k, z_{k+1}
        offsets: List[int] = []
        pos = 0
        for ell in range(1, k + 1):
            offsets.append(pos); pos += n[ell]   # z_ell
            offsets.append(pos); pos += n[ell]   # a_ell
        offsets.append(pos)                       # z_{k+1}
        free_dim = pos + n[k + 1]

        rows, cols, vals = [], [], []

        # Diagonal blocks A_ell for ell = 1, ..., k
        for ell in range(1, k + 1):
            z_off = offsets[2 * (ell - 1)]
            a_off = offsets[2 * (ell - 1) + 1]

            R = torch.diag(masks[ell - 1].to(self.dtype))
            I_ell = torch.eye(n[ell], device=self.device, dtype=self.dtype)
            W_next = self.weights[ell]

            _add(I_ell + R,                  z_off, z_off, rows, cols, vals)
            _add(-R,                         z_off, a_off, rows, cols, vals)
            _add(-R,                         a_off, z_off, rows, cols, vals)
            _add(I_ell + W_next.T @ W_next,  a_off, a_off, rows, cols, vals)

        # Final block I_{n_{k+1}} at z_{k+1}
        z_last = offsets[-1]
        I_out = torch.eye(n[k + 1], device=self.device, dtype=self.dtype)
        _add(I_out, z_last, z_last, rows, cols, vals)

        # Off-diagonal blocks C_ell for ell = 2, ..., k
        for ell in range(2, k + 1):
            W_ell = self.weights[ell - 1]
            z_off = offsets[2 * (ell - 1)]
            a_prev = offsets[2 * (ell - 2) + 1]
            _add(-W_ell,   z_off,  a_prev, rows, cols, vals)
            _add(-W_ell.T, a_prev, z_off,  rows, cols, vals)

        # D_{k+1}: couples z_{k+1} to a_k via W_{k+1}
        if k >= 1:
            W_last = self.weights[k]
            a_k = offsets[2 * (k - 1) + 1]
            _add(-W_last,   z_last, a_k,    rows, cols, vals)
            _add(-W_last.T, a_k,    z_last, rows, cols, vals)

        return self._assemble_sparse(rows, cols, vals, (free_dim, free_dim))

    # ------------------------------------------------------------------ #
    #  Private: L_F[Omega, U] boundary coupling                            #
    # ------------------------------------------------------------------ #

    def _build_L_boundary_sparse(
        self, masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build L_F[Omega, U] coupling free coordinates to boundary data (sparse).

        Free row order:  z_1, a_1, ..., a_k, z_{k+1}.
        Boundary col order: [x, 1_{n_1}] (from v_x),
                             1_{n_2} (from v_{a_1}), ...,
                             1_{n_{k+1}} (from v_{a_k}).
        """
        k = self.k
        n = self.layer_dims
        _add = self._accumulate_block

        # Free row offsets (same order as _build_L_free_sparse)
        free_offsets: List[int] = []
        pos = 0
        for ell in range(1, k + 1):
            free_offsets.append(pos); pos += n[ell]   # z_ell
            free_offsets.append(pos); pos += n[ell]   # a_ell
        free_offsets.append(pos)                       # z_{k+1}
        free_dim = pos + n[k + 1]

        # Boundary column offsets
        # Block 0: v_x = [x(n_0), 1_{n_1}]  -> dim n_0 + n_1
        # Block ell: 1_{n_{ell+1}} from v_{a_ell}  (ell = 1..k)
        bnd_offsets = [0]
        bnd_dim = n[0] + n[1]
        for ell in range(1, k + 1):
            bnd_offsets.append(bnd_dim)
            bnd_dim += n[ell + 1]

        rows, cols, vals = [], [], []

        # Weight edge e_{z_1}: L[z_1, v_x] = -W_bar_1 = -[W_1 | B_1]
        W1 = self.weights[0]
        B1 = torch.diag(self.biases[0].squeeze(-1))
        W_bar_1 = torch.cat([W1, B1], dim=1)
        _add(-W_bar_1, free_offsets[0], bnd_offsets[0], rows, cols, vals)

        # Weight edges e_{z_ell} for ell = 2, ..., k+1
        for ell in range(2, k + 2):
            B_ell = torch.diag(self.biases[ell - 1].squeeze(-1))
            z_off = free_offsets[2 * (ell - 1)]
            bnd_col = bnd_offsets[ell - 1]

            # L[z_ell, ones at a_{ell-1}] = -B_ell
            _add(-B_ell, z_off, bnd_col, rows, cols, vals)

            # L[a_{ell-1}, ones at a_{ell-1}] = W_ell^T B_ell
            if ell - 1 <= k:
                W_ell = self.weights[ell - 1]
                a_prev_off = free_offsets[2 * (ell - 2) + 1]
                _add(W_ell.T @ B_ell, a_prev_off, bnd_col, rows, cols, vals)

        return self._assemble_sparse(rows, cols, vals, (free_dim, bnd_dim))

    # ------------------------------------------------------------------ #
    #  Private: full Laplacian via local formula                           #
    # ------------------------------------------------------------------ #

    def _build_L_full_sparse(
        self, masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build the full sheaf Laplacian on the extended cochain space (sparse).

        Vertex order: v_x, v_{z_1}, v_{a_1}, ..., v_{a_k}, v_{z_{k+1}}.
        Extended stalks: v_x has dim n_0 + n_1, v_{a_ell} has dim
        n_ell + n_{ell+1}.

        Uses the local Laplacian formula: for each edge with restriction
        maps F_src, F_dst, add F_src^T F_src and F_dst^T F_dst on diagonal,
        -F_src^T F_dst and -F_dst^T F_src on off-diagonal.
        """
        k = self.k
        n = self.layer_dims
        _add = self._accumulate_block

        # Vertex stalk dimensions (extended representation)
        vtx_dims = [n[0] + n[1]]                     # v_x
        for ell in range(1, k + 1):
            vtx_dims.append(n[ell])                   # v_{z_ell}
            vtx_dims.append(n[ell] + n[ell + 1])     # v_{a_ell}
        vtx_dims.append(n[k + 1])                     # v_{z_{k+1}}

        # Cumulative offsets into the global coordinate vector
        vtx_offsets: List[int] = []
        pos = 0
        for d in vtx_dims:
            vtx_offsets.append(pos)
            pos += d
        total_dim = pos

        # Vertex offset accessors (index into vtx_offsets)
        def _vx() -> int:
            return vtx_offsets[0]

        def _vz(ell: int) -> int:           # ell = 1, ..., k+1
            return vtx_offsets[2 * ell - 1]

        def _va(ell: int) -> int:           # ell = 1, ..., k
            return vtx_offsets[2 * ell]

        rows, cols, vals = [], [], []

        def _add_edge(F_src, src_off, F_dst, dst_off):
            """Add Laplacian contributions from one edge."""
            _add(F_src.T @ F_src,  src_off, src_off, rows, cols, vals)
            _add(F_dst.T @ F_dst,  dst_off, dst_off, rows, cols, vals)
            _add(-F_src.T @ F_dst, src_off, dst_off, rows, cols, vals)
            _add(-F_dst.T @ F_src, dst_off, src_off, rows, cols, vals)

        # Weight edges
        for ell in range(1, k + 2):
            W = self.weights[ell - 1]
            B = torch.diag(self.biases[ell - 1].squeeze(-1))
            W_bar = torch.cat([W, B], dim=1)
            I_ell = torch.eye(n[ell], device=self.device, dtype=self.dtype)

            src_off = _vx() if ell == 1 else _va(ell - 1)
            _add_edge(W_bar, src_off, I_ell, _vz(ell))

        # Activation edges
        for ell in range(1, k + 1):
            R = torch.diag(masks[ell - 1].to(self.dtype))
            dim_a = n[ell] + n[ell + 1]
            P = torch.zeros(n[ell], dim_a, device=self.device, dtype=self.dtype)
            P[:, :n[ell]] = torch.eye(
                n[ell], device=self.device, dtype=self.dtype,
            )
            _add_edge(R, _vz(ell), P, _va(ell))

        return self._assemble_sparse(rows, cols, vals, (total_dim, total_dim))
