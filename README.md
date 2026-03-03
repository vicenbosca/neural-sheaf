# Neural Sheaf

A PyTorch library implementing the framework from **"Neural Networks as Local-to-Global Computations"** (Bosca & Ghrist, 2026).

The core idea: a feedforward ReLU network can be encoded as a **cellular sheaf**. The network's forward pass is the *harmonic extension* of boundary data on this sheaf — and the sheaf heat equation recovers it from any initialization. Training becomes simultaneous diffusion on two coupled sheaves: one governing activations, one governing weights.

## Installation

```bash
git clone https://github.com/vicenbosca/neural-sheaf.git
cd neural-sheaf
pip install -e .
```

For interactive plotly visualizations and notebook support:

```bash
pip install -e ".[notebooks]"
```

## Quickstart

```python
import torch
from neural_sheaf import NeuralSheaf, SheafDynamics

torch.set_default_dtype(torch.float64)

# Build a small network: 2 inputs → 4 hidden → 1 output
net = NeuralSheaf([2, 4, 1], seed=0)

# Standard forward pass
x = torch.randn(2, 1)
y_fwd = net.forward(x)

# Sheaf diffusion from random initialization → same result
dyn = SheafDynamics(net)
state = net.init_stalks(x, method='random', seed=99)
result = dyn.run(state, x, n_steps=300, dt=0.05)

print(f"Forward pass:    {y_fwd.item():.6f}")
print(f"Sheaf diffusion: {result['z'][-1].item():.6f}")
```

See `quickstart.py` for a complete training example.

## What's in the library

The `neural_sheaf` package provides three layers of functionality:

**Core** — `sheaf.py`, `dynamics.py`, `trainer.py`, `baseline.py`
Sheaf construction, heat equation inference, joint stalk/weight training dynamics, and an SGD baseline for comparison.

**Support** — `activations.py`, `losses.py`, `datasets.py`, `pinning.py`, `tasks.py`
ReLU/sigmoid/softmax activations, loss functions, benchmark data generators, hard/soft pinning mechanisms, and reusable training infrastructure.

**Analysis** — `spectral.py`, `discord.py`, `visualization.py`
Eigenvalue spectra of the sheaf Laplacian, per-edge discord (discrepancy) analysis, and a comprehensive matplotlib plotting suite.

## Notebooks

The `notebooks/` directory contains four pedagogical walkthroughs:

| Notebook | Topic |
|---|---|
| `01_inference.ipynb` | Sheaf diffusion recovers the forward pass |
| `02_training.ipynb` | Sheaf training vs. SGD on regression & classification |
| `03_spectral.ipynb` | Eigenvalue spectra and spectral gap evolution |
| `04_discord.ipynb` | Per-edge discord analysis for sheaf and SGD models |

## Repository layout

```
neural-sheaf/
├── pyproject.toml
├── README.md
├── quickstart.py
├── notebooks/
│   ├── 01_inference.ipynb
│   ├── 02_training.ipynb
│   ├── 03_spectral.ipynb
│   └── 04_discord.ipynb
└─── neural_sheaf/
    ├── __init__.py
    ├── sheaf.py
    ├── dynamics.py
    ├── trainer.py
    ├── baseline.py
    ├── activations.py
    ├── losses.py
    ├── datasets.py
    ├── pinning.py
    ├── tasks.py
    ├── spectral.py
    ├── discord.py
    └── visualization.py
```

## Conventions

All tensors use `torch.float64` with shape `(dim, batch_size)`. Weight matrices are `(n_out, n_in)`, matching PyTorch convention. The library is device-agnostic — pass `device='cuda'` to run on GPU.

## Citation

```bibtex
@article{bosca2026neural,
  title   = {Neural Networks as Local-to-Global Computations},
  author  = {Bosca, Vicente and Ghrist, Robert},
  year    = {2026}
}
```

## License

MIT
