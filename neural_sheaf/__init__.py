"""
neural_sheaf — Neural networks as cellular sheaves.

Implements the framework from "Neural Networks as Local-to-Global Computations"
for encoding feedforward ReLU networks as cellular sheaves, with inference via
heat equation dynamics and training via joint stalk/weight dynamics.
"""

__version__ = "0.1.1"

# Public API — importable as `from neural_sheaf import NeuralSheaf` etc.
from .sheaf import NeuralSheaf
from .dynamics import SheafDynamics
from .trainer import SheafTrainer
from .baseline import TraditionalNN
from .pinning import HardPin, SoftPin

__all__ = [
    "NeuralSheaf",
    "SheafDynamics",
    "SheafTrainer",
    "TraditionalNN",
    "HardPin",
    "SoftPin",
]
