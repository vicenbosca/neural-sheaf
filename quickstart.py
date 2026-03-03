"""
quickstart.py — Minimal training demo for the Neural Sheaf library.

Place this file inside the neural_sheaf/ folder and run:

    python quickstart.py

By default it trains on the paraboloid regression task.
To switch to multiclass classification, comment/uncomment the blocks
marked ── REGRESSION and ── CLASSIFICATION below.

To use your own data, replace X_train, Y_train, X_test, Y_test with
your own tensors.  The expected layout is:

    X : shape (input_dim, n_samples), dtype float64
    Y : shape (output_dim, n_samples), dtype float64
        For multiclass: one-hot encoded, e.g. shape (n_classes, n_samples).
"""

import logging
import torch
import matplotlib.pyplot as plt

from neural_sheaf.sheaf    import NeuralSheaf
from neural_sheaf.trainer  import SheafTrainer
from neural_sheaf.datasets import (
    paraboloid, generate_regression_data,
    generate_blob_data,
)
from neural_sheaf.visualization import (
    plot_training_curves,
    plot_regression_surfaces,
    plot_multiclass_boundaries,
)

torch.set_default_dtype(torch.float64)


# ====================================================================
# Hyperparameters  (edit here)
# ====================================================================

SEED      = 42
N_TRAIN   = 300
N_TEST    = 100
VERBOSE   = True        # print epoch-by-epoch progress (set False to silence)

# ── REGRESSION (default) ─────────────────────────────────────────────
ARCH              = [2, 30, 1]
OUTPUT_ACTIVATION = "identity"      # linear output for regression
INIT_METHOD       = "random"        # stalks start random
ALPHA  = 1.0                        # stalk dynamics rate
BETA   = 1.0 / N_TRAIN              # weight dynamics rate  (β = 1/n)
DT     = 0.005                      # Euler step size
EPOCHS = 100                        # outer epochs
STEPS  = 1000                       # dynamics steps per epoch

# ── CLASSIFICATION (uncomment this block, comment the one above) ─────
# ARCH              = [2, 25, 4]     # 4-class → output dim 4
# OUTPUT_ACTIVATION = "softmax"
# INIT_METHOD       = "forward_pass" # start at current equilibrium
# ALPHA  = 1.0
# BETA   = 1.0 / N_TRAIN
# DT     = 0.005
# EPOCHS = 100
# STEPS  = 1000


# ====================================================================
# Data  (replace with your own tensors here)
# ====================================================================

torch.manual_seed(SEED)

# ── REGRESSION (default) ─────────────────────────────────────────────
X_train, Y_train, X_test, Y_test = generate_regression_data(
    paraboloid, n_train=N_TRAIN, n_test=N_TEST, seed=SEED,
)

# ── CLASSIFICATION (uncomment and comment the block above) ───────────
# X_train, Y_train, X_test, Y_test = generate_blob_data(
#     n_train=N_TRAIN, n_test=N_TEST, seed=SEED,
# )


# ====================================================================
# Train
# ====================================================================

# Enable epoch-by-epoch progress from the trainer
if VERBOSE:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    logging.getLogger("neural_sheaf.trainer").setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)

sheaf   = NeuralSheaf(ARCH, output_activation=OUTPUT_ACTIVATION, seed=SEED)
trainer = SheafTrainer(sheaf, alpha=ALPHA, beta=BETA, dt=DT)

print(f"\nArchitecture: {ARCH}   activation: {OUTPUT_ACTIVATION}")
print(f"α={ALPHA}  β={BETA:.4f}  dt={DT}  "
      f"epochs={EPOCHS}×{STEPS} steps")
print(f"Training...\n")

history = trainer.train(
    X_train, Y_train, X_test, Y_test,
    epochs=EPOCHS, steps_per_epoch=STEPS,
    warm_start=True, init_method=INIT_METHOD, seed=SEED,
)

print(f"\nDone! Final losses:")
print(f"  Train: {history['train_loss'][-1]:.6f}")
print(f"  Test:  {history['test_loss'][-1]:.6f}")


# ====================================================================
# Plots
# ====================================================================

print("\nPlotting training curves...")
plot_training_curves([history], ["Sheaf"], title="Training Curves")
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("  Saved training_curves.png")
plt.show()

# ── REGRESSION surface plot (comment out for classification) ─────────
print("Plotting learned surface...")
plot_regression_surfaces(
    [trainer], ["Sheaf"],
    target_func=paraboloid, input_range=(-2.0, 2.0),
    suptitle="Learned Surface",
)
plt.savefig("learned_surface.png", dpi=150, bbox_inches="tight")
print("  Saved learned_surface.png")
plt.show()

# ── CLASSIFICATION plots (uncomment for classification) ──────────────
# print("Plotting decision boundaries...")
# plot_multiclass_boundaries(
#     [trainer], ["Sheaf"],
#     X_train, Y_train, n_classes=4,
#     suptitle="Decision Boundaries",
# )
# plt.savefig("decision_boundaries.png", dpi=150, bbox_inches="tight")
# print("  Saved decision_boundaries.png")
# plt.show()
#
# # Per-class accuracy
# with torch.no_grad():
#     preds = trainer.predict_classes(X_test)
#     true  = Y_test.argmax(dim=0)
#     acc   = (preds == true).float().mean().item()
#     print(f"\nTest accuracy: {acc:.1%}")
#     for c in range(Y_test.shape[0]):
#         mask = true == c
#         if mask.sum() > 0:
#             print(f"  Class {c}: {(preds[mask] == c).float().mean().item():.1%}")
