"""Linearized Langevin scaffolding for future OPO noise calculations.

The present implementation only constructs the bookkeeping objects needed for
future quantum-noise calculations. It intentionally avoids a full physical
derivation while establishing the interfaces used by the workflow and plotting
layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .opo_model import OPOModelResult


@dataclass(frozen=True)
class OPOLangevinModel:
    """Container for a linearized Langevin state-space placeholder."""

    quadrature_labels: tuple[str, ...]
    drift_matrix: np.ndarray
    input_matrix: np.ndarray
    noise_coupling_matrix: np.ndarray
    notes: tuple[str, ...]


def build_langevin_model(model: OPOModelResult) -> OPOLangevinModel:
    """Construct a minimal state-space scaffold for a degenerate below-threshold OPO."""
    linewidth_hz = max(model.cavity_linewidth_Hz, 0.0)
    linewidth_rad_s = 2.0 * np.pi * linewidth_hz
    detuning_rad_s = 2.0 * np.pi * model.cavity_detuning_Hz
    sigma = model.pump_parameter

    drift_matrix = np.array(
        [
            [-(linewidth_rad_s) * (1.0 - sigma), detuning_rad_s],
            [-detuning_rad_s, -(linewidth_rad_s) * (1.0 + sigma)],
        ],
        dtype=float,
    )

    identity = np.eye(2, dtype=float)
    return OPOLangevinModel(
        quadrature_labels=("X1", "X2"),
        drift_matrix=drift_matrix,
        input_matrix=identity.copy(),
        noise_coupling_matrix=identity,
        notes=(
            "Placeholder linearized Langevin scaffold.",
            "Matrices are structured for future degenerate OPO quadrature calculations.",
        ),
    )


__all__ = [
    "OPOLangevinModel",
    "build_langevin_model",
]
