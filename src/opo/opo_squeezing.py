"""Squeezing-spectrum placeholders built on the OPO Langevin scaffold.

This module defines the analysis-frequency grid and returns a minimal output
shape for squeezing and antisqueezing spectra. The current spectra are flat
shot-noise placeholders so the workflow can be developed before the full
Langevin solution is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .opo_langevin import OPOLangevinModel
from .opo_model import OPOModelResult, OPOParameters


@dataclass(frozen=True)
class OPOSqueezingSpectrum:
    """Frequency-domain OPO squeezing output."""

    frequency_Hz: np.ndarray
    squeezing_spectrum: np.ndarray
    antisqueezing_spectrum: np.ndarray
    shot_noise_reference: np.ndarray
    notes: tuple[str, ...]


def build_analysis_frequency_grid(parameters: OPOParameters) -> np.ndarray:
    """Build the analysis-frequency axis used for spectrum calculations."""
    f_min, f_max = parameters.analysis_span_Hz
    if parameters.n_analysis_points < 2:
        raise ValueError("n_analysis_points must be at least 2")
    if f_min < 0.0 or f_max <= f_min:
        raise ValueError("analysis_span_Hz must satisfy 0 <= f_min < f_max")
    return np.linspace(f_min, f_max, parameters.n_analysis_points, dtype=float)


def compute_squeezing_spectra(
    parameters: OPOParameters,
    model: OPOModelResult,
    langevin: OPOLangevinModel,
) -> OPOSqueezingSpectrum:
    """Return placeholder spectra for a future below-threshold degenerate OPO model."""
    del model, langevin  # Reserved for the future full implementation.

    frequency_hz = build_analysis_frequency_grid(parameters)
    shot_noise = np.ones_like(frequency_hz)

    return OPOSqueezingSpectrum(
        frequency_Hz=frequency_hz,
        squeezing_spectrum=shot_noise.copy(),
        antisqueezing_spectrum=shot_noise.copy(),
        shot_noise_reference=shot_noise,
        notes=(
            "Placeholder unity spectra.",
            "Future versions should solve the Langevin equations and include detection loss consistently.",
        ),
    )


def spectrum_to_dict(spectrum: OPOSqueezingSpectrum) -> dict[str, list[float] | list[str]]:
    """Convert the squeezing spectrum dataclass into a JSON-friendly mapping."""
    return {
        "frequency_Hz": spectrum.frequency_Hz.tolist(),
        "squeezing_spectrum": spectrum.squeezing_spectrum.tolist(),
        "antisqueezing_spectrum": spectrum.antisqueezing_spectrum.tolist(),
        "shot_noise_reference": spectrum.shot_noise_reference.tolist(),
        "notes": list(spectrum.notes),
    }


__all__ = [
    "OPOSqueezingSpectrum",
    "build_analysis_frequency_grid",
    "compute_squeezing_spectra",
    "spectrum_to_dict",
]
