"""Plotting helpers for OPO operating-point and squeezing-spectrum views."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_opo_spectrum_summary(spectrum: dict[str, list[float]]):
    """Plot squeezing, antisqueezing, and shot-noise reference spectra."""
    frequency_hz = np.asarray(spectrum["frequency_Hz"], dtype=float)
    squeezing = np.asarray(spectrum["squeezing_spectrum"], dtype=float)
    antisqueezing = np.asarray(spectrum["antisqueezing_spectrum"], dtype=float)
    shot_noise = np.asarray(spectrum["shot_noise_reference"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frequency_hz, squeezing, lw=2, label="Squeezing")
    ax.plot(frequency_hz, antisqueezing, lw=2, label="Antisqueezing")
    ax.plot(frequency_hz, shot_noise, "--", lw=1.5, label="Shot noise")
    ax.set_xlabel("Analysis frequency [Hz]")
    ax.set_ylabel("Normalized noise")
    ax.set_title("OPO squeezing spectrum")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_opo_operating_point_summary(model_result: dict[str, float | bool]):
    """Plot a compact summary of the current OPO operating point."""
    labels = [
        "pump / threshold",
        "escape eff.",
        "detection eff.",
    ]
    values = [
        float(model_result["pump_parameter"]),
        float(model_result["escape_efficiency"]),
        float(model_result["detection_efficiency"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.5)
    ax.set_ylim(0.0, max(1.05, max(values) * 1.2))
    ax.set_title("OPO operating-point summary")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_opo_spectrum_summary",
    "plot_opo_operating_point_summary",
]
