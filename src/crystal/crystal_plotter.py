"""Plotting utilities for crystal phase-matching and mode-matching outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .crystal_boyd_kleinman import boyd_kleinman_efficiency


def plot_phase_matching_temperature_scan(scan: dict[str, np.ndarray]):
    """Plot phase-matching power and mismatch versus temperature."""
    T_K = scan["T_K"]
    pm = scan["pm_power"]
    dk = scan["delta_k_rad_per_m"]
    dk_eff = scan["delta_k_eff_rad_per_m"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(T_K, pm, lw=2)
    axes[0].set_ylabel("PM power factor")
    axes[0].grid(alpha=0.3)

    axes[1].plot(T_K, dk, lw=1.8, label="Delta k")
    axes[1].plot(T_K, dk_eff, lw=1.8, label="Delta k eff")
    axes[1].set_xlabel("Temperature [K]")
    axes[1].set_ylabel("Mismatch [rad/m]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_mode_matching_summary(mode_result: dict[str, float]):
    """Plot a compact bar chart for main mode-matching quantities."""
    labels = [
        "waist [um]",
        "zR [mm]",
        "confocal [mm]",
        "xi [-]",
    ]
    values = [
        mode_result["waist_crystal_m"] * 1e6,
        mode_result["rayleigh_range_m"] * 1e3,
        mode_result["confocal_parameter_m"] * 1e3,
        mode_result["focusing_parameter"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title("Crystal mode-matching summary")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_boyd_kleinman_vs_focusing_parameter(
    xi_values: np.ndarray,
    crystal_length_m: float,
    delta_k_rad_per_m: float,
):
    """Plot BK efficiency factor versus focusing parameter."""
    xi_values = np.asarray(xi_values, dtype=float)
    eff = np.empty_like(xi_values)
    for i, xi in enumerate(xi_values):
        if xi <= 0:
            eff[i] = np.nan
            continue
        z_r = crystal_length_m / (2.0 * xi)
        eff[i] = boyd_kleinman_efficiency(
            waist_m=1.0,
            rayleigh_range_m=z_r,
            crystal_length_m=crystal_length_m,
            delta_k=delta_k_rad_per_m,
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xi_values, eff, lw=2)
    ax.set_xlabel("Focusing parameter xi")
    ax.set_ylabel("Boyd-Kleinman efficiency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_boyd_kleinman_vs_delta_k(
    delta_k_values: np.ndarray,
    waist_m: float,
    rayleigh_range_m: float,
    crystal_length_m: float,
):
    """Plot BK efficiency factor versus phase mismatch Δk."""
    delta_k_values = np.asarray(delta_k_values, dtype=float)
    eff = np.array(
        [
            boyd_kleinman_efficiency(
                waist_m=waist_m,
                rayleigh_range_m=rayleigh_range_m,
                crystal_length_m=crystal_length_m,
                delta_k=dk,
            )
            for dk in delta_k_values
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(delta_k_values, eff, lw=2)
    ax.set_xlabel("Delta k [rad/m]")
    ax.set_ylabel("Boyd-Kleinman efficiency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_phase_matching_temperature_scan",
    "plot_mode_matching_summary",
    "plot_boyd_kleinman_vs_focusing_parameter",
    "plot_boyd_kleinman_vs_delta_k",
]
