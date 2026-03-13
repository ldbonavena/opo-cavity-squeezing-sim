"""Focused-beam phase-matching helpers based on Boyd-Kleinman formalism."""

from __future__ import annotations

import numpy as np


def compute_focusing_parameter(crystal_length_m: float, rayleigh_range_m: float) -> float:
    """Return Boyd-Kleinman focusing parameter ``xi = L/(2 zR)``."""
    if rayleigh_range_m <= 0:
        return np.nan
    return float(crystal_length_m / (2.0 * rayleigh_range_m))


def boyd_kleinman_integral(
    xi: float,
    delta_k: float,
    crystal_length_m: float,
    n_points: int = 4001,
) -> complex:
    """Return the normalized focused-beam overlap integral.

    The integrand follows a standard single-pass Gaussian-beam model:
    ``exp(i*Delta_k*z) / (1 + i*z/zR)``, integrated over the crystal length.
    """
    if not np.isfinite(xi) or xi <= 0 or crystal_length_m <= 0:
        return 0.0 + 0.0j

    z_r = crystal_length_m / (2.0 * xi)
    z = np.linspace(-0.5 * crystal_length_m, 0.5 * crystal_length_m, int(n_points))
    integrand = np.exp(1j * delta_k * z) / (1.0 + 1j * z / z_r)
    return complex(np.trapz(integrand, z) / crystal_length_m)


def boyd_kleinman_efficiency(
    waist_m: float,
    rayleigh_range_m: float,
    crystal_length_m: float,
    delta_k: float,
) -> float:
    """Return focused-beam BK efficiency correction factor (dimensionless)."""
    _ = waist_m
    xi = compute_focusing_parameter(crystal_length_m, rayleigh_range_m)
    integral = boyd_kleinman_integral(xi, delta_k, crystal_length_m)
    return float(np.abs(integral) ** 2)


__all__ = [
    "compute_focusing_parameter",
    "boyd_kleinman_integral",
    "boyd_kleinman_efficiency",
]
