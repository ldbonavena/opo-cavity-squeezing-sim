"""Material and thermo-optic helpers for crystal simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class SellmeierCoefficients:
    """Generic Sellmeier coefficient container.

    Uses the common form:
    n^2(λ) = A + B/(λ^2 - C) + D/(λ^2 - E) + F*λ^2, with λ in micrometers.
    """

    A: float
    B: float
    C: float
    D: float = 0.0
    E: float = 0.0
    F: float = 0.0


def central_diff(f: Callable[[float], float], x: float, dx: float) -> float:
    """Return the central finite-difference derivative of ``f`` at ``x``."""
    return (f(x + dx) - f(x - dx)) / (2.0 * dx)


def n_sellmeier_um(lambda_um: np.ndarray | float, coeffs: SellmeierCoefficients) -> np.ndarray | float:
    """Evaluate refractive index from a standard Sellmeier model."""
    lam2 = np.asarray(lambda_um, dtype=float) ** 2
    n2 = (
        coeffs.A
        + coeffs.B / (lam2 - coeffs.C)
        + (coeffs.D / (lam2 - coeffs.E) if coeffs.D != 0.0 else 0.0)
        + coeffs.F * lam2
    )
    n = np.sqrt(n2)
    return n if isinstance(lambda_um, np.ndarray) else float(n)


def n_from_model(
    wavelength_m: np.ndarray | float,
    T_K: float,
    n_lambda: Callable[[np.ndarray | float], np.ndarray | float],
    dn_dT_perK: Optional[Callable[[float], float]] = None,
    T_ref_K: float = 293.15,
) -> np.ndarray | float:
    """Evaluate ``n(λ, T)`` from base ``n(λ)`` and an optional linear ``dn/dT`` term."""
    n0 = n_lambda(wavelength_m)
    if dn_dT_perK is None:
        return n0

    wl = np.asarray(wavelength_m, dtype=float)
    if wl.ndim == 0:
        dn = dn_dT_perK(float(wl))
        return n0 + dn * (T_K - T_ref_K)

    dn_vec = np.vectorize(lambda w: dn_dT_perK(float(w)), otypes=[float])(wl)
    return np.asarray(n0, dtype=float) + dn_vec * (T_K - T_ref_K)


def dn_dT_numeric(
    n_of_T: Callable[[float], float],
    T_K: float,
    dT_K: float = 1e-2,
) -> float:
    """Estimate ``dn/dT`` at temperature ``T_K`` via central finite difference."""
    return central_diff(n_of_T, T_K, dT_K)


__all__ = [
    "SellmeierCoefficients",
    "central_diff",
    "n_sellmeier_um",
    "n_from_model",
    "dn_dT_numeric",
]
