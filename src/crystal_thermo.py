

# %%
"""crystal_thermo.py

Utilities to model crystal thermo-optic and QPM/phase-matching quantities.

This module is meant to be *called* by the cavity + OPO model:
- your cavity code provides geometry/mode parameters (e.g. waist w0 in the crystal)
- this module provides temperature-dependent material/QPM factors

Conventions (keep consistent across the repo):
- SI units everywhere unless explicitly stated
- Wavelengths are in meters (m)
- Temperature is in Kelvin (K)
- Wave-vectors are in rad/m

The functions below are intentionally "base" building blocks.
You can later add crystal-specific coefficient sets (e.g. PPKTP) and polarization handling.
"""

from __future__ import annotations

# %%
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# %%
# -----------------------------
# Helpers / small utilities
# -----------------------------

def sinc(x: np.ndarray | float) -> np.ndarray | float:
    """Normalized sinc: sin(x)/x with safe handling of x=0.

    Note: NumPy's np.sinc is sin(pi x)/(pi x), i.e. a different convention.
    """
    x_arr = np.asarray(x)
    out = np.ones_like(x_arr, dtype=float)
    mask = np.abs(x_arr) > 0
    out[mask] = np.sin(x_arr[mask]) / x_arr[mask]
    return out if isinstance(x, np.ndarray) else float(out)


def central_diff(f: Callable[[float], float], x: float, dx: float) -> float:
    """Simple central finite difference derivative."""
    return (f(x + dx) - f(x - dx)) / (2.0 * dx)


# %%
# -----------------------------
# Sellmeier / refractive index
# -----------------------------

@dataclass(frozen=True)
class SellmeierCoefficients:
    """Generic Sellmeier-like model container.

    This is intentionally generic: different crystals/polarizations use different
    functional forms and coefficient sets.

    The default evaluator implemented below uses the common form:

        n^2(λ) = A + B/(λ^2 - C) + D/(λ^2 - E) + F*λ^2

    with λ in micrometers.

    If you want a different model, provide your own evaluator function.
    """

    A: float
    B: float
    C: float
    D: float = 0.0
    E: float = 0.0
    F: float = 0.0


def n_sellmeier_um(lambda_um: np.ndarray | float, coeffs: SellmeierCoefficients) -> np.ndarray | float:
    """Compute refractive index from a common Sellmeier form.

    Parameters
    ----------
    lambda_um:
        Wavelength in micrometers.
    coeffs:
        Sellmeier coefficients.

    Returns
    -------
    n:
        Refractive index (dimensionless).
    """
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
    """Generic thermo-optic wrapper: n(λ, T) = n(λ) + (dn/dT)*(T - T_ref).

    Many crystals use more elaborate temperature models; this is a useful baseline.

    Parameters
    ----------
    wavelength_m:
        Wavelength in meters.
    T_K:
        Temperature in Kelvin.
    n_lambda:
        Function that returns n(λ) at reference temperature.
    dn_dT_perK:
        Optional function returning dn/dT [1/K]. If None, assumes 0.
    T_ref_K:
        Reference temperature for n_lambda.

    Returns
    -------
    n:
        Refractive index at (λ, T).
    """
    n0 = n_lambda(wavelength_m)
    if dn_dT_perK is None:
        return n0
    return n0 + dn_dT_perK(float(wavelength_m if np.isscalar(wavelength_m) else np.asarray(wavelength_m).mean())) * (T_K - T_ref_K)


def dn_dT_numeric(
    n_of_T: Callable[[float], float],
    T_K: float,
    dT_K: float = 1e-2,
) -> float:
    """Numerical derivative dn/dT at a given temperature."""
    return central_diff(n_of_T, T_K, dT_K)


# %%
# -----------------------------
# Wave-vectors and phase-matching
# -----------------------------

def k_of_n(wavelength_m: np.ndarray | float, n: np.ndarray | float) -> np.ndarray | float:
    """Wave-vector magnitude k = 2π n / λ (rad/m)."""
    wl = np.asarray(wavelength_m, dtype=float)
    nn = np.asarray(n, dtype=float)
    k = 2.0 * np.pi * nn / wl
    return k if isinstance(wavelength_m, np.ndarray) or isinstance(n, np.ndarray) else float(k)


def delta_k_three_wave(
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_p: float,
    n_s: float,
    n_i: float,
) -> float:
    """Compute phase-mismatch Δk = k_p - k_s - k_i (rad/m)."""
    k_p = k_of_n(wavelength_p_m, n_p)
    k_s = k_of_n(wavelength_s_m, n_s)
    k_i = k_of_n(wavelength_i_m, n_i)
    return float(k_p - k_s - k_i)


def qpm_grating_k(Lambda_m: float) -> float:
    """QPM grating wave-vector: K_g = 2π/Λ (rad/m)."""
    return float(2.0 * np.pi / Lambda_m)


def delta_k_qpm(
    delta_k_rad_per_m: float,
    Lambda_m: float,
    m: int = 1,
) -> float:
    """QPM-adjusted mismatch: Δk_eff = Δk - m*(2π/Λ)."""
    return float(delta_k_rad_per_m - m * qpm_grating_k(Lambda_m))


def pm_amplitude_factor(delta_k_eff: float, L_crystal_m: float) -> float:
    """Amplitude phase-matching factor for a uniform crystal: sinc(Δk L/2)."""
    return float(sinc(0.5 * delta_k_eff * L_crystal_m))


def pm_power_factor(delta_k_eff: float, L_crystal_m: float) -> float:
    """Power/efficiency factor ~ |sinc(Δk L/2)|^2."""
    a = pm_amplitude_factor(delta_k_eff, L_crystal_m)
    return float(a * a)


# %%
# -----------------------------
# Thermal expansion (poling period, lengths)
# -----------------------------

def expand_linear(L0: float, T_K: float, T0_K: float, alpha_perK: float) -> float:
    """Linear expansion: L(T) = L0 * (1 + α (T - T0))."""
    return float(L0 * (1.0 + alpha_perK * (T_K - T0_K)))


def poling_period_T(
    Lambda0_m: float,
    T_K: float,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
) -> float:
    """Temperature-adjusted poling period Λ(T) via linear expansion.

    This is a baseline model. Some crystals use higher-order polynomials.
    """
    return expand_linear(Lambda0_m, T_K, T0_K, alpha_perK)


# %%
# -----------------------------
# Convenience: compute Δk_eff(T) from callbacks
# -----------------------------

def delta_k_eff_T(
    T_K: float,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_p_of_T: Callable[[float], float],
    n_s_of_T: Callable[[float], float],
    n_i_of_T: Callable[[float], float],
    Lambda0_m: float,
    L_crystal_m: float,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
) -> tuple[float, float, float]:
    """Compute (Δk, Δk_eff, pm_power) at temperature T.

    This is the typical bridge between "material" and "OPO" code.

    Returns
    -------
    delta_k:
        Bare mismatch k_p - k_s - k_i.
    delta_k_eff:
        QPM-adjusted mismatch delta_k - m*2π/Λ(T).
    pm_pow:
        |sinc(Δk_eff L/2)|^2.
    """
    n_p = float(n_p_of_T(T_K))
    n_s = float(n_s_of_T(T_K))
    n_i = float(n_i_of_T(T_K))

    dk = delta_k_three_wave(wavelength_p_m, wavelength_s_m, wavelength_i_m, n_p, n_s, n_i)
    Lambda_T = poling_period_T(Lambda0_m, T_K, T0_K=T0_K, alpha_perK=alpha_perK)
    dk_eff = delta_k_qpm(dk, Lambda_T, m=qpm_order_m)
    pm_pow = pm_power_factor(dk_eff, L_crystal_m)
    return float(dk), float(dk_eff), float(pm_pow)


# %%
if __name__ == "__main__":
    # Minimal smoke-test: check that helpers run.
    # Replace the below with real coefficient sets/callbacks in your main code.

    # Example: constant n(T) callbacks
    n_p_of_T = lambda T: 1.8
    n_s_of_T = lambda T: 1.75
    n_i_of_T = lambda T: 1.75

    T = 300.0
    wl_p = 775e-9
    wl_s = 1550e-9
    wl_i = 1550e-9

    dk, dk_eff, pm_pow = delta_k_eff_T(
        T,
        wl_p,
        wl_s,
        wl_i,
        n_p_of_T,
        n_s_of_T,
        n_i_of_T,
        Lambda0_m=9.0e-6,
        L_crystal_m=10e-3,
        alpha_perK=0.0,
    )

    print("Δk       =", dk, "rad/m")
    print("Δk_eff   =", dk_eff, "rad/m")
    print("PM power =", pm_pow)
# %%
