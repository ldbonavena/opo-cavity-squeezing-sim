"""Material and thermo-optic helpers for crystal simulations.

The default refractive-index baseline is ``Kato2002``. Alternative literature
models are exposed for comparison, while one selected model is applied
consistently across the crystal workflow through ``build_refractive_index_model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


DEFAULT_CRYSTAL_MODEL = "Kato2002"


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


@dataclass(frozen=True)
class ThermoOpticCoefficients:
    """Wavelength-dependent polynomial coefficients for quadratic thermo-optic terms."""

    a0: float
    a1: float
    a2: float
    a3: float


@dataclass(frozen=True)
class AxisModel:
    """One axis refractive-index model built from a base Sellmeier and thermo-optic terms."""

    base_index_um: Callable[[np.ndarray | float], np.ndarray | float]
    reference_temperature_C: float
    thermo_linear: ThermoOpticCoefficients | None = None
    thermo_quadratic: ThermoOpticCoefficients | None = None


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


def _to_lambda_um(wavelength_m: np.ndarray | float) -> np.ndarray | float:
    lam_um = np.asarray(wavelength_m, dtype=float) * 1e6
    return lam_um if isinstance(wavelength_m, np.ndarray) else float(lam_um)


def _sellmeier_reference_um(lambda_um: np.ndarray | float, coeffs: SellmeierCoefficients) -> np.ndarray | float:
    """Evaluate the five-coefficient Sellmeier form used for Kato2002."""
    return n_sellmeier_um(lambda_um, coeffs)


def _sellmeier_fan_konig_um(
    lambda_um: np.ndarray | float,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
) -> np.ndarray | float:
    """Evaluate the reduced Sellmeier form used in Fan1987 and Konig2004."""
    lam = np.asarray(lambda_um, dtype=float)
    n = np.sqrt(b0 + b1 / (1.0 - b2 / lam**2) - b3 * lam**2)
    return n if isinstance(lambda_um, np.ndarray) else float(n)


def _sellmeier_fradkin_um(
    lambda_um: np.ndarray | float,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: float,
) -> np.ndarray | float:
    """Evaluate the six-coefficient Sellmeier form used by Fradkin1999."""
    lam = np.asarray(lambda_um, dtype=float)
    n = np.sqrt(b0 + b1 / (1.0 - b2 / lam**2) + b3 / (1.0 - b4 / lam**2) - b5 * lam**2)
    return n if isinstance(lambda_um, np.ndarray) else float(n)


def _deltan_coeff(lambda_um: np.ndarray | float, coeffs: ThermoOpticCoefficients) -> np.ndarray | float:
    """Evaluate the wavelength-dependent thermo-optic polynomial from the reference model."""
    lam = np.asarray(lambda_um, dtype=float)
    dn = coeffs.a0 + coeffs.a1 / lam + coeffs.a2 / lam**2 + coeffs.a3 / lam**3
    return dn if isinstance(lambda_um, np.ndarray) else float(dn)


def _evaluate_axis_model(
    wavelength_m: np.ndarray | float,
    T_K: float,
    axis_model: AxisModel,
) -> np.ndarray | float:
    """Evaluate one axis model at wavelength ``wavelength_m`` and temperature ``T_K``."""
    lambda_um = _to_lambda_um(wavelength_m)
    base_n = axis_model.base_index_um(lambda_um)

    if axis_model.thermo_linear is None and axis_model.thermo_quadratic is None:
        return base_n

    delta_T_C = float(T_K) - 273.15 - axis_model.reference_temperature_C
    n = np.asarray(base_n, dtype=float)

    if axis_model.thermo_linear is not None:
        n = n + np.asarray(_deltan_coeff(lambda_um, axis_model.thermo_linear), dtype=float) * delta_T_C
    if axis_model.thermo_quadratic is not None:
        n = n + np.asarray(_deltan_coeff(lambda_um, axis_model.thermo_quadratic), dtype=float) * delta_T_C**2

    return n if isinstance(wavelength_m, np.ndarray) else float(n)


_COMMON_NX_MODEL = AxisModel(
    base_index_um=lambda lam_um: _sellmeier_reference_um(
        lam_um,
        SellmeierCoefficients(3.29100, 0.04140, 0.03978, 9.35522, 31.45571),
    ),
    reference_temperature_C=25.0,
)

_THERMO_NY_LINEAR = ThermoOpticCoefficients(6.2897e-6, 6.3061e-6, -6.0629e-6, 2.6486e-6)
_THERMO_NY_QUADRATIC = ThermoOpticCoefficients(-0.14445e-8, 2.2244e-8, -3.5770e-8, 1.3470e-8)
_THERMO_NZ_LINEAR = ThermoOpticCoefficients(9.9587e-6, 9.9228e-6, -8.9603e-6, 4.1010e-6)
_THERMO_NZ_QUADRATIC = ThermoOpticCoefficients(-1.1882e-8, 10.459e-8, -9.8136e-8, 3.1481e-8)

_MODEL_SPECS: dict[str, dict[str, AxisModel]] = {
    "Kato2002": {
        "n_x_of_T": _COMMON_NX_MODEL,
        "n_y_of_T": AxisModel(
            base_index_um=lambda lam_um: _sellmeier_reference_um(
                lam_um,
                SellmeierCoefficients(3.45018, 0.04341, 0.04597, 16.98825, 39.43799),
            ),
            reference_temperature_C=20.0,
            thermo_linear=_THERMO_NY_LINEAR,
            thermo_quadratic=_THERMO_NY_QUADRATIC,
        ),
        "n_z_of_T": AxisModel(
            base_index_um=lambda lam_um: _sellmeier_reference_um(
                lam_um,
                SellmeierCoefficients(4.59423, 0.06206, 0.04763, 110.80672, 86.12171),
            ),
            reference_temperature_C=20.0,
            thermo_linear=_THERMO_NZ_LINEAR,
            thermo_quadratic=_THERMO_NZ_QUADRATIC,
        ),
    },
    "Fan1987": {
        "n_x_of_T": _COMMON_NX_MODEL,
        "n_y_of_T": AxisModel(
            base_index_um=lambda lam_um: _sellmeier_fan_konig_um(lam_um, 2.19229, 0.83547, 0.04970, 0.01621),
            reference_temperature_C=25.0,
            thermo_linear=_THERMO_NY_LINEAR,
            thermo_quadratic=_THERMO_NY_QUADRATIC,
        ),
        "n_z_of_T": AxisModel(
            base_index_um=lambda lam_um: _sellmeier_fan_konig_um(lam_um, 2.25411, 1.06543, 0.05486, 0.02140),
            reference_temperature_C=25.0,
            thermo_linear=_THERMO_NZ_LINEAR,
            thermo_quadratic=_THERMO_NZ_QUADRATIC,
        ),
    },
    "Konig2004": {
        "n_x_of_T": _COMMON_NX_MODEL,
        "n_y_of_T": AxisModel(
            base_index_um=lambda lam_um: _sellmeier_fan_konig_um(lam_um, 2.09930, 0.922683, 0.0467695, 0.0138408),
            reference_temperature_C=25.0,
            thermo_linear=_THERMO_NY_LINEAR,
            thermo_quadratic=_THERMO_NY_QUADRATIC,
        ),
        "n_z_of_T": AxisModel(
            base_index_um=lambda lam_um: _sellmeier_fradkin_um(
                lam_um,
                2.12725,
                1.18431,
                0.0514852,
                0.66030,
                100.00507,
                9.68956e-3,
            ),
            reference_temperature_C=25.0,
            thermo_linear=_THERMO_NZ_LINEAR,
            thermo_quadratic=_THERMO_NZ_QUADRATIC,
        ),
    },
}


def supported_crystal_models() -> tuple[str, ...]:
    """Return supported literature models, with ``Kato2002`` as the default baseline."""
    return tuple(_MODEL_SPECS)


def _validate_model_name(model_name: str) -> str:
    if model_name not in _MODEL_SPECS:
        supported = ", ".join(supported_crystal_models())
        raise ValueError(f"Unknown crystal model '{model_name}'. Supported models: {supported}")
    return model_name


def build_refractive_index_model(model_name: str = DEFAULT_CRYSTAL_MODEL) -> dict[str, Any]:
    """Build axis functions for one literature model.

    ``Kato2002`` is the default physically consistent baseline. ``Fan1987`` and
    ``Konig2004`` are available for comparison only.
    """

    selected_model = _validate_model_name(model_name)
    model_spec = _MODEL_SPECS[selected_model]

    def _axis_function(axis_key: str) -> Callable[[np.ndarray | float, float], np.ndarray | float]:
        axis_model = model_spec[axis_key]
        return lambda wavelength_m, T_K: _evaluate_axis_model(wavelength_m, T_K, axis_model)

    return {
        "model_name": selected_model,
        "n_x_of_T": _axis_function("n_x_of_T"),
        "n_y_of_T": _axis_function("n_y_of_T"),
        "n_z_of_T": _axis_function("n_z_of_T"),
    }


def nx(
    lambda_m: np.ndarray | float,
    T_K: float,
    model: str = DEFAULT_CRYSTAL_MODEL,
) -> np.ndarray | float:
    """Return ``n_x(λ, T)`` for the selected literature model."""
    return build_refractive_index_model(model)["n_x_of_T"](lambda_m, T_K)


def ny(
    lambda_m: np.ndarray | float,
    T_K: float,
    model: str = DEFAULT_CRYSTAL_MODEL,
) -> np.ndarray | float:
    """Return ``n_y(λ, T)`` for the selected literature model."""
    return build_refractive_index_model(model)["n_y_of_T"](lambda_m, T_K)


def nz(
    lambda_m: np.ndarray | float,
    T_K: float,
    model: str = DEFAULT_CRYSTAL_MODEL,
) -> np.ndarray | float:
    """Return ``n_z(λ, T)`` for the selected literature model."""
    return build_refractive_index_model(model)["n_z_of_T"](lambda_m, T_K)


__all__ = [
    "DEFAULT_CRYSTAL_MODEL",
    "SellmeierCoefficients",
    "ThermoOpticCoefficients",
    "AxisModel",
    "central_diff",
    "n_sellmeier_um",
    "n_from_model",
    "dn_dT_numeric",
    "supported_crystal_models",
    "build_refractive_index_model",
    "nx",
    "ny",
    "nz",
]
