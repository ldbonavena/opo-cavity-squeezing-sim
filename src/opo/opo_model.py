"""Minimal OPO model definitions and below-threshold operating-point helpers.

This module holds the structured parameter containers for the degenerate
below-threshold OPO layer. The current implementation keeps the physics light
and focuses on providing a stable data model that can later be extended with
more complete threshold and nonlinear coupling calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OPOParameters:
    """User-facing OPO parameters for one workflow run."""

    pump_power_W: float
    threshold_power_W: float
    signal_wavelength_m: float
    pump_wavelength_m: float
    analysis_sideband_Hz: float
    analysis_span_Hz: tuple[float, float]
    n_analysis_points: int
    detection_efficiency: float


@dataclass(frozen=True)
class OPOModelResult:
    """Compact collection of derived OPO quantities used downstream."""

    pump_parameter: float
    threshold_power_W: float
    pump_power_W: float
    below_threshold: bool
    escape_efficiency: float
    cavity_linewidth_Hz: float
    cavity_detuning_Hz: float
    signal_wavelength_m: float
    pump_wavelength_m: float
    notes: tuple[str, ...]


def build_opo_parameters(config: dict[str, Any]) -> OPOParameters:
    """Build a validated OPO parameter object from a plain configuration mapping."""
    return OPOParameters(
        pump_power_W=float(config["pump_power_W"]),
        threshold_power_W=float(config["threshold_power_W"]),
        signal_wavelength_m=float(config["signal_wavelength_m"]),
        pump_wavelength_m=float(config["pump_wavelength_m"]),
        analysis_sideband_Hz=float(config["analysis_sideband_Hz"]),
        analysis_span_Hz=tuple(float(v) for v in config["analysis_span_Hz"]),
        n_analysis_points=int(config["n_analysis_points"]),
        detection_efficiency=float(config["detection_efficiency"]),
    )


def derive_opo_quantities(
    parameters: OPOParameters,
    cavity_data: dict[str, Any],
    crystal_data: dict[str, Any],
) -> OPOModelResult:
    """Build a minimal below-threshold OPO operating point from cavity/crystal outputs."""
    cavity_results = cavity_data.get("results", {})
    crystal_results = crystal_data.get("results", {})
    phase_matching = crystal_results.get("phase_matching", {})

    if parameters.threshold_power_W <= 0.0:
        raise ValueError("threshold_power_W must be positive")

    pump_parameter = parameters.pump_power_W / parameters.threshold_power_W
    linewidth_hz = float(cavity_results.get("kappa_total_Hz", 0.0))
    detuning_hz = float(cavity_data.get("inputs", {}).get("detuning_Hz", 0.0))
    escape_efficiency = float(cavity_results.get("escape_efficiency", 0.0))

    notes = [
        "Initial placeholder OPO model.",
        "Threshold and nonlinear coupling are currently user-parameterized.",
    ]
    if "pm_power_best" in phase_matching:
        notes.append("Crystal phase-matching output is loaded and available for future coupling calibration.")

    return OPOModelResult(
        pump_parameter=float(pump_parameter),
        threshold_power_W=float(parameters.threshold_power_W),
        pump_power_W=float(parameters.pump_power_W),
        below_threshold=bool(pump_parameter < 1.0),
        escape_efficiency=escape_efficiency,
        cavity_linewidth_Hz=linewidth_hz,
        cavity_detuning_Hz=detuning_hz,
        signal_wavelength_m=float(parameters.signal_wavelength_m),
        pump_wavelength_m=float(parameters.pump_wavelength_m),
        notes=tuple(notes),
    )


__all__ = [
    "OPOParameters",
    "OPOModelResult",
    "build_opo_parameters",
    "derive_opo_quantities",
]
