"""High-level workflow assembly for crystal and mode-matching simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from common.results_paths import ensure_geometry_results_subdirs, get_cavity_results_dir, get_crystal_results_dir
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.results_paths import ensure_geometry_results_subdirs, get_cavity_results_dir, get_crystal_results_dir

from .crystal_mode_matching import (
    ModeMatchingResult,
    build_mode_matching_context_from_cavity_output,
    estimate_mode_matching_quantities,
)
from .crystal_phase_matching import scan_phase_matching_vs_temperature


@dataclass(frozen=True)
class CrystalContext:
    """Structured context loaded from cavity simulation outputs."""

    geometry: str | None
    cavity_output_path: str
    crystal_length_m: float
    wavelength_m: float
    n_crystal: float
    beam_waist_crystal_m: float
    cavity_data: dict[str, Any]


@dataclass(frozen=True)
class CrystalSimulationResult:
    """Combined crystal workflow output."""

    context: CrystalContext
    phase_matching: dict[str, Any]
    mode_matching: ModeMatchingResult


def _load_cavity_output_data(path: str | Path) -> dict[str, Any]:
    cavity_output_path = Path(path)
    if not cavity_output_path.exists():
        raise FileNotFoundError(f"Cavity simulation output not found: {cavity_output_path}")
    with cavity_output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cavity_context_for_crystal(
    geometry: str,
    cavity_output_path: str | Path | None = None,
) -> CrystalContext:
    """Load cavity JSON output and build the crystal workflow context."""
    if cavity_output_path is None:
        cavity_output_path = get_cavity_results_dir(geometry) / "cavity_simulation_output.json"
    cavity_data = _load_cavity_output_data(cavity_output_path)

    inputs = cavity_data.get("inputs", {})
    results = cavity_data.get("results", {})
    waist_um = results.get("beam_waist_crystal_um")

    if waist_um is None:
        raise ValueError("Cavity output missing results.beam_waist_crystal_um")

    return CrystalContext(
        geometry=cavity_data.get("geometry", geometry),
        cavity_output_path=str(Path(cavity_output_path)),
        crystal_length_m=float(inputs["crystal_length_m"]),
        wavelength_m=float(inputs["wavelength_m"]),
        n_crystal=float(inputs["n_crystal"]),
        beam_waist_crystal_m=float(waist_um) * 1e-6,
        cavity_data=cavity_data,
    )


def _phase_matching_scan_to_output(scan: dict[str, Any]) -> dict[str, Any]:
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in scan.items()}


def compute_crystal_phase_matching(
    context: CrystalContext,
    n_p_of_T: Callable[[float], float],
    n_s_of_T: Callable[[float], float],
    n_i_of_T: Callable[[float], float],
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    Lambda0_m: float,
    T_min_K: float,
    T_max_K: float,
    n_T: int,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
) -> dict[str, Any]:
    """Compute the phase-matching temperature scan from the cavity-derived context."""
    scan = scan_phase_matching_vs_temperature(
        T_min_K,
        T_max_K,
        n_T,
        wavelength_p_m,
        wavelength_s_m,
        wavelength_i_m,
        n_p_of_T,
        n_s_of_T,
        n_i_of_T,
        Lambda0_m,
        context.crystal_length_m,
        T0_K=T0_K,
        alpha_perK=alpha_perK,
        qpm_order_m=qpm_order_m,
    )
    return _phase_matching_scan_to_output(scan)


def compute_crystal_mode_matching(
    context: CrystalContext,
    n_crystal: float | None = None,
    delta_k_rad_per_m: float = 0.0,
) -> ModeMatchingResult:
    """Compute focusing and mode-matching quantities from the cavity-derived beam parameters."""
    cavity_mode_matching_context = build_mode_matching_context_from_cavity_output(context.cavity_data)
    waist_crystal_m = cavity_mode_matching_context.waist_crystal_m or context.beam_waist_crystal_m
    medium_index = context.n_crystal if n_crystal is None else float(n_crystal)
    return estimate_mode_matching_quantities(
        waist_crystal_m=float(waist_crystal_m),
        crystal_length_m=context.crystal_length_m,
        wavelength_m=context.wavelength_m,
        n_crystal=medium_index,
        delta_k_rad_per_m=delta_k_rad_per_m,
    )


def build_crystal_simulation_result(
    context: CrystalContext,
    phase_matching: dict[str, Any],
    mode_matching: ModeMatchingResult,
) -> CrystalSimulationResult:
    """Build the structured crystal workflow result."""
    return CrystalSimulationResult(
        context=context,
        phase_matching=phase_matching,
        mode_matching=mode_matching,
    )


def print_crystal_summary(result: CrystalSimulationResult) -> None:
    """Print concise phase-matching and mode-matching summary."""
    phase = result.phase_matching
    mode = result.mode_matching
    t_best = float(phase["T_best_K"][0])
    pm_best = float(phase["pm_power_best"][0])

    print("Crystal simulation summary")
    print("-------------------------")
    print(f"Geometry: {result.context.geometry}")
    print(f"Best phase-matching temperature: {t_best:.3f} K")
    print(f"Best phase-matching power factor: {pm_best:.6f}")
    print(f"Beam waist in crystal: {mode.waist_crystal_m*1e6:.3f} um")
    print(f"Rayleigh range in crystal: {mode.rayleigh_range_m*1e3:.3f} mm")
    print(f"Focusing parameter xi: {mode.focusing_parameter_xi:.6f}")
    print(f"Boyd-Kleinman factor: {mode.boyd_kleinman_factor:.6f}")
    print(f"Effective nonlinear overlap: {mode.effective_nonlinear_overlap:.6f}")


def _build_crystal_inputs_payload(context: CrystalContext) -> dict[str, Any]:
    return {
        "cavity_output_path": context.cavity_output_path,
        "crystal_length_m": context.crystal_length_m,
        "wavelength_m": context.wavelength_m,
        "n_crystal": context.n_crystal,
        "beam_waist_crystal_m": context.beam_waist_crystal_m,
    }


def _build_crystal_results_payload(result: CrystalSimulationResult) -> dict[str, Any]:
    return {
        "phase_matching": result.phase_matching,
        "mode_matching": asdict(result.mode_matching),
    }


def build_crystal_simulation_output(result: CrystalSimulationResult) -> dict[str, Any]:
    """Build JSON-serializable crystal simulation output."""
    return {
        "geometry": result.context.geometry,
        "inputs": _build_crystal_inputs_payload(result.context),
        "results": _build_crystal_results_payload(result),
    }


def save_crystal_outputs(
    geometry: str,
    output: dict[str, Any],
    fig_phase,
    fig_mode,
    results_root: str | Path | None = None,
) -> dict[str, str]:
    """Save crystal JSON and plots under ``results/<geometry>/crystal/``."""
    ensure_geometry_results_subdirs(geometry, results_root=results_root)
    result_dir = get_crystal_results_dir(geometry, results_root=results_root)
    project_root = Path(__file__).resolve().parents[2]

    json_path = result_dir / "crystal_simulation_output.json"
    phase_path = result_dir / "phase_matching_scan.png"
    mode_path = result_dir / "mode_matching_summary.png"

    def _repo_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if fig_phase is not None:
        fig_phase.savefig(phase_path, dpi=300, bbox_inches="tight")
    if fig_mode is not None:
        fig_mode.savefig(mode_path, dpi=300, bbox_inches="tight")

    return {
        "result_dir": _repo_relative(result_dir),
        "crystal_output_json": _repo_relative(json_path),
        "phase_matching_scan_png": _repo_relative(phase_path),
        "mode_matching_summary_png": _repo_relative(mode_path),
    }


__all__ = [
    "CrystalContext",
    "CrystalSimulationResult",
    "load_cavity_context_for_crystal",
    "compute_crystal_phase_matching",
    "compute_crystal_mode_matching",
    "build_crystal_simulation_result",
    "build_crystal_simulation_output",
    "print_crystal_summary",
    "save_crystal_outputs",
]
