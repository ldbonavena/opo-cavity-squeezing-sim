"""High-level workflow assembly for OPO simulations.

The OPO layer consumes the exported cavity and crystal results for a selected
geometry, builds a minimal below-threshold degenerate OPO operating point,
prepares a Langevin-model scaffold, constructs placeholder squeezing spectra,
and saves the resulting outputs in ``results/<geometry>/opo/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

try:
    from common.results_paths import (
        ensure_geometry_results_subdirs,
        get_cavity_results_dir,
        get_crystal_results_dir,
        get_opo_results_dir,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.results_paths import (
        ensure_geometry_results_subdirs,
        get_cavity_results_dir,
        get_crystal_results_dir,
        get_opo_results_dir,
    )

from .opo_langevin import OPOLangevinModel, build_langevin_model
from .opo_model import OPOModelResult, OPOParameters, build_opo_parameters, derive_opo_quantities
from .opo_squeezing import OPOSqueezingSpectrum, compute_squeezing_spectra, spectrum_to_dict


@dataclass(frozen=True)
class OPOContext:
    """Structured OPO input context loaded from cavity and crystal outputs."""

    geometry: str
    cavity_output_path: str
    crystal_output_path: str
    cavity_data: dict[str, Any]
    crystal_data: dict[str, Any]


@dataclass(frozen=True)
class OPOSimulationResult:
    """Combined OPO workflow output."""

    context: OPOContext
    parameters: OPOParameters
    model: OPOModelResult
    langevin: OPOLangevinModel
    spectrum: OPOSqueezingSpectrum


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Simulation output not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_opo_context(
    geometry: str,
    cavity_output_path: str | Path | None = None,
    crystal_output_path: str | Path | None = None,
) -> OPOContext:
    """Load cavity and crystal outputs needed by the OPO workflow."""
    if cavity_output_path is None:
        cavity_output_path = get_cavity_results_dir(geometry) / "cavity_simulation_output.json"
    if crystal_output_path is None:
        crystal_output_path = get_crystal_results_dir(geometry) / "crystal_simulation_output.json"

    cavity_data = _load_json(cavity_output_path)
    crystal_data = _load_json(crystal_output_path)

    return OPOContext(
        geometry=str(cavity_data.get("geometry", geometry)),
        cavity_output_path=str(Path(cavity_output_path)),
        crystal_output_path=str(Path(crystal_output_path)),
        cavity_data=cavity_data,
        crystal_data=crystal_data,
    )


def compute_opo_model(context: OPOContext, config: dict[str, Any]) -> tuple[OPOParameters, OPOModelResult]:
    """Build OPO parameters and the minimal operating-point model."""
    parameters = build_opo_parameters(config)
    model = derive_opo_quantities(parameters, context.cavity_data, context.crystal_data)
    return parameters, model


def compute_opo_langevin(model: OPOModelResult) -> OPOLangevinModel:
    """Build the placeholder Langevin scaffold for the current OPO model."""
    return build_langevin_model(model)


def compute_opo_squeezing(
    parameters: OPOParameters,
    model: OPOModelResult,
    langevin: OPOLangevinModel,
) -> OPOSqueezingSpectrum:
    """Build the placeholder squeezing-spectrum payload."""
    return compute_squeezing_spectra(parameters, model, langevin)


def build_opo_simulation_result(
    context: OPOContext,
    parameters: OPOParameters,
    model: OPOModelResult,
    langevin: OPOLangevinModel,
    spectrum: OPOSqueezingSpectrum,
) -> OPOSimulationResult:
    """Assemble the structured OPO workflow result."""
    return OPOSimulationResult(
        context=context,
        parameters=parameters,
        model=model,
        langevin=langevin,
        spectrum=spectrum,
    )


def print_opo_summary(result: OPOSimulationResult) -> None:
    """Print a concise OPO operating-point summary."""
    print("OPO simulation summary")
    print("----------------------")
    print(f"Geometry: {result.context.geometry}")
    print(f"Pump power: {result.parameters.pump_power_W:.6f} W")
    print(f"Threshold power: {result.parameters.threshold_power_W:.6f} W")
    print(f"Pump parameter sigma: {result.model.pump_parameter:.6f}")
    print(f"Below threshold: {result.model.below_threshold}")
    print(f"Escape efficiency: {result.model.escape_efficiency:.6f}")
    print(f"Detection efficiency: {result.parameters.detection_efficiency:.6f}")
    print(f"Analysis sideband: {result.parameters.analysis_sideband_Hz:.3f} Hz")


def _build_opo_inputs_payload(context: OPOContext, parameters: OPOParameters) -> dict[str, Any]:
    return {
        "cavity_output_path": context.cavity_output_path,
        "crystal_output_path": context.crystal_output_path,
        **asdict(parameters),
    }


def _build_opo_results_payload(result: OPOSimulationResult) -> dict[str, Any]:
    model_payload = asdict(result.model)
    model_payload["notes"] = list(result.model.notes)
    model_payload["detection_efficiency"] = float(result.parameters.detection_efficiency)

    langevin_payload = {
        "quadrature_labels": list(result.langevin.quadrature_labels),
        "drift_matrix": result.langevin.drift_matrix.tolist(),
        "input_matrix": result.langevin.input_matrix.tolist(),
        "noise_coupling_matrix": result.langevin.noise_coupling_matrix.tolist(),
        "notes": list(result.langevin.notes),
    }

    return {
        "model": model_payload,
        "langevin": langevin_payload,
        "spectrum": spectrum_to_dict(result.spectrum),
    }


def build_opo_simulation_output(result: OPOSimulationResult) -> dict[str, Any]:
    """Build JSON-serializable OPO simulation output."""
    return {
        "geometry": result.context.geometry,
        "inputs": _build_opo_inputs_payload(result.context, result.parameters),
        "results": _build_opo_results_payload(result),
    }


def save_opo_outputs(
    geometry: str,
    output: dict[str, Any],
    fig_spectrum,
    fig_summary,
    results_root: str | Path | None = None,
) -> dict[str, str]:
    """Save OPO JSON and plots under ``results/<geometry>/opo/``."""
    ensure_geometry_results_subdirs(geometry, results_root=results_root)
    result_dir = get_opo_results_dir(geometry, results_root=results_root)
    project_root = Path(__file__).resolve().parents[2]

    json_path = result_dir / "opo_simulation_output.json"
    spectrum_path = result_dir / "opo_squeezing_spectrum.png"
    summary_path = result_dir / "opo_operating_point_summary.png"

    def _repo_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path)

    outputs_info = {
        "result_dir": _repo_relative(result_dir),
        "opo_output_json": _repo_relative(json_path),
        "opo_squeezing_spectrum_png": _repo_relative(spectrum_path),
        "opo_operating_point_summary_png": _repo_relative(summary_path),
    }
    output["outputs"] = outputs_info

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if fig_spectrum is not None:
        fig_spectrum.savefig(spectrum_path, dpi=300, bbox_inches="tight")
    if fig_summary is not None:
        fig_summary.savefig(summary_path, dpi=300, bbox_inches="tight")

    return outputs_info


__all__ = [
    "OPOContext",
    "OPOSimulationResult",
    "load_opo_context",
    "compute_opo_model",
    "compute_opo_langevin",
    "compute_opo_squeezing",
    "build_opo_simulation_result",
    "build_opo_simulation_output",
    "print_opo_summary",
    "save_opo_outputs",
]
