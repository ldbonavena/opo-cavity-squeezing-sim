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
from .crystal_boyd_kleinman import (
    BKAnalysisConfig,
    bk_analysis_result_to_dict,
    run_bk_analysis,
)
from .crystal_phase_matching import compute_design_poling_period as compute_design_poling_period_from_material_model
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
    bk_analysis: dict[str, Any] | None = None


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


def _to_json_compatible(value: Any) -> Any:
    """Recursively convert NumPy-heavy workflow payloads into JSON-safe types."""
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


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


def compute_design_poling_period(
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    temperature_K: float,
    n_p_of_lambda_T: Callable[[float, float], float],
    n_s_of_lambda_T: Callable[[float, float], float],
    n_i_of_lambda_T: Callable[[float, float], float],
    qpm_order_m: int = 1,
) -> Any:
    """Compute the design poling period from wavelengths and one design temperature."""
    return compute_design_poling_period_from_material_model(
        wavelength_p_m=wavelength_p_m,
        wavelength_s_m=wavelength_s_m,
        wavelength_i_m=wavelength_i_m,
        temperature_K=temperature_K,
        n_p_of_lambda_T=n_p_of_lambda_T,
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
        qpm_order_m=qpm_order_m,
    )


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


def compute_boyd_kleinman_analysis(
    context: CrystalContext,
    mode_matching: ModeMatchingResult,
    n_p_of_T: Callable[[float], float],
    n_s_of_T: Callable[[float], float],
    n_i_of_T: Callable[[float], float],
    n_p_of_lambda_T: Callable[[float, float], float],
    n_s_of_lambda_T: Callable[[float, float], float],
    n_i_of_lambda_T: Callable[[float, float], float],
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
    phase_matching: dict[str, Any] | None = None,
    bk_config: BKAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Orchestrate BK analysis and expose the plotting-compatible dictionary payload.

    The workflow layer delegates all BK-specific reference construction,
    sweeps, normalization, defaults, and metadata assembly to
    ``crystal_boyd_kleinman``.
    """
    del T_min_K, T_max_K
    bk_result = run_bk_analysis(
        context=context,
        mode_matching=mode_matching,
        n_p_of_T=n_p_of_T,
        n_s_of_T=n_s_of_T,
        n_i_of_T=n_i_of_T,
        n_p_of_lambda_T=n_p_of_lambda_T,
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
        wavelength_p_m=wavelength_p_m,
        wavelength_s_m=wavelength_s_m,
        wavelength_i_m=wavelength_i_m,
        Lambda0_m=Lambda0_m,
        T0_K=T0_K,
        alpha_perK=alpha_perK,
        qpm_order_m=qpm_order_m,
        phase_matching=phase_matching,
        n_temperature=n_T,
        bk_config=bk_config,
    )
    return bk_analysis_result_to_dict(bk_result)


def build_crystal_simulation_result(
    context: CrystalContext,
    phase_matching: dict[str, Any],
    mode_matching: ModeMatchingResult,
    bk_analysis: dict[str, Any] | None = None,
) -> CrystalSimulationResult:
    """Build the structured crystal workflow result."""
    return CrystalSimulationResult(
        context=context,
        phase_matching=phase_matching,
        mode_matching=mode_matching,
        bk_analysis=bk_analysis,
    )


def print_crystal_summary(result: CrystalSimulationResult) -> None:
    """Print concise phase-matching and mode-matching summary."""
    phase = result.phase_matching
    mode = result.mode_matching
    bk_analysis = result.bk_analysis
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

    if bk_analysis is not None:
        reference = bk_analysis.get("reference", {})
        sigma_reference = reference.get("sigma_reference")
        xi_reference = reference.get("xi_reference")
        bk_reference_factor = reference.get("bk_reference_factor", mode.boyd_kleinman_factor)
        bk_master_sigma_opt = bk_analysis.get("bk_master_sigma_opt")
        bk_master_xi_opt = bk_analysis.get("bk_master_xi_opt")
        bk_master_h_opt = bk_analysis.get("bk_master_h_opt")

        if sigma_reference is not None and xi_reference is not None:
            print(f"BK reference point (sigma, xi): ({float(sigma_reference):.6f}, {float(xi_reference):.6f})")
        if bk_reference_factor is not None:
            print(f"BK reference factor: {float(bk_reference_factor):.6f}")
        if bk_master_sigma_opt is not None and bk_master_xi_opt is not None:
            print(
                "BK master-map optimum (sigma, xi): "
                f"({float(bk_master_sigma_opt):.6f}, {float(bk_master_xi_opt):.6f})"
            )
        if bk_master_h_opt is not None:
            print(f"BK master-map optimum factor: {float(bk_master_h_opt):.6f}")


def _build_crystal_inputs_payload(context: CrystalContext) -> dict[str, Any]:
    return {
        "cavity_output_path": context.cavity_output_path,
        "crystal_length_m": context.crystal_length_m,
        "wavelength_m": context.wavelength_m,
        "n_crystal": context.n_crystal,
        "beam_waist_crystal_m": context.beam_waist_crystal_m,
    }


def _build_crystal_results_payload(result: CrystalSimulationResult) -> dict[str, Any]:
    payload = {
        "phase_matching": result.phase_matching,
        "mode_matching": asdict(result.mode_matching),
    }
    if result.bk_analysis is not None:
        payload["boyd_kleinman_analysis"] = result.bk_analysis
    return payload


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
    fig_bk_master=None,
    fig_qpm=None,
    fig_bk=None,
    legacy_fig_unused=None,
    results_root: str | Path | None = None,
) -> dict[str, str]:
    """Save crystal JSON and plots under ``results/<geometry>/crystal/``."""
    # Backward compatibility:
    # `save_crystal_outputs(geometry, output, fig_bk)`
    # `save_crystal_outputs(geometry, output, fig_phase, fig_mode)`
    # `save_crystal_outputs(geometry, output, fig_bk_master, fig_qpm, fig_bk)`
    if fig_bk is None and fig_qpm is None and fig_bk_master is not None and hasattr(fig_bk_master, "savefig"):
        fig_bk = fig_bk_master
        fig_bk_master = None

    if results_root is None and isinstance(legacy_fig_unused, (str, Path)):
        results_root = legacy_fig_unused
        legacy_fig_unused = None

    if fig_bk is None and fig_qpm is not None and hasattr(fig_qpm, "savefig"):
        fig_bk = fig_qpm
        fig_qpm = None

    if (
        fig_bk is not None
        and fig_qpm is not None
        and hasattr(fig_bk_master, "savefig")
        and hasattr(fig_qpm, "savefig")
        and not hasattr(fig_bk, "savefig")
    ):
        fig_bk = fig_qpm
        fig_qpm = None

    if results_root is None and legacy_fig_unused is not None and not isinstance(legacy_fig_unused, (str, Path)):
        if fig_bk is None and hasattr(legacy_fig_unused, "savefig"):
            fig_bk = legacy_fig_unused
        legacy_fig_unused = None

    ensure_geometry_results_subdirs(geometry, results_root=results_root)
    result_dir = get_crystal_results_dir(geometry, results_root=results_root)
    project_root = Path(__file__).resolve().parents[2]

    json_path = result_dir / "crystal_simulation_output.json"
    bk_master_path = result_dir / "boyd_kleinman_master_map.png"
    qpm_path = result_dir / "qpm_length_poling_map.png"
    bk_path = result_dir / "boyd_kleinman_analysis.png"
    old_phase_path = result_dir / "phase_matching_scan.png"
    old_mode_path = result_dir / "mode_matching_summary.png"

    def _repo_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_compatible(output), f, indent=2)

    # Remove legacy plot files so the crystal results directory reflects the
    # current single-figure BK workflow after a fresh save.
    for legacy_path in (old_phase_path, old_mode_path):
        if legacy_path.exists():
            legacy_path.unlink()

    if fig_bk_master is not None:
        fig_bk_master.savefig(bk_master_path, dpi=300, bbox_inches="tight")
    if fig_qpm is not None:
        fig_qpm.savefig(qpm_path, dpi=300, bbox_inches="tight")
    if fig_bk is not None:
        fig_bk.savefig(bk_path, dpi=300, bbox_inches="tight")

    outputs = {
        "result_dir": _repo_relative(result_dir),
        "crystal_output_json": _repo_relative(json_path),
    }
    if fig_bk_master is not None:
        outputs["boyd_kleinman_master_map_png"] = _repo_relative(bk_master_path)
    if fig_qpm is not None:
        outputs["qpm_length_poling_map_png"] = _repo_relative(qpm_path)
    if fig_bk is not None:
        bk_relpath = _repo_relative(bk_path)
        outputs["boyd_kleinman_analysis_png"] = bk_relpath
        outputs["phase_matching_scan_png"] = bk_relpath
        outputs["mode_matching_summary_png"] = bk_relpath
    return outputs


__all__ = [
    "CrystalContext",
    "CrystalSimulationResult",
    "load_cavity_context_for_crystal",
    "compute_crystal_phase_matching",
    "compute_design_poling_period",
    "compute_crystal_mode_matching",
    "compute_boyd_kleinman_analysis",
    "build_crystal_simulation_result",
    "build_crystal_simulation_output",
    "print_crystal_summary",
    "save_crystal_outputs",
]
