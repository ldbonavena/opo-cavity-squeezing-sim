# %%
"""Bridge layer connecting cavity outputs to crystal calculations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

# Support both package execution and direct interactive execution.
try:
    from .crystal_mode_matching import build_mode_matching_context_from_cavity_output
    from .crystal_plotter import plot_mode_matching_summary, plot_phase_matching_temperature_scan
    from .crystal_workflow import (
        CrystalSimulationResult,
        build_crystal_simulation_output,
        compute_crystal_mode_matching,
        compute_crystal_phase_matching,
        load_cavity_context_for_crystal,
        print_crystal_summary,
        save_crystal_outputs,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from crystal.crystal_mode_matching import build_mode_matching_context_from_cavity_output
    from crystal.crystal_plotter import plot_mode_matching_summary, plot_phase_matching_temperature_scan
    from crystal.crystal_workflow import (
        CrystalSimulationResult,
        build_crystal_simulation_output,
        compute_crystal_mode_matching,
        compute_crystal_phase_matching,
        load_cavity_context_for_crystal,
        print_crystal_summary,
        save_crystal_outputs,
    )


# %%
# Helper functions

def load_cavity_simulation_output(path: str | Path) -> dict[str, Any]:
    """Load cavity simulation JSON from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cavity simulation output not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_crystal_simulation_from_cavity(
    geometry: str,
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
    cavity_output_path: str | Path | None = None,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """High-level bridge: load cavity output, run crystal workflow, optionally save outputs."""
    context = load_cavity_context_for_crystal(geometry, cavity_output_path=cavity_output_path)

    phase = compute_crystal_phase_matching(
        context,
        n_p_of_T=n_p_of_T,
        n_s_of_T=n_s_of_T,
        n_i_of_T=n_i_of_T,
        wavelength_p_m=wavelength_p_m,
        wavelength_s_m=wavelength_s_m,
        wavelength_i_m=wavelength_i_m,
        Lambda0_m=Lambda0_m,
        T_min_K=T_min_K,
        T_max_K=T_max_K,
        n_T=n_T,
        T0_K=T0_K,
        alpha_perK=alpha_perK,
        qpm_order_m=qpm_order_m,
    )
    mode = compute_crystal_mode_matching(context)

    result = CrystalSimulationResult(context=context, phase_matching=phase, mode_matching=mode)
    print_crystal_summary(result)

    output = build_crystal_simulation_output(result)
    outputs_info = None

    if save_outputs:
        fig_phase = plot_phase_matching_temperature_scan(
            {
                k: _to_array(v)
                for k, v in phase.items()
                if k in {"T_K", "pm_power", "delta_k_rad_per_m", "delta_k_eff_rad_per_m"}
            }
        )
        fig_mode = plot_mode_matching_summary(output["results"]["mode_matching"])
        outputs_info = save_crystal_outputs(geometry, output, fig_phase, fig_mode)

    cavity_mm_context = build_mode_matching_context_from_cavity_output(context.cavity_data)

    return {
        "output": output,
        "saved_outputs": outputs_info,
        "mode_matching_context": cavity_mm_context,
    }


def _to_array(value):
    import numpy as np

    return np.asarray(value, dtype=float)


__all__ = [
    "load_cavity_simulation_output",
    "run_crystal_simulation_from_cavity",
]


# %%
# Simulation configuration

GEOMETRY = "bowtie"

WAVELENGTH_P_M = 775e-9
WAVELENGTH_S_M = 1550e-9
WAVELENGTH_I_M = 1550e-9

LAMBDA0_M = 9e-6

T_MIN_K = 280.0
T_MAX_K = 340.0
N_T = 201

CAVITY_OUTPUT_PATH = None
T0_K = 293.15
ALPHA_PER_K = 0.0
QPM_ORDER_M = 1
SAVE_OUTPUTS = True


def n_p_of_T(T_K: float) -> float:
    """Pump refractive index model versus temperature."""
    _ = T_K
    return 1.8


def n_s_of_T(T_K: float) -> float:
    """Signal refractive index model versus temperature."""
    _ = T_K
    return 1.75


def n_i_of_T(T_K: float) -> float:
    """Idler refractive index model versus temperature."""
    _ = T_K
    return 1.75


# %%
# Crystal simulation workflow

def run_current_simulation() -> dict[str, Any]:
    """Run the crystal simulation using the configuration defined above.

    This is intentionally wrapped in a function so that opening the file or
    executing earlier cells in the VS Code interactive window does not
    automatically launch the simulation.
    """
    simulation = run_crystal_simulation_from_cavity(
        geometry=GEOMETRY,
        n_p_of_T=n_p_of_T,
        n_s_of_T=n_s_of_T,
        n_i_of_T=n_i_of_T,
        wavelength_p_m=WAVELENGTH_P_M,
        wavelength_s_m=WAVELENGTH_S_M,
        wavelength_i_m=WAVELENGTH_I_M,
        Lambda0_m=LAMBDA0_M,
        T_min_K=T_MIN_K,
        T_max_K=T_MAX_K,
        n_T=N_T,
        cavity_output_path=CAVITY_OUTPUT_PATH,
        T0_K=T0_K,
        alpha_perK=ALPHA_PER_K,
        qpm_order_m=QPM_ORDER_M,
        save_outputs=SAVE_OUTPUTS,
    )

    output = simulation["output"]
    saved_outputs = simulation["saved_outputs"]
    mode_matching_context = simulation["mode_matching_context"]

    if saved_outputs is not None:
        print(f"Saved crystal output to: {saved_outputs['crystal_output_json']}")
        print(f"Saved phase-matching plot to: {saved_outputs['phase_matching_scan_png']}")
        print(f"Saved mode-matching plot to: {saved_outputs['mode_matching_summary_png']}")

    return {
        "simulation": simulation,
        "output": output,
        "saved_outputs": saved_outputs,
        "mode_matching_context": mode_matching_context,
    }


# To run the simulation manually in the interactive window, execute:
# crystal_run = run_current_simulation()

# %%
