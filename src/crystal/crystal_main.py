# %%
"""
Main entry point for crystal simulation.

This script orchestrates the crystal simulation workflow:
cavity context -> phase matching -> mode matching -> nonlinear efficiency -> export
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# -------------------------------------------------------------------
# Main simulation script
# -------------------------------------------------------------------
# This file acts as the entry point for the crystal simulation.
# The detailed physics is implemented in the corresponding workflow
# and module files. This script orchestrates the workflow and exports
# results for the selected geometry.
# -------------------------------------------------------------------

# Support both package execution and direct interactive execution.
try:
    from .crystal_materials import build_refractive_index_model
    from .crystal_mode_matching import build_mode_matching_context_from_cavity_output
    from .crystal_plotter import (
        plot_bk_master_map_sigma_xi,
        plot_boyd_kleinman_analysis,
        plot_qpm_length_poling_map,
    )
    from .crystal_workflow import (
        build_crystal_simulation_output,
        build_crystal_simulation_result,
        compute_boyd_kleinman_analysis,
        compute_design_poling_period,
        compute_crystal_mode_matching,
        compute_crystal_phase_matching,
        load_cavity_context_for_crystal,
        print_crystal_summary,
        save_crystal_outputs,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from crystal.crystal_materials import build_refractive_index_model
    from crystal.crystal_mode_matching import build_mode_matching_context_from_cavity_output
    from crystal.crystal_plotter import (
        plot_bk_master_map_sigma_xi,
        plot_boyd_kleinman_analysis,
        plot_qpm_length_poling_map,
    )
    from crystal.crystal_workflow import (
        build_crystal_simulation_output,
        build_crystal_simulation_result,
        compute_boyd_kleinman_analysis,
        compute_design_poling_period,
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


def _to_array(value):
    import numpy as np

    return np.asarray(value, dtype=float)


__all__ = [
    "load_cavity_simulation_output",
]


# %%
# Simulation configuration

GEOMETRY = "bowtie"  # Cavity geometry used to load upstream cavity results (e.g. waist, kappa, cavity output JSON)
# Multiple literature models can exist, but one global selection is enforced per run.
CRYSTAL_MODEL = "Kato2002"  # Refractive-index model used consistently for all crystal axes. Choices: "Kato2002", "Fan1987", "Konig2004"

WAVELENGTH_P_M = 775e-9   # Pump wavelength [m]
WAVELENGTH_S_M = 1550e-9  # Signal wavelength [m]
WAVELENGTH_I_M = 1550e-9  # Idler wavelength [m] (equal to signal in the degenerate case)

PHASE_MATCHING_MODE = "design"  # "design" derives the QPM period from wavelengths + temperature; "analysis" uses ANALYSIS_LAMBDA0_M directly
DESIGN_TEMPERATURE_K = 318.15   # Design temperature [K] used to derive the QPM period in design mode
ANALYSIS_LAMBDA0_M = 27.7e-6    # QPM poling period Λ [m] used only when PHASE_MATCHING_MODE = "analysis"

T_MIN_K = 280.0  # Minimum temperature of the phase-matching scan [K]
T_MAX_K = 340.0  # Maximum temperature of the phase-matching scan [K]
N_T = 201        # Number of temperature points in the scan

CAVITY_OUTPUT_PATH = None  # Optional manual override for cavity output path (None = use default results/<geometry>/cavity/)
T0_K = 293.15              # Reference temperature [K] for thermo-optic / thermal-expansion models
ALPHA_PER_K = 0.0          # Linear thermal expansion coefficient [1/K]
QPM_ORDER_M = 1            # Quasi-phase-matching order m (m = 1 is first-order QPM)
SAVE_OUTPUTS = True        # If True, save JSON results and plots to disk


def _build_temperature_index_function(axis_model, wavelength_m: float):
    """Bind one axis model to a fixed wavelength for a temperature scan."""

    def _index_of_T(T_K: float) -> float:
        return float(axis_model(wavelength_m, T_K))

    return _index_of_T


def _build_wavelength_temperature_index_function(axis_model):
    """Expose one axis model directly as ``n(lambda, T)``."""

    def _index_of_lambda_T(wavelength_m: float, T_K: float) -> float:
        return float(axis_model(wavelength_m, T_K))

    return _index_of_lambda_T


refractive_index_model = build_refractive_index_model(CRYSTAL_MODEL)
# Bind pump/signal/idler to one model so nx/ny/nz never come from mixed sources.
n_p_of_T = _build_temperature_index_function(refractive_index_model["n_z_of_T"], WAVELENGTH_P_M)
n_s_of_T = _build_temperature_index_function(refractive_index_model["n_y_of_T"], WAVELENGTH_S_M)
n_i_of_T = _build_temperature_index_function(refractive_index_model["n_y_of_T"], WAVELENGTH_I_M)
n_p_of_lambda_T = _build_wavelength_temperature_index_function(refractive_index_model["n_z_of_T"])
n_s_of_lambda_T = _build_wavelength_temperature_index_function(refractive_index_model["n_y_of_T"])
n_i_of_lambda_T = _build_wavelength_temperature_index_function(refractive_index_model["n_y_of_T"])


# %%
# Load cavity context

context = load_cavity_context_for_crystal(
    GEOMETRY,
    cavity_output_path=CAVITY_OUTPUT_PATH,
)

# %%
# Compute design poling period

if PHASE_MATCHING_MODE == "design":
    design_poling = compute_design_poling_period(
        wavelength_p_m=WAVELENGTH_P_M,
        wavelength_s_m=WAVELENGTH_S_M,
        wavelength_i_m=WAVELENGTH_I_M,
        temperature_K=DESIGN_TEMPERATURE_K,
        n_p_of_lambda_T=n_p_of_lambda_T,
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
        qpm_order_m=QPM_ORDER_M,
    )
    Lambda0_m = design_poling.Lambda0_design_m
elif PHASE_MATCHING_MODE == "analysis":
    design_poling = None
    Lambda0_m = ANALYSIS_LAMBDA0_M
else:
    raise ValueError(f"Unknown PHASE_MATCHING_MODE: {PHASE_MATCHING_MODE}")

# %%
# Compute phase matching

phase = compute_crystal_phase_matching(
    context,
    n_p_of_T=n_p_of_T,
    n_s_of_T=n_s_of_T,
    n_i_of_T=n_i_of_T,
    wavelength_p_m=WAVELENGTH_P_M,
    wavelength_s_m=WAVELENGTH_S_M,
    wavelength_i_m=WAVELENGTH_I_M,
    Lambda0_m=Lambda0_m,
    T_min_K=T_MIN_K,
    T_max_K=T_MAX_K,
    n_T=N_T,
    T0_K=T0_K,
    alpha_perK=ALPHA_PER_K,
    qpm_order_m=QPM_ORDER_M,
)

# %%
# Determine operating temperature and refractive index for mode matching

phase_temperature_for_mode_matching_K = float(phase["T_best_K"][0])
mode_matching_n_crystal = float(
    refractive_index_model["n_y_of_T"](
        WAVELENGTH_S_M,
        phase_temperature_for_mode_matching_K,
    )
)

# %%
# Compute mode matching

mode = compute_crystal_mode_matching(
    context,
    n_crystal=mode_matching_n_crystal,
)

# %%
# Compute BK analysis

bk_data = compute_boyd_kleinman_analysis(
    context=context,
    phase_matching=phase,
    mode_matching=mode,
    n_p_of_T=n_p_of_T,
    n_s_of_T=n_s_of_T,
    n_i_of_T=n_i_of_T,
    n_p_of_lambda_T=n_p_of_lambda_T,
    n_s_of_lambda_T=n_s_of_lambda_T,
    n_i_of_lambda_T=n_i_of_lambda_T,
    wavelength_p_m=WAVELENGTH_P_M,
    wavelength_s_m=WAVELENGTH_S_M,
    wavelength_i_m=WAVELENGTH_I_M,
    Lambda0_m=Lambda0_m,
    T_min_K=T_MIN_K,
    T_max_K=T_MAX_K,
    n_T=N_T,
    T0_K=T0_K,
    alpha_perK=ALPHA_PER_K,
    qpm_order_m=QPM_ORDER_M,
)

# %%
# Build simulation result

result = build_crystal_simulation_result(
    context=context,
    phase_matching=phase,
    mode_matching=mode,
    bk_analysis=bk_data,
)

# %%
# Print summary

print_crystal_summary(result)

# %%
# Build simulation output

output = build_crystal_simulation_output(result)
output["inputs"]["crystal_model"] = CRYSTAL_MODEL
output["inputs"]["n_crystal"] = mode_matching_n_crystal
output["inputs"]["phase_matching_mode"] = PHASE_MATCHING_MODE
output["inputs"]["design_temperature_K"] = DESIGN_TEMPERATURE_K if PHASE_MATCHING_MODE == "design" else None
output["inputs"]["Lambda0_m"] = Lambda0_m
if design_poling is not None:
    output["inputs"]["delta_k_bulk_design_rad_per_m"] = design_poling.delta_k_bulk_rad_per_m

# %%
# Generate plots

fig_bk_master = plot_bk_master_map_sigma_xi(bk_data)
fig_qpm = plot_qpm_length_poling_map(bk_data)
fig_bk = plot_boyd_kleinman_analysis(bk_data)

# %%
# Save outputs

outputs_info = None
if SAVE_OUTPUTS:
    outputs_info = save_crystal_outputs(
        GEOMETRY,
        output,
        fig_bk_master=fig_bk_master,
        fig_qpm=fig_qpm,
        fig_bk=fig_bk,
    )
    print(f"Saved crystal output to: {outputs_info['crystal_output_json']}")
    print(f"Saved BK master map to: {outputs_info['boyd_kleinman_master_map_png']}")
    print(f"Saved QPM / poling-length map to: {outputs_info['qpm_length_poling_map_png']}")
    print(f"Saved BK analysis plot to: {outputs_info['boyd_kleinman_analysis_png']}")

# %%
# Build mode-matching context

mode_matching_context = build_mode_matching_context_from_cavity_output(context.cavity_data)

# %%
