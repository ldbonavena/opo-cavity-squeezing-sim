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
    from .crystal_mode_matching import build_mode_matching_context_from_cavity_output
    from .crystal_plotter import plot_mode_matching_summary, plot_phase_matching_temperature_scan
    from .crystal_workflow import (
        build_crystal_simulation_output,
        build_crystal_simulation_result,
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
        build_crystal_simulation_output,
        build_crystal_simulation_result,
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
# Load cavity context

context = load_cavity_context_for_crystal(
    GEOMETRY,
    cavity_output_path=CAVITY_OUTPUT_PATH,
)

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
    Lambda0_m=LAMBDA0_M,
    T_min_K=T_MIN_K,
    T_max_K=T_MAX_K,
    n_T=N_T,
    T0_K=T0_K,
    alpha_perK=ALPHA_PER_K,
    qpm_order_m=QPM_ORDER_M,
)

# %%
# Compute mode matching

mode = compute_crystal_mode_matching(context)

# %%
# Build simulation result

result = build_crystal_simulation_result(
    context=context,
    phase_matching=phase,
    mode_matching=mode,
)

# %%
# Print summary

print_crystal_summary(result)

# %%
# Build simulation output

output = build_crystal_simulation_output(result)

# %%
# Generate plots

fig_phase = plot_phase_matching_temperature_scan(
    {
        k: _to_array(v)
        for k, v in phase.items()
        if k in {"T_K", "pm_power", "delta_k_rad_per_m", "delta_k_eff_rad_per_m"}
    }
)
fig_mode = plot_mode_matching_summary(output["results"]["mode_matching"])

# %%
# Save outputs

outputs_info = None
if SAVE_OUTPUTS:
    outputs_info = save_crystal_outputs(GEOMETRY, output, fig_phase, fig_mode)
    print(f"Saved crystal output to: {outputs_info['crystal_output_json']}")
    print(f"Saved phase-matching plot to: {outputs_info['phase_matching_scan_png']}")
    print(f"Saved mode-matching plot to: {outputs_info['mode_matching_summary_png']}")

# %%
# Build mode-matching context

mode_matching_context = build_mode_matching_context_from_cavity_output(context.cavity_data)

# %%
