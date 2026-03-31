# %%
"""
Main entry point for OPO simulation.

This script orchestrates the initial OPO workflow:
cavity/crystal outputs -> OPO model -> Langevin scaffold -> squeezing placeholders -> export
"""

from __future__ import annotations

from pathlib import Path

# -------------------------------------------------------------------
# Main simulation script
# -------------------------------------------------------------------
# This file acts as the entry point for the OPO simulation.
# The detailed physics is implemented in the corresponding workflow
# and module files. This script is intentionally readable top-to-bottom
# and organized in notebook-style cells for interactive development.
# -------------------------------------------------------------------

# Support both package execution and direct interactive execution.
try:
    from .opo_plotter import plot_opo_operating_point_summary, plot_opo_spectrum_summary
    from .opo_workflow import (
        build_opo_simulation_output,
        build_opo_simulation_result,
        compute_opo_langevin,
        compute_opo_model,
        compute_opo_squeezing,
        load_opo_context,
        print_opo_summary,
        save_opo_outputs,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from opo.opo_plotter import plot_opo_operating_point_summary, plot_opo_spectrum_summary
    from opo.opo_workflow import (
        build_opo_simulation_output,
        build_opo_simulation_result,
        compute_opo_langevin,
        compute_opo_model,
        compute_opo_squeezing,
        load_opo_context,
        print_opo_summary,
        save_opo_outputs,
    )


# %%
# Simulation configuration

GEOMETRY = "bowtie"

CAVITY_OUTPUT_PATH = None
CRYSTAL_OUTPUT_PATH = None
SAVE_OUTPUTS = True

# Minimal below-threshold degenerate OPO configuration.
OPO_CONFIG = {
    "pump_power_W": 25e-3,
    "threshold_power_W": 100e-3,
    "signal_wavelength_m": 1550e-9,
    "pump_wavelength_m": 775e-9,
    "analysis_sideband_Hz": 5e6,
    "analysis_span_Hz": (1e5, 20e6),
    "n_analysis_points": 400,
    "detection_efficiency": 0.95,
}


# %%
# Load upstream simulation outputs

context = load_opo_context(
    GEOMETRY,
    cavity_output_path=CAVITY_OUTPUT_PATH,
    crystal_output_path=CRYSTAL_OUTPUT_PATH,
)


# %%
# Build OPO model

parameters, model = compute_opo_model(context, OPO_CONFIG)


# %%
# Build Langevin scaffold

langevin = compute_opo_langevin(model)


# %%
# Compute squeezing placeholders

spectrum = compute_opo_squeezing(parameters, model, langevin)


# %%
# Build structured result

result = build_opo_simulation_result(
    context=context,
    parameters=parameters,
    model=model,
    langevin=langevin,
    spectrum=spectrum,
)


# %%
# Print summary

print_opo_summary(result)


# %%
# Build export payload

output = build_opo_simulation_output(result)


# %%
# Generate plots

fig_spectrum = plot_opo_spectrum_summary(output["results"]["spectrum"])
fig_summary = plot_opo_operating_point_summary(output["results"]["model"])


# %%
# Save outputs

outputs_info = None
if SAVE_OUTPUTS:
    outputs_info = save_opo_outputs(GEOMETRY, output, fig_spectrum, fig_summary)
    print(f"Saved OPO output to: {outputs_info['opo_output_json']}")
    print(f"Saved OPO spectrum plot to: {outputs_info['opo_squeezing_spectrum_png']}")
    print(f"Saved OPO summary plot to: {outputs_info['opo_operating_point_summary_png']}")

# %%
