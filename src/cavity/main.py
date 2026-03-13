# %%
"""
Main entry point for cavity simulation.

This script runs the full cavity geometry and optical mode simulation workflow:
- geometry selection
- stability analysis
- cavity eigenmode computation
- derived cavity quantities
- export of simulation results

The script is designed to be executed interactively in VS Code using `# %%` cells.
"""

# -------------------------------------------------------------------
# Main simulation script
# -------------------------------------------------------------------
# This file acts as the entry point for the cavity simulation.
# The detailed physics is implemented in the corresponding workflow
# and module files. This script orchestrates the workflow and exports
# results for the selected geometry.
# -------------------------------------------------------------------

import numpy as np

from cavity_plotter import CavityPlotter
from cavity_workflow import (
    build_cavity_context,
    build_cavity_simulation_output,
    build_cavity_simulation_result,
    build_geometry_estimators,
    compute_cavity_derived_quantities,
    compute_cavity_operating_point,
    print_derived_cavity_quantities,
    print_geometry_info,
    print_single_point_summary,
    save_cavity_outputs,
)

# %%
# Geometry selection

# Choose: "bowtie", "linear", "triangle", or "hemilithic"
GEOMETRY = "bowtie"

# %%

# Numeric parameters

c_num = 299792458.0

f_crystal_length = 16e-3
f_n_crystal = 1.82
f_RoC = 50e-3
f_wavelength = 1550e-9

# Squeezing/OPO parameters
f_T_ext = 0.10
f_L_rt = 0.01
f_detuning_Hz = 0.0

# Bow-tie parameters
f_theta_AOI = 6 * np.pi / 180.0
f_short_axis = np.arange(56e-3, 71e-3, 0.01e-3)
f_long_axis = np.arange(70e-3, 120e-3, 0.5e-3)
mesh_short_axis, mesh_long_axis = np.meshgrid(f_short_axis, f_long_axis)

# Linear parameters
f_L_cav = 100e-3

# Hemilithic parameters
f_L_air = 20e-3

# Triangle parameters
f_triangle_width = np.arange(max(f_crystal_length + 1e-3, 40e-3), 140e-3, 0.5e-3)
f_triangle_height = np.arange(10e-3, 80e-3, 0.5e-3)
mesh_triangle_width, mesh_triangle_height = np.meshgrid(f_triangle_width, f_triangle_height)

parameters = {
    "f_crystal_length": f_crystal_length,
    "f_n_crystal": f_n_crystal,
    "f_RoC": f_RoC,
    "f_wavelength": f_wavelength,
    "f_T_ext": f_T_ext,
    "f_L_rt": f_L_rt,
    "f_detuning_Hz": f_detuning_Hz,
    "f_theta_AOI": f_theta_AOI,
    "f_L_cav": f_L_cav,
    "f_L_air": f_L_air,
    "mesh_short_axis": mesh_short_axis,
    "mesh_long_axis": mesh_long_axis,
    "mesh_triangle_width": mesh_triangle_width,
    "mesh_triangle_height": mesh_triangle_height,
}

# %%
# Geometry info

print_geometry_info(GEOMETRY)

# %%
# Geometry-dependent estimators

estimators = build_geometry_estimators(GEOMETRY, parameters)
context = build_cavity_context(GEOMETRY, parameters, estimators=estimators)

# %%
# Stability and waist plots

plotter = CavityPlotter(GEOMETRY)
fig_stability = plotter.make_stability_plot(
    estimate_m_factor_s=estimators.estimate_m_factor_s,
    crystal_length=f_crystal_length,
    n_crystal=f_n_crystal,
    radius_of_curvature=f_RoC,
    incidence_angle=f_theta_AOI,
    mesh_x=estimators.mesh_x,
    mesh_y=estimators.mesh_y,
)

fig_waist = plotter.make_waist_plot(
    estimate_q_sagittal=estimators.estimate_q_sagittal,
    crystal_length=f_crystal_length,
    n_crystal=f_n_crystal,
    wavelength=f_wavelength,
    radius_of_curvature=f_RoC,
    incidence_angle=f_theta_AOI,
    mesh_x=estimators.mesh_x,
    mesh_y=estimators.mesh_y,
)

# %%
# Single-point evaluation

# Explicit single-point selections used for the detailed evaluation step.
single_point_parameters = {
    "bowtie_short_axis_m": 68e-3,
    "bowtie_long_axis_m": 90e-3,
    "bowtie_theta_AOI_rad": f_theta_AOI,
    "linear_cavity_length_m": f_L_cav,
    "triangle_width_m": 80e-3,
    "triangle_height_m": 30e-3,
    "hemilithic_air_gap_m": f_L_air,
}

single_point = compute_cavity_operating_point(context, single_point_parameters)
print_single_point_summary(GEOMETRY, single_point)

# %%
# Derived cavity figures

derived = compute_cavity_derived_quantities(
    context,
    single_point,
    c_m_per_s=c_num,
    T_ext=f_T_ext,
    L_rt=f_L_rt,
    detuning_Hz=f_detuning_Hz,
)
print_derived_cavity_quantities(derived)

# %%
# Export simulation output

cavity_result = build_cavity_simulation_result(context, single_point, derived)
simulation_output = build_cavity_simulation_output(cavity_result, c_m_per_s=c_num)
saved_outputs = save_cavity_outputs(GEOMETRY, simulation_output, fig_stability, fig_waist)
simulation_output["outputs"] = saved_outputs

print(f"Saved simulation output to: {saved_outputs['cavity_output_json']}")
print(f"Saved stability map to: {saved_outputs['stability_map_png']}")
print(f"Saved waist map to: {saved_outputs['waist_map_png']}")

# %%
