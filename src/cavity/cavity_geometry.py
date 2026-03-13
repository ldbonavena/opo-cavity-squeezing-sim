# %%
"""Standalone cavity simulation workflow script."""

import json
from pathlib import Path

import numpy as np

from cavity_plotter import CavityPlotter
from cavity_workflow import (
    build_geometry_estimators,
    build_geometry_inputs_for_export,
    compute_derived_cavity_quantities,
    evaluate_single_point,
    print_derived_cavity_quantities,
    print_geometry_info,
    print_single_point_summary,
)

# %%
# Geometry selection

# Choose: "bowtie", "linear", "triangle", or "hemilithic"
GEOMETRY = "bowtie"
RESULT_DIR = Path(__file__).resolve().parents[2] / "results" / GEOMETRY
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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

single_point = evaluate_single_point(
    GEOMETRY,
    parameters,
    estimators,
    single_point_parameters,
)
print_single_point_summary(GEOMETRY, single_point)

qs = single_point["qs"]
qt = single_point["qt"]
m_factor_dict = single_point["m_factor"]
cavity_length = single_point["cavity_length"]
optical_crystal_length = single_point["optical_crystal_length"]

# %%
# Derived cavity figures

derived = compute_derived_cavity_quantities(
    GEOMETRY,
    qs,
    qt,
    m_factor_dict,
    cavity_length,
    optical_crystal_length,
    f_wavelength,
    f_n_crystal,
    c_num,
    f_T_ext,
    f_L_rt,
    f_detuning_Hz,
)
print_derived_cavity_quantities(derived)

# %%
# Export simulation output

geometry_inputs, m_factor_export = build_geometry_inputs_for_export(GEOMETRY, single_point)

simulation_output = {
    "geometry": GEOMETRY,
    "constants": {"c_m_per_s": float(c_num)},
    "inputs": {
        "crystal_length_m": float(f_crystal_length),
        "n_crystal": float(f_n_crystal),
        "RoC_m": float(f_RoC),
        "wavelength_m": float(f_wavelength),
        "T_ext": float(f_T_ext),
        "L_rt": float(f_L_rt),
        "detuning_Hz": float(f_detuning_Hz),
        "geometry_specific": geometry_inputs,
    },
    "results": {
        "q_sagittal": {"real": float(np.real(qs)), "imag": float(np.imag(qs))},
        "q_tangential": {"real": float(np.real(qt)), "imag": float(np.imag(qt))},
        "m_factor": m_factor_export,
        "beam_waist_crystal_um": float(derived["beam_waist_crystal_um"]),
        "cavity_length_m": float(derived["cavity_length_m"]),
        "optical_crystal_length_m": float(derived["optical_crystal_length_m"]),
        "optical_roundtrip_length_m": float(derived["optical_roundtrip_length_m"]),
        "fsr_Hz": float(derived["fsr_Hz"]),
        "kappa_ext_rad_s": float(derived["kappa_ext_rad_s"]),
        "kappa_loss_rad_s": float(derived["kappa_loss_rad_s"]),
        "kappa_total_rad_s": float(derived["kappa_total_rad_s"]),
        "kappa_total_Hz": float(derived["kappa_total_Hz"]),
        "escape_efficiency": float(derived["escape_efficiency"]),
        "detuning_rad_s": float(derived["detuning_rad_s"]),
        "gouy_phase_sagittal_rad": float(derived["gouy_phase_sagittal_rad"]),
        "gouy_phase_tangential_rad": float(derived["gouy_phase_tangential_rad"]),
    },
}

stability_map_path = RESULT_DIR / "stability_map.png"
waist_map_path = RESULT_DIR / "waist_map.png"
fig_stability.savefig(stability_map_path, dpi=300, bbox_inches="tight")
fig_waist.savefig(waist_map_path, dpi=300, bbox_inches="tight")

simulation_output["outputs"] = {
    "result_dir": str(RESULT_DIR),
    "stability_map_png": str(stability_map_path),
    "waist_map_png": str(waist_map_path),
}

output_path = RESULT_DIR / "cavity_simulation_output.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(simulation_output, f, indent=2)

print(f"Saved simulation output to: {output_path}")
print(f"Saved stability map to: {stability_map_path}")
print(f"Saved waist map to: {waist_map_path}")

# %%
