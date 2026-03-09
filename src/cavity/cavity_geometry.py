# %%
"""Standalone cavity simulation workflow script."""

import json
from pathlib import Path

import numpy as np

from cavity_analysis import beam_waist_from_q
from cavity_plotter import CavityPlotter
from cavity_workflow import (
    build_geometry_estimators,
    build_geometry_inputs_for_export,
    evaluate_single_point,
    print_geometry_info,
    print_single_point_summary,
)

# %%
# Geometry selection

# Choose: "bowtie", "linear", "triangle", or "hemilithic"
GEOMETRY = "bowtie"
RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / GEOMETRY
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

w_um = beam_waist_from_q(qs, f_wavelength, refractive_index=f_n_crystal) * 1e6
print(f"Beam waist in crystal (from q): {w_um:.3f} um")

L_optical = float(cavity_length + (f_n_crystal - 1.0) * optical_crystal_length)
fsr = float(c_num / L_optical)

print(f"Geometric cavity length: {cavity_length:.6f} m")
print(f"Optical round-trip length: {L_optical:.6f} m")
print(f"FSR: {fsr:.6f} Hz ({fsr/1e6:.6f} MHz)")

kappa_ext = (c_num / (2.0 * L_optical)) * f_T_ext
kappa_loss = (c_num / (2.0 * L_optical)) * f_L_rt
kappa = kappa_ext + kappa_loss
eta_escape = kappa_ext / kappa if kappa != 0 else np.nan
kappa_Hz = kappa / (2.0 * np.pi)
Delta_rad_s = 2.0 * np.pi * f_detuning_Hz

print("\nCavity parameters for squeezing / OPO models:")
print(f"w0 in crystal (TEM00): {w_um:.3f} um")
print(f"kappa_ext  = {kappa_ext:.3e} rad/s   (kappa_ext/2π = {kappa_ext/(2*np.pi):.3e} Hz)")
print(f"kappa_loss = {kappa_loss:.3e} rad/s   (kappa_loss/2π = {kappa_loss/(2*np.pi):.3e} Hz)")
print(f"kappa      = {kappa:.3e} rad/s   (kappa/2π = {kappa_Hz:.3e} Hz)")
print(f"Escape efficiency η_escape = {eta_escape:.4f}")
print(f"Detuning Δ = {Delta_rad_s:.3e} rad/s   (Δ/2π = {f_detuning_Hz:.3e} Hz)")

if GEOMETRY in ("bowtie", "triangle"):
    psi_sagittal = np.arccos(m_factor_dict["sagittal"])
    psi_tangential = np.arccos(m_factor_dict["tangential"])
else:
    psi_sagittal = np.arccos(m_factor_dict["sagittal"])
    psi_tangential = psi_sagittal

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
        "beam_waist_crystal_um": float(w_um),
        "cavity_length_m": float(cavity_length),
        "optical_crystal_length_m": float(optical_crystal_length),
        "optical_roundtrip_length_m": float(L_optical),
        "fsr_Hz": float(fsr),
        "kappa_ext_rad_s": float(kappa_ext),
        "kappa_loss_rad_s": float(kappa_loss),
        "kappa_total_rad_s": float(kappa),
        "kappa_total_Hz": float(kappa_Hz),
        "escape_efficiency": float(eta_escape),
        "detuning_rad_s": float(Delta_rad_s),
        "gouy_phase_sagittal_rad": float(psi_sagittal),
        "gouy_phase_tangential_rad": float(psi_tangential),
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
