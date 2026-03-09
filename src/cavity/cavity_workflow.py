"""Geometry-specific workflow helpers for cavity simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cavity_analysis import make_m_factor_estimator, make_q_estimator
from cavity_plotter import print_geometry_ascii


@dataclass
class GeometryEstimators:
    """Container for geometry-dependent estimators and plotting meshes."""

    estimate_m_factor_s: callable
    estimate_m_factor_t: callable
    estimate_q_sagittal: callable
    estimate_q_tangential: callable
    mesh_x: np.ndarray | None
    mesh_y: np.ndarray | None


def print_geometry_info(geometry: str) -> None:
    """Print geometry parameter definitions and an ASCII sketch."""
    if geometry == "bowtie":
        print("Bow-tie geometry parameters:")
        print(" (0) crystal length [mm]")
        print(" (1) crystal refractive index")
        print(" (2) short axis [mm]")
        print(" (3) long axis [mm]")
        print(" (4) mirror radius of curvature [mm]")
        print(" (5) AOI [deg]")
    elif geometry == "linear":
        print("Linear geometry parameters:")
        print(" (0) crystal length [mm]")
        print(" (1) crystal refractive index")
        print(" (2) cavity length [mm]")
        print(" (3) mirror radius of curvature [mm]")
    elif geometry == "triangle":
        print("Triangle geometry parameters:")
        print(" (0) crystal length [mm]")
        print(" (1) crystal refractive index")
        print(" (2) triangle width [mm]")
        print(" (3) triangle height [mm]")
        print(" (4) mirror radius of curvature [mm]")
    elif geometry == "hemilithic":
        print("Hemilithic geometry parameters:")
        print(" (0) crystal length [mm]")
        print(" (1) crystal refractive index")
        print(" (2) air gap [mm]")
        print(" (3) mirror radius of curvature [mm]")
    else:
        raise ValueError("GEOMETRY must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")

    print_geometry_ascii(geometry)


def build_geometry_estimators(geometry: str, parameters: dict) -> GeometryEstimators:
    """Build geometry-dependent m-factor/q estimators and plot meshes."""
    if geometry == "bowtie":
        return GeometryEstimators(
            estimate_m_factor_s=make_m_factor_estimator("bowtie", plane="sagittal"),
            estimate_m_factor_t=make_m_factor_estimator("bowtie", plane="tangential"),
            estimate_q_sagittal=make_q_estimator("bowtie", plane="sagittal"),
            estimate_q_tangential=make_q_estimator("bowtie", plane="tangential"),
            mesh_x=parameters["mesh_short_axis"],
            mesh_y=parameters["mesh_long_axis"],
        )

    if geometry == "linear":
        m = make_m_factor_estimator("linear")
        q = make_q_estimator("linear")
        return GeometryEstimators(
            estimate_m_factor_s=m,
            estimate_m_factor_t=m,
            estimate_q_sagittal=q,
            estimate_q_tangential=q,
            mesh_x=None,
            mesh_y=None,
        )

    if geometry == "triangle":
        return GeometryEstimators(
            estimate_m_factor_s=make_m_factor_estimator("triangle", plane="sagittal"),
            estimate_m_factor_t=make_m_factor_estimator("triangle", plane="tangential"),
            estimate_q_sagittal=make_q_estimator("triangle", plane="sagittal"),
            estimate_q_tangential=make_q_estimator("triangle", plane="tangential"),
            mesh_x=parameters["mesh_triangle_width"],
            mesh_y=parameters["mesh_triangle_height"],
        )

    if geometry == "hemilithic":
        m = make_m_factor_estimator("hemilithic")
        q = make_q_estimator("hemilithic")
        return GeometryEstimators(
            estimate_m_factor_s=m,
            estimate_m_factor_t=m,
            estimate_q_sagittal=q,
            estimate_q_tangential=q,
            mesh_x=None,
            mesh_y=None,
        )

    raise ValueError("GEOMETRY must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")


def evaluate_single_point(
    geometry: str,
    parameters: dict,
    estimators: GeometryEstimators,
    point_parameters: dict,
) -> dict:
    """Evaluate one representative cavity point for the selected geometry."""
    if geometry == "bowtie":
        short_axis_val = point_parameters["bowtie_short_axis_m"]
        long_axis_val = point_parameters["bowtie_long_axis_m"]
        theta = point_parameters.get("bowtie_theta_AOI_rad", parameters["f_theta_AOI"])
        diagonal_val = float((long_axis_val + short_axis_val) / 2 * np.cos(theta))

        m_x = float(
            estimators.estimate_m_factor_s(
                long_axis_val,
                short_axis_val,
                theta,
                parameters["f_crystal_length"],
                parameters["f_RoC"],
                parameters["f_n_crystal"],
            )
        )
        m_y = float(
            estimators.estimate_m_factor_t(
                long_axis_val,
                short_axis_val,
                theta,
                parameters["f_crystal_length"],
                parameters["f_RoC"],
                parameters["f_n_crystal"],
            )
        )
        qs = complex(
            estimators.estimate_q_sagittal(
                long_axis_val,
                short_axis_val,
                theta,
                parameters["f_crystal_length"],
                parameters["f_RoC"],
                parameters["f_n_crystal"],
            )
        )
        qt = complex(
            estimators.estimate_q_tangential(
                long_axis_val,
                short_axis_val,
                theta,
                parameters["f_crystal_length"],
                parameters["f_RoC"],
                parameters["f_n_crystal"],
            )
        )

        return {
            "qs": qs,
            "qt": qt,
            "m_factor": {"sagittal": m_x, "tangential": m_y},
            "geometry_values": {
                "short_axis_m": short_axis_val,
                "long_axis_m": long_axis_val,
                "diagonal_m": diagonal_val,
                "theta_AOI_rad": float(theta),
                "theta_AOI_deg": float(np.degrees(theta)),
            },
            "cavity_length": float(long_axis_val + short_axis_val + 2 * diagonal_val),
            "optical_crystal_length": float(parameters["f_crystal_length"]),
        }

    if geometry == "linear":
        L_cav_val = point_parameters.get("linear_cavity_length_m", parameters["f_L_cav"])
        m_val = float(
            estimators.estimate_m_factor_s(
                parameters["f_RoC"],
                L_cav_val,
                parameters["f_crystal_length"],
                parameters["f_n_crystal"],
            )
        )
        qs = complex(
            estimators.estimate_q_sagittal(
                parameters["f_RoC"],
                L_cav_val,
                parameters["f_crystal_length"],
                parameters["f_n_crystal"],
            )
        )

        return {
            "qs": qs,
            "qt": qs,
            "m_factor": {"sagittal": m_val, "tangential": m_val},
            "geometry_values": {"L_cav_m": float(L_cav_val)},
            "cavity_length": float(2 * L_cav_val),
            "optical_crystal_length": float(parameters["f_crystal_length"]),
        }

    if geometry == "triangle":
        width_val = point_parameters["triangle_width_m"]
        height_val = point_parameters["triangle_height_m"]
        diagonal_val = float(np.sqrt((width_val / 2) ** 2 + height_val**2))
        theta_half_val = float(np.arcsin(height_val / diagonal_val) / 2)

        m_x = float(
            estimators.estimate_m_factor_s(
                width_val,
                height_val,
                parameters["f_crystal_length"],
                parameters["f_RoC"],
                parameters["f_n_crystal"],
            )
        )
        m_y = float(
            estimators.estimate_m_factor_t(
                width_val,
                height_val,
                parameters["f_crystal_length"],
                parameters["f_RoC"],
                parameters["f_n_crystal"],
            )
        )
        qs = complex(
            estimators.estimate_q_sagittal(
                width_val,
                height_val,
                parameters["f_crystal_length"],
                parameters["f_RoC"],
                parameters["f_n_crystal"],
            )
        )
        qt = complex(
            estimators.estimate_q_tangential(
                width_val,
                height_val,
                parameters["f_crystal_length"],
                parameters["f_RoC"],
                parameters["f_n_crystal"],
            )
        )

        return {
            "qs": qs,
            "qt": qt,
            "m_factor": {"sagittal": m_x, "tangential": m_y},
            "geometry_values": {
                "triangle_width_m": float(width_val),
                "triangle_height_m": float(height_val),
                "triangle_diagonal_m": float(diagonal_val),
                "mirror_aoi_half_angle_rad": float(theta_half_val),
                "mirror_aoi_half_angle_deg": float(np.degrees(theta_half_val)),
            },
            "cavity_length": float(width_val + 2 * diagonal_val),
            "optical_crystal_length": float(parameters["f_crystal_length"]),
        }

    if geometry == "hemilithic":
        L_air_val = point_parameters.get("hemilithic_air_gap_m", parameters["f_L_air"])
        m_val = float(
            estimators.estimate_m_factor_s(
                parameters["f_RoC"],
                L_air_val,
                parameters["f_crystal_length"],
                parameters["f_n_crystal"],
            )
        )
        qs = complex(
            estimators.estimate_q_sagittal(
                parameters["f_RoC"],
                L_air_val,
                parameters["f_crystal_length"],
                parameters["f_n_crystal"],
            )
        )

        return {
            "qs": qs,
            "qt": qs,
            "m_factor": {"sagittal": m_val, "tangential": m_val},
            "geometry_values": {"L_air_m": float(L_air_val)},
            "cavity_length": float(2 * (L_air_val + parameters["f_crystal_length"])),
            "optical_crystal_length": float(2 * parameters["f_crystal_length"]),
        }

    raise ValueError("GEOMETRY must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")


def print_single_point_summary(geometry: str, result: dict) -> None:
    """Print a concise single-point summary for the selected geometry."""
    qs = result["qs"]
    qt = result["qt"]
    m = result["m_factor"]
    g = result["geometry_values"]

    if geometry == "bowtie":
        print(
            f"Geometrical parameters: long axis = {g['long_axis_m']*1e3:.3f} mm, short axis = {g['short_axis_m']*1e3:.3f} mm, "
            f"diagonal = {g['diagonal_m']*1e3:.3f} mm, AOI = {g['theta_AOI_deg']:.3f} deg"
        )
        print(f"m_sagittal = {m['sagittal']:.6f}, m_tangential = {m['tangential']:.6f}")
        print(f"qs parameter in the crystal: {qs:.5f}")
        print(f"qt parameter in the crystal: {qt:.5f}")

    elif geometry == "linear":
        print(f"Linear cavity parameters: L_cav = {g['L_cav_m']*1e3:.3f} mm")
        print(f"m = {m['sagittal']:.6f} (stable if |m|<1)")
        print(f"q parameter at crystal center: {qs:.5f}")

    elif geometry == "triangle":
        print(
            f"Triangle cavity parameters: width = {g['triangle_width_m']*1e3:.3f} mm, "
            f"height = {g['triangle_height_m']*1e3:.3f} mm, diagonal = {g['triangle_diagonal_m']*1e3:.3f} mm"
        )
        print(f"m_sagittal = {m['sagittal']:.6f}, m_tangential = {m['tangential']:.6f}")
        print(f"qs parameter in the crystal: {qs:.5f}")
        print(f"qt parameter in the crystal: {qt:.5f}")

    elif geometry == "hemilithic":
        print(f"Hemilithic cavity parameters: L_air = {g['L_air_m']*1e3:.3f} mm")
        print(f"m = {m['sagittal']:.6f} (stable if |m|<1)")
        print(f"q parameter at crystal input face: {qs:.5f}")

    else:
        raise ValueError("GEOMETRY must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")


def build_geometry_inputs_for_export(geometry: str, result: dict) -> tuple[dict, dict]:
    """Build geometry-specific export fields for JSON output."""
    return result["geometry_values"], result["m_factor"]


__all__ = [
    "GeometryEstimators",
    "print_geometry_info",
    "build_geometry_estimators",
    "evaluate_single_point",
    "print_single_point_summary",
    "build_geometry_inputs_for_export",
]
