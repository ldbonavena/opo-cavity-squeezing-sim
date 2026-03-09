"""Cavity physics analysis utilities built on ABCD matrices."""

from __future__ import annotations

import numpy as np
import sympy as sp

from cavity_abcd import CavityAbcdBuilder
from optics_abcd import Abcd


def cavity_stability(matrix):
    """Return cavity stability m-factor = (A + D) / 2."""
    A, _, _, D = Abcd.parameters(matrix)
    return (A + D) / 2


def cavity_q_parameter(matrix):
    """Return stable cavity q parameter from round-trip ABCD matrix."""
    A, B, C, D = Abcd.parameters(matrix)
    return -1 * (D - A) / (2 * C) + sp.I * sp.sqrt(1 - ((D + A) / 2) ** 2) / sp.Abs(C)


def beam_waist_from_q(q_parameter, wavelength, refractive_index=1):
    """Compute beam waist radius from q parameter imaginary part."""
    q_im = np.imag(q_parameter)
    return np.sqrt(wavelength * q_im / (refractive_index * np.pi))


def optical_roundtrip_length(cavity_length_m, optical_crystal_length_m, n_crystal):
    """Return optical round-trip cavity length in meters."""
    return float(cavity_length_m + (n_crystal - 1.0) * optical_crystal_length_m)


def fsr_from_roundtrip_length(L_optical_m, c_m_per_s):
    """Return free spectral range in Hz from optical round-trip length."""
    return float(c_m_per_s / L_optical_m)


def compute_decay_rates(L_optical_m, c_m_per_s, T_ext, L_rt):
    """Compute cavity decay rates and escape efficiency."""
    kappa_ext = (c_m_per_s / (2.0 * L_optical_m)) * T_ext
    kappa_loss = (c_m_per_s / (2.0 * L_optical_m)) * L_rt
    kappa_total = kappa_ext + kappa_loss
    return {
        "kappa_ext_rad_s": float(kappa_ext),
        "kappa_loss_rad_s": float(kappa_loss),
        "kappa_total_rad_s": float(kappa_total),
        "kappa_total_Hz": float(kappa_total / (2.0 * np.pi)),
        "escape_efficiency": float(kappa_ext / kappa_total) if kappa_total != 0 else np.nan,
    }


def gouy_phases_from_m_factor(geometry, m_factor_dict):
    """Return sagittal/tangential Gouy phases in radians from m-factors."""
    psi_sagittal = np.arccos(m_factor_dict["sagittal"])
    if geometry in ("bowtie", "triangle"):
        psi_tangential = np.arccos(m_factor_dict["tangential"])
    else:
        psi_tangential = psi_sagittal
    return {
        "gouy_phase_sagittal_rad": float(psi_sagittal),
        "gouy_phase_tangential_rad": float(psi_tangential),
    }


def bowtie_m_factor(long_axis, short_axis, incidence_angle, crystal_length, radius_of_curvature, refractive_index, plane="sagittal"):
    """Return bow-tie cavity m-factor for a selected plane."""
    matrix = CavityAbcdBuilder.bowtie_roundtrip(
        long_axis,
        short_axis,
        crystal_length,
        radius_of_curvature,
        refractive_index,
        incidence_angle,
        plane=plane,
    )
    return cavity_stability(matrix)


def bowtie_q_parameter(long_axis, short_axis, incidence_angle, crystal_length, radius_of_curvature, refractive_index, plane="sagittal"):
    """Return bow-tie cavity q parameter for a selected plane."""
    matrix = CavityAbcdBuilder.bowtie_roundtrip(
        long_axis,
        short_axis,
        crystal_length,
        radius_of_curvature,
        refractive_index,
        incidence_angle,
        plane=plane,
    )
    return cavity_q_parameter(matrix)


def linear_m_factor(radius_of_curvature, cavity_length, crystal_length, refractive_index):
    """Return m-factor for a symmetric linear cavity."""
    matrix = CavityAbcdBuilder.linear_roundtrip(
        cavity_length,
        crystal_length,
        radius_of_curvature,
        radius_of_curvature,
        refractive_index,
    )
    return cavity_stability(matrix)


def linear_q_parameter(radius_of_curvature, cavity_length, crystal_length, refractive_index):
    """Return q parameter for a symmetric linear cavity."""
    matrix = CavityAbcdBuilder.linear_roundtrip(
        cavity_length,
        crystal_length,
        radius_of_curvature,
        radius_of_curvature,
        refractive_index,
    )
    return cavity_q_parameter(matrix)


def hemilithic_m_factor(radius_of_curvature, air_gap, crystal_length, refractive_index):
    """Return m-factor for a hemilithic cavity."""
    matrix = CavityAbcdBuilder.hemilithic_roundtrip(
        air_gap,
        crystal_length,
        radius_of_curvature,
        refractive_index,
    )
    return cavity_stability(matrix)


def hemilithic_q_parameter(radius_of_curvature, air_gap, crystal_length, refractive_index):
    """Return q parameter for a hemilithic cavity."""
    matrix = CavityAbcdBuilder.hemilithic_roundtrip(
        air_gap,
        crystal_length,
        radius_of_curvature,
        refractive_index,
    )
    return cavity_q_parameter(matrix)


def triangle_m_factor(width, height, crystal_length, radius_of_curvature, refractive_index, plane="sagittal"):
    """Return m-factor for a triangular cavity."""
    matrix = CavityAbcdBuilder.triangle_roundtrip(
        width,
        height,
        crystal_length,
        radius_of_curvature,
        refractive_index,
        plane=plane,
    )
    return cavity_stability(matrix)


def triangle_q_parameter(width, height, crystal_length, radius_of_curvature, refractive_index, plane="sagittal"):
    """Return q parameter for a triangular cavity."""
    matrix = CavityAbcdBuilder.triangle_roundtrip(
        width,
        height,
        crystal_length,
        radius_of_curvature,
        refractive_index,
        plane=plane,
    )
    return cavity_q_parameter(matrix)


def make_m_factor_estimator(geometry: str, plane: str = "sagittal"):
    """Build a NumPy-callable m-factor estimator for a geometry."""
    if geometry == "bowtie":
        long_axis, short_axis, incidence_angle, crystal_length, radius_of_curvature, refractive_index = sp.symbols(
            "long_axis short_axis incidence_angle crystal_length radius_of_curvature refractive_index", positive=True, real=True
        )
        expr = bowtie_m_factor(
            long_axis,
            short_axis,
            incidence_angle,
            crystal_length,
            radius_of_curvature,
            refractive_index,
            plane=plane,
        )
        return sp.lambdify(
            (long_axis, short_axis, incidence_angle, crystal_length, radius_of_curvature, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )

    if geometry == "linear":
        radius_of_curvature, cavity_length, crystal_length, refractive_index = sp.symbols(
            "radius_of_curvature cavity_length crystal_length refractive_index", positive=True, real=True
        )
        expr = linear_m_factor(radius_of_curvature, cavity_length, crystal_length, refractive_index)
        return sp.lambdify(
            (radius_of_curvature, cavity_length, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )

    if geometry == "hemilithic":
        radius_of_curvature, air_gap, crystal_length, refractive_index = sp.symbols(
            "radius_of_curvature air_gap crystal_length refractive_index", positive=True, real=True
        )
        expr = hemilithic_m_factor(radius_of_curvature, air_gap, crystal_length, refractive_index)
        return sp.lambdify(
            (radius_of_curvature, air_gap, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )

    if geometry == "triangle":
        width, height, crystal_length, radius_of_curvature, refractive_index = sp.symbols(
            "width height crystal_length radius_of_curvature refractive_index", positive=True, real=True
        )
        expr = triangle_m_factor(width, height, crystal_length, radius_of_curvature, refractive_index, plane=plane)
        return sp.lambdify(
            (width, height, crystal_length, radius_of_curvature, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )

    raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")


def make_q_estimator(geometry: str, plane: str = "sagittal"):
    """Build a NumPy-callable q-parameter estimator for a geometry."""
    if geometry == "bowtie":
        long_axis, short_axis, incidence_angle, crystal_length, radius_of_curvature, refractive_index = sp.symbols(
            "long_axis short_axis incidence_angle crystal_length radius_of_curvature refractive_index", positive=True, real=True
        )
        expr = bowtie_q_parameter(
            long_axis,
            short_axis,
            incidence_angle,
            crystal_length,
            radius_of_curvature,
            refractive_index,
            plane=plane,
        )
        return sp.lambdify(
            (long_axis, short_axis, incidence_angle, crystal_length, radius_of_curvature, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )

    if geometry == "linear":
        radius_of_curvature, cavity_length, crystal_length, refractive_index = sp.symbols(
            "radius_of_curvature cavity_length crystal_length refractive_index", positive=True, real=True
        )
        expr = linear_q_parameter(radius_of_curvature, cavity_length, crystal_length, refractive_index)
        return sp.lambdify(
            (radius_of_curvature, cavity_length, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )

    if geometry == "hemilithic":
        radius_of_curvature, air_gap, crystal_length, refractive_index = sp.symbols(
            "radius_of_curvature air_gap crystal_length refractive_index", positive=True, real=True
        )
        expr = hemilithic_q_parameter(radius_of_curvature, air_gap, crystal_length, refractive_index)
        return sp.lambdify(
            (radius_of_curvature, air_gap, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )

    if geometry == "triangle":
        width, height, crystal_length, radius_of_curvature, refractive_index = sp.symbols(
            "width height crystal_length radius_of_curvature refractive_index", positive=True, real=True
        )
        expr = triangle_q_parameter(width, height, crystal_length, radius_of_curvature, refractive_index, plane=plane)
        return sp.lambdify(
            (width, height, crystal_length, radius_of_curvature, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )

    raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")


__all__ = [
    "cavity_stability",
    "cavity_q_parameter",
    "beam_waist_from_q",
    "optical_roundtrip_length",
    "fsr_from_roundtrip_length",
    "compute_decay_rates",
    "gouy_phases_from_m_factor",
    "bowtie_m_factor",
    "bowtie_q_parameter",
    "linear_m_factor",
    "linear_q_parameter",
    "hemilithic_m_factor",
    "hemilithic_q_parameter",
    "triangle_m_factor",
    "triangle_q_parameter",
    "make_m_factor_estimator",
    "make_q_estimator",
]
