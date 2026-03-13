"""Plotting utilities for cavity stability, waist maps, and cavity layouts."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from cavity_analysis import beam_waist_from_q


def print_geometry_ascii(geometry: str) -> None:
    """Print an ASCII sketch of the selected cavity geometry."""
    if geometry == "linear":
        print(
            "      T_in       T_out\n"
            "   -----(--[###]--)----->\n"
            "         <- - - -> d"
        )
    elif geometry == "triangle":
        print(
            "              __ M3,R         ___\n"
            "              /\\               |\n"
            "             /  \\              |\n"
            "            /    \\             | h\n"
            "           /      \\            |\n"
            "     T_in / [###]  \\ T_out     |\n"
            "    -----/----------\\----->   _|_\n"
            "         <- - - - - > w"
        )
    elif geometry == "bowtie":
        print(
            "          <- - - -> d1\n"
            "    M3,R /--[###]--\\ M4,R\n"
            "          \\       /\n"
            "            \\   /\n"
            "              X\n"
            "             / \\\n"
            "           /     \\\n"
            "    T_in /         \\ T_out\n"
            "   ----/-------------\\----->\n"
            "       <- - - - - - -> d2"
        )
    elif geometry == "hemilithic":
        print(
            "   T_out                HR crystal face\n"
            "   -----(---- air ----[###)----->\n"
            "         <-- L_air -->"
        )
    else:
        raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")


class CavityPlotter:
    """Plotter for cavity maps and simple geometric layouts."""

    def __init__(self, geometry: str):
        self.geometry = geometry

    @staticmethod
    def mirror_patch(anchor, angle_deg, diameter_inch=0.5):
        """Return rectangle patches approximating a coated mirror."""
        x0, y0 = anchor
        inch = 25.4
        coating_thickness = 1
        substrate_thickness = 6
        diameter = inch * diameter_inch
        substrate_top = plt.Rectangle((x0, y0), substrate_thickness, diameter / 2, angle=angle_deg, fc="C0", alpha=0.5)
        substrate_bottom = plt.Rectangle((x0, y0), substrate_thickness, -diameter / 2, angle=angle_deg, fc="C0", alpha=0.5)
        coating_top = plt.Rectangle((x0, y0), coating_thickness, diameter / 2, angle=angle_deg, fc="k", alpha=1.0)
        coating_bottom = plt.Rectangle((x0, y0), coating_thickness, -diameter / 2, angle=angle_deg, fc="k", alpha=1.0)
        return [substrate_top, substrate_bottom, coating_top, coating_bottom]

    @staticmethod
    def crystal_patch(anchor, length_mm, width_mm=1.0):
        """Return a rectangle patch for the nonlinear crystal."""
        x0, y0 = anchor
        return plt.Rectangle((x0 - length_mm / 2, y0 - width_mm / 2), length_mm, width_mm, fc="C2", alpha=0.5)

    @staticmethod
    def beam_radius(z, waist, wavelength):
        """Gaussian beam radius evolution around a waist at z=0."""
        rayleigh = np.pi * waist**2 / wavelength
        return waist * np.sqrt(1 + (z / rayleigh) ** 2)

    @staticmethod
    def plot_beam_envelope(start, end, z_offset, waist, wavelength, ax):
        """Plot a filled TEM00 beam envelope between two anchor points."""
        x0, y0 = start
        x1, y1 = end
        x = np.linspace(x0, x1, 200)
        y = np.linspace(y0, y1, 200)
        distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        angle = np.arctan2(y1 - y0, x1 - x0)
        w = CavityPlotter.beam_radius(z_offset + distance, waist, wavelength)
        x_top = x - w * np.sin(angle)
        y_top = y + w * np.cos(angle)
        x_bottom = x + w * np.sin(angle)
        y_bottom = y - w * np.cos(angle)
        ax.fill_between(x_top, y_top, y_bottom, color="C3", alpha=0.5)
        ax.fill_betweenx(y_top, x_top, x_bottom, color="C3", alpha=0.5)

    def make_stability_plot(
        self,
        estimate_m_factor_s,
        crystal_length,
        n_crystal,
        radius_of_curvature=None,
        incidence_angle=None,
        mesh_x=None,
        mesh_y=None,
        x_label=None,
        y_label=None,
        title=None,
    ):
        """Create a stability map figure for the configured geometry."""
        fig = plt.figure()

        if self.geometry == "bowtie":
            if mesh_x is None or mesh_y is None or radius_of_curvature is None or incidence_angle is None:
                raise ValueError("bowtie stability plot requires mesh_x, mesh_y, radius_of_curvature, and incidence_angle")
            stable_map = np.abs(
                estimate_m_factor_s(mesh_y, mesh_x, incidence_angle, crystal_length, radius_of_curvature, n_crystal)
            ) < 1
            plt.contourf(mesh_x * 1e3, mesh_y * 1e3, stable_map)
            plt.xlabel(x_label or "Short axis [mm]")
            plt.ylabel(y_label or "Long axis [mm]")
            plt.title(title or "Bow-tie stability (|m|<1)")

        elif self.geometry == "linear":
            cavity_scan = np.arange(crystal_length + 0.5e-3, 120e-3, 0.5e-3)
            roc_scan = np.arange(10e-3, 150e-3, 0.5e-3)
            mesh_cavity, mesh_roc = np.meshgrid(cavity_scan, roc_scan)
            stable_map = np.abs(estimate_m_factor_s(mesh_roc, mesh_cavity, crystal_length, n_crystal)) < 1
            plt.contourf(mesh_cavity * 1e3, mesh_roc * 1e3, stable_map)
            plt.xlabel(x_label or "Cavity length [mm]")
            plt.ylabel(y_label or "RoC [mm]")
            plt.title(title or "Linear cavity stability (|m|<1)")

        elif self.geometry == "triangle":
            if mesh_x is None or mesh_y is None or radius_of_curvature is None:
                raise ValueError("triangle stability plot requires mesh_x, mesh_y, and radius_of_curvature")
            stable_map = np.abs(estimate_m_factor_s(mesh_x, mesh_y, crystal_length, radius_of_curvature, n_crystal)) < 1
            plt.contourf(mesh_x * 1e3, mesh_y * 1e3, stable_map)
            plt.xlabel(x_label or "Triangle width [mm]")
            plt.ylabel(y_label or "Triangle height [mm]")
            plt.title(title or "Triangle cavity stability (|m|<1)")

        elif self.geometry == "hemilithic":
            air_scan = np.arange(0.5e-3, 120e-3, 0.5e-3)
            roc_scan = np.arange(10e-3, 150e-3, 0.5e-3)
            mesh_air, mesh_roc = np.meshgrid(air_scan, roc_scan)
            stable_map = np.abs(estimate_m_factor_s(mesh_roc, mesh_air, crystal_length, n_crystal)) < 1
            plt.contourf(mesh_air * 1e3, mesh_roc * 1e3, stable_map)
            plt.xlabel(x_label or "Air gap [mm]")
            plt.ylabel(y_label or "RoC [mm]")
            plt.title(title or "Hemilithic cavity stability (|m|<1)")

        else:
            raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")

        plt.colorbar(label="stable")
        plt.grid(True)
        return fig

    def make_waist_plot(
        self,
        estimate_q_sagittal,
        crystal_length,
        n_crystal,
        wavelength,
        radius_of_curvature=None,
        incidence_angle=None,
        mesh_x=None,
        mesh_y=None,
        x_label=None,
        y_label=None,
        title=None,
    ):
        """Create a beam-waist map figure for the configured geometry."""
        fig = plt.figure()

        if self.geometry == "bowtie":
            if mesh_x is None or mesh_y is None or radius_of_curvature is None or incidence_angle is None:
                raise ValueError("bowtie waist plot requires mesh_x, mesh_y, radius_of_curvature, and incidence_angle")
            q = estimate_q_sagittal(mesh_y, mesh_x, incidence_angle, crystal_length, radius_of_curvature, n_crystal)
            waist_um = beam_waist_from_q(q, wavelength, refractive_index=n_crystal) * 1e6
            plt.contourf(mesh_x * 1e3, mesh_y * 1e3, waist_um)
            plt.xlabel(x_label or "Short axis [mm]")
            plt.ylabel(y_label or "Long axis [mm]")
            plt.title(title or "Bow-tie waist map")

        elif self.geometry == "linear":
            cavity_scan = np.arange(crystal_length + 0.5e-3, 120e-3, 0.5e-3)
            roc_scan = np.arange(10e-3, 150e-3, 0.5e-3)
            mesh_cavity, mesh_roc = np.meshgrid(cavity_scan, roc_scan)
            q = estimate_q_sagittal(mesh_roc, mesh_cavity, crystal_length, n_crystal)
            waist_um = beam_waist_from_q(q, wavelength, refractive_index=n_crystal) * 1e6
            plt.contourf(mesh_cavity * 1e3, mesh_roc * 1e3, waist_um)
            plt.xlabel(x_label or "Cavity length [mm]")
            plt.ylabel(y_label or "RoC [mm]")
            plt.title(title or "Linear cavity waist map")

        elif self.geometry == "triangle":
            if mesh_x is None or mesh_y is None or radius_of_curvature is None:
                raise ValueError("triangle waist plot requires mesh_x, mesh_y, and radius_of_curvature")
            q = estimate_q_sagittal(mesh_x, mesh_y, crystal_length, radius_of_curvature, n_crystal)
            waist_um = beam_waist_from_q(q, wavelength, refractive_index=n_crystal) * 1e6
            plt.contourf(mesh_x * 1e3, mesh_y * 1e3, waist_um)
            plt.xlabel(x_label or "Triangle width [mm]")
            plt.ylabel(y_label or "Triangle height [mm]")
            plt.title(title or "Triangle cavity waist map")

        elif self.geometry == "hemilithic":
            air_scan = np.arange(0.5e-3, 120e-3, 0.5e-3)
            roc_scan = np.arange(10e-3, 150e-3, 0.5e-3)
            mesh_air, mesh_roc = np.meshgrid(air_scan, roc_scan)
            q = estimate_q_sagittal(mesh_roc, mesh_air, crystal_length, n_crystal)
            waist_um = beam_waist_from_q(q, wavelength, refractive_index=n_crystal) * 1e6
            plt.contourf(mesh_air * 1e3, mesh_roc * 1e3, waist_um)
            plt.xlabel(x_label or "Air gap [mm]")
            plt.ylabel(y_label or "RoC [mm]")
            plt.title(title or "Hemilithic cavity waist map")

        else:
            raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")

        plt.colorbar(label="waist [um]")
        return fig


__all__ = ["CavityPlotter", "print_geometry_ascii"]
