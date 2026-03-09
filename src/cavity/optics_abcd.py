"""Generic ABCD matrix utilities for paraxial optics."""

from __future__ import annotations

import sympy as sp


def validate_nonnegative(value, name: str) -> None:
    """Raise ValueError if a numeric value is negative."""
    if isinstance(value, sp.Basic):
        if value.is_number and float(sp.N(value)) < 0:
            raise ValueError(f"{name} must be >= 0")
    else:
        if float(value) < 0:
            raise ValueError(f"{name} must be >= 0")


def validate_plane(plane: str) -> None:
    """Validate sagittal/tangential plane selection."""
    if plane not in {"sagittal", "tangential"}:
        raise ValueError("plane must be 'sagittal' or 'tangential'")


class Abcd:
    """Static helpers for ABCD matrix elements and operations."""

    @staticmethod
    def propagation(length):
        """Free-space (or uniform-medium) propagation matrix."""
        return sp.Matrix([[1, length], [0, 1]])

    @staticmethod
    def planar_interface(n1, n2):
        """Planar dielectric interface matrix."""
        return sp.Matrix([[1, 0], [0, n1 / n2]])

    @staticmethod
    def curved_interface(n1, n2, radius):
        """Curved dielectric interface matrix."""
        return sp.Matrix([[1, 0], [(n1 - n2) / (n2 * radius), n1 / n2]])

    @staticmethod
    def thin_lens(focal_length):
        """Thin lens matrix."""
        return sp.Matrix([[1, 0], [-1 / focal_length, 1]])

    @staticmethod
    def mirror(radius_of_curvature, incidence_angle=0.0, plane: str = "tangential"):
        """Spherical mirror reflection matrix with optional astigmatism plane."""
        validate_plane(plane)
        if plane == "tangential":
            radius_eff = radius_of_curvature * sp.cos(incidence_angle)
        else:
            radius_eff = radius_of_curvature / sp.cos(incidence_angle)
        return sp.Matrix([[1, 0], [-2 / radius_eff, 1]])

    @staticmethod
    def chain(*elements, simplify: bool = True):
        """Multiply a sequence of ABCD elements in order."""
        matrix = sp.eye(2)
        for element in elements:
            matrix = matrix @ element
        return sp.simplify(matrix) if simplify else matrix

    @staticmethod
    def parameters(matrix):
        """Extract (A, B, C, D) from a 2x2 ABCD matrix."""
        return matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]


__all__ = [
    "Abcd",
    "validate_nonnegative",
    "validate_plane",
]
