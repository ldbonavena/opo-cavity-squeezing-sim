"""Cavity-specific ABCD round-trip matrix builders."""

from __future__ import annotations

import sympy as sp

from optics_abcd import Abcd, validate_nonnegative, validate_plane


def bowtie_diagonal(long_axis, short_axis, incidence_angle):
    """Return bow-tie diagonal segment length from long/short axes and AOI."""
    return (long_axis + short_axis) / 2 * sp.cos(incidence_angle)


class CavityAbcdBuilder:
    """Factory for round-trip cavity ABCD matrices across supported geometries."""

    @staticmethod
    def bowtie_roundtrip(
        long_axis,
        short_axis,
        crystal_length,
        radius_of_curvature,
        refractive_index,
        incidence_angle,
        plane: str = "sagittal",
    ):
        """Round-trip matrix for a bow-tie cavity from crystal center."""
        validate_plane(plane)
        mirror = Abcd.mirror
        diagonal = bowtie_diagonal(long_axis, short_axis, incidence_angle)
        return Abcd.chain(
            Abcd.propagation(crystal_length / 2),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation((short_axis - crystal_length) / 2),
            mirror(radius_of_curvature, incidence_angle, plane),
            Abcd.propagation(long_axis + 2 * diagonal),
            mirror(radius_of_curvature, incidence_angle, plane),
            Abcd.propagation((short_axis - crystal_length) / 2),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(crystal_length / 2),
        )

    @staticmethod
    def linear_roundtrip(
        cavity_length,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
    ):
        """Round-trip matrix for a linear cavity with centered crystal."""
        air_total = cavity_length - crystal_length
        validate_nonnegative(air_total, "cavity_length - crystal_length")
        air_half = air_total / 2
        return Abcd.chain(
            Abcd.propagation(air_half),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_half),
            Abcd.mirror(radius_2, 0, "tangential"),
            Abcd.propagation(air_half),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_half),
            Abcd.mirror(radius_1, 0, "tangential"),
        )

    @staticmethod
    def hemilithic_roundtrip(
        air_gap,
        crystal_length,
        mirror_radius,
        refractive_index,
    ):
        """Round-trip matrix for a hemilithic cavity."""
        validate_nonnegative(air_gap, "air_gap")
        return Abcd.chain(
            Abcd.propagation(air_gap),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_gap),
            Abcd.mirror(mirror_radius, 0, "tangential"),
        )

    @staticmethod
    def triangle_roundtrip(
        width,
        height,
        crystal_length,
        radius_of_curvature,
        refractive_index,
        plane: str = "sagittal",
    ):
        """Round-trip matrix for a triangular cavity with crystal in base arm."""
        validate_plane(plane)
        diagonal = sp.sqrt((width / 2) ** 2 + height**2)
        fold_angle = sp.asin(height / diagonal)
        side_length = (width - crystal_length) / 2
        validate_nonnegative(side_length, "width - crystal_length")
        return Abcd.chain(
            Abcd.propagation(crystal_length / 2),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(side_length),
            Abcd.mirror(radius_of_curvature, fold_angle / 2, plane),
            Abcd.propagation(2 * diagonal),
            Abcd.mirror(radius_of_curvature, fold_angle / 2, plane),
            Abcd.propagation(side_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(crystal_length / 2),
        )

    @staticmethod
    def build(geometry: str, **kwargs):
        """Build a round-trip matrix for a named geometry."""
        if geometry == "bowtie":
            return CavityAbcdBuilder.bowtie_roundtrip(
                kwargs["long_axis"],
                kwargs["short_axis"],
                kwargs["crystal_length"],
                kwargs["radius_of_curvature"],
                kwargs["refractive_index"],
                kwargs["incidence_angle"],
                kwargs.get("plane", "sagittal"),
            )
        if geometry == "linear":
            r1 = kwargs.get("radius_1", kwargs.get("radius_of_curvature"))
            r2 = kwargs.get("radius_2", kwargs.get("radius_of_curvature"))
            if r1 is None or r2 is None:
                raise ValueError("linear geometry requires radius_1/radius_2 or radius_of_curvature")
            return CavityAbcdBuilder.linear_roundtrip(
                kwargs["cavity_length"],
                kwargs["crystal_length"],
                r1,
                r2,
                kwargs["refractive_index"],
            )
        if geometry == "hemilithic":
            radius = kwargs.get("mirror_radius", kwargs.get("radius_of_curvature"))
            if radius is None:
                raise ValueError("hemilithic geometry requires mirror_radius or radius_of_curvature")
            return CavityAbcdBuilder.hemilithic_roundtrip(
                kwargs["air_gap"],
                kwargs["crystal_length"],
                radius,
                kwargs["refractive_index"],
            )
        if geometry == "triangle":
            return CavityAbcdBuilder.triangle_roundtrip(
                kwargs["width"],
                kwargs["height"],
                kwargs["crystal_length"],
                kwargs["radius_of_curvature"],
                kwargs["refractive_index"],
                kwargs.get("plane", "sagittal"),
            )
        raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', or 'hemilithic'")


__all__ = [
    "bowtie_diagonal",
    "CavityAbcdBuilder",
]
