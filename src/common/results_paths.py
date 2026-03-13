"""Shared helpers for organizing simulation results on disk."""

from __future__ import annotations

from pathlib import Path


RESULT_SUBDIRS = ("cavity", "crystal", "opo")


def get_geometry_results_dir(geometry: str, results_root: str | Path | None = None) -> Path:
    """Return the base results directory for a given geometry."""
    if results_root is None:
        results_root = Path(__file__).resolve().parents[2] / "results"
    return Path(results_root) / geometry


def ensure_geometry_results_subdirs(geometry: str, results_root: str | Path | None = None) -> Path:
    """Create the standard cavity/crystal/opo subdirectories for a geometry."""
    geometry_dir = get_geometry_results_dir(geometry, results_root=results_root)
    for subdir_name in RESULT_SUBDIRS:
        (geometry_dir / subdir_name).mkdir(parents=True, exist_ok=True)
    return geometry_dir


def get_geometry_results_subdir(
    geometry: str,
    subdir_name: str,
    results_root: str | Path | None = None,
) -> Path:
    """Return one of the standard results subdirectories for a geometry."""
    geometry_dir = ensure_geometry_results_subdirs(geometry, results_root=results_root)
    return geometry_dir / subdir_name
