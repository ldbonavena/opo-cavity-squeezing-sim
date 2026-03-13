"""Shared physical constants used across simulation modules."""

from __future__ import annotations

import numpy as np


C_M_PER_S = 299792458.0  # Speed of light in vacuum [m/s]

# Mathematical constants
PI = np.pi  # Archimedes' constant pi [-]

# Electromagnetic constants
EPSILON_0_F_PER_M = 8.8541878128e-12  # Vacuum permittivity [F/m]
MU_0_H_PER_M = 1.25663706212e-6  # Vacuum permeability [H/m]
Z0_OHM = 376.730313668  # Vacuum impedance [ohm]

# Quantum constants
H_J_S = 6.62607015e-34  # Planck constant [J s]
HBAR_J_S = 1.054571817e-34  # Reduced Planck constant [J s]

# Fundamental constants
K_B_J_PER_K = 1.380649e-23  # Boltzmann constant [J/K]
E_CHARGE_C = 1.602176634e-19  # Elementary charge [C]
