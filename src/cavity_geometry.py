# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

import matplotlib as mpl
from matplotlib import rc
from cycler import cycler

# %%
# Symbolic parameters

short_axis = sp.symbols("l", positive=True, real=True)
long_axis = sp.symbols("L", positive=True, real=True)
crystal_length = sp.symbols("d_crystal", positive=True, real=True)
n_crystal = sp.symbols("n_crystal", positive=True, real=True)
RoC = sp.symbols("RoC", positive=True, real=True)
theta_AOI = sp.symbols("theta_AOI", positive=True, real=True)
RoC_lin = sp.symbols("RoC_lin", positive=True, real=True)
L_cav = sp.symbols("L_cav", positive=True, real=True)
RoC_hemi = sp.symbols("RoC_hemi", positive=True, real=True)
L_air = sp.symbols("L_air", positive=True, real=True)



# %%
# Geometry selection

# Choose: "bowtie", "linear", or "hemilithic" 
GEOMETRY = "hemilithic"

# Numeric parameters

c_num = 299792458.0

f_crystal_length = 16e-3
f_n_crystal = 1.82

f_RoC = 50e-3
f_wavelength = 1550e-9

# --- Bow-tie cavity parameters (used only if GEOMETRY == "bowtie") ---
f_theta_AOI = 6 * np.pi / 180.0
f_short_axis = np.arange(56e-3, 71e-3, 0.01e-3)
f_long_axis = np.arange(70e-3, 120e-3, 0.5e-3)

# Mesh 2D for bow-tie
mesh_short_axis, mesh_long_axis = np.meshgrid(f_short_axis, f_long_axis)

def get_diagonal(long_axis, short_axis, theta_AOI):
    diagonal = (long_axis + short_axis) / 2 * sp.cos(theta_AOI)
    return diagonal

# --- Linear cavity parameters (used only if GEOMETRY == "linear") ---
# L_cav is the geometric mirror-to-mirror separation (round-trip geometric length is 2*L_cav)
f_L_cav = 100e-3

# --- Hemilithic cavity parameters (used only if GEOMETRY == "hemilithic") ---
# Geometry: curved mirror -> air gap -> crystal with HR-coated back face
f_L_air = 20e-3

# %%
# ABCD matrices for single optical elements

def mirror_ref_RoC_tangential(effective_RoC, theta_AOI):
    R_t = effective_RoC * sp.cos(theta_AOI)
    matrix = [[1, 0], [-2 / R_t, 1]]
    return sp.Matrix(matrix)


def mirror_ref_RoC_sagittal(effective_RoC, theta_AOI):
    R_s = effective_RoC / sp.cos(theta_AOI)
    matrix = [[1, 0], [-2 / R_s, 1]]
    return sp.Matrix(matrix)


def free_propagation(length):
    matrix = [[1, length], [0, 1]]
    return sp.Matrix(matrix)


def dielectric_interface(ref_idx1, ref_idx2):
    matrix1 = sp.Matrix([[1, 0], [0, ref_idx1 / ref_idx2]])
    return matrix1


def dielectric_interface_curved(ref_idx1, ref_idx2, RoC):
    matrix1 = sp.Matrix([[1, 0], [(ref_idx1 - ref_idx2) / (ref_idx2 * RoC), ref_idx1 / ref_idx2]])
    return matrix1


def propagation_in_medium(length, refractive_index):
    matrix = [[1, length / refractive_index], [0, 1]]
    return sp.Matrix(matrix)


# %%
# ABCD matrices for different geometries
def get_abcd_matrix_bowtie(long_axis, short_axis, diagonal, crystal_length, RoC, n_crystal, plane="sagittal"):
    if plane == "sagittal":
        mirror_fn = mirror_ref_RoC_sagittal
    elif plane == "tangential":
        mirror_fn = mirror_ref_RoC_tangential
    else:
        raise ValueError("plane must be 'sagittal' or 'tangential'")

    M = sp.eye(2)
    # Start at crystal center: propagate half-crystal in medium, exit to air.
    M = M @ free_propagation(crystal_length / 2)
    M = M @ dielectric_interface(1, n_crystal )
    M = M @ free_propagation( (short_axis - crystal_length) / 2)
    M = M @ mirror_fn(RoC, theta_AOI)
    M = M @ free_propagation(long_axis+ 2 * diagonal)
    M = M @ mirror_fn(RoC, theta_AOI)
    M = M @ free_propagation( (short_axis- crystal_length) / 2)
    M = M @ dielectric_interface(n_crystal, 1 )
    M = M @ free_propagation(crystal_length / 2)
    return sp.simplify(M)


def get_abcd_matrix_sagittal(long_axis, short_axis, diagonal, crystal_length, RoC, n_crystal):
    return get_abcd_matrix_bowtie(long_axis, short_axis, diagonal, crystal_length, RoC, n_crystal, plane="sagittal")


def get_abcd_matrix_tangential(long_axis, short_axis, diagonal, crystal_length, RoC, n_crystal):
    return get_abcd_matrix_bowtie(long_axis, short_axis, diagonal, crystal_length, RoC, n_crystal, plane="tangential")

def get_abcd_matrix_linear(RoC1, RoC2, L_cav, crystal_length, n_crystal, theta_AOI=0.0):
    # Split the cavity into: air/medium halves around a centered crystal
    L_air_total = L_cav - crystal_length
    # Allow symbolic values during lambdify; validate numeric calls.
    if isinstance(L_air_total, sp.Basic):
        if L_air_total.is_number and float(sp.N(L_air_total)) < 0:
            raise ValueError("crystal_length must be <= L_cav")
    else:
        if float(L_air_total) < 0:
            raise ValueError("crystal_length must be <= L_cav")
    L_air_half = L_air_total / 2

    M = sp.eye(2)

    # By default the AOI is zero, so sagittal and tangential are the same in linear cavity
    # Start just after reflection on Mirror1
    M = M @ free_propagation(L_air_half)
    M = M @ dielectric_interface(1, n_crystal)
    M = M @ free_propagation(crystal_length)
    M = M @ dielectric_interface(n_crystal, 1)
    M = M @ free_propagation(L_air_half)

    # Reflect on Mirror2 (theta=0 => sagittal=tangential)
    M = M @ mirror_ref_RoC_tangential(RoC2, 0)

    # Propagate back
    M = M @ free_propagation(L_air_half)
    M = M @ dielectric_interface(1, n_crystal)
    M = M @ free_propagation(crystal_length)
    M = M @ dielectric_interface(n_crystal, 1)
    M = M @ free_propagation(L_air_half)

    # Reflect on Mirror1 to complete the round trip
    M = M @ mirror_ref_RoC_tangential(RoC1, 0)

    return sp.simplify(M)


def get_abcd_matrix_hemilithic(RoC_mirror, L_air, crystal_length, n_crystal, theta_AOI=0.0):
    if isinstance(L_air, sp.Basic):
        if L_air.is_number and float(sp.N(L_air)) < 0:
            raise ValueError("L_air must be >= 0")
    else:
        if float(L_air) < 0:
            raise ValueError("L_air must be >= 0")

    M = sp.eye(2)

    # Forward path: mirror -> air -> crystal
    M = M @ free_propagation(L_air)
    M = M @ dielectric_interface(1, n_crystal)
    M = M @ free_propagation(crystal_length)

    # HR coated back face is planar: reflection matrix is identity in ABCD.

    # Return path: crystal -> air -> mirror
    # By default the AOI is zero for the hemilithic case

    M = M @ free_propagation(crystal_length)
    M = M @ dielectric_interface(n_crystal, 1)
    M = M @ free_propagation(L_air)
    M = M @ mirror_ref_RoC_tangential(RoC_mirror, 0)

    return sp.simplify(M)


# %%
# Helper functions to extract ABCD parameters and compute stability and q-parameter
def get_abcd_param(abcd_matrix):
    A = abcd_matrix[0, 0]
    B = abcd_matrix[0, 1]
    C = abcd_matrix[1, 0]
    D = abcd_matrix[1, 1]
    return A, B, C, D


def get_cavity_stability(abcd_matrix):
    A, _, _, D = get_abcd_param(abcd_matrix)
    m_factor = (A + D) / 2
    return m_factor

# m-factor helpers

def get_m_factor(abcd_sp_function, long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal):
    diagonal = get_diagonal(long_axis, short_axis, theta_AOI)
    matrix = abcd_sp_function(long_axis, short_axis, diagonal, crystal_length, RoC, n_crystal)
    m_factor = get_cavity_stability(matrix)
    return m_factor


# %%
# Geometry-dependent lambdify: m-factor

if GEOMETRY == "bowtie":
    estimate_m_factor_s = sp.lambdify(
        (long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal),
        get_m_factor(get_abcd_matrix_sagittal, long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal),
        modules="numpy",
        cse=True,
    )

    estimate_m_factor_t = sp.lambdify(
        (long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal),
        get_m_factor(get_abcd_matrix_tangential, long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal),
        modules="numpy",
        cse=True,
    )

elif GEOMETRY == "linear":
    # Symmetric linear cavity: single RoC for both mirrors
    def _m_linear(RoC_, L_cav_, crystal_length_, n_crystal_):
        M = get_abcd_matrix_linear(RoC_, RoC_, L_cav_, crystal_length_, n_crystal_, theta_AOI=0.0)
        return get_cavity_stability(M)

    estimate_m_factor_s = sp.lambdify(
        (RoC_lin, L_cav, crystal_length, n_crystal),
        _m_linear(RoC_lin, L_cav, crystal_length, n_crystal),
        modules="numpy",
        cse=True,
    )
    estimate_m_factor_t = estimate_m_factor_s

elif GEOMETRY == "hemilithic":
    def _m_hemilithic(RoC_, L_air_, crystal_length_, n_crystal_):
        M = get_abcd_matrix_hemilithic(RoC_, L_air_, crystal_length_, n_crystal_, theta_AOI=0.0)
        return get_cavity_stability(M)

    estimate_m_factor_s = sp.lambdify(
        (RoC_hemi, L_air, crystal_length, n_crystal),
        _m_hemilithic(RoC_hemi, L_air, crystal_length, n_crystal),
        modules="numpy",
        cse=True,
    )
    estimate_m_factor_t = estimate_m_factor_s

else:
    raise ValueError("GEOMETRY must be 'bowtie', 'linear', or 'hemilithic'")


# %%
# Stability plots

if GEOMETRY == "bowtie":
    plt.figure()
    plt.contourf(
        mesh_short_axis * 1e3,
        mesh_long_axis * 1e3,
        np.abs(estimate_m_factor_s(mesh_long_axis, mesh_short_axis, f_theta_AOI, f_crystal_length, f_RoC, f_n_crystal)) < 1,
    )
    plt.xlabel("Short axis [mm]")
    plt.ylabel("Long axis [mm]")
    plt.title("Bow-tie stability (|m|<1)")
    plt.colorbar(label="stable")
    plt.grid(True)

elif GEOMETRY == "linear":
    # Linear cavity: 2D stability map vs (L_cav, RoC)
    L_cav_scan = np.arange(f_crystal_length + 0.5e-3, 120e-3, 0.5e-3)
    RoC_scan = np.arange(10e-3, 150e-3, 0.5e-3)
    mesh_L_cav, mesh_RoC = np.meshgrid(L_cav_scan, RoC_scan)
    stable_map = np.abs(estimate_m_factor_s(mesh_RoC, mesh_L_cav, f_crystal_length, f_n_crystal)) < 1

    plt.figure()
    plt.contourf(mesh_L_cav * 1e3, mesh_RoC * 1e3, stable_map)
    plt.xlabel("Cavity length [mm]")
    plt.ylabel("RoC [mm]")
    plt.title("Linear cavity stability (|m|<1)")
    plt.colorbar(label="stable")
    plt.grid(True)

else:
    # Hemilithic cavity: 2D stability map vs (L_air, RoC)
    L_air_scan = np.arange(0.5e-3, 120e-3, 0.5e-3)
    RoC_scan = np.arange(10e-3, 150e-3, 0.5e-3)
    mesh_L_air, mesh_RoC = np.meshgrid(L_air_scan, RoC_scan)
    stable_map = np.abs(estimate_m_factor_s(mesh_RoC, mesh_L_air, f_crystal_length, f_n_crystal)) < 1

    plt.figure()
    plt.contourf(mesh_L_air * 1e3, mesh_RoC * 1e3, stable_map)
    plt.xlabel("Air gap [mm]")
    plt.ylabel("RoC [mm]")
    plt.title("Hemilithic cavity stability (|m|<1)")
    plt.colorbar(label="stable")
    plt.grid(True)


# %%
# q-parameter helpers

def get_cavity_q(abcd_matrix):
    A, B, C, D = get_abcd_param(abcd_matrix)
    q = -1 * (D - A) / (2 * C) + sp.I * sp.sqrt(1 - ((D + A) / 2) ** 2) / sp.Abs(C)
    return q


def get_q_fsr_condition(abcd_sp_function, long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal):
    diagonal = get_diagonal(long_axis, short_axis, theta_AOI)
    matrix = abcd_sp_function(long_axis, short_axis, diagonal, crystal_length, RoC, n_crystal)
    q = get_cavity_q(matrix)
    return q


# %%
# Geometry-dependent q

if GEOMETRY == "bowtie":
    estimate_q_sagittal = sp.lambdify(
        (long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal),
        get_q_fsr_condition(get_abcd_matrix_sagittal, long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal),
        modules="numpy",
        cse=True,
    )

    estimate_q_tangential = sp.lambdify(
        (long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal),
        get_q_fsr_condition(get_abcd_matrix_tangential, long_axis, short_axis, theta_AOI, crystal_length, RoC, n_crystal),
        modules="numpy",
        cse=True,
    )

elif GEOMETRY == "linear":
    def _q_linear(RoC_, L_cav_, crystal_length_, n_crystal_):
        M = get_abcd_matrix_linear(RoC_, RoC_, L_cav_, crystal_length_, n_crystal_, theta_AOI=0.0)
        return get_cavity_q(M)

    estimate_q_sagittal = sp.lambdify(
        (RoC_lin, L_cav, crystal_length, n_crystal),
        _q_linear(RoC_lin, L_cav, crystal_length, n_crystal),
        modules="numpy",
        cse=True,
    )
    estimate_q_tangential = estimate_q_sagittal

else:
    def _q_hemilithic(RoC_, L_air_, crystal_length_, n_crystal_):
        M = get_abcd_matrix_hemilithic(RoC_, L_air_, crystal_length_, n_crystal_, theta_AOI=0.0)
        return get_cavity_q(M)

    estimate_q_sagittal = sp.lambdify(
        (RoC_hemi, L_air, crystal_length, n_crystal),
        _q_hemilithic(RoC_hemi, L_air, crystal_length, n_crystal),
        modules="numpy",
        cse=True,
    )
    estimate_q_tangential = estimate_q_sagittal


def estimate_beam_waist(q_factor, wavelength, refractive_index=1):
    q_img = np.imag(q_factor)
    radius = np.sqrt(1.0 * wavelength * q_img / (refractive_index * np.pi))
    return radius


# %%
# Plotting beam waist heatmap
if GEOMETRY == "bowtie":
    f_q_sagittal_waist = estimate_q_sagittal(
        mesh_long_axis, mesh_short_axis, f_theta_AOI, f_crystal_length, f_RoC, f_n_crystal
    )

    plt.figure()
    plt.contourf(
        mesh_short_axis * 1e3,
        mesh_long_axis * 1e3,
        estimate_beam_waist(f_q_sagittal_waist, f_wavelength, refractive_index=f_n_crystal) * 1e6,
    )
    plt.xlabel("Short axis [mm]")
    plt.ylabel("Long axis [mm]")
    plt.title("Bow-tie waist map")
    plt.colorbar(label="waist [um]")
elif GEOMETRY == "linear":
    L_cav_scan = np.arange(f_crystal_length + 0.5e-3, 120e-3, 0.5e-3)
    RoC_scan = np.arange(10e-3, 150e-3, 0.5e-3)
    mesh_L_cav, mesh_RoC = np.meshgrid(L_cav_scan, RoC_scan)
    f_q_sagittal_waist = estimate_q_sagittal(mesh_RoC, mesh_L_cav, f_crystal_length, f_n_crystal)

    plt.figure()
    plt.contourf(
        mesh_L_cav * 1e3,
        mesh_RoC * 1e3,
        estimate_beam_waist(f_q_sagittal_waist, f_wavelength, refractive_index=f_n_crystal) * 1e6,
    )
    plt.xlabel("Cavity length [mm]")
    plt.ylabel("RoC [mm]")
    plt.title("Linear cavity waist map")
    plt.colorbar(label="waist [um]")
else:
    L_air_scan = np.arange(0.5e-3, 120e-3, 0.5e-3)
    RoC_scan = np.arange(10e-3, 150e-3, 0.5e-3)
    mesh_L_air, mesh_RoC = np.meshgrid(L_air_scan, RoC_scan)
    f_q_sagittal_waist = estimate_q_sagittal(mesh_RoC, mesh_L_air, f_crystal_length, f_n_crystal)

    plt.figure()
    plt.contourf(
        mesh_L_air * 1e3,
        mesh_RoC * 1e3,
        estimate_beam_waist(f_q_sagittal_waist, f_wavelength, refractive_index=f_n_crystal) * 1e6,
    )
    plt.xlabel("Air gap [mm]")
    plt.ylabel("RoC [mm]")
    plt.title("Hemilithic cavity waist map")
    plt.colorbar(label="waist [um]")


# %%
# Single-point evaluation

if GEOMETRY == "bowtie":
    short_axis_val = 68e-3
    long_axis_val = 90e-3

    # get_diagonal uses sympy, so force a numeric float here
    _diagonal_sym = get_diagonal(long_axis_val, short_axis_val, f_theta_AOI)
    diagonal_val = float(sp.N(_diagonal_sym))

    m_x_sym = get_m_factor(get_abcd_matrix_sagittal, long_axis_val, short_axis_val, theta_AOI, f_crystal_length, f_RoC, f_n_crystal)
    m_y_sym = get_m_factor(get_abcd_matrix_tangential, long_axis_val, short_axis_val, theta_AOI, f_crystal_length, f_RoC, f_n_crystal)

    # evaluate symbolic expressions at the numeric AOI
    m_x = float(sp.N(m_x_sym.subs(theta_AOI, f_theta_AOI)))
    m_y = float(sp.N(m_y_sym.subs(theta_AOI, f_theta_AOI)))

    qs = estimate_q_sagittal(long_axis_val, short_axis_val, f_theta_AOI, f_crystal_length, f_RoC, f_n_crystal)
    qt = estimate_q_tangential(long_axis_val, short_axis_val, f_theta_AOI, f_crystal_length, f_RoC, f_n_crystal)

    print(
        f"Geometrical parameters: Long axis = {long_axis_val*1e3:.3f} mm, Short axis = {short_axis_val*1e3:.3f} mm, "
        f"diagonal = {diagonal_val*1e3:.3f} mm, AOI = {f_theta_AOI*180/np.pi:.3f} deg"
    )
    print(f"m_sagittal = {m_x:.6f}, m_tangential = {m_y:.6f}")
    print(f"qs parameter in the crystal: {qs:.5f}")
    print(f"qt parameter in the crystal: {qt:.5f}")

    cavity_length = float(long_axis_val + short_axis_val + 2 * diagonal_val)

elif GEOMETRY == "linear":
    # Linear cavity example point (symmetric)
    L_cav_val = 100e-3

    # Evaluate from symbolic expressions for consistency with bow-tie flow
    subs_linear = {
        RoC_lin: f_RoC,
        L_cav: L_cav_val,
        crystal_length: f_crystal_length,
        n_crystal: f_n_crystal,
    }
    m_sym = _m_linear(RoC_lin, L_cav, crystal_length, n_crystal)
    q_sym = _q_linear(RoC_lin, L_cav, crystal_length, n_crystal)

    m_val = float(sp.N(m_sym.subs(subs_linear)))
    qs = complex(sp.N(q_sym.subs(subs_linear)))
    qt = qs

    print(f"Linear cavity parameters: L_cav = {L_cav_val*1e3:.3f} mm, RoC = {f_RoC*1e3:.3f} mm")
    print(f"m = {m_val:.6f} (stable if |m|<1)")
    print(f"q parameter at the crystal center: {qs:.5f}")

    cavity_length = float(2 * L_cav_val)  # geometric round-trip length for linear cavity
    optical_crystal_length = f_crystal_length

else:
    # Hemilithic cavity example point
    L_air_val = 20e-3

    # Evaluate from symbolic expressions for consistency with bow-tie flow
    subs_hemi = {
        RoC_hemi: f_RoC,
        L_air: L_air_val,
        crystal_length: f_crystal_length,
        n_crystal: f_n_crystal,
    }
    m_sym = _m_hemilithic(RoC_hemi, L_air, crystal_length, n_crystal)
    q_sym = _q_hemilithic(RoC_hemi, L_air, crystal_length, n_crystal)

    m_val = float(sp.N(m_sym.subs(subs_hemi)))
    qs = complex(sp.N(q_sym.subs(subs_hemi)))
    qt = qs

    print(f"Hemilithic cavity parameters: L_air = {L_air_val*1e3:.3f} mm, RoC = {f_RoC*1e3:.3f} mm")
    print(f"m = {m_val:.6f} (stable if |m|<1)")
    print(f"q parameter at crystal input face: {qs:.5f}")

    cavity_length = float(2 * (L_air_val + f_crystal_length))
    optical_crystal_length = 2 * f_crystal_length

if GEOMETRY == "bowtie":
    optical_crystal_length = f_crystal_length

# Waist estimate at the crystal (works for both)
w_um = estimate_beam_waist(qs, f_wavelength, refractive_index=f_n_crystal) * 1e6
print(f"Beam waist in crystal (from q): {w_um:.3f} um")


# %%
# Cavity FSR

L_optical = float(cavity_length + (f_n_crystal - 1.0) * optical_crystal_length)
fsr = float(c_num / L_optical)

print(f"Geometric cavity length: {cavity_length:.6f} m")
print(f"Optical round-trip length: {L_optical:.6f} m")
print(f"FSR: {fsr:.6f} Hz ({fsr/1e6:.6f} MHz)")


# %%
# Gouy phase accumulation

if GEOMETRY == "bowtie":
    psi_sagittal = np.arccos(m_x)
    psi_tangential = np.arccos(m_y)
    print(f"Psi_sagittal = {psi_sagittal:.6f} rad ({np.degrees(psi_sagittal):.3f} deg)")
    print(f"Psi_tangential = {psi_tangential:.6f} rad ({np.degrees(psi_tangential):.6f} deg)")
else:
    psi_sagittal = np.arccos(float(m_val))
    psi_tangential = psi_sagittal


# %%
