# OPO Cavity Geometry Toolkit

This repository provides a classical cavity geometry and eigenmode analysis tool based on ABCD matrices (paraxial Gaussian optics). The main script is `src/cavity_geometry.py`.

Supported geometries (selected via the `GEOMETRY` variable inside the script):
- Bow-tie (ring) cavity
- Linear standing-wave cavity (symmetric mirrors)
- Hemilithic cavity (curved mirror + air gap + crystal with HR-coated back face)

## What the code computes

- Stability maps using the cavity m-factor (stable when |m| < 1)
- Cavity eigenmode (q-parameter) from the round-trip ABCD matrix
- Beam waist maps derived from the q-parameter
- Single-point evaluation of m-factor, q-parameter, and beam waist in the crystal
- Free Spectral Range (FSR) from the optical round-trip length (including crystal refractive index)
- Gouy phase per round trip (including sagittal/tangential behavior for bow-tie cavities)

The outputs are intended to provide the geometric and mode parameters required as inputs for later squeezing/OPO quantum simulations.

---

## Examples
For now, geometry is selected inside `src/cavity_geometry.py` via `GEOMETRY`.

- Bow-tie: set `GEOMETRY = "bowtie"`
- Linear: set `GEOMETRY = "linear"`
- Hemilithic: set `GEOMETRY = "hemilithic"`

Then run:

python src/cavity_geometry.py

---

