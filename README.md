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

## Modeling details – Crystal ABCD matrices

The crystal section is modeled using **decoupled ABCD elements**, rather than a single “plane dielectric slab” matrix. The round-trip construction explicitly separates:

- Propagation segments (using the standard propagation matrix)
- Dielectric interface(s)

**Important convention:** in this ray-vector convention, propagation through free space and through a uniform medium of constant refractive index uses the **same** propagation matrix. The effect of the refractive index is accounted for via the **dielectric interface matrices** (which scale the ray angle), and when converting the q-parameter into a beam waist (the waist formula includes the refractive index).

This modular approach makes the cavity construction transparent and allows easy extension (e.g., inclusion of curved interfaces or thermal lensing).

---

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/ldbonavena/opo-cavity-squeezing-sim.git
cd opo-cavity-squeezing-sim
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the code

Select the desired geometry inside `src/cavity_geometry.py` by setting:

```python
GEOMETRY = "bowtie"      # or "linear" or "hemilithic"
```

Then run:

```bash
python src/cavity_geometry.py
```

## Examples
For now, geometry is selected inside `src/cavity_geometry.py` via `GEOMETRY`.

- Bow-tie: set `GEOMETRY = "bowtie"`
- Linear: set `GEOMETRY = "linear"`
- Hemilithic: set `GEOMETRY = "hemilithic"`

Then run:

python src/cavity_geometry.py

---

