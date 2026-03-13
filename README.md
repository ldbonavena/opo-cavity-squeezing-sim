# OPO Cavity Simulation Toolkit

This repository provides a modular simulation toolkit for designing and analyzing optical cavities used in Optical Parametric Oscillators (OPO) and squeezing experiments.  
The code is based on paraxial Gaussian optics and ABCD matrix formalism and is organized as a small Python package to keep cavity geometry, crystal physics, and OPO models clearly separated.

The current implementation focuses on **cavity geometry and eigenmode analysis**, producing the optical parameters required for later nonlinear and quantum simulations.

---

# Project Structure

```
src/
    cavity/            # Cavity geometry and optical mode analysis
        cavity_geometry.py
        cavity_abcd.py
        cavity_analysis.py
        cavity_plotter.py
        cavity_workflow.py
        optics_abcd.py

    crystal/           # Crystal physics and nonlinear interaction modeling
        crystal_workflow.py
        crystal_materials.py
        crystal_phase_matching.py
        crystal_mode_matching.py
        crystal_boyd_kleinman.py
        crystal_plotter.py
        crystal_thermo.py

    common/            # Shared utilities (constants, helpers)

    opo/               # Future nonlinear and squeezing simulations
```

The project is structured so that each module has a clear responsibility:

- **cavity/**: geometry definition, ABCD matrices, stability analysis, beam modes
- **crystal/**: crystal material models, phase‑matching calculations, mode matching, and focused‑beam nonlinear interaction (Boyd–Kleinman theory)
- **opo/**: nonlinear OPO dynamics and squeezing simulations (future work)
- **common/**: reusable helpers and constants

---

# What the cavity simulation computes

The cavity module performs a complete **Gaussian eigenmode analysis** of several cavity geometries.

Supported geometries:

- Bow‑tie (ring) cavity
- Linear standing‑wave cavity
- Hemilithic cavity
- Triangle cavity

For a given geometry the code computes:

- Stability maps using the cavity **m‑factor** (stable when |m| < 1)
- Cavity eigenmodes via the **round‑trip ABCD matrix**
- Beam waist maps across parameter space
- Single‑point cavity evaluation including:

  - m‑factor(s)
  - q‑parameter(s)
  - beam waist inside the crystal
  - geometric round‑trip length
  - optical round‑trip length

- Derived cavity quantities:

  - Free Spectral Range (FSR)
  - decay rates: `kappa_ext`, `kappa_loss`, `kappa_total`
  - escape efficiency
  - detuning
  - Gouy phases

These quantities are the required inputs for later simulations of:

- nonlinear gain
- OPO threshold
- squeezing spectra

---

# Output files

All simulation outputs are written to the local `results/` directory.  
This folder is intended for locally generated simulation data and is not required to be version‑controlled.

```
results/
    <geometry>/
        cavity_simulation_output.json
        stability_map.png
        waist_map.png
```

Each run produces:

**cavity_simulation_output.json**

Contains all relevant simulation inputs and computed parameters, including:

- cavity geometry parameters
- q‑parameters
- waist sizes
- FSR
- decay rates
- escape efficiency
- Gouy phases

This JSON file is intended to be loaded by the next simulation layer (crystal physics or OPO model).

**stability_map.png**

Visualization of cavity stability across the scanned parameter space.

**waist_map.png**

Beam waist map corresponding to the same parameter scan.

---

# Modeling approach

The cavity model uses the **ABCD matrix formalism** for paraxial Gaussian beams.

The crystal section is modeled using **decoupled ABCD elements** rather than a single dielectric slab matrix:

- free‑space propagation
- dielectric interfaces
- crystal propagation

In this convention, propagation in free space and in a uniform refractive index medium use the **same propagation matrix**.  
The refractive index enters through the dielectric interface matrices and through the waist calculation.

This modular construction makes it straightforward to extend the simulation with:

- thermal lensing
- curved interfaces
- nonlinear effects

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ldbonavena/opo-cavity-squeezing-sim.git
cd opo-cavity-squeezing-sim
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the cavity simulation

The main entry point is:

```
src/cavity/cavity_geometry.py
```

Select the geometry by editing:

```python
GEOMETRY = "bowtie"
```

Available options:

```
bowtie
linear
triangle
hemilithic
```

Run the simulation:

```bash
python src/cavity/cavity_geometry.py
```

The script will:

1. Print a short description of the selected cavity geometry
2. Display an ASCII sketch of the cavity
3. Generate stability and waist maps
4. Evaluate a representative cavity configuration
5. Export all parameters to `results/<geometry>/cavity_simulation_output.json`

---

# Examples

Example scripts are available in the `examples/` folder:

```
examples/
    run_bowtie.py
    run_linear.py
    run_hemilithic.py
```

These scripts run predefined cavity configurations for quick testing.

---

# Future extensions

Planned modules include:

- crystal thermo‑optic modeling
- phase‑matching calculations
- nonlinear coupling estimation
- OPO threshold simulations
- squeezing spectrum computation

The modular structure of the repository is designed to allow these layers to build directly on the cavity results exported by the current code.

---

# License

See the `LICENSE` file for details.
