# OPO Cavity Simulation Toolkit

This repository provides a modular simulation toolkit for designing and analyzing optical cavities used in Optical Parametric Oscillators (OPO) and squeezing experiments.  
The code is based on paraxial Gaussian optics and ABCD matrix formalism and is organized as a small Python package to keep cavity geometry, crystal physics, and OPO models clearly separated.

The current implementation focuses on **cavity geometry and eigenmode analysis**, producing the optical parameters required for later nonlinear and quantum simulations.

---

# Project Structure

```
opo-cavity-squeezing-sim/
    docs/
    results/        # local simulation outputs; not intended for version control
    src/
        cavity/     # Cavity geometry and optical mode analysis
            main.py
            cavity_workflow.py
            cavity_abcd.py
            cavity_analysis.py
            cavity_plotter.py
            optics_abcd.py

        crystal/    # Crystal physics and nonlinear interaction modeling
            main.py
            crystal_workflow.py
            crystal_materials.py
            crystal_phase_matching.py
            crystal_mode_matching.py
            crystal_boyd_kleinman.py
            crystal_plotter.py

        common/     # Shared utilities (constants, helpers)
            __init__.py
            constants.py
            results_paths.py
        opo/        # Future nonlinear and squeezing simulations

    LICENSE
    README.md
    requirements.txt
```

The project is structured so that each module has a clear responsibility:

- **cavity/**: geometry definition, ABCD matrices, stability analysis, beam modes
- **crystal/**: crystal material models, phase‑matching calculations, mode matching, and focused‑beam nonlinear interaction (Boyd–Kleinman theory)
- **opo/**: nonlinear OPO dynamics and squeezing simulations (future work)
- **common/**: reusable constants and shared helpers such as results-path management

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
The directory structure is created automatically when simulations are executed.

```
results/
    <geometry>/
        cavity/
            cavity_simulation_output.json
            stability_map.png
            waist_map.png
        crystal/
            crystal_simulation_output.json
            phase_matching_scan.png
            mode_matching_summary.png
        opo/
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

**crystal_simulation_output.json**

Contains the crystal-layer inputs derived from the cavity simulation together with:

- phase-matching scan results
- mode-matching summary values
- cavity output reference used to build the crystal context

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

# Running simulations

Run the simulation layers directly from their main entry points:

- `src/cavity/main.py`
- `src/crystal/main.py`

Typical usage:

```bash
python -m src.cavity.main
python -m src.crystal.main
```

Both scripts are designed to be run interactively in VS Code using `# %%` cells or as plain Python entry points.

The cavity script generates geometry-dependent cavity results under `results/<geometry>/cavity/`.  
The crystal script loads the cavity output from `results/<geometry>/cavity/cavity_simulation_output.json` and writes crystal results under `results/<geometry>/crystal/`.

Shared utilities used by both layers live in:

- `src/common/constants.py`
- `src/common/results_paths.py`

---

# Future extensions

Planned developments include:

- full thermo‑optic crystal models
- nonlinear coupling estimation
- OPO threshold simulations
- squeezing spectrum computation
- quantum noise and detection modeling

The modular structure of the repository is designed so that each layer (cavity → crystal → OPO) builds directly on the results exported by the previous stage.

---

# License

See the `LICENSE` file for details.
