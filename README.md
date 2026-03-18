# QPIT-SQZsim

A modular simulation framework for optical parametric oscillators (OPOs) and squeezed light generation, combining cavity geometry, crystal physics, and nonlinear interaction modeling.

The project is structured in layers:

cavity → crystal → OPO → squeezing

---

# Project Structure

```
QPIT-SQZsim/
    docs/
    results/        # local simulation outputs; not intended for version control
    src/
        cavity/     # Cavity geometry and optical mode analysis
            cavity_main.py
            cavity_workflow.py
            cavity_abcd.py
            cavity_analysis.py
            cavity_plotter.py
            optics_abcd.py

        crystal/    # Crystal physics and nonlinear interaction modeling
            crystal_main.py
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

For a given geometry the code computes:

- Stability maps (|m| < 1)
- Eigenmodes via the round-trip ABCD matrix
- Beam waist maps
- Single-point evaluation:
  - q-parameters
  - waist in the crystal
  - round-trip lengths (geometric and optical)

Derived quantities:

- Free Spectral Range (FSR)
- decay rates: `kappa_ext`, `kappa_loss`, `kappa_total`
- escape efficiency and detuning
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

This JSON file is used by the next simulation layer (crystal or OPO).

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

In the crystal section, propagation is modeled using decoupled ABCD elements instead of a single dielectric slab matrix:

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

## Installation

```bash
git clone <repository-url> QPIT-SQZsim
cd QPIT-SQZsim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# Running simulations

Simulations are executed per module (no global pipeline yet).

Run the simulation layers directly from their main entry points:

- `src/cavity/cavity_main.py`
- `src/crystal/crystal_main.py`

Typical usage:

```bash
python -m src.cavity.cavity_main
python -m src.crystal.crystal_main
```

Both scripts are designed to be run interactively in VS Code using `# %%` cells or as plain Python entry points.

The cavity script generates geometry-dependent cavity results under `results/<geometry>/cavity/`.  
The crystal script loads the cavity output from `results/<geometry>/cavity/cavity_simulation_output.json` and writes crystal results under `results/<geometry>/crystal/`.

Shared utilities used by both layers live in:

- `src/common/constants.py`
- `src/common/results_paths.py`

---

# Design principles

- Modular separation of physics layers (cavity → crystal → OPO)
- Explicit data flow via JSON outputs
- Interactive workflow using `# %%` cells
- Minimal dependencies and transparent numerical implementation

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

## Documentation

Detailed documentation is available in the `docs/` folder:

- architecture overview
- cavity theory and outputs
- crystal modeling

---

# License

See the `LICENSE` file for details.
