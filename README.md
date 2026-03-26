# QPIT-SQZsim

A modular simulation framework for optical parametric oscillators (OPOs) and squeezed light generation, combining cavity geometry, crystal design physics, and nonlinear interaction modeling.

The project is structured in layers:

cavity → crystal → OPO → squeezing

---

# Project Structure

```
QPIT-SQZsim/
    docs/           # Project documentation
        overview.md
        architecture.md
        cavity.md
        crystal.md
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
- **crystal/**: crystal design and analysis workflow including dispersion, phase matching, derived QPM poling, mode matching, and focused‑beam nonlinear interaction (Boyd–Kleinman theory)
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
            boyd_kleinman_master_map.png
            qpm_length_poling_map.png
            boyd_kleinman_analysis.png
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
- Boyd-Kleinman analysis payload
- cavity output reference used to build the crystal context

The crystal results directory now also includes:

- `boyd_kleinman_master_map.png`: universal `h_BK(\sigma,\xi)` map with the system reference operating point and the theoretical master-map optimum
- `qpm_length_poling_map.png`: normalized QPM / poling-length map with first-order QPM guide
- `boyd_kleinman_analysis.png`: system-specific BK sweep analysis around the current operating point

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

## Crystal Workflow

The crystal layer is now organized as a design-oriented OPO workflow.

In the default `design` mode, the user specifies the target wavelengths, crystal model, and a design temperature. The code then derives the required QPM poling period from the bulk phase-matching condition and uses that derived period for the downstream phase-matching scan, mode matching, and BK analysis.

An `analysis` mode is also available when you want to study a chosen crystal configuration directly using an explicit poling period.

The high-level crystal execution order is:

1. Load cavity context from `results/<geometry>/cavity/`
2. Derive the design poling period from wavelengths and design temperature, or use the configured analysis period
3. Scan phase matching versus temperature
4. Determine the operating temperature from the phase-matching optimum
5. Evaluate the crystal refractive index at that operating temperature
6. Compute mode matching
7. Run Boyd-Kleinman analysis
8. Build structured result and JSON output
9. Print a summary
10. Generate plots
11. Save outputs

The BK analysis is integrated into the structured result, the console summary, the exported JSON, and the saved plots. The current crystal plotting outputs are:

- BK master map
- QPM / poling-length map
- BK sweep analysis plot

The wavelength sweeps in the BK analysis use fixed pump wavelength and derive the idler from exact three-wave energy conservation.

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

# What Changed

Recent crystal updates:

- the main crystal workflow is now design-oriented rather than assuming a fixed input poling period
- the execution order now follows design poling -> phase matching -> operating-point mode matching -> BK analysis
- BK analysis is included in the summary, JSON output, and plots
- the crystal output now includes BK master-map and QPM / poling-length figures
- BK wavelength sweeps enforce energy conservation and the QPM guide is labeled as a first-order QPM guide

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
