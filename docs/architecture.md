# Architecture

QPIT-SQZsim is organized as a staged simulation stack. The top-level scripts select parameters and orchestrate the run, while the lower-level modules implement the actual optics and nonlinear-physics calculations.

## `src/` Layout

### `src/cavity/`

Implements resonator geometry, round-trip ABCD matrices, Gaussian eigenmode extraction, and derived cavity quantities.

- `cavity_main.py`: cavity entry point and parameter selection
- `cavity_workflow.py`: high-level assembly of the cavity simulation
- `cavity_abcd.py`: geometry-specific round-trip matrix builders
- `cavity_analysis.py`: physics derived from the ABCD matrix (`m`, `q`, FSR, `kappa`, Gouy phase)
- `optics_abcd.py`: reusable ABCD elements
- `cavity_plotter.py`: stability and waist visualizations

### `src/crystal/`

Consumes cavity output and runs the crystal design-and-analysis workflow for the intended nonlinear interaction.

- `crystal_main.py`: entry point for the crystal layer
- `crystal_workflow.py`: orchestrates cavity-output loading, design poling, phase matching, mode matching, BK analysis, and export
- `crystal_materials.py`: refractive-index and thermo-optic helper functions
- `crystal_phase_matching.py`: `Delta k`, QPM, design poling, and temperature scans
- `crystal_mode_matching.py`: converts cavity beam data into focusing and overlap metrics
- `crystal_boyd_kleinman.py`: focused-beam overlap model and BK/QPM analysis helpers
- `crystal_plotter.py`: crystal result visualizations

### `src/common/`

Shared infrastructure used by multiple simulation layers.

- `constants.py`: physical constants
- `results_paths.py`: standardized result-directory creation

### `src/opo/`

Reserved for future OPO dynamics and squeezing calculations. The current architecture already leaves a clean handoff point for this layer.

## Execution Pattern

The code follows a consistent pattern:

`*_main.py -> workflow -> physics modules`

The entry-point scripts define a geometry and numerical parameters, then call workflow functions that assemble structured contexts and results. Those workflows delegate the physics details to smaller modules.

This is the core architectural choice of the project:

- the `*_main.py` entry points stay thin and readable
- workflow modules define the sequence of calculations
- lower-level physics modules stay reusable and easier to test or extend

## Cavity Data Flow

The cavity layer starts from geometry-dependent parameters and produces a JSON payload that captures both inputs and derived quantities.

High-level flow:

1. `src/cavity/cavity_main.py` chooses a geometry and parameter set.
2. `src/cavity/cavity_workflow.py` builds geometry-specific estimators.
3. `src/cavity/cavity_abcd.py` constructs the round-trip matrix for the chosen resonator.
4. `src/cavity/cavity_analysis.py` extracts the stable mode and computes derived figures.
5. `cavity_workflow.py` serializes the result to `cavity_simulation_output.json`.

The exported cavity JSON contains:

- general cavity inputs
- geometry-specific operating-point values
- `q_sagittal` and `q_tangential`
- `m_factor`
- `beam_waist_crystal_um`
- round-trip lengths
- `fsr_Hz`
- `kappa_ext_rad_s`, `kappa_loss_rad_s`, `kappa_total_rad_s`
- `escape_efficiency`
- `detuning_rad_s`
- Gouy phases

## Crystal Data Flow

The crystal layer is intentionally downstream of the cavity layer rather than coupled directly to cavity internals.

High-level flow:

1. `src/crystal/crystal_main.py` selects the crystal model, wavelengths, design temperature, and phase-matching mode.
2. `src/crystal/crystal_workflow.py` loads `results/<geometry>/cavity/cavity_simulation_output.json`.
3. The cavity JSON is converted into a `CrystalContext`.
4. In design mode, `crystal_phase_matching.py` derives the required QPM period from the chosen wavelengths and design temperature. In analysis mode, the configured period is used directly.
5. `crystal_phase_matching.py` scans temperature-dependent mismatch and QPM metrics to determine the operating temperature.
6. `crystal_mode_matching.py` converts cavity waist data into Rayleigh range, focusing parameter, and overlap factors using the refractive index at the phase-matching operating temperature.
7. `crystal_boyd_kleinman.py` assembles BK analysis products, including the BK master map, QPM / poling-length map, and system-specific BK sweeps.
8. The result is serialized to `results/<geometry>/crystal/crystal_simulation_output.json` together with the BK analysis payload and plot outputs.

Today, the crystal layer relies most directly on:

- `beam_waist_crystal_um`
- `crystal_length_m`
- `wavelength_m`
- `n_crystal`
- optionally the exported `q` parameters for richer downstream mode context

This output is the intended handoff to a future OPO layer, where nonlinear coupling and cavity response will be combined into threshold and squeezing predictions.

## Results Layout

Results are grouped first by geometry, then by simulation layer:

```text
results/<geometry>/cavity/
results/<geometry>/crystal/
```

The directory helpers in `src/common/results_paths.py` always create the standard subdirectories:

```text
results/<geometry>/cavity/
results/<geometry>/crystal/
results/<geometry>/opo/
```

This layout reflects the staged physical workflow rather than the implementation details. A single geometry therefore accumulates all cavity, crystal, and future OPO results in one place.

## Why the Design Is Modular

The modular structure was chosen for physical and software reasons.

On the physics side, the cavity, crystal, and OPO problems are related but not identical. The cavity layer is mostly paraxial resonator optics; the crystal layer adds material dispersion and nonlinear overlap; the OPO layer will add dynamical response and quantum noise. Keeping these layers separated avoids mixing incompatible abstractions.

On the software side, the JSON handoff makes dependencies explicit. The crystal code does not need to know how the cavity geometry was built internally; it only needs the exported optical mode and cavity parameters. That makes it easier to:

- add new cavity geometries without rewriting the crystal layer
- swap in richer crystal models without changing resonator code
- develop the future OPO module against a stable, file-based interface

The result is a pipeline that mirrors the physical modeling hierarchy of the experiment.
