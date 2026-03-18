# Crystal Module

In QPIT-SQZsim, the crystal layer takes the cavity-derived intracavity mode and asks whether the nonlinear medium supports efficient three-wave interaction under those optical conditions. It combines material dispersion, quasi-phase matching, temperature dependence, and focused-beam overlap.

## Refractive Index

The starting point is the refractive index model of the nonlinear crystal. Conceptually, this is a Sellmeier-type description: refractive index depends on wavelength, and may also change with temperature through a thermo-optic correction.

The project keeps this logic in `src/crystal/crystal_materials.py`:

- `SellmeierCoefficients` stores dispersion parameters
- `n_sellmeier_um(...)` evaluates a standard Sellmeier model
- `n_from_model(...)` adds an optional linear temperature dependence

The important point for the rest of the pipeline is that phase matching depends on refractive indices evaluated at pump, signal, and idler wavelengths.

## Phase Mismatch `Delta k`

Three-wave mixing is controlled by the wave-vector mismatch

`Delta k = k_p - k_s - k_i`

If `Delta k = 0`, the nonlinear polarization stays in phase with the generated fields and conversion is efficient. If not, the interaction oscillates along the crystal and net conversion is reduced.

This logic is implemented in `src/crystal/crystal_phase_matching.py`, primarily through:

- `k_of_n(...)`
- `delta_k_three_wave(...)`
- `pm_amplitude_factor(...)`
- `pm_power_factor(...)`

The code uses the usual sinc-like reduction factor for a finite crystal length rather than a long symbolic derivation.

## Quasi-Phase Matching

When the material’s natural dispersion does not give `Delta k = 0`, the code can include a QPM grating vector. The effective mismatch becomes

`Delta k_eff = Delta k - m K_g`

where `K_g = 2 pi / Lambda` and `m` is the QPM order.

This is handled by:

- `qpm_grating_k(...)`
- `delta_k_qpm(...)`
- `poling_period_T(...)`

in `src/crystal/crystal_phase_matching.py`.

Conceptually, periodic poling compensates the bulk mismatch by resetting the nonlinear phase accumulation. That is why the crystal layer treats the poling period as a first-class input.

## Temperature Tuning

Temperature affects the crystal model in two ways:

- refractive indices shift with temperature
- the poling period can change through thermal expansion

The function `scan_phase_matching_vs_temperature(...)` evaluates these effects over a temperature grid and returns the best operating point together with the full scan arrays. This gives a practical answer to the experimental question: at what crystal temperature is the intended interaction best phase matched?

## Gaussian Beam Focusing

Phase matching alone is not enough. The field distribution inside the crystal also matters, because nonlinear coupling depends on how tightly the Gaussian mode is focused.

The cavity layer provides the crystal waist, and the crystal layer converts that waist into:

- Rayleigh range
- confocal parameter
- focusing parameter

These calculations are implemented in `src/crystal/crystal_mode_matching.py`.

The relevant physical tradeoff is standard:

- tight focusing increases intensity
- excessive focusing shortens the effective interaction length

The code captures that tradeoff with a compact focused-beam model.

## Focusing Parameter `xi`

The normalized focusing parameter is

`xi = L / (2 z_R)`

with crystal length `L` and Rayleigh range `z_R` inside the medium. This parameter compares the crystal length to the beam’s diffraction length.

`xi` is computed in both `src/crystal/crystal_mode_matching.py` and `src/crystal/crystal_boyd_kleinman.py`. It is the natural dimensionless quantity for deciding whether the mode is under-focused, over-focused, or near the useful interaction regime.

## Boyd-Kleinman Theory

The project uses a simplified Boyd-Kleinman-style treatment to estimate focused-beam nonlinear overlap. Rather than only asking whether the crystal is phase matched, it evaluates how the Gaussian beam profile and longitudinal phase accumulation combine over the crystal length.

This is implemented in `src/crystal/crystal_boyd_kleinman.py`:

- `compute_focusing_parameter(...)`
- `boyd_kleinman_integral(...)`
- `boyd_kleinman_efficiency(...)`

The exported `boyd_kleinman_factor` and `effective_nonlinear_overlap` are therefore not purely material properties. They are cavity-conditioned quantities because they depend on the mode that the cavity creates inside the crystal.

## How Cavity Output Is Used

The cavity-to-crystal handoff is explicit and file-based.

### Waist and position

`load_cavity_context_for_crystal(...)` in `src/crystal/crystal_workflow.py` reads `beam_waist_crystal_um` from the cavity JSON and converts it into the waist used for all focusing calculations. The reference plane is already the crystal region, so the crystal layer receives the beam where the nonlinear interaction is evaluated.

### Phase-matching inputs

The same cavity JSON also provides:

- `crystal_length_m`
- `wavelength_m`
- `n_crystal`

These become the default optical and geometric context for the crystal workflow. The phase-matching scan then combines them with the user-supplied pump/signal/idler wavelength model and refractive-index functions.

### Mode context

`build_mode_matching_context_from_cavity_output(...)` in `src/crystal/crystal_mode_matching.py` also reconstructs complex `q_sagittal` and `q_tangential` from the cavity JSON. Even though the current overlap calculation mainly uses the scalar waist, the exported `q` values keep the interface ready for more detailed astigmatic or longitudinal mode treatments.

## How Phase Matching Is Computed

The workflow is:

1. Evaluate `n_p(T)`, `n_s(T)`, and `n_i(T)`.
2. Convert refractive indices into wave vectors.
3. Compute bulk `Delta k`.
4. Apply the QPM grating correction to get `Delta k_eff`.
5. Convert `Delta k_eff` into a finite-length sinc-squared power factor.
6. Repeat across temperature and locate the best operating point.

This sequencing is implemented by `compute_crystal_phase_matching(...)` in `src/crystal/crystal_workflow.py`, which delegates the actual physics to `src/crystal/crystal_phase_matching.py`.

## How Mode Matching Is Handled

The workflow converts the cavity waist into a medium-adjusted Rayleigh range and then into `xi`. That `xi`, together with any supplied phase mismatch, is passed into the Boyd-Kleinman overlap integral.

So mode matching here means more than geometric alignment: it is the compatibility between the cavity-supported Gaussian mode and the finite nonlinear interaction volume of the crystal.

The core file responsibilities are:

- `src/crystal/crystal_materials.py`: dispersion and thermo-optic material models
- `src/crystal/crystal_phase_matching.py`: `Delta k`, QPM, and temperature scans
- `src/crystal/crystal_mode_matching.py`: cavity-output parsing and focusing quantities
- `src/crystal/crystal_boyd_kleinman.py`: focused-beam overlap model
- `src/crystal/crystal_workflow.py`: orchestration and export

Taken together, these files define the crystal side of the project’s pipeline: cavity mode -> phase matching and focusing -> nonlinear overlap metrics.
