# Cavity Module

The cavity layer computes the resonator eigenmode and the linear cavity parameters needed by later nonlinear and quantum calculations. Its core formalism is the round-trip ABCD matrix of a paraxial Gaussian beam.

## ABCD Matrix Method

Each supported geometry is reduced to a sequence of optical elements:

- free-space propagation
- planar dielectric interfaces
- curved mirrors

These elements are multiplied into a round-trip ABCD matrix referenced to the crystal region. The geometry-specific assembly lives in `src/cavity/cavity_abcd.py`.

This representation is useful because the cavity mode is then extracted from a single matrix equation rather than by ray tracing many passes explicitly.

## Stability Condition

The code uses the standard cavity stability quantity

`m = (A + D) / 2`

from the round-trip matrix. A resonator is stable when `|m| < 1`. In that regime, the cavity supports a bounded Gaussian eigenmode; outside it, the mode diverges and the cavity is not physically usable for the intended operation.

The stability helper is implemented in `src/cavity/cavity_analysis.py`.

For bow-tie and triangle resonators, sagittal and tangential planes can differ because folding angles introduce astigmatism, so the code evaluates both planes separately.

## `q` Parameter and Gaussian Beam

Once the round-trip matrix is known, the code solves for the self-consistent cavity `q` parameter. This is the compact Gaussian-beam quantity that encodes both wavefront curvature and spot size evolution.

The imaginary part of `q` determines the local beam waist scale. In practice, the code uses the crystal-region `q` parameter to recover the intracavity waist through `beam_waist_from_q(...)` in `src/cavity/cavity_analysis.py`.

The exported fields:

- `q_sagittal`
- `q_tangential`
- `beam_waist_crystal_um`

are the most important bridge from resonator optics into crystal modeling.

## Waist and Waist Position

The relevant waist in this project is the waist inside the crystal section, because that is where the nonlinear interaction occurs. The cavity matrices are built with the reference plane located in the crystal region, so the computed `q` parameter is already tied to the beam state where the nonlinear medium matters most.

Conceptually, the waist sets the local intensity scale:

- smaller waist -> higher intensity -> stronger nonlinear interaction
- larger waist -> weaker interaction but reduced focusing sensitivity

Even when the documentation refers to "waist position," the operational point in this code is the cavity mode evaluated in the crystal reference region rather than a separate longitudinal beam-tracing output.

## Gouy Phase

The Gouy phase is derived from the stability factor and exported separately for sagittal and tangential planes. It quantifies the transverse-mode phase advance accumulated over a round trip and matters for mode structure and resonance spacing.

The calculation is performed by `gouy_phases_from_m_factor(...)` in `src/cavity/cavity_analysis.py`.

In folded cavities, unequal sagittal and tangential Gouy phases are a direct signature of astigmatic cavity behavior.

## Cavity Decay Rates

The module also computes the cavity linewidth parameters used later in OPO and squeezing models:

- `kappa_ext`: coupling through the designated output coupler
- `kappa_loss`: internal round-trip loss contribution
- `kappa_total = kappa_ext + kappa_loss`

These are computed from the optical round-trip length and the specified transmission/loss parameters in `compute_decay_rates(...)` inside `src/cavity/cavity_analysis.py`.

Physically:

- larger `kappa_ext` means stronger coupling to the outside world
- larger `kappa_loss` means more parasitic dissipation
- `kappa_total` sets the cavity bandwidth

## Escape Efficiency

Escape efficiency is exported as

`eta_escape = kappa_ext / kappa_total`

This is a compact measure of how much of the cavity decay occurs through the useful output channel instead of internal loss. For squeezing experiments, that distinction is critical because internal loss degrades observable nonclassical noise suppression.

## Downstream Use of Cavity Outputs

This is the most important role of the cavity module: it does not stop at geometry analysis, it prepares the inputs for later nonlinear and quantum models.

### Waist -> nonlinear interaction

The intracavity waist in the crystal sets the focusing strength and therefore the peak field intensity. The crystal layer uses this quantity to compute Rayleigh range, focusing parameter `xi`, and Boyd-Kleinman overlap. This handoff is performed in `src/crystal/crystal_workflow.py` and `src/crystal/crystal_mode_matching.py`.

### `kappa` -> squeezing spectrum

The cavity decay rates determine how quickly intracavity fields respond to perturbations. In a future OPO model, `kappa_total` sets the resonator bandwidth, while the balance between `kappa_ext` and `kappa_loss` controls how much squeezing exits the cavity versus being lost internally.

### Detuning -> frequency response

The exported `detuning_rad_s` is the offset from exact cavity resonance. In an OPO or squeezing calculation, detuning modifies the resonator response function and therefore changes both gain and noise transfer as a function of analysis frequency.

## Code Map

The cavity layer is split by responsibility:

- `src/cavity/cavity_abcd.py`: builds round-trip matrices for `bowtie`, `linear`, `triangle`, and `hemilithic` geometries
- `src/cavity/cavity_analysis.py`: derives `m`, `q`, waist, FSR, decay rates, and Gouy phase from those matrices
- `src/cavity/cavity_workflow.py`: assembles contexts, evaluates operating points, and exports the JSON consumed by the crystal layer

Together these files define the cavity side of the project’s pipeline: geometry -> eigenmode -> exported physical parameters.
