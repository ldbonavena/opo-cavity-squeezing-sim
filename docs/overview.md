# Overview

QPIT-SQZsim models the optical chain needed to analyze an Optical Parametric Oscillator from cavity geometry up to quantities that matter for squeezing. The current implementation covers the linear cavity eigenmode problem and the crystal-layer calculations that depend on that mode.

## Physical Scope

The code solves a staged problem:

1. Build the resonator round-trip optics from a chosen cavity geometry.
2. Extract the stable Gaussian eigenmode inside the cavity crystal region.
3. Use that intracavity mode to evaluate crystal phase matching and focused-beam overlap.
4. Export the quantities needed by a later OPO and squeezing model.

In that sense, the project pipeline is:

`cavity -> crystal -> OPO -> squeezing`

Only the first two layers are implemented today, but the interfaces already reflect the full intended flow.

## Conceptual Module Interaction

The `cavity` layer answers: what mode does the resonator support, and what are its linear dynamical parameters? It produces beam waists, `q` parameters, round-trip length, decay rates, Gouy phases, and detuning-related quantities.

The `crystal` layer answers: given that cavity mode, what crystal operating point and QPM period are required, and how well does the nonlinear medium support the intended interaction? It computes refractive-index-dependent phase matching, derives a design poling period when requested, determines the operating temperature, and evaluates focused-beam overlap through a Boyd-Kleinman-style model.

The `opo` layer is reserved for the next step: combining cavity loss, coupling, detuning, and nonlinear interaction strength into threshold and squeezing calculations.

## Why the Separation Matters

The main design choice is to keep resonator optics, crystal physics, and future quantum/OPO dynamics loosely coupled. Each layer consumes a compact result from the previous one instead of reaching directly into internal functions. That makes the physical pipeline explicit:

- cavity geometry determines the intracavity Gaussian mode
- the Gaussian mode determines nonlinear overlap in the crystal
- cavity linewidth and detuning determine the frequency response seen by the OPO model

The `docs/architecture.md`, `docs/cavity.md`, and `docs/crystal.md` files expand these layers in more detail.
