# Development Notes

## 2026-02-18

### Scope
- Inspected `src/prommis/nanofiltration/multi_component_diafiltration.py`.
- Checked related tests in `src/prommis/nanofiltration/tests/test_multi_component_diafiltration.py`.

### Current understanding
- The unit model represents a steady-state multi-component diafiltration membrane with one common anion and one to three cations.
- The model discretizes module length and membrane thickness with backward finite differences.
- It builds its own `time` set as `[0]` for compatibility with the property package.
- Core physics include retentate balances, permeate bulk flux relations, osmotic-pressure-driven water flux, membrane transport coefficients, electroneutrality, and interfacial partitioning.
- The implementation fixes/deactivates anion derivative terms that Pyomo creates automatically but the model does not use directly.

### Initial review points
- The initializer seeds retentate/permeate/membrane states using simple fractions of the feed concentration and arbitrary derivative values of `1`.
- Several boundary conditions at `x=0` enforce small positive `numerical_zero_tolerance` values instead of literal zero for numerical stability.
- The retentate inlet concentration boundary condition is only written for cations; anions appear to be recovered through electroneutrality constraints rather than a direct inlet mixing equation.
- The editable environment currently points to the fork checkout in this repo.

### Questions to resolve
- Confirm intended handling of inlet anion concentration at `x=0`.
- Confirm whether repeated membrane-flux equations over every `z` point are intentional or part of the stabilization strategy.
- Confirm expected workflow for changing membrane charge, permeability, and discretization in experiments.

### Clarifications from Molly
- The retentate inlet mixing condition is intentionally written only for cations; the retentate electroneutrality equation is expected to recover the anion concentration and serve as a consistency check.
- The membrane anion concentration at `x=0` is intentionally omitted; only cation membrane concentrations need explicit boundary treatment there.
- The flux equation is written for every `z` point because it contains `z`-dependent variables, though the implementation may still be revisited if there is a cleaner formulation.
- The current initializer values are pragmatic guesses intended to help BT initialization, not a final physics-based initialization strategy.
- There are no current plans to add a real time domain.
- The current scaling strategy is based on observed numerical behavior and is still open to improvement.

### Current focus
- Improve the initialization path so it reflects model physics better and reduces the burden on the nonlinear solve.
- Revisit the membrane flux formulation to make sure the `z`-indexed implementation is mathematically tight and numerically sensible.
- Expand scaling coverage in a targeted way based on the worst-conditioned variables and constraints.

### Additional clarifications
- The first common failure mode is BT initialization failure.
- Single-salt cases appear somewhat more stable than two-salt and three-salt cases, but this has not been characterized rigorously.
- The main numerical offenders seen so far are:
  - `volume_flux_water`
  - `membrane_D_tilde`
  - `membrane_cross_diffusion_coefficient_bilinear`
  - `membrane_convection_coefficient_bilinear`
  - `membrane_cross_diffusion_coefficient`
  - `membrane_convection_coefficient`
  - `lumped_water_flux`
- A staged, physics-informed initializer is preferred.
- It is acceptable to temporarily fix variables or deactivate constraints during initialization if that materially improves robustness.
- The initializer should be designed to work generically for one-, two-, and three-salt systems.

### Working hypothesis
- BT is likely failing because the current initializer seeds the highly coupled membrane transport block with weak guesses, especially for the coefficient calculations tied to `membrane_D_tilde`.
- A better path is to initialize from a reduced transport picture first:
  - build consistent inlet retentate and anion states from mixing + electroneutrality
  - estimate osmotic pressure and water flux from bulk states
  - initialize permeate with a simple low-rejection or moderate-rejection assumption
  - initialize membrane interface concentrations from partitioning relationships
  - initialize the membrane thickness profile from a simple interpolation between interfaces
  - only then release the full cross-diffusion constraints
