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

### Implemented changes
- Replaced the old constant-fraction initializer in `MultiComponentDiafiltrationInitializer` with a staged heuristic initializer.
- The new routine now:
  - computes the retentate inlet state from feed/diafiltrate mixing
  - computes anion concentrations from electroneutrality instead of arbitrary guesses
  - initializes permeate cation concentrations using a valence-based sieving heuristic where lower-valence cations permeate more strongly
  - computes osmotic pressure from the initialized bulk concentrations
  - computes water flux from `L_p (P - \Delta \pi)` with a positive floor
  - initializes membrane interface cation concentrations from partition coefficients
  - computes membrane anion concentrations from membrane electroneutrality
  - fills the membrane interior with a linear profile in `z`
  - initializes membrane transport coefficients from the seeded membrane concentrations
  - initializes derivative variables from finite differences instead of arbitrary `1`s
- Added a membrane-interface projection step for cation concentrations so the provisional membrane interface state remains compatible with electroneutrality and nonnegative anion concentration before building the membrane profile.

### Verification
- `python -m py_compile src/prommis/nanofiltration/multi_component_diafiltration.py` passed.
- Targeted pytest build slice passed for representative Li and Li/Co cases.
- Direct smoke tests that build, initialize, and solve the Li and Li/Co cases both succeeded in `prommis-codex`.
- User ran:
  - `test_diagnostics_Li_Co`
  - `test_diagnostics_Li_Co_Al`
  and both passed in `prommis-codex`.
- A direct custom Li/Co run written to `/tmp/Li_Co_ipopt.log` converged in 3 IPOPT iterations.
- After adding the membrane-interface projection step, the same direct custom Li/Co and Li/Co/Al runs both initialized and solved successfully.
- Current direct-run IPOPT baseline:
  - `Li_Co`: 3 iterations
  - `Li_Co_Al`: 3 iterations
- Denser direct-run benchmark (`NFE_module_length=20`, `NFE_membrane_thickness=10`):
  - `Li_Co_dense`: 3 iterations
  - `Li_Co_Al_dense`: 3 iterations
- Extreme-concentration stress sweep also remained robust, with all tested cases initializing and solving successfully in 3 IPOPT iterations:
  - `Li_Co_base`
  - `Li_Co_high2x`
  - `Li_Co_high4x`
  - `Li_Co_skewedLi`
  - `Li_Co_skewedCo`
  - `Li_Co_Al_base`
  - `Li_Co_Al_high2x`
  - `Li_Co_Al_high4x`
  - `Li_Co_Al_Alrich`

### Stress-sweep concentrations
- All concentrations below are in `mol/m^3`.
- Li/Co feed cases:
  - `Li_Co_base`: feed `Li=170`, `Co=170`, `Cl=510`; diafiltrate `Li=10`, `Co=10`, `Cl=30`
  - `Li_Co_high2x`: feed `Li=340`, `Co=340`, `Cl=1020`; diafiltrate `Li=10`, `Co=10`, `Cl=30`
  - `Li_Co_high4x`: feed `Li=680`, `Co=680`, `Cl=2040`; diafiltrate `Li=10`, `Co=10`, `Cl=30`
  - `Li_Co_skewedLi`: feed `Li=900`, `Co=50`, `Cl=1000`; diafiltrate `Li=10`, `Co=10`, `Cl=20`
  - `Li_Co_skewedCo`: feed `Li=50`, `Co=450`, `Cl=950`; diafiltrate `Li=10`, `Co=10`, `Cl=20`
- Li/Co/Al feed cases:
  - `Li_Co_Al_base`: feed `Li=100`, `Co=100`, `Al=100`, `Cl=600`; diafiltrate `Li=10`, `Co=10`, `Al=10`, `Cl=60`
  - `Li_Co_Al_high2x`: feed `Li=200`, `Co=200`, `Al=200`, `Cl=1200`; diafiltrate `Li=10`, `Co=10`, `Al=10`, `Cl=60`
  - `Li_Co_Al_high4x`: feed `Li=400`, `Co=400`, `Al=400`, `Cl=2400`; diafiltrate `Li=10`, `Co=10`, `Al=10`, `Cl=60`
  - `Li_Co_Al_Alrich`: feed `Li=50`, `Co=50`, `Al=500`, `Cl=1650`; diafiltrate `Li=10`, `Co=10`, `Al=10`, `Cl=30`

### Low-pressure-margin benchmarks
- Goal: keep `applied_pressure > osmotic_pressure` so `volume_flux_water` remains positive while reducing the pressure margin.
- Li/Co high-concentration case (`feed Li=340, Co=340, Cl=1020`; diafiltrate `Li=10, Co=10, Cl=30`):
  - `Li_Co_margin_large`: `P=6.0 bar`, `osmotic_pressure=2.790789 bar`, `volume_flux_water=0.032092`
  - `Li_Co_margin_med`: `P=4.5 bar`, `osmotic_pressure=2.098419 bar`, `volume_flux_water=0.024016`
  - `Li_Co_margin_small`: `P=3.5 bar`, `osmotic_pressure=1.636327 bar`, `volume_flux_water=0.018637`
  - `Li_Co_margin_tight`: `P=3.0 bar`, `osmotic_pressure=1.404705 bar`, `volume_flux_water=0.015953`
- Li/Co/Al high-concentration case (`feed Li=200, Co=200, Al=200, Cl=1200`; diafiltrate `Li=10, Co=10, Al=10, Cl=60`):
  - `Li_Co_Al_margin_large`: `P=10.0 bar`, `osmotic_pressure=8.657132 bar`, `volume_flux_water=0.013429`
  - `Li_Co_Al_margin_med`: `P=8.0 bar`, `osmotic_pressure=6.945219 bar`, `volume_flux_water=0.010548`
  - `Li_Co_Al_margin_small`: `P=6.8 bar`, `osmotic_pressure=5.913371 bar`, `volume_flux_water=0.008866`
  - `Li_Co_Al_margin_tight`: `P=6.2 bar`, `osmotic_pressure=5.396107 bar`, `volume_flux_water=0.008039`
- All of the above low-margin benchmark cases initialized and solved successfully in 3 IPOPT iterations.

### Remaining risks / next steps
- The initializer is still heuristic; it is more physics-informed than before, but not yet based on a reduced solve sequence with temporary constraint activation/deactivation.
- The transport-coefficient block and water-flux block are still the main likely sources of poor conditioning.
- The next iteration should probably:
  - add staged activation/deactivation during initialization
  - broaden scaling coverage using more detailed diagnostics than the current pass/fail checks
  - compare solve iteration counts and robustness before/after scaling changes on Li/Co and Li/Co/Al systems
