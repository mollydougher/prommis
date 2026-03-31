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
