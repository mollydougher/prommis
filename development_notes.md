# Multi-Component Diafiltration Development Notes

## Working Context

- Date started: 2026-05-05
- Branch: `codex-debug-partition`
- Conda environment requested: `prommis-codex`
- Primary model: `src/prommis/nanofiltration/multi_component_diafiltration.py`
- Example flowsheet: `src/prommis/nanofiltration/diafiltration_flowsheet_two_salt.py`

## Initial Inspection

- The example flowsheet uses a two-cation, one-anion system: Li, Co, and Cl.
- The unit is built with `include_boundary_layer=True`, `NFE_module_length=10`, `NFE_boundary_layer_thickness=5`, and `NFE_membrane_thickness=5`.
- The currently flagged constraints are:
  - `cation_equilibrium_boundary_layer_membrane_interface`
  - `cation_equilibrium_membrane_permeate_interface`
- Both flagged constraints are written as product/power Donnan equilibrium relationships using concentration variables raised to charge-based powers.
- The documented partitioning equations include partition coefficients at the solution-membrane interfaces, but the implementation currently has the partition coefficient terms commented out in both flagged constraints.
- Concentration variables have lower bounds of `1e-11`, which prevents invalid zero/negative bases for power expressions but may still permit badly scaled residuals when concentrations become very small.
- The common anion is selected as `anion_list[0]`, and the model currently supports only one anion.

## Diagnostics Run

- Reproduced the reported initializer failure in `prommis-codex`.
- A full pytest run of the two-salt flowsheet did not produce useful output quickly and was stopped.
- A focused initializer-only run reproduced `BlockTriangularizationInitializer` failure with `TerminationCondition.maxIterations`.
- The initializer currently calculates membrane chloride as `-33 mol/m^3` for the Li/Co/Cl example. This comes from the current membrane cation guesses:
  - Li membrane guess: `0.2 * 50 = 10`
  - Co membrane guess: `0.01 * 50 = 0.5`
  - membrane electroneutrality: `-44 + 10 + 2*0.5 - Cl = 0`, so `Cl = -33`
- This violates the `1e-11` lower bound repeatedly before the block-triangular solve begins.
- A Donnan-consistent membrane concentration initializer eliminated the negative chloride warnings, but did not by itself produce a verified solve.
- A log-equivalent interface form and an explicit Donnan-factor interface form were tried locally. Neither was kept because neither produced a verified improvement across the initializer and solve path.
- Reducing applied pressure can avoid a separate over-recovery issue, but pressure reduction alone does not fix the original negative chloride initialization failure.
- All unverified model-code experiments were reverted. Only this notes file remains from the diagnostic pass.

## Patch 1: Membrane Initialization

- Implemented a narrow initialization change in `multi_component_diafiltration.py`.
- The initializer now computes membrane concentration guesses from H=1 Donnan partitioning and membrane electroneutrality instead of using charge-class multipliers (`0.2` for monovalent cations and `1e-2` for higher-valence cations).
- For the Li/Co/Cl example, applying only the initializer guesses gives positive membrane chloride everywhere:
  - minimum membrane chloride: `1e-10 mol/m^3` at the inlet boundary
  - maximum membrane chloride: about `105.35 mol/m^3`
- The focused `BlockTriangularizationInitializer` run still fails after this patch:
  - with `max_iter=20`: `TerminationCondition.iterationLimit`
  - with `max_iter=200`: `TerminationCondition.locallyInfeasible`
- The remaining negative chloride warnings appear after the SCC solve attempts, not from the initial values.
- A full-space IPOPT diagnostic from the repaired guesses still lands locally infeasible. The visible residuals are no longer dominated by the two cation equilibrium interface constraints; instead, the largest reported residuals are `lumped_water_flux` and `osmotic_pressure_calculation`, with `osmotic_pressure` pressed near its lower bound.

## Pressure Sweep

- Tested applied pressures from 10 bar down to 1 bar using the Li/Co/Cl example, patched membrane concentration guesses, and a full-space IPOPT diagnostic after applying initializer guesses.
- Lowering pressure reduces the water-flux value as expected, but it does not move `osmotic_pressure` away from its lower bound:
  - 10 bar: `osmotic_pressure` min/max = `1e-11`, `volume_flux_water` max about `0.10`
  - 5 bar: `osmotic_pressure` min/max = `1e-11`, `volume_flux_water` max about `0.05`
  - 1 bar: `osmotic_pressure` min/max = `1e-11`, `volume_flux_water` max about `0.01`
- At 1 bar, the visible residuals are only `osmotic_pressure_calculation` residuals of about `1.44e-4`, so lowering pressure helps the water-flux side but does not resolve the osmotic-pressure lower-bound issue.
- No pressure change was retained in the example flowsheet because pressure reduction alone did not produce a converged solve.

## Patch 2: Permeate Initialization

- Implemented a targeted permeate concentration guess in `multi_component_diafiltration.py`.
- The previous initializer used `permeate_conc_mol_comp = 0.8 * feed_conc`, while the boundary layer was initialized at `0.75 * feed_conc`. This made the initial osmotic-pressure expression want a negative value.
- The initializer now scales all permeate ion concentrations together to match the current `osmotic_pressure` guess while preserving electroneutrality.
- For the Li/Co/Cl example with the default `osmotic_pressure` guess of 4 bar:
  - permeate guesses are about Li `22.82`, Co `22.82`, and Cl `68.47 mol/m^3`
  - initial `osmotic_pressure_calculation` residual is essentially zero (`~1.8e-15`) at both `x=0.1` and `x=1.0`
- Focused initializer diagnostics:
  - `BlockTriangularizationInitializer` still fails with `TerminationCondition.locallyInfeasible`
  - out-of-bounds membrane chloride warnings are gone
  - a full-space IPOPT diagnostic still lands locally infeasible after solving; visible residuals include `osmotic_pressure_calculation` and downstream flux/bulk balance constraints, especially near `x=1`

## Early Numerical Concerns To Investigate

- Product/power interface constraints can become poorly scaled when concentrations span orders of magnitude, especially for divalent or trivalent cations.
- Omitting partition coefficients may make the implemented equilibrium relationship differ from the documented formulation and from the property-package assumptions.
- The same algebraic form may be creating large derivatives near lower concentration bounds.
- A log-form or ratio-form reformulation might improve conditioning, but this should be confirmed against the intended physical model before implementation.
- Initialization currently sets membrane concentrations by charge class, then calculates the common anion from membrane electroneutrality for nonzero module-length points.
- The current membrane initialization was likely written for small divalent/trivalent partition coefficients. With all partition coefficients set to 1, divalent cations should not be initialized at `1e-2 * feed_conc` because that makes membrane electroneutrality impossible for this dilute, negatively charged membrane case.

## Clarification Questions

- Partition coefficient terms were intentionally removed; the intended current assumption is all partition coefficients equal 1.
- A log-form Donnan equilibrium is acceptable to try, but prior attempts also had numerical convergence issues.
- Li/Co/Cl is the immediate target system. Other cation combinations are out of scope until this example works.
- The current representative failure is `BlockTriangularizationInitializer` hitting `maxIterations`.
- The current `1e-11` concentration lower bounds are numerical guards. They can be revisited, but should remain small enough to represent zero.
- For now, the membrane should be treated as negatively charged.

## Next Steps

- First candidate change: revise membrane concentration initialization so the initial membrane cation guesses are compatible with negative fixed charge and H=1 partitioning. A Donnan-consistent profile is a good starting point, but it should be introduced separately from equation reformulation. Status: implemented; it fixes the negative initial chloride guesses but does not fully converge initialization.
- Second candidate change: inspect the first failing SCC after Patch 2 and determine whether the remaining issue is downstream over-recovery/flux coupling, the product-form interface constraints, or another bound.
- Third candidate change: after the remaining active block is isolated, compare product-form, log-form, and Donnan-factor interface forms one at a time if the cation equilibrium constraints are still central to the failure.
- Keep each experiment isolated and verified before retaining code changes.
