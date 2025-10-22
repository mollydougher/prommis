from pyomo.dae import (
    ContinuousSet,
    DerivativeVar,
)
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    maximize,
    NonNegativeReals,
    Objective,
    Param,
    SolverFactory,
    Suffix,
    TransformationFactory,
    units,
    value,
    Var,
)
from idaes.core.util.constants import Constants
from idaes.core.util.model_diagnostics import DiagnosticsToolbox

from idaes.core.util.scaling import (
    extreme_jacobian_columns,
    extreme_jacobian_rows,
    constraint_autoscale_large_jac,
)

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


def main():
    m = build_model()
    discretize_model(m, NFEx=10, NFEz=8)
    dt = DiagnosticsToolbox(m)
    dt.assert_no_structural_warnings()

    # Create a scaled version of the model to solve
    set_scaling(m)
    scaling = TransformationFactory("core.scale_model")
    scaled_model = scaling.create_using(m, rename=False)
    solve_model(scaled_model)
    # Propagate results back to unscaled model
    scaling.propagate_solution(scaled_model, m)

    # solve_model(m)
    # dt.assert_no_numerical_warnings()
    dt.report_numerical_issues()
    # dt.display_variables_with_extreme_jacobians()
    # dt.display_constraints_with_extreme_jacobians()
    # dt.display_variables_at_or_outside_bounds()

    # unfix_dof(m)
    # optimize(m)

    plot_results(m)
    plot_membrane_results(m)


def build_model():
    """
    Adds model equations for the two-salt (with common anion) diafiltraiton
    system without the inclusion of a boundary layer.
    Considers convenction and diffustion and electromigration transport
    mechanisms.

    References:
        diffusion coefficients: https://www.aqion.de/site/diffusion-coefficients

    Returns:
        m: the DAE system, a pyomo model
    """
    # create the model
    m = ConcreteModel()

    # define the model parameters
    m.n = Param(
        initialize=3,
        units=units.dimensionless,
        doc="Number of dissociated ions in solution",
    )
    m.l = Param(
        initialize=1e-7,  # TODO: verify 100nm reasonable
        units=units.m,
        doc="Thickness of membrane (z-direction)",
    )
    m.w = Param(
        initialize=1,
        units=units.m,
        doc="Width of the membrane (x-direction)",
    )
    m.z_lithium = Param(
        initialize=1,
        units=units.dimensionless,
        doc="Charge of lithium ion",
    )
    m.z_cobalt = Param(
        initialize=2,
        units=units.dimensionless,
        doc="Charge of cobalt ion",
    )
    m.z_chlorine = Param(
        initialize=-1,
        units=units.dimensionless,
        doc="Charge of coblat ion",
    )
    m.molar_mass_lithium = Param(
        initialize=0.006941,
        units=units.kg / units.mol,
        doc="Molar mass of lithium",
    )
    m.molar_mass_cobalt = Param(
        initialize=0.05893,
        units=units.kg / units.mol,
        doc="Molar mass of cobalt",
    )
    m.molar_mass_chlorine = Param(
        initialize=0.03545,
        units=units.kg / units.mol,
        doc="Molar mass of chlorine",
    )
    m.D_lithium = Param(
        initialize=3.7e-6,
        units=units.m**2 / units.h,
        doc="Diffusion coefficient for lithium ion in water",
    )
    m.D_cobalt = Param(
        initialize=2.64e-6,
        units=units.m**2 / units.h,
        doc="Diffusion coefficient for cobalt ion in water",
    )
    m.D_chlorine = Param(
        initialize=7.3e-6,
        units=units.m**2 / units.h,
        doc="Diffusion coefficient for chlorine ion in water",
    )
    m.Lp = Param(
        initialize=0.01,  # TODO: verify
        units=units.m / units.h / units.bar,
        doc="Hydraulic permeability coefficient",
    )
    m.sigma_lithium = Param(
        initialize=1,
        units=units.dimensionless,
        doc="Thermodynamic reflection coefficient for lithium ion",
    )
    m.sigma_cobalt = Param(
        initialize=1,
        units=units.dimensionless,
        doc="Thermodynamic reflection coefficient for cobalt ion",
    )
    m.sigma_chlorine = Param(
        initialize=1,
        units=units.dimensionless,
        doc="Thermodynamic reflection coefficient for chlorine ion",
    )
    m.feed_flow_volume = Param(
        initialize=100,
        units=units.m**3 / units.h,
        doc="Volumetric flow rate of the feed",
    )
    m.feed_conc_mass_lithium = Param(
        initialize=1.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of lithium in the feed",
    )
    m.feed_conc_mass_cobalt = Param(
        initialize=17,
        units=units.kg / units.m**3,
        doc="Mass concentration of cobalt in the feed",
    )
    m.feed_conc_mass_chlorine = Param(
        initialize=29,
        units=units.kg / units.m**3,
        doc="Mass concentration of chlorine in the feed",
    )
    m.diafiltrate_flow_volume = Param(
        initialize=30,
        units=units.m**3 / units.h,
        doc="Volumetric flow rate of the diafiltrate",
    )
    m.diafiltrate_conc_mass_lithium = Param(
        initialize=0.1,
        units=units.kg / units.m**3,
        doc="Mass concentration of lithium in the diafiltrate",
    )
    m.diafiltrate_conc_mass_cobalt = Param(
        initialize=0.2,
        units=units.kg / units.m**3,
        doc="Mass concentration of cobalt in the diafiltrate",
    )
    m.diafiltrate_conc_mass_chlorine = Param(
        initialize=0.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of chlorine in the diafiltrate",
    )

    # define length scales
    m.x_bar = ContinuousSet(bounds=(0, 1))
    m.z_bar = ContinuousSet(bounds=(0, 1))

    # define remaining algebraic variables
    ## independent of x,z length scale
    m.L = Var(
        initialize=100,
        units=units.m,
        domain=NonNegativeReals,
        doc="Length of the membrane, wound radially",
    )
    m.L.fix()  # fix for simulation
    m.dP = Var(
        initialize=10,  # TODO: verify 10 bar reasoanable
        units=units.bar,
        domain=NonNegativeReals,
        doc="Pressure applied to membrane",
    )
    m.dP.fix()  # fix for simulation

    ## dependent on x
    m.volume_flux_water = Var(
        m.x_bar,
        initialize=0.03,
        units=units.m**3 / units.m**2 / units.h,
        domain=NonNegativeReals,
        doc="Volumetric water flux of water across the membrane",
    )
    m.mass_flux_lithium = Var(
        m.x_bar,
        initialize=0.05,  # TODO: verify good value
        units=units.kg / units.m**2 / units.h,
        domain=NonNegativeReals,
        doc="Mass flux of lithium across the membrane (z-direction, x-dependent)",
    )
    m.mass_flux_cobalt = Var(
        m.x_bar,
        initialize=0.05,  # TODO: verify good value
        units=units.kg / units.m**2 / units.h,
        domain=NonNegativeReals,
        doc="Mass flux of cobalt across the membrane (z-direction, x-dependent)",
    )
    m.mass_flux_chlorine = Var(
        m.x_bar,
        initialize=0.05,  # TODO: verify good value
        units=units.kg / units.m**2 / units.h,
        domain=NonNegativeReals,
        doc="Mass flux of chlorine across the membrane (z-direction, x-dependent)",
    )
    m.retentate_flow_volume = Var(
        m.x_bar,
        initialize=130,
        units=units.m**3 / units.h,
        domain=NonNegativeReals,
        doc="Volumetric flow rate of the retentate, x-dependent",
    )
    m.retentate_conc_mass_lithium = Var(
        m.x_bar,
        initialize=1.33,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of lithium in the retentate, x-dependent",
    )
    m.retentate_conc_mass_cobalt = Var(
        m.x_bar,
        initialize=13.1,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of cobalt in the retentate, x-dependent",
    )
    m.retentate_conc_mass_chlorine = Var(
        m.x_bar,
        initialize=22.5,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of chlorine in the retentate, x-dependent",
    )
    m.permeate_flow_volume = Var(
        m.x_bar,
        initialize=0,
        units=units.m**3 / units.h,
        domain=NonNegativeReals,
        doc="Volumetric flow rate of the permeate, x-dependent",
    )
    m.permeate_conc_mass_lithium = Var(
        m.x_bar,
        initialize=0,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of lithium in the permeate, x-dependent",
    )
    m.permeate_conc_mass_cobalt = Var(
        m.x_bar,
        initialize=0,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of cobalt in the permeate, x-dependent",
    )
    m.permeate_conc_mass_chlorine = Var(
        m.x_bar,
        initialize=0,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of chlorine in the retentate, x-dependent",
    )
    m.osmotic_pressure = Var(
        m.x_bar,
        initialize=5,
        units=units.bar,
        domain=NonNegativeReals,
        doc="Osmostic pressure of the feed-side fluid",
    )

    ## dependent on z_hat and x_hat
    m.membrane_conc_mass_lithium = Var(
        m.x_bar,
        m.z_bar,
        initialize=1.7,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of lithium in the membrane, x- and z-dependent",
    )
    m.membrane_conc_mass_cobalt = Var(
        m.x_bar,
        m.z_bar,
        initialize=13.1,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of cobalt in the membrane, x- and z-dependent",
    )
    m.membrane_conc_mass_chlorine = Var(
        m.x_bar,
        m.z_bar,
        initialize=22.5,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of chlorine in the membrane, x- and z-dependent",
    )
    m.D_lithium_lithium = Var(
        m.x_bar,
        m.z_bar,
        initialize=-1e-9,  # TODO: verify good initial value
        units=units.m**2 / units.h,
        doc="Linearized cross diffusion coefficent for lithium-lithium",
    )
    m.D_lithium_cobalt = Var(
        m.x_bar,
        m.z_bar,
        initialize=-1e-11,  # TODO: verify good initial value
        units=units.m**2 / units.h,
        doc="Linearized cross diffusion coefficent for lithium-cobalt",
    )
    m.D_cobalt_lithium = Var(
        m.x_bar,
        m.z_bar,
        initialize=-1e-10,  # TODO: verify good initial value
        units=units.m**2 / units.h,
        doc="Linearized cross diffusion coefficent for cobalt-lithium",
    )
    m.D_cobalt_cobalt = Var(
        m.x_bar,
        m.z_bar,
        initialize=-1e-9,  # TODO: verify good initial value
        units=units.m**2 / units.h,
        doc="Linearized cross diffusion coefficent for cobalt-cobalt",
    )

    # define the (partial) derivative variables
    m.d_retentate_conc_mass_lithium_dx = DerivativeVar(
        m.retentate_conc_mass_lithium,
        wrt=m.x_bar,
        units=units.kg / units.m**3,
    )
    m.d_retentate_conc_mass_cobalt_dx = DerivativeVar(
        m.retentate_conc_mass_cobalt,
        wrt=m.x_bar,
        units=units.kg / units.m**3,
    )
    m.d_retentate_flow_volume_dx = DerivativeVar(
        m.retentate_flow_volume,
        wrt=m.x_bar,
        units=units.m**3 / units.h,
    )
    m.d_membrane_conc_mass_lithium_dz = DerivativeVar(
        m.membrane_conc_mass_lithium,
        wrt=m.z_bar,
        units=units.kg / units.m**3,
    )
    m.d_membrane_conc_mass_cobalt_dz = DerivativeVar(
        m.membrane_conc_mass_cobalt,
        wrt=m.z_bar,
        units=units.kg / units.m**3,
    )

    # define the constraints
    ## mass balance constraints
    def _overall_mass_balance(m, x):
        return m.d_retentate_flow_volume_dx[x] == (-m.volume_flux_water[x] * m.L * m.w)

    m.overall_mass_balance = Constraint(m.x_bar, rule=_overall_mass_balance)

    def _lithium_mass_balance(m, x):
        return (m.retentate_flow_volume[x] * m.d_retentate_conc_mass_lithium_dx[x]) == (
            (
                m.volume_flux_water[x] * m.retentate_conc_mass_lithium[x]
                - m.mass_flux_lithium[x]
            )
            * m.L
            * m.w
        )

    m.lithium_mass_balance = Constraint(m.x_bar, rule=_lithium_mass_balance)

    def _cobalt_mass_balance(m, x):
        return (m.retentate_flow_volume[x] * m.d_retentate_conc_mass_cobalt_dx[x]) == (
            (
                m.volume_flux_water[x] * m.retentate_conc_mass_cobalt[x]
                - m.mass_flux_cobalt[x]
            )
            * m.L
            * m.w
        )

    m.cobalt_mass_balance = Constraint(m.x_bar, rule=_cobalt_mass_balance)

    ## transport constraints
    def _geometric_flux_equation_overall(m, x):
        if x == 0:
            return Constraint.Skip
        return m.permeate_flow_volume[x] == m.volume_flux_water[x] * x * m.L * m.w

    m.geometric_flux_equation_overall = Constraint(
        m.x_bar, rule=_geometric_flux_equation_overall
    )

    def _geometric_flux_equation_lithium(m, x):
        if x == 0:
            return Constraint.Skip
        return m.mass_flux_lithium[x] == (
            m.permeate_conc_mass_lithium[x] * m.volume_flux_water[x]
        )

    m.geometric_flux_equation_lithium = Constraint(
        m.x_bar, rule=_geometric_flux_equation_lithium
    )

    def _geometric_flux_equation_cobalt(m, x):
        if x == 0:
            return Constraint.Skip
        return m.mass_flux_cobalt[x] == (
            m.permeate_conc_mass_cobalt[x] * m.volume_flux_water[x]
        )

    m.geometric_flux_equation_cobalt = Constraint(
        m.x_bar, rule=_geometric_flux_equation_cobalt
    )

    def _lumped_water_flux(m, x):
        if x == 0:
            return Constraint.Skip
        return m.volume_flux_water[x] == (m.Lp * (m.dP - m.osmotic_pressure[x]))

    m.lumped_water_flux = Constraint(m.x_bar, rule=_lumped_water_flux)

    def _D_lithium_lithium_calculation(m, x, z):
        return m.D_lithium_lithium[x, z] == (
            (-3.87e-6 * units.m**2 / units.h)
            + (
                (-6.56e-8 * units.m**5 / units.kg / units.h)
                * (m.membrane_conc_mass_lithium[x, z])
            )
            + (
                (2.58e-8 * units.m**5 / units.kg / units.h)
                * (m.membrane_conc_mass_cobalt[x, z])
            )
        )

    m.D_lithium_lithium_calculation = Constraint(
        m.x_bar, m.z_bar, rule=_D_lithium_lithium_calculation
    )

    def _D_lithium_cobalt_calculation(m, x, z):
        return m.D_lithium_cobalt[x, z] == (
            (-4.50e-7 * units.m**2 / units.h)
            + (
                (-1.70e-7 * units.m**5 / units.kg / units.h)
                * (m.membrane_conc_mass_lithium[x, z])
            )
            + (
                (6.67e-8 * units.m**5 / units.kg / units.h)
                * (m.membrane_conc_mass_cobalt[x, z])
            )
        )

    m.D_lithium_cobalt_calculation = Constraint(
        m.x_bar, m.z_bar, rule=_D_lithium_cobalt_calculation
    )

    def _D_cobalt_lithium_calculation(m, x, z):
        return m.D_cobalt_lithium[x, z] == (
            (-6.47e-7 * units.m**2 / units.h)
            + (
                (4.10e-8 * units.m**5 / units.kg / units.h)
                * (m.membrane_conc_mass_lithium[x, z])
            )
            + (
                (-1.61e-8 * units.m**5 / units.kg / units.h)
                * (m.membrane_conc_mass_cobalt[x, z])
            )
        )

    m.D_cobalt_lithium_calculation = Constraint(
        m.x_bar, m.z_bar, rule=_D_cobalt_lithium_calculation
    )

    def _D_cobalt_cobalt_calculation(m, x, z):
        return m.D_cobalt_cobalt[x, z] == (
            (-3.56e-6 * units.m**2 / units.h)
            + (
                (3.91e-7 * units.m**5 / units.kg / units.h)
                * (m.membrane_conc_mass_lithium[x, z])
            )
            + (
                (-1.53e-7 * units.m**5 / units.kg / units.h)
                * (m.membrane_conc_mass_cobalt[x, z])
            )
        )

    m.D_cobalt_cobalt_calculation = Constraint(
        m.x_bar, m.z_bar, rule=_D_cobalt_cobalt_calculation
    )

    def _lithium_flux_membrane(m, x, z):
        if z == 0:
            return Constraint.Skip

        return m.mass_flux_lithium[x] == (
            m.membrane_conc_mass_lithium[x, z] * m.volume_flux_water[x]
            + (
                m.D_lithium_lithium[x, z]
                / m.l
                * m.d_membrane_conc_mass_lithium_dz[x, z]
            )
            + (m.D_lithium_cobalt[x, z] / m.l * m.d_membrane_conc_mass_cobalt_dz[x, z])
        )

    m.lithium_flux_membrane = Constraint(m.x_bar, m.z_bar, rule=_lithium_flux_membrane)

    def _cobalt_flux_membrane(m, x, z):
        if z == 0:
            return Constraint.Skip

        return m.mass_flux_cobalt[x] == (
            m.membrane_conc_mass_cobalt[x, z] * m.volume_flux_water[x]
            + (m.D_cobalt_lithium[x, z] / m.l * m.d_membrane_conc_mass_lithium_dz[x, z])
            + (m.D_cobalt_cobalt[x, z] / m.l * m.d_membrane_conc_mass_cobalt_dz[x, z])
        )

    m.cobalt_flux_membrane = Constraint(m.x_bar, m.z_bar, rule=_cobalt_flux_membrane)

    def _chlorine_flux_membrane(m, x):
        return m.mass_flux_chlorine[x] == -(
            (m.z_lithium / m.z_chlorine)
            * (m.molar_mass_chlorine / m.molar_mass_lithium)
            * m.mass_flux_lithium[x]
        ) - (
            (m.z_cobalt / m.z_chlorine)
            * (m.molar_mass_chlorine / m.molar_mass_cobalt)
            * m.mass_flux_cobalt[x]
        )

    m.chlorine_flux_membrane = Constraint(m.x_bar, rule=_chlorine_flux_membrane)

    ## other physical constaints
    def _osmotic_pressure_calculation(m, x):
        return m.osmotic_pressure[x] == units.convert(
            (
                (
                    m.n
                    * Constants.gas_constant  # J / mol / K
                    * 298
                    * units.K  # assume room temp
                )
                * (
                    m.sigma_lithium
                    / m.molar_mass_lithium
                    * (
                        m.retentate_conc_mass_lithium[x]
                        - m.permeate_conc_mass_lithium[x]
                    )
                    + m.sigma_cobalt
                    / m.molar_mass_cobalt
                    * (m.retentate_conc_mass_cobalt[x] - m.permeate_conc_mass_cobalt[x])
                    + m.sigma_chlorine
                    / m.molar_mass_chlorine
                    * (
                        m.retentate_conc_mass_chlorine[x]
                        - m.permeate_conc_mass_chlorine[x]
                    )
                )
            ),
            to_units=units.bar,
        )

    m.osmotic_pressure_calcualation = Constraint(
        m.x_bar, rule=_osmotic_pressure_calculation
    )

    ## boundary conditions
    def _retentate_membrane_interface_lithium(m, x):
        if x == 0:
            return Constraint.Skip
        return m.retentate_conc_mass_lithium[x] == m.membrane_conc_mass_lithium[x, 0]

    m.retentate_membrane_interface_lithium = Constraint(
        m.x_bar, rule=_retentate_membrane_interface_lithium
    )

    def _retentate_membrane_interface_cobalt(m, x):
        if x == 0:
            return Constraint.Skip
        return m.retentate_conc_mass_cobalt[x] == m.membrane_conc_mass_cobalt[x, 0]

    m.retentate_membrane_interface_cobalt = Constraint(
        m.x_bar, rule=_retentate_membrane_interface_cobalt
    )

    def _retentate_membrane_interface_chlorine(m, x):
        if x == 0:
            return Constraint.Skip
        return m.retentate_conc_mass_chlorine[x] == m.membrane_conc_mass_chlorine[x, 0]

    m.retentate_membrane_interface_chlorine = Constraint(
        m.x_bar, rule=_retentate_membrane_interface_chlorine
    )

    def _membrane_permeate_interface_lithium(m, x):
        return m.permeate_conc_mass_lithium[x] == m.membrane_conc_mass_lithium[x, 1]

    m.membrane_permeate_interface_lithium = Constraint(
        m.x_bar, rule=_membrane_permeate_interface_lithium
    )

    def _membrane_permeate_interface_cobalt(m, x):
        return m.permeate_conc_mass_cobalt[x] == m.membrane_conc_mass_cobalt[x, 1]

    m.membrane_permeate_interface_cobalt = Constraint(
        m.x_bar, rule=_membrane_permeate_interface_cobalt
    )

    def _membrane_permeate_interface_chlorine(m, x):
        return m.permeate_conc_mass_chlorine[x] == m.membrane_conc_mass_chlorine[x, 1]

    m.membrane_permeate_interface_chlorine = Constraint(
        m.x_bar, rule=_membrane_permeate_interface_chlorine
    )

    def _electroneutrality_retentate(m, x):
        return 0 == (
            m.z_lithium * m.retentate_conc_mass_lithium[x] / m.molar_mass_lithium
            + m.z_cobalt * m.retentate_conc_mass_cobalt[x] / m.molar_mass_cobalt
            + m.z_chlorine * m.retentate_conc_mass_chlorine[x] / m.molar_mass_chlorine
        )

    m.electroneutrality_retentate = Constraint(
        m.x_bar, rule=_electroneutrality_retentate
    )

    def _electroneutrality_membrane(m, x, z):
        if z == 0:
            return Constraint.Skip
        return 0 == (
            m.z_lithium * m.membrane_conc_mass_lithium[x, z] / m.molar_mass_lithium
            + m.z_cobalt * m.membrane_conc_mass_cobalt[x, z] / m.molar_mass_cobalt
            + m.z_chlorine * m.membrane_conc_mass_chlorine[x, z] / m.molar_mass_chlorine
        )

    m.electroneutrality_membrane = Constraint(
        m.x_bar, m.z_bar, rule=_electroneutrality_membrane
    )

    ## initial/final conditions
    def _initial_retentate_flow_volume(m):
        return m.retentate_flow_volume[0] == (
            m.feed_flow_volume + m.diafiltrate_flow_volume
        )

    m.initial_retentate_flow_volume = Constraint(rule=_initial_retentate_flow_volume)

    def _initial_permeate_flow_volume(m):
        return m.permeate_flow_volume[0] == (0 * units.m**3 / units.h)

    m.initial_permeate_flow_volume = Constraint(rule=_initial_permeate_flow_volume)

    def _initial_retentate_conc_mass_lithium(m):
        return m.retentate_conc_mass_lithium[0] == (
            (
                m.feed_flow_volume * m.feed_conc_mass_lithium
                + m.diafiltrate_flow_volume * m.diafiltrate_conc_mass_lithium
            )
            / (m.feed_flow_volume + m.diafiltrate_flow_volume)
        )

    m.initial_retentate_conc_mass_lithium = Constraint(
        rule=_initial_retentate_conc_mass_lithium
    )

    def _initial_retentate_conc_mass_cobalt(m):
        return m.retentate_conc_mass_cobalt[0] == (
            (
                m.feed_flow_volume * m.feed_conc_mass_cobalt
                + m.diafiltrate_flow_volume * m.diafiltrate_conc_mass_cobalt
            )
            / (m.feed_flow_volume + m.diafiltrate_flow_volume)
        )

    m.initial_retentate_conc_mass_cobalt = Constraint(
        rule=_initial_retentate_conc_mass_cobalt
    )

    def _initial_membrane_interface_lithium(m):
        return m.membrane_conc_mass_lithium[0, 0] == (0 * units.kg / units.m**3)

    m.initial_membrane_interface_lithium = Constraint(
        rule=_initial_membrane_interface_lithium
    )

    def _initial_membrane_interface_cobalt(m):
        return m.membrane_conc_mass_cobalt[0, 0] == (0 * units.kg / units.m**3)

    m.initial_membrane_interface_cobalt = Constraint(
        rule=_initial_membrane_interface_cobalt
    )

    def _initial_membrane_interface_chlorine(m):
        return m.membrane_conc_mass_chlorine[0, 0] == (0 * units.kg / units.m**3)

    m.initial_membrane_interface_chlorine = Constraint(
        rule=_initial_membrane_interface_chlorine
    )

    def _initial_permeate_conc_mass_lithium(m):
        return m.permeate_conc_mass_lithium[0] == (0 * units.kg / units.m**3)

    m.initial_permeate_conc_mass_lithium = Constraint(
        rule=_initial_permeate_conc_mass_lithium
    )

    def _initial_permeate_conc_mass_cobalt(m):
        return m.permeate_conc_mass_cobalt[0] == (0 * units.kg / units.m**3)

    m.initial_permeate_conc_mass_cobalt = Constraint(
        rule=_initial_permeate_conc_mass_cobalt
    )

    def _general_mass_balance_lithium(m, x):
        if x == 0:
            return Constraint.Skip
        return (
            m.retentate_conc_mass_lithium[x] * m.retentate_flow_volume[x]
            + m.permeate_conc_mass_lithium[x] * m.permeate_flow_volume[x]
        ) == (
            m.feed_flow_volume * m.feed_conc_mass_lithium
            + m.diafiltrate_flow_volume * m.diafiltrate_conc_mass_lithium
        )

    m.general_mass_balance_lithium = Constraint(
        m.x_bar, rule=_general_mass_balance_lithium
    )

    def _general_mass_balance_cobalt(m, x):
        if x == 0:
            return Constraint.Skip
        return (
            m.retentate_conc_mass_cobalt[x] * m.retentate_flow_volume[x]
            + m.permeate_conc_mass_cobalt[x] * m.permeate_flow_volume[x]
        ) == (
            m.feed_flow_volume * m.feed_conc_mass_cobalt
            + m.diafiltrate_flow_volume * m.diafiltrate_conc_mass_cobalt
        )

    m.general_mass_balance_cobalt = Constraint(
        m.x_bar, rule=_general_mass_balance_cobalt
    )

    def _initial_d_retentate_conc_mass_lithium_dx(m):
        return m.d_retentate_conc_mass_lithium_dx[0] == (0 * units.kg / units.m**3)

    m.initial_d_retentate_conc_mass_lithium_dx = Constraint(
        rule=_initial_d_retentate_conc_mass_lithium_dx
    )

    def _initial_d_retentate_conc_mass_cobalt_dx(m):
        return m.d_retentate_conc_mass_cobalt_dx[0] == (0 * units.kg / units.m**3)

    m.initial_d_retentate_conc_mass_cobalt_dx = Constraint(
        rule=_initial_d_retentate_conc_mass_cobalt_dx
    )

    def _initial_d_retentate_flow_volume_dx(m):
        return m.d_retentate_flow_volume_dx[0] == (0 * units.m**3 / units.h)

    m._initial_d_retentate_flow_volume_dx = Constraint(
        rule=_initial_d_retentate_flow_volume_dx
    )

    return m


def discretize_model(m, NFEx, NFEz):
    discretizer_findif = TransformationFactory("dae.finite_difference")
    discretizer_findif.apply_to(m, wrt=m.x_bar, nfe=NFEx, scheme="FORWARD")
    discretizer_findif.apply_to(m, wrt=m.z_bar, nfe=NFEz, scheme="FORWARD")


def solve_model(m):
    solver = SolverFactory("ipopt")
    # solver.options = {"max_iter":5000}
    solver.solve(m, tee=True)


def set_scaling(m):
    """
    Apply scaling factors to certain constraints to improve solver performance

    Args:
        m: Pyomo model
    """
    m.scaling_factor = Suffix(direction=Suffix.EXPORT)

    # Add scaling factors for poorly scaled variables
    for x in m.x_bar:
        for z in m.z_bar:
            m.scaling_factor[m.D_lithium_lithium[x, z]] = 1e6
            m.scaling_factor[m.D_lithium_cobalt[x, z]] = 1e6
            m.scaling_factor[m.D_cobalt_lithium[x, z]] = 1e6
            m.scaling_factor[m.D_cobalt_cobalt[x, z]] = 1e6

            m.scaling_factor[m.volume_flux_water[x]] = 1e2
            m.scaling_factor[m.mass_flux_lithium[x]] = 1e2
            m.scaling_factor[m.mass_flux_cobalt[x]] = 1e2
            m.scaling_factor[m.mass_flux_chlorine[x]] = 1e2

    # Add scaling factors for poorly scaled constraints
    for x in m.x_bar:
        for z in m.z_bar:
            m.scaling_factor[m.D_lithium_lithium_calculation[x, z]] = 1e12
            m.scaling_factor[m.D_lithium_cobalt_calculation[x, z]] = 1e12
            m.scaling_factor[m.D_cobalt_lithium_calculation[x, z]] = 1e12
            m.scaling_factor[m.D_cobalt_cobalt_calculation[x, z]] = 1e12

            if z != 0:
                m.scaling_factor[m.lithium_flux_membrane[x, z]] = 1e2
                m.scaling_factor[m.cobalt_flux_membrane[x, z]] = 1e2


def unfix_dof(m):
    m.L.unfix()


def optimize(m):
    @m.Expression()
    def lithium_recovery(m):
        return (
            m.permeate_conc_mass_lithium[value(m.w)] / m.retentate_conc_mass_lithium[0]
        )

    m.lithium_objective = Objective(expr=m.lithium_recovery, sense=maximize)

    solve_model(m)


def plot_results(m):
    x_plot = []
    conc_ret_lith = []
    conc_perm_lith = []
    conc_ret_cob = []
    conc_perm_cob = []

    water_flux = []
    lithium_flux = []

    for x_val in m.x_bar:
        x_plot.append(x_val * value(m.w))
        conc_ret_lith.append(value(m.retentate_conc_mass_lithium[x_val]))
        conc_perm_lith.append(value(m.permeate_conc_mass_lithium[x_val]))
        conc_ret_cob.append(value(m.retentate_conc_mass_cobalt[x_val]))
        conc_perm_cob.append(value(m.permeate_conc_mass_cobalt[x_val]))

        water_flux.append(value(m.volume_flux_water[x_val]))
        lithium_flux.append(value(m.mass_flux_lithium[x_val]))

    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, dpi=125, figsize=(10, 7)
    )

    ax1.plot(x_plot, conc_ret_lith, linewidth=2)
    ax1.set_ylim(1.3, 1.4)
    ax1.set_ylabel(
        "Retentate-side Lithium\n Concentration (kg/m3)", fontsize=10, fontweight="bold"
    )
    ax1.tick_params(direction="in", labelsize=10)

    ax2.plot(x_plot, conc_perm_lith, linewidth=2)
    ax2.set_ylabel(
        "Permeate-side Lithium\n Concentration (kg/m3)", fontsize=10, fontweight="bold"
    )
    ax2.tick_params(direction="in", labelsize=10)

    ax3.plot(x_plot, conc_ret_cob, linewidth=2)
    ax3.set_ylim(13, 13.5)
    ax3.set_ylabel(
        "Retentate-side Cobalt\n Concentration (kg/m3)", fontsize=10, fontweight="bold"
    )
    ax3.tick_params(direction="in", labelsize=10)

    ax4.plot(x_plot, conc_perm_cob, linewidth=2)
    ax4.set_ylabel(
        "Permeate-side Cobalt\n Concentration (kg/m3)", fontsize=10, fontweight="bold"
    )
    ax4.tick_params(direction="in", labelsize=10)

    ax5.plot(x_plot, water_flux, linewidth=2)
    ax5.set_xlabel("Membrane Length (m)", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Water Flux (m3/m2/h)", fontsize=10, fontweight="bold")
    ax5.tick_params(direction="in", labelsize=10)

    ax6.plot(x_plot, lithium_flux, linewidth=2)
    ax6.set_xlabel("Membrane Length (m)", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Mass Flux of Lithium\n (kg/m2/h)", fontsize=10, fontweight="bold")
    ax6.tick_params(direction="in", labelsize=10)

    plt.show()


def plot_membrane_results(m):
    x_vals = []
    z_vals = []

    for x_val in m.x_bar:
        x_vals.append(x_val)
    for z_val in m.z_bar:
        z_vals.append(z_val)

    c_lith_mem = []
    c_cob_mem = []
    c_chl_mem = []

    c_lith_mem_dict = {}
    c_cob_mem_dict = {}
    c_chl_mem_dict = {}

    for z_val in m.z_bar:
        for x_val in m.x_bar:
            c_lith_mem.append(value(m.membrane_conc_mass_lithium[x_val, z_val]))
            c_cob_mem.append(value(m.membrane_conc_mass_cobalt[x_val, z_val]))
            c_chl_mem.append(value(m.membrane_conc_mass_chlorine[x_val, z_val]))

        c_lith_mem_dict[f"{z_val}"] = c_lith_mem
        c_cob_mem_dict[f"{z_val}"] = c_cob_mem
        c_chl_mem_dict[f"{z_val}"] = c_chl_mem
        c_lith_mem = []
        c_cob_mem = []
        c_chl_mem = []

    c_lith_mem_df = DataFrame(index=x_vals, data=c_lith_mem_dict)
    c_cob_mem_df = DataFrame(index=x_vals, data=c_cob_mem_dict)
    c_chl_mem_df = DataFrame(index=x_vals, data=c_chl_mem_dict)

    figs, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=125, figsize=(15, 7))
    sns.heatmap(
        ax=ax1,
        data=c_lith_mem_df,
        cmap="mako",
    )
    ax1.tick_params(axis="x", labelrotation=45)
    ax1.set_xlabel("z (dimensionless)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("x (dimensionless)", fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_title(
        "Lithium Concentration\n in Membrane (kg/m3)", fontsize=10, fontweight="bold"
    )
    ax1.tick_params(direction="in", labelsize=10)

    sns.heatmap(
        ax=ax2,
        data=c_cob_mem_df,
        cmap="mako",
    )
    ax2.tick_params(axis="x", labelrotation=45)
    ax2.set_xlabel("z (dimensionless)", fontsize=10, fontweight="bold")
    # ax2.set_ylabel("x (dimensionless)", fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_title(
        "Cobalt Concentration\n in Membrane (kg/m3)", fontsize=10, fontweight="bold"
    )
    ax2.tick_params(direction="in", labelsize=10)

    sns.heatmap(
        ax=ax3,
        data=c_chl_mem_df,
        cmap="mako",
    )
    ax3.tick_params(axis="x", labelrotation=45)
    ax3.set_xlabel("z (dimensionless)", fontsize=10, fontweight="bold")
    # ax3.set_ylabel("x (dimensionless)", fontsize=10, fontweight="bold")
    ax3.invert_yaxis()
    ax3.set_title(
        "Chlorine Concentration\n in Membrane (kg/m3)", fontsize=10, fontweight="bold"
    )
    ax3.tick_params(direction="in", labelsize=10)

    plt.show()


if __name__ == "__main__":
    main()
