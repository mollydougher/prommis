from pyomo.dae import (
    ContinuousSet,
    DerivativeVar,
)
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    maximize,
    NonNegativeReals,
    Objective,
    Param,
    SolverFactory,
    TransformationFactory,
    units,
    value,
    Var,
)
from idaes.core.util.constants import Constants
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import report_statistics

import matplotlib.pyplot as plt


def main():
    m = build_model()

    report_statistics(m)

    dt = DiagnosticsToolbox(m)
    # dt.assert_no_structural_warnings()
    dt.report_structural_issues()
    dt.display_underconstrained_set()
    dt.display_overconstrained_set()

    solve_model(m)
    # dt.assert_no_numerical_warnings()
    dt.report_numerical_issues()
    dt.display_constraints_with_large_residuals()
    dt.display_variables_at_or_outside_bounds()
    # dt.display_near_parallel_constraints()
    # dt.display_unused_variables()

    # svd = dt.prepare_svd_toolbox()
    # svd.display_rank_of_equality_constraints()

    # dh = dt.prepare_degeneracy_hunter(solver="gurobi")
    # dh.report_irreducible_degenerate_sets()

    discretize_model(m, NFEx=15, NFEz=10)
    report_statistics(m)

    dt = DiagnosticsToolbox(m)
    dt.report_structural_issues()
    # dt.display_underconstrained_set()
    # dt.display_overconstrained_set()
    # dt.display_components_with_inconsistent_units()
    dt.display_unused_variables()

    # solve_model(m)
    # dt.report_numerical_issues()
    # dt.display_constraints_with_large_residuals()
    # dt.display_variables_at_or_outside_bounds()

    # unfix_dof(m)
    # optimize(m)

    # m.L.display()

    plot_results(m)


# testing


def build_model():
    """
    Adds model equations for the single-salt diafiltraiton system without
    the inclusion of a boundary layer.
    Considers convenction and diffustion (and electromigration) transport
    mechanisms.

    References:
        diffusion coefficients: https://www.aqion.de/site/diffusion-coefficients

    Returns:
        m: the DAE system, a pyomo model
    """
    # create the model
    m = ConcreteModel()

    # define the model parameters
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
    m.z_chlorine = Param(
        initialize=-1,
        units=units.dimensionless,
        doc="Charge of coblat ion",
    )
    m.molar_mass_lithium = Param(
        initialize=0.006941, units=units.kg / units.mol, doc="Molar mass of lithium"
    )
    m.molar_mass_chlorine = Param(
        initialize=0.03545, units=units.kg / units.mol, doc="Molar mass of chlorine"
    )
    m.D_lithium = Param(
        initialize=3.7e-6,
        units=units.m**2 / units.h,
        doc="Diffusion coefficient for lithium ion in water",
    )
    m.D_chlorine = Param(
        initialize=7.3e-6,
        units=units.m**2 / units.h,
        doc="Diffusion coefficient for chlorine ion in water",
    )
    m.Lp = Param(
        initialize=0.003,  # TODO: verify
        units=units.m / units.h / units.bar,
        doc="Hydraulic permeability coefficient",
    )
    m.sigma_lithium = Param(
        initialize=1,
        units=units.dimensionless,
        doc="Thermodynamic reflection coefficient for lithium ion",
    )
    m.sigma_chlorine = Param(
        initialize=1,
        units=units.dimensionless,
        doc="Thermodynamic reflection coefficient for chlorine ion",
    )
    m.feed_flow_volume = Param(
        initialize=100,
        units=units.m**3 / units.h,
        doc="Volumetric flow rate of the membrane feed",
    )
    m.feed_conc_mass_lithium = Param(
        initialize=1.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of lithium in the membrane feed",
    )
    m.feed_conc_mass_chlorine = Param(
        initialize=8.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of chlorine in the membrane feed",
    )
    m.diafiltrate_flow_volume = Param(
        initialize=30,
        units=units.m**3 / units.h,
        doc="Volumetric flow rate of the membrane diafiltrate",
    )
    m.diafiltrate_conc_mass_lithium = Param(
        initialize=0.1,
        units=units.kg / units.m**3,
        doc="Mass concentration of lithium in the membrane diafiltrate",
    )
    m.diafiltrate_conc_mass_chlorine = Param(
        initialize=0.5,
        units=units.kg / units.m**3,
        doc="Mass concentration of chlorine in the membrane diafiltrate",
    )

    # define length scales
    m.x = ContinuousSet(bounds=(0, value(m.w)))
    m.z = ContinuousSet(bounds=(0, value(m.l)))

    # define remaining algebraic variables
    ## independent of x,z length scale
    m.L = Var(
        initialize=500,
        units=units.m,
        doc="Length of the membrane, wound radially",
    )
    m.L.fix()  # fix for simulation
    m.dP = Var(
        initialize=10,  # TODO: verify 10 bar reasoanable
        units=units.bar,
        doc="Pressure applied to membrane",
    )
    m.dP.fix()  # fix for simulation

    ## dependent on x
    m.volume_flux_water = Var(
        m.x,
        initialize=0.03,
        units=units.m**3 / units.m**2 / units.h,
        domain=NonNegativeReals,
        doc="Volumetric water flux of water across the membrane",
    )
    m.mass_flux_lithium = Var(
        m.x,
        initialize=0.05,  # TODO: verify good value
        units=units.kg / units.m**2 / units.h,
        domain=NonNegativeReals,
        doc="Mass flux of lithium across the membrane (z-direction, x-dependent)",
    )
    m.mass_flux_chlorine = Var(
        m.x,
        initialize=0.05,  # TODO: verify good value
        units=units.kg / units.m**2 / units.h,
        domain=NonNegativeReals,
        doc="Mass flux of chlorine across the membrane (z-direction, x-dependent)",
    )
    m.retentate_flow_volume = Var(
        m.x,
        initialize=130,
        units=units.m**3 / units.h,
        domain=NonNegativeReals,
        doc="Volumetric flow rate of the retentate, x-dependent",
    )
    m.retentate_conc_mass_lithium = Var(
        m.x,
        initialize=1.33,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of lithium in the retentate, x-dependent",
    )
    m.retentate_conc_mass_chlorine = Var(
        m.x,
        initialize=6.81,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of chlorine in the retentate, x-dependent",
    )
    m.permeate_flow_volume = Var(
        m.x,
        initialize=0,
        units=units.m**3 / units.h,
        domain=NonNegativeReals,
        doc="Volumetric flow rate of the permeate, x-dependent",
    )
    m.permeate_conc_mass_lithium = Var(
        m.x,
        initialize=0,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of lithium in the permeate, x-dependent",
    )
    m.permeate_conc_mass_chlorine = Var(
        m.x,
        initialize=0,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of chlorine in the retentate, x-dependent",
    )
    m.osmotic_pressure = Var(
        m.x,
        initialize=5,
        units=units.bar,
        domain=NonNegativeReals,
        doc="Osmostic pressure of the feed-side fluid",
    )

    ## dependent on z_hat and x_hat
    m.membrane_conc_mass_lithium = Var(
        m.x,
        m.z,
        initialize=1.7,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of lithium in the membrane, x- and z-dependent",
    )
    m.membrane_conc_mass_chlorine = Var(
        m.x,
        m.z,
        initialize=8.7,
        units=units.kg / units.m**3,
        domain=NonNegativeReals,
        doc="Mass concentration of chlorine in the membrane, x- and z-dependent",
    )

    # define the (partial) derivative variables
    m.d_retentate_conc_mass_lithium_dx = DerivativeVar(
        m.retentate_conc_mass_lithium,
        wrt=m.x,
        units=units.kg / units.m**3 / units.m,
    )
    m.d_retentate_flow_volume_dx = DerivativeVar(
        m.retentate_flow_volume,
        wrt=m.x,
        units=units.m**3 / units.h / units.m,
    )
    m.d_membrane_conc_mass_lithium_dz = DerivativeVar(
        m.membrane_conc_mass_lithium,
        wrt=m.z,
        units=units.kg / units.m**3 / units.m,
    )

    # define the constraints
    ## mass balance constraints
    def _overall_mass_balance(m, x):
        # if x == 0 or x == value(m.w):
        if x == 0:
            return Constraint.Skip
        return m.d_retentate_flow_volume_dx[x] == (-m.volume_flux_water[x] * m.L)

    m.overall_mass_balance = Constraint(m.x, rule=_overall_mass_balance)

    def _lithium_mass_balance(m, x):
        if x == 0:
            return Constraint.Skip
        return (m.retentate_flow_volume[x] * m.d_retentate_conc_mass_lithium_dx[x]) == (
            (
                m.volume_flux_water[x] * m.retentate_conc_mass_lithium[x]
                - m.mass_flux_lithium[x]
            )
            * m.L
        )

    m.lithium_mass_balance = Constraint(m.x, rule=_lithium_mass_balance)

    ## transport constraints
    def _geometric_flux_equation_overall(m, x):
        if x == 0:
            return Constraint.Skip
        return m.permeate_flow_volume[x] == m.volume_flux_water[x] * x * units.m * m.L

    m.geometric_flux_equation_overall = Constraint(
        m.x, rule=_geometric_flux_equation_overall
    )

    def _geometric_flux_equation_lithium(m, x):
        # if x == 0:
        #     return Constraint.Skip
        return m.mass_flux_lithium[x] == (
            m.permeate_conc_mass_lithium[x] * m.volume_flux_water[x]
        )

    m.geometric_flux_equation_lithium = Constraint(
        m.x, rule=_geometric_flux_equation_lithium
    )

    def _lumped_water_flux(m, x):
        return m.volume_flux_water[x] == (m.Lp * (m.dP - m.osmotic_pressure[x]))

    m.lumped_water_flux = Constraint(m.x, rule=_lumped_water_flux)

    def _lithium_flux_membrane(m, x, z):
        if x == 0 and z == 0:
            # if x == 0:
            return Constraint.Skip
        if x == value(m.w) and z == 0:
            # if x == value(m.w):
            return Constraint.Skip
        return m.mass_flux_lithium[x] == (
            m.membrane_conc_mass_lithium[x, z] * m.volume_flux_water[x]
            - (
                m.D_lithium
                * m.D_chlorine
                * (m.z_lithium - m.z_chlorine)
                / (m.D_lithium * m.z_lithium - m.D_chlorine * m.z_chlorine)
            )
            * m.d_membrane_conc_mass_lithium_dz[x, z]
        )

    m.lithium_flux_membrane = Constraint(m.x, m.z, rule=_lithium_flux_membrane)

    def _chlorine_flux_membrane(m, x):
        return (
            m.mass_flux_chlorine[x]
            == m.molar_mass_chlorine
            * (-m.z_lithium / m.z_chlorine * m.mass_flux_lithium[x])
            / m.molar_mass_lithium
        )

    m.chlorine_flux_membrane = Constraint(m.x, rule=_chlorine_flux_membrane)

    ## other physical constaints
    def _osmotic_pressure_calculation(m, x):
        return m.osmotic_pressure[x] == units.convert(
            (
                (
                    2  # two dissociated ions (Li+ and Cl-)
                    * Constants.gas_constant  # J / mol / K
                    * 298
                    * units.K  # assume room temp
                )
                * (
                    m.sigma_lithium
                    * (
                        m.retentate_conc_mass_lithium[x]
                        - m.permeate_conc_mass_lithium[x]
                    )
                    / m.molar_mass_lithium
                    + m.sigma_chlorine
                    * (
                        m.retentate_conc_mass_chlorine[x]
                        - m.permeate_conc_mass_chlorine[x]
                    )
                    / m.molar_mass_chlorine
                )
            ),
            to_units=units.bar,
        )

    m.osmotic_pressure_calcualation = Constraint(
        m.x, rule=_osmotic_pressure_calculation
    )

    ## boundary conditions
    def _retentate_membrane_interface_lithium(m, x):
        return m.retentate_conc_mass_lithium[x] == m.membrane_conc_mass_lithium[x, 0]

    m.retentate_membrane_interface_lithium = Constraint(
        m.x, rule=_retentate_membrane_interface_lithium
    )

    def _retentate_membrane_interface_chlorine(m, x):
        return m.retentate_conc_mass_chlorine[x] == m.membrane_conc_mass_chlorine[x, 0]

    m.retentate_membrane_interface_chlorine = Constraint(
        m.x, rule=_retentate_membrane_interface_chlorine
    )

    def _membrane_permeate_interface_lithium(m, x):
        return (
            m.permeate_conc_mass_lithium[x]
            == m.membrane_conc_mass_lithium[x, value(m.l)]
        )

    m.membrane_permeate_interface_lithium = Constraint(
        m.x, rule=_membrane_permeate_interface_lithium
    )

    def _membrane_permeate_interface_chlorine(m, x):
        return (
            m.permeate_conc_mass_chlorine[x]
            == m.membrane_conc_mass_chlorine[x, value(m.l)]
        )

    m.membrane_permeate_interface_chlorine = Constraint(
        m.x, rule=_membrane_permeate_interface_chlorine
    )

    def _electroneutrality_retentate(m, x):
        return 0 == (
            m.z_lithium * m.retentate_conc_mass_lithium[x] / m.molar_mass_lithium
            + m.z_chlorine * m.retentate_conc_mass_chlorine[x] / m.molar_mass_chlorine
        )

    m.electroneutrality_retentate = Constraint(m.x, rule=_electroneutrality_retentate)

    def _electroneutrality_membrane(m, x, z):
        if z == 0:
            return Constraint.Skip
        return 0 == (
            m.z_lithium * m.membrane_conc_mass_lithium[x, z] / m.molar_mass_lithium
            + m.z_chlorine * m.membrane_conc_mass_chlorine[x, z] / m.molar_mass_chlorine
        )

    m.electroneutrality_membrane = Constraint(
        m.x, m.z, rule=_electroneutrality_membrane
    )

    ## initial/final conditions
    def _initial_retentate_flow_volume(m):
        return (
            (value(m.feed_flow_volume) + value(m.diafiltrate_flow_volume))
            * units.m**3
            / units.h
        )

    m.initial_retentate_flow_volume = Expression(rule=_initial_retentate_flow_volume)

    m.retentate_flow_volume[0].fix(m.initial_retentate_flow_volume)

    # m.permeate_flow_volume[0].fix(1e-4)
    # m.permeate_conc_mass_lithium[0].fix(1e-4)

    def _general_flow_balance(m, x):
        if x != 0 and x != value(m.w):
        # if x != value(m.w):
            return Constraint.Skip
        return m.permeate_flow_volume[x] == (
            (
                (value(m.feed_flow_volume) + value(m.diafiltrate_flow_volume))
                * units.m**3
                / units.h
            )
            - m.retentate_flow_volume[x]
        )

    m.general_flow_balance = Constraint(m.x, rule=_general_flow_balance)

    # def _final_flow_balance(m):
    #     return m.permeate_flow_volume[value(m.w)] == (
    #         (
    #             (value(m.feed_flow_volume) + value(m.diafiltrate_flow_volume))
    #             * units.m**3
    #             / units.h
    #         )
    #         - m.retentate_flow_volume[value(m.w)]
    #     )

    # m.final_flow_balance = Constraint(rule=_final_flow_balance)

    def _initial_retentate_conc_mass_lithium(m):
        return (
            (
                value(m.feed_flow_volume)
                * units.m**3
                / units.h
                * value(m.feed_conc_mass_lithium)
                * units.kg
                / units.m**3
            )
            + (
                value(m.diafiltrate_flow_volume)
                * units.m**3
                / units.h
                * value(m.diafiltrate_conc_mass_lithium)
                * units.kg
                / units.m**3
            )
        ) / (
            (value(m.feed_flow_volume) + value(m.diafiltrate_flow_volume))
            * units.m**3
            / units.h
        )

    m.initial_retentate_conc_mass_lithium = Expression(
        rule=_initial_retentate_conc_mass_lithium
    )

    m.retentate_conc_mass_lithium[0].fix(m.initial_retentate_conc_mass_lithium)

    def _general_mass_balance_lithium(m, x):
        if x != 0 and x != value(m.w):
        # if x != value(m.w):
            return Constraint.Skip
        return m.permeate_conc_mass_lithium[x] * m.permeate_flow_volume[x] == (
            (
                value(m.feed_flow_volume)
                * units.m**3
                / units.h
                * value(m.feed_conc_mass_lithium)
                * units.kg
                / units.m**3
            )
            + (
                value(m.diafiltrate_flow_volume)
                * units.m**3
                / units.h
                * value(m.diafiltrate_conc_mass_lithium)
                * units.kg
                / units.m**3
            )
            - m.retentate_conc_mass_lithium[x] * m.retentate_flow_volume[x]
        )

    m.general_mass_balance_lithium = Constraint(m.x, rule=_general_mass_balance_lithium)

    # def _final_mass_balance_lithium(m):
    #     return m.permeate_conc_mass_lithium[value(m.w)] * m.permeate_flow_volume[value(m.w)] == (
    #         (
    #             value(m.feed_flow_volume)
    #             * units.m**3
    #             / units.h
    #             * value(m.feed_conc_mass_lithium)
    #             * units.kg
    #             / units.m**3
    #         )
    #         + (
    #             value(m.diafiltrate_flow_volume)
    #             * units.m**3
    #             / units.h
    #             * value(m.diafiltrate_conc_mass_lithium)
    #             * units.kg
    #             / units.m**3
    #         )
    #         - m.retentate_conc_mass_lithium[value(m.w)] * m.retentate_flow_volume[value(m.w)]
    #     )

    # m.final_mass_balance_lithium = Constraint(
    #     rule=_final_mass_balance_lithium
    # )

    m.d_retentate_conc_mass_lithium_dx[value(m.w)].fix(0)

    return m


def discretize_model(m, NFEx, NFEz):
    discretizer_findif = TransformationFactory("dae.finite_difference")
    discretizer_findif.apply_to(m, wrt=m.x, nfe=NFEx, scheme="FORWARD")
    discretizer_findif.apply_to(m, wrt=m.z, nfe=NFEz, scheme="FORWARD")

    # discretizer_col = TransformationFactory("dae.collocation")
    # discretizer_col.apply_to(m, wrt=m.x, nfe=NFEx, ncp=3, scheme="LAGRANGE-RADAU")
    # discretizer_col.apply_to(m, wrt=m.z, nfe=NFEz, ncp=3, scheme="LAGRANGE-RADAU")


def solve_model(m):
    solver = SolverFactory("ipopt")
    solver.solve(m, tee=True)


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
    x = []
    conc_ret_lith = []
    conc_perm_lith = []

    water_flux = []
    lithium_flux = []

    for x_val in m.x:
        x.append(x_val)
        conc_ret_lith.append(value(m.retentate_conc_mass_lithium[x_val]))
        conc_perm_lith.append(value(m.permeate_conc_mass_lithium[x_val]))

        water_flux.append(value(m.volume_flux_water[x_val]))
        lithium_flux.append(value(m.mass_flux_lithium[x_val]))

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.plot(x, conc_ret_lith)
    ax1.set_xlabel("Membrane Length (m)")
    ax1.set_ylabel("Retentate-side Lithium Concentration (kg/m3)")

    ax2.plot(x, conc_perm_lith)
    ax2.set_xlabel("Membrane Length (m)")
    ax2.set_ylabel("Permeate-side Lithium Concentration (kg/m3)")

    ax3.plot(x, water_flux)
    ax3.set_xlabel("Membrane Length (m)")
    ax3.set_ylabel("Water Flux (m3/m2/h)")

    ax4.plot(x, lithium_flux)
    ax4.set_xlabel("Membrane Length (m)")
    ax4.set_ylabel("Mass Flux of Lithium (kg/m2/h)")

    plt.show()


if __name__ == "__main__":
    main()
