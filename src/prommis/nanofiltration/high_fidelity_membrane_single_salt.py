from pyomo.dae import (
    ContinuousSet,
    DerivativeVar,
)
from pyomo.environ import (
    ConcreteModel,
    Constraint,
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


def main():
    m = build_model()

    report_statistics(m)

    dt = DiagnosticsToolbox(m)
    dt.report_structural_issues()

    discretize_model(m)
    report_statistics(m)

    dt = DiagnosticsToolbox(m)
    dt.report_structural_issues()

    solve_model(m)


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
        initialize=100,
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
        initialize=0.1,
        units=units.m**3 / units.m**2 / units.h,
        doc="Volumetric water flux of water across the membrane",
    )
    m.mass_flux_lithium = Var(
        m.x,
        initialize=1,  # TODO: verify good value
        units=units.kg / units.m**2 / units.h,
        doc="Mass flux of lithium across the membrane",
    )
    m.mass_flux_chlorine = Var(
        m.x,
        initialize=1,  # TODO: verify good value
        units=units.kg / units.m**2 / units.h,
        doc="Mass flux of chlorine across the membrane",
    )
    m.retentate_flow_volume = Var(
        m.x,
        initialize=98,
        units=units.m**3 / units.h,
        doc="Volumetric flow rate of the retentate, x-dependent",
    )
    m.retentate_conc_mass_lithium = Var(
        m.x,
        initialize=1.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of lithium in the retentate, x-dependent",
    )
    m.retentate_conc_mass_chlorine = Var(
        m.x,
        initialize=8.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of chlorine in the retentate, x-dependent",
    )
    m.permeate_flow_volume = Var(
        m.x,
        initialize=1,
        units=units.m**3 / units.h,
        doc="Volumetric flow rate of the permeate, x-dependent",
    )
    m.permeate_conc_mass_lithium = Var(
        m.x,
        initialize=1.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of lithium in the permeate, x-dependent",
    )
    m.permeate_conc_mass_chlorine = Var(
        m.x,
        initialize=8.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of chlorine in the retentate, x-dependent",
    )
    m.osmotic_pressure = Var(
        m.x,
        initialize=5,
        units=units.bar,
        doc="Osmostic pressure of the feed-side fluid",
    )

    ## dependent on z_hat and x_hat
    m.membrane_conc_mass_lithium = Var(
        m.x,
        m.z,
        initialize=1.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of lithium in the membrane, x- and z-dependent",
    )
    m.membrane_conc_mass_chlorine = Var(
        m.x,
        m.z,
        initialize=8.7,
        units=units.kg / units.m**3,
        doc="Mass concentration of chlorine in the membrane, x- and z-dependent",
    )

    # define the (partial) derivative variables
    m.d_membrane_conc_mass_lithium_dz = DerivativeVar(
        m.membrane_conc_mass_lithium,
        wrt=m.z,
        units=units.kg / units.m**3 / units.m,
    )

    # define the constraints
    ## mass balance constraints
    def _overall_mass_balance(m, x):
        return (m.feed_flow_volume + m.diafiltrate_flow_volume) == (
            m.retentate_flow_volume[x] + m.permeate_flow_volume[x]
        )

    m.overall_mass_balance = Constraint(m.x, rule=_overall_mass_balance)

    # TODO: re-cast this constraint as a diff. eq.
    def _lithium_mass_balance(m, x):
        return (
            (m.feed_flow_volume * m.feed_conc_mass_lithium)
            + (m.diafiltrate_flow_volume * m.diafiltrate_conc_mass_lithium)
        ) == (
            (m.retentate_flow_volume[x] * m.retentate_conc_mass_lithium[x])
            + (m.permeate_flow_volume[x] * m.permeate_conc_mass_lithium[x])
        )

    m.lithium_mass_balance = Constraint(m.x, rule=_lithium_mass_balance)

    # chlorine mass balance accounted for with electroneutrality

    ## transport constraints
    def _geometric_flux_equation(m, x):
        return m.permeate_flow_volume[x] == m.volume_flux_water[x] * (x * units.m) * m.L

    m.geometric_flux_equation = Constraint(m.x, rule=_geometric_flux_equation)

    def _lumped_water_flux(m, x):
        return m.volume_flux_water[x] == (m.Lp * (m.dP - m.osmotic_pressure[x]))

    m.lumped_water_flux = Constraint(m.x, rule=_lumped_water_flux)

    def _lithium_flux_membrane(m, x, z):
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
        return m.mass_flux_chlorine[x] == (
            - m.z_lithium / m.z_chlorine
            * m.mass_flux_lithium[x]
        )

    m.chlorine_flux_membrane = Constraint(m.x, rule=_chlorine_flux_membrane)

    def _lithium_flux_bulk(m, x):
        return m.mass_flux_lithium[x] == (
            m.retentate_conc_mass_lithium[x] * m.volume_flux_water[x]
        )

    m.lithium_flux_bulk = Constraint(m.x, rule=_lithium_flux_bulk)

    ## other physical constaints
    def _osmotic_pressure_calculation(m, x):
        return m.osmotic_pressure[x] == units.convert(
            (
                2  # two dissociated ions (Li+ and Cl-)
                * Constants.gas_constant  # J / mol / K
                * 298
                * units.K  # assume room temp
            )
            * (
                m.sigma_lithium
                * (m.retentate_conc_mass_lithium[x] - m.permeate_conc_mass_lithium[x])
                / m.molar_mass_lithium
                + m.sigma_chlorine
                * (m.retentate_conc_mass_chlorine[x] - m.permeate_conc_mass_chlorine[x])
                / m.molar_mass_chlorine
            ),
            to_units=units.bar,
        )

    m.osmotic_pressure_calcualation = Constraint(
        m.x, rule=_osmotic_pressure_calculation
    )

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
            m.permeate_conc_mass_lithium[x] == m.membrane_conc_mass_lithium[x, value(m.l)]
        )

    m.membrane_permeate_interface_lithium = Constraint(
        m.x, rule=_membrane_permeate_interface_lithium
    )

    def _membrane_permeate_interface_chlorine(m, x):
        return (
            m.permeate_conc_mass_chlorine[x] == m.membrane_conc_mass_chlorine[x, value(m.l)]
        )

    m.membrane_permeate_interface_chlorine = Constraint(
        m.x, rule=_membrane_permeate_interface_chlorine
    )

    def _electroneutrality_retentate(m,x):
        return 0 == (
            m.z_lithium * m.retentate_conc_mass_lithium[x]
            + m.z_chlorine * m.retentate_conc_mass_chlorine[x]
        )

    m._electroneutrality_retentate = Constraint(m.x, rule=_electroneutrality_retentate)

    def _electroneutrality_membrane(m, x, z):
        return 0 == (
            m.z_lithium * m.membrane_conc_mass_lithium[x,z]
            + m.z_chlorine * m.membrane_conc_mass_chlorine[x,z]
        )

    m.electroneutrality_membrane = Constraint(m.x, m.z, rule=_electroneutrality_membrane)

    return m


def discretize_model(m):
    discretizer_findif = TransformationFactory("dae.finite_difference")
    discretizer_findif.apply_to(m, wrt=m.x, nfe=2, scheme="FORWARD")
    # discretizer_findif.apply_to(m, wrt=m.z, nfe=2, scheme="FORWARD")

    discretizer_col = TransformationFactory("dae.collocation")
    # discretizer_col.apply_to(m,wrt=m.x,nfe=2,ncp=2,scheme="LAGRANGE-RADAU")
    discretizer_col.apply_to(m,wrt=m.z,nfe=2,ncp=3,scheme="LAGRANGE-RADAU")


def solve_model(m):
    solver = SolverFactory("ipopt")
    solver.solve(m, tee=True)


if __name__ == "__main__":
    main()
