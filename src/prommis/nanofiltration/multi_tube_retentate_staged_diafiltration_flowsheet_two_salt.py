#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Sample flowsheet for the multi-component diafiltration cascade.

Author: Molly Dougher
"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    maximize,
    units,
    value,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

import matplotlib.pyplot as plt
from pandas import DataFrame

from prommis.nanofiltration.multi_component_diafiltration_stream_properties import (
    MultiComponentDiafiltrationStreamParameter,
)
from prommis.nanofiltration.multi_component_diafiltration_solute_properties import (
    MultiComponentDiafiltrationSoluteParameter,
)
from prommis.nanofiltration.multi_component_diafiltration import (
    MultiComponentDiafiltration,
)


def main():
    """ """
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    cation_list = ["Na", "Ca"]
    anion_list = ["Cl"]
    non_Donnan_partition_dict = {
        "Na": 0.7,
        "Ca": 0.3,
        "Cl": 0.1,
    }  # informed from data and sensitivity analysis

    m.fs.stream_properties = MultiComponentDiafiltrationStreamParameter(
        cation_list=cation_list,
        anion_list=anion_list,
    )
    m.fs.properties = MultiComponentDiafiltrationSoluteParameter(
        cation_list=cation_list,
        anion_list=anion_list,
        non_Donnan_partition_dict=non_Donnan_partition_dict,
    )

    # add feed blocks for feed and diafiltrate streams
    m.fs.feed_block_stage_one = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block_stage_one = Feed(property_package=m.fs.stream_properties)
    m.fs.feed_block_stage_two = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block_stage_two = Feed(property_package=m.fs.stream_properties)

    Dm_Cl = units.convert(
        m.fs.properties.membrane_diffusion_coefficient[anion_list[0]],
        to_units=units.um**2 / units.s,
    )
    Dm_over_l_value = 40
    l_um = value(Dm_Cl) / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m

    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    # add the membrane unit models
    m.fs.membrane_one = MultiComponentDiafiltration(
        property_package=m.fs.properties,
        cation_list=cation_list,
        anion_list=anion_list,
        include_boundary_layer=include_boundary_layer,
        total_membrane_thickness=l_m,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )
    # m.fs.membrane_one.total_module_length.set_value(8)
    m.fs.membrane_two = MultiComponentDiafiltration(
        property_package=m.fs.properties,
        cation_list=cation_list,
        anion_list=anion_list,
        include_boundary_layer=include_boundary_layer,
        total_membrane_thickness=l_m,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # add product blocks for retentate and permeate
    m.fs.retentate_block_stage_one = Product(property_package=m.fs.stream_properties)
    m.fs.permeate_block_stage_one = Product(property_package=m.fs.stream_properties)
    m.fs.retentate_block_stage_two = Product(property_package=m.fs.stream_properties)
    m.fs.permeate_block_stage_two = Product(property_package=m.fs.stream_properties)

    inlet_flow_volume = {"feed": 12.5, "diafiltrate": 3.75}
    inlet_concentration = {
        "feed": {"Na": 245, "Ca": 288, "Cl": 821},
        "diafiltrate": {"Na": 14, "Ca": 3, "Cl": 20},
    }

    # fix the degrees of freedom
    fix_variables(m, inlet_flow_volume, inlet_concentration)

    # initialize membrane models
    initialized_membrane_one = m.fs.membrane_one.default_initializer(
        fudge_factor_H_feed=1.2,  # increase can help solver performance
        fudge_factor_H_perm=1,
    )
    initialized_membrane_one.initialize(m.fs.membrane_one)

    dt_membrane_one = DiagnosticsToolbox(m.fs.membrane_one)
    dt_membrane_one.assert_no_structural_warnings()
    solve_model(m.fs.membrane_one)

    for t in m.fs.membrane_one.time:
        m.fs.membrane_two.feed_flow_volume[t].fix(
            value(m.fs.membrane_one.retentate_flow_volume[t, 1])
        )
        for j in m.fs.membrane_one.solutes:
            m.fs.membrane_two.feed_conc_mol_comp[t, j].fix(
                value(m.fs.membrane_one.retentate_conc_mol_comp[t, 1, j])
            )

    initialized_membrane_two = m.fs.membrane_two.default_initializer(
        fudge_factor_H_feed=1.2,  # increase can help solver performance
        fudge_factor_H_perm=1,
    )
    initialized_membrane_two.initialize(m.fs.membrane_two)

    dt_membrane_two = DiagnosticsToolbox(m.fs.membrane_two)
    dt_membrane_two.assert_no_structural_warnings()
    solve_model(m.fs.membrane_two)

    for t in m.fs.membrane_one.time:
        m.fs.membrane_two.feed_flow_volume[t].unfix()
        for j in m.fs.membrane_one.solutes:
            m.fs.membrane_two.feed_conc_mol_comp[t, j].unfix()

    # add and connect flowsheet streams
    add_and_connect_streams(m)

    # check structural warnings
    dt = DiagnosticsToolbox(m)
    dt.assert_no_structural_warnings()

    # solve flowsheet
    solve_model(m)
    dt.assert_no_numerical_warnings()

    # set_objective(m)
    # m.calcium_recovery.display()
    # m.sodium_recovery.display()
    # solve_model(m)
    # dt.assert_no_numerical_warnings()
    # # dt.report_numerical_issues()

    m.fs.membrane_one.applied_pressure.display()
    m.fs.membrane_two.applied_pressure.display()
    m.fs.membrane_one.diafiltrate_flow_volume.display()
    m.fs.membrane_two.diafiltrate_flow_volume.display()
    # m.calcium_recovery.display()
    # m.sodium_recovery.display()

    m.fs.membrane_one.retentate_flow_volume.display()
    m.fs.membrane_one.retentate_conc_mol_comp.display()
    m.fs.membrane_one.permeate_local_conc_mol_comp.display()
    m.fs.membrane_one.permeate_outlet_conc_mol_comp.display()

    m.fs.membrane_two.retentate_flow_volume.display()
    m.fs.membrane_two.retentate_conc_mol_comp.display()
    m.fs.membrane_two.permeate_local_conc_mol_comp.display()
    m.fs.membrane_two.permeate_outlet_conc_mol_comp.display()

    # m.fs.membrane_one.total_feed_conc_mol_comp.display()
    # m.fs.membrane_two.total_feed_flow_volume.display()

    # m.fs.membrane_one.feed_flow_volume.display()
    # m.fs.membrane_one.diafiltrate_flow_volume.display()
    # m.fs.membrane_one.feed_conc_mol_comp.display()
    # m.fs.membrane_one.diafiltrate_conc_mol_comp.display()

    # visualize the results
    plot_results_by_length(m.fs.membrane_one)
    plot_results_by_length(m.fs.membrane_two)
    plt.show()


def fix_variables(m, inlet_flow_volume, inlet_concentration):
    # fix degrees of freedom in the membrane
    m.fs.membrane_one.total_module_length.fix()
    m.fs.membrane_one.total_membrane_length.fix()
    m.fs.membrane_one.applied_pressure.fix(15)

    m.fs.membrane_two.total_module_length.fix()
    m.fs.membrane_two.total_membrane_length.fix()
    m.fs.membrane_two.applied_pressure.fix(15)

    m.fs.membrane_one.feed_flow_volume.fix(inlet_flow_volume["feed"])
    m.fs.membrane_one.diafiltrate_flow_volume.fix(inlet_flow_volume["diafiltrate"])
    m.fs.membrane_two.diafiltrate_flow_volume.fix(
        value(m.fs.membrane_two.numerical_zero_tolerance)
    )

    for t in m.fs.membrane_one.time:
        for j in m.fs.membrane_one.solutes:
            m.fs.membrane_one.feed_conc_mol_comp[t, j].fix(
                inlet_concentration["feed"][j]
            )
            m.fs.membrane_one.diafiltrate_conc_mol_comp[t, j].fix(
                inlet_concentration["diafiltrate"][j]
            )
            m.fs.membrane_two.diafiltrate_conc_mol_comp[t, j].fix(
                value(m.fs.membrane_two.numerical_zero_tolerance)
            )

    # def _permeate_staged_flows(blk):
    #     return (
    #         blk.fs.membrane_one.permeate_flow_volume[0, 1]
    #         == blk.fs.membrane_two.feed_flow_volume[0]
    #     )

    # m.permeate_staged_flows = Constraint(
    #     rule=_permeate_staged_flows,
    # )

    # def _permeate_staged_conc_mol_comp(blk, k):
    #     return (
    #         blk.fs.membrane_one.permeate_conc_mol_comp[0, 1, k]
    #         == blk.fs.membrane_two.feed_conc_mol_comp[0, k]
    #     )

    # m.permeate_staged_conc_mol_comp = Constraint(
    #     m.fs.membrane_one.solutes,
    #     rule=_permeate_staged_conc_mol_comp,
    # )


def add_and_connect_streams(m):
    m.fs.feed_stream_stage_one = Arc(
        source=m.fs.feed_block_stage_one.outlet,
        destination=m.fs.membrane_one.feed_inlet,
    )
    m.fs.diafiltrate_stream_stage_one = Arc(
        source=m.fs.diafiltrate_block_stage_one.outlet,
        destination=m.fs.membrane_one.diafiltrate_inlet,
    )
    m.fs.retentate_stream_stage_one = Arc(
        source=m.fs.membrane_one.retentate_outlet,
        destination=m.fs.membrane_two.feed_inlet,
        # destination=m.fs.retentate_block_stage_one.inlet,
    )
    m.fs.permeate_stream_stage_one = Arc(
        source=m.fs.membrane_one.permeate_outlet,
        destination=m.fs.permeate_block_stage_one.inlet,
    )
    m.fs.diafiltrate_stream_stage_two = Arc(
        source=m.fs.diafiltrate_block_stage_two.outlet,
        destination=m.fs.membrane_two.diafiltrate_inlet,
    )
    m.fs.retentate_stream_stage_two = Arc(
        source=m.fs.membrane_two.retentate_outlet,
        destination=m.fs.retentate_block_stage_two.inlet,
    )
    m.fs.permeate_stream_stage_two = Arc(
        source=m.fs.membrane_two.permeate_outlet,
        destination=m.fs.permeate_block_stage_two.inlet,
    )

    TransformationFactory("network.expand_arcs").apply_to(m)


def set_objective(m):
    """
    Unfixes applied pressure and sets average water flux target.
    """
    m.fs.membrane_one.applied_pressure.unfix()
    # m.fs.membrane_two.applied_pressure.unfix()
    m.fs.membrane_one.diafiltrate_flow_volume.unfix()
    # m.fs.membrane_two.diafiltrate_flow_volume.unfix()

    # def _equal_pressure_constraint(m):
    #     return m.fs.membrane_one.applied_pressure[0] == m.fs.membrane_two.applied_pressure[0]

    # m.equal_pressure_constraint = Constraint(
    #     rule=_equal_pressure_constraint
    # )

    def _calcium_recovery(m):
        return (
            (
                m.fs.membrane_one.retentate_conc_mol_comp[0, 1, "Ca"]
                * m.fs.membrane_one.retentate_flow_volume[0, 1]
            )
            # + (
            #     m.fs.membrane_two.retentate_conc_mol_comp[0, 1, "Ca"]
            #     * m.fs.membrane_two.retentate_flow_volume[0, 1]
            # )
        ) / (
            m.fs.membrane_one.total_feed_conc_mol_comp[0, "Ca"]
            * (
                m.fs.membrane_one.feed_flow_volume[0]
                + m.fs.membrane_one.diafiltrate_flow_volume[0]
            )
        )

    m.calcium_recovery = Expression(rule=_calcium_recovery)

    def _sodium_recovery(m):
        return (
            # (m.fs.membrane_two.permeate_outlet_conc_mol_comp[0, "Na"]
            # * m.fs.membrane_two.permeate_outlet_flow[0])
            # + (
                m.fs.membrane_one.permeate_outlet_conc_mol_comp[0, "Na"]
                            * m.fs.membrane_one.permeate_outlet_flow[0]
            # )
        ) / (
            m.fs.membrane_one.total_feed_conc_mol_comp[0, "Na"]
            * (
                m.fs.membrane_one.feed_flow_volume[0]
                + m.fs.membrane_one.diafiltrate_flow_volume[0]
            )
        )

    m.sodium_recovery = Expression(rule=_sodium_recovery)

    def _percent_recovery_membrane_one(m):
        return (
            m.fs.membrane_one.permeate_outlet_flow[0]
            / (
                m.fs.membrane_one.feed_flow_volume[0]
                + m.fs.membrane_one.diafiltrate_flow_volume[0]
            )
            * 100
        )

    m.percent_recovery_membrane_one = Expression(rule=_percent_recovery_membrane_one)

    def _percent_recovery_membrane_one_constraint(m):
        return m.percent_recovery_membrane_one <= 75

    m.percent_recovery_membrane_one_constraint = Constraint(
        rule=_percent_recovery_membrane_one_constraint
    )

    # def _percent_recovery_membrane_two(m):
    #     return (
    #         m.fs.membrane_two.permeate_outlet_flow[0]
    #         / (
    #             m.fs.membrane_two.feed_flow_volume[0]
    #             + m.fs.membrane_two.diafiltrate_flow_volume[0]
    #         )
    #         * 100
    #     )

    # m.percent_recovery_membrane_two = Expression(rule=_percent_recovery_membrane_two)

    # def _percent_recovery_membrane_two_constraint(m):
    #     return m.percent_recovery_membrane_two <= 75

    # m.percent_recovery_membrane_two_constraint = Constraint(
    #     rule=_percent_recovery_membrane_two_constraint
    # )


    def _inlet_bound_membrane_one(m):
        return m.fs.membrane_one.feed_flow_volume[0] + m.fs.membrane_one.diafiltrate_flow_volume[0] <= 16.25

    m.inlet_bound_membrane_one = Constraint(
        rule=_inlet_bound_membrane_one
    )

    # def _inlet_bound_membrane_two(m):
    #     return m.fs.membrane_two.feed_flow_volume[0] + m.fs.membrane_two.diafiltrate_flow_volume[0] <= 16.25

    # m.inlet_bound_membrane_two = Constraint(
    #     rule=_inlet_bound_membrane_two
    # )

    

    def _calcium_recovery_constraint(m):
        return m.calcium_recovery >= 0.75
    m.calcium_recovery_constraint = Constraint(rule=_calcium_recovery_constraint)

    # def _sodium_recovery_constraint(m):
    #     return m.sodium_recovery >= 0.5
    # m.sodium_recovery_constraint = Constraint(rule=_sodium_recovery_constraint)

    m.sodium_recovery_objective = Objective(rule=_sodium_recovery, sense=maximize)
    # m.calcium_recovery_objective = Objective(rule=_calcium_recovery, sense=maximize)


def solve_model(m):
    """
    Solves scaled model.

    Args:
        m: Pyomo model
    """
    scaling = TransformationFactory("core.scale_model")
    scaled_model = scaling.create_using(m, rename=False)

    solver = SolverFactory("ipopt")
    results = solver.solve(scaled_model, tee=True)
    assert_optimal_termination(results)

    scaling.propagate_solution(scaled_model, m)


def plot_results_by_length(membrane):
    """
    Plots concentration and flux variables across the length of the membrane module.

    Args:
        m: Pyomo model
    """
    # store values for x-coordinate
    x_axis_values = []

    # store values for concentration of Na in the retentate
    conc_ret_na = []
    # store values for concentration of Na at interface of BL and M
    conc_int_na = []
    # store values for concentration of Na in the permeate
    conc_perm_na = []
    # store values for concentration of Ca in the retentate
    conc_ret_ca = []
    # store values for concentration of Ca at interface of BL and M
    conc_int_ca = []
    # store values for concentration of Ca in the permeate
    conc_perm_ca = []

    # store values for water flux across membrane
    water_flux = []
    # store values for mol flux of Na across membrane
    Na_flux = []
    # store values for mol flux of Ca across membrane
    Ca_flux = []

    # store values for percent recovery
    percent_recovery = []

    # store values for Na rejection (observed)
    Na_rejection_obs = []
    # store values for Na rejection (actual)
    Na_rejection_act = []
    # store values for Ca rejection (observed)
    Ca_rejection_obs = []
    # store values for Ca rejection (actual)
    Ca_rejection_act = []

    for x_val in membrane.dimensionless_module_length:
        if x_val != 0:
            x_axis_values.append(x_val * value(membrane.total_module_length))
            conc_ret_na.append(value(membrane.retentate_conc_mol_comp[0, x_val, "Na"]))
            conc_int_na.append(
                value(membrane.boundary_layer_conc_mol_comp[0, x_val, 1, "Na"])
            )
            conc_perm_na.append(value(membrane.permeate_local_conc_mol_comp[0, x_val, "Na"]))
            conc_ret_ca.append(value(membrane.retentate_conc_mol_comp[0, x_val, "Ca"]))
            conc_int_ca.append(
                value(membrane.boundary_layer_conc_mol_comp[0, x_val, 1, "Ca"])
            )
            conc_perm_ca.append(value(membrane.permeate_local_conc_mol_comp[0, x_val, "Ca"]))

            water_flux.append(value(membrane.volume_flux_water[0, x_val]))
            Na_flux.append(value(membrane.molar_ion_flux[0, x_val, "Na"]))
            Ca_flux.append(value(membrane.molar_ion_flux[0, x_val, "Ca"]))

            Na_rejection_obs.append(
                value(membrane.local_observed_rejection_percent[0, x_val, "Na"])
            )
            Na_rejection_act.append(
                value(membrane.local_actual_rejection_percent[0, x_val, "Na"])
            )
            Ca_rejection_obs.append(
                value(membrane.local_observed_rejection_percent[0, x_val, "Ca"])
            )
            Ca_rejection_act.append(
                value(membrane.local_actual_rejection_percent[0, x_val, "Ca"])
            )

            percent_recovery.append(
                (
                    (value(membrane.total_feed_flow_volume[0]) - value(membrane.retentate_flow_volume[0, x_val]))
                    / (
                        value(membrane.feed_flow_volume[0])
                        + value(membrane.diafiltrate_flow_volume[0])
                    )
                    * 100
                )
            )

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, dpi=100, figsize=(12, 10)
    )

    ax1.plot(x_axis_values, conc_ret_na, linewidth=2, label="retentate")
    ax1.plot(x_axis_values, conc_int_na, linewidth=2, label="interface")
    ax1.plot(x_axis_values, conc_perm_na, linewidth=2, label="permeate")
    ax1.set_ylabel(
        "Sodium Concentration (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.tick_params(direction="in", labelsize=10)
    ax1.legend()

    ax2.plot(x_axis_values, conc_ret_ca, linewidth=2, label="retentate")
    ax2.plot(x_axis_values, conc_int_ca, linewidth=2, label="interface")
    ax2.plot(x_axis_values, conc_perm_ca, linewidth=2, label="permeate")
    ax2.set_ylabel(
        "Calcium Concentration (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax2.tick_params(direction="in", labelsize=10)
    ax2.legend()

    ax3.plot(x_axis_values, water_flux, linewidth=2)
    ax3.set_ylabel("Water Flux (m$^3$/m$^2$/h)", fontsize=10, fontweight="bold")
    ax3.tick_params(direction="in", labelsize=10)

    ax4.plot(x_axis_values, Na_flux, linewidth=2, label="Na")
    ax4.plot(x_axis_values, Ca_flux, linewidth=2, label="Ca")
    ax4.set_ylabel("Solute Molar Flux (mol/m$^2$/h)", fontsize=10, fontweight="bold")
    ax4.tick_params(direction="in", labelsize=10)
    ax4.legend()

    ax5.plot(x_axis_values, Na_rejection_obs, linewidth=2, label="Na (observed)")
    ax5.plot(
        x_axis_values,
        Na_rejection_act,
        "--",
        linewidth=2,
        label="Na (actual)",
    )
    ax5.plot(x_axis_values, Ca_rejection_obs, linewidth=2, label="Ca (observed)")
    ax5.plot(x_axis_values, Ca_rejection_act, "--", linewidth=2, label="Ca (actual)")
    ax5.set_xlabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Solute Rejection (%)", fontsize=10, fontweight="bold")
    ax5.tick_params(direction="in", labelsize=10)
    ax5.legend()

    ax6.plot(x_axis_values, percent_recovery, linewidth=2)
    ax6.set_xlabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Percent Recovery (%)", fontsize=10, fontweight="bold")
    ax6.tick_params(direction="in", labelsize=10)

    return fig


def plot_results_by_thickness(m, phase):
    """
    Plots concentrations within the boundary layer or membrane.

    Args:
        m: Pyomo model
    """
    x_axis_values = []
    z_axis_values = []

    # store values for concentration of Li
    conc_na = []
    conc_na_dict = {}
    # store values for concentration of Ca
    conc_ca = []
    conc_ca_dict = {}
    # store values for concentration of Cl
    conc_chl = []
    conc_chl_dict = {}

    for x_val in m.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            x_axis_values.append(x_val * value(m.fs.membrane.total_module_length))

    if phase == "Boundary Layer":
        for z_val in m.fs.membrane.dimensionless_boundary_layer_thickness:
            z_axis_values.append(
                z_val * value(m.fs.membrane.total_boundary_layer_thickness) * 1e6
            )
            for x_val in m.fs.membrane.dimensionless_module_length:
                if x_val != 0:
                    conc_na.append(
                        value(
                            m.fs.membrane.boundary_layer_conc_mol_comp[
                                0, x_val, z_val, "Na"
                            ]
                        )
                    )
                    conc_ca.append(
                        value(
                            m.fs.membrane.boundary_layer_conc_mol_comp[
                                0, x_val, z_val, "Ca"
                            ]
                        )
                    )
                    conc_chl.append(
                        value(
                            m.fs.membrane.boundary_layer_conc_mol_comp[
                                0, x_val, z_val, "Cl"
                            ]
                        )
                    )

            conc_na_dict[f"{z_val}"] = conc_na
            conc_ca_dict[f"{z_val}"] = conc_ca
            conc_chl_dict[f"{z_val}"] = conc_chl
            conc_na = []
            conc_ca = []
            conc_chl = []

    elif phase == "Membrane":
        for z_val in m.fs.membrane.dimensionless_membrane_thickness:
            z_axis_values.append(
                z_val * value(m.fs.membrane.total_membrane_thickness) * 1e9
            )
            for x_val in m.fs.membrane.dimensionless_module_length:
                if x_val != 0:
                    conc_na.append(
                        value(
                            m.fs.membrane.membrane_conc_mol_comp[0, x_val, z_val, "Na"]
                        )
                    )
                    conc_ca.append(
                        value(
                            m.fs.membrane.membrane_conc_mol_comp[0, x_val, z_val, "Ca"]
                        )
                    )
                    conc_chl.append(
                        value(
                            m.fs.membrane.membrane_conc_mol_comp[0, x_val, z_val, "Cl"]
                        )
                    )

            conc_na_dict[f"{z_val}"] = conc_na
            conc_ca_dict[f"{z_val}"] = conc_ca
            conc_chl_dict[f"{z_val}"] = conc_chl
            conc_na = []
            conc_ca = []
            conc_chl = []

    conc_na_df = DataFrame(index=x_axis_values, data=conc_na_dict)
    conc_ca_df = DataFrame(index=x_axis_values, data=conc_ca_dict)
    conc_chl_df = DataFrame(index=x_axis_values, data=conc_chl_dict)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=125, figsize=(15, 7))
    Na_plot = ax1.pcolor(z_axis_values, x_axis_values, conc_na_df, cmap="Greens")
    if phase == "Boundary Layer":
        ax1.set_xlabel("Boundary Layer Thickness (um)", fontsize=10, fontweight="bold")
    elif phase == "Membrane":
        ax1.set_xlabel("Membrane Thickness (nm)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"Sodium Concentration\n in {phase} (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.tick_params(direction="in", labelsize=10)
    fig.colorbar(Na_plot, ax=ax1)

    Ca_plot = ax2.pcolor(z_axis_values, x_axis_values, conc_ca_df, cmap="Blues")
    if phase == "Boundary Layer":
        ax2.set_xlabel("Boundary Layer Thickness (um)", fontsize=10, fontweight="bold")
    elif phase == "Membrane":
        ax2.set_xlabel("Membrane Thickness (nm)", fontsize=10, fontweight="bold")
    ax2.set_title(
        f"Calcium Concentration\n in {phase} (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax2.tick_params(direction="in", labelsize=10)
    fig.colorbar(Ca_plot, ax=ax2)

    Cl_plot = ax3.pcolor(z_axis_values, x_axis_values, conc_chl_df, cmap="Oranges")
    if phase == "Boundary Layer":
        ax3.set_xlabel("Boundary Layer Thickness (um)", fontsize=10, fontweight="bold")
    elif phase == "Membrane":
        ax3.set_xlabel("Membrane Thickness (nm)", fontsize=10, fontweight="bold")
    ax3.set_title(
        f"Chloride Concentration\n in {phase} (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax3.tick_params(direction="in", labelsize=10)
    fig.colorbar(Cl_plot, ax=ax3)

    return fig


if __name__ == "__main__":
    main()
