#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Sample flowsheet for the diafiltration cascade.

Author: Molly Dougher
"""

from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    value,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

import matplotlib.pyplot as plt
from pandas import DataFrame

from prommis.nanofiltration.diafiltration_stream_properties import (
    DiafiltrationStreamParameter,
)
from prommis.nanofiltration.diafiltration_solute_properties import SoluteParameter
from prommis.nanofiltration.diafiltration_two_salt import TwoSaltDiafiltration


def main():
    """
    Builds and solves flowsheet with two-salt diafiltration unit model.
    """
    # build flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.stream_properties = DiafiltrationStreamParameter()
    m.fs.properties = SoluteParameter()

    # update parameter inputs if desired
    build_membrane_parameters(m)

    # add feed blocks for feed and diafiltrate
    m.fs.feed_block_stage_one = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block_stage_one = Feed(property_package=m.fs.stream_properties)
    m.fs.feed_block_stage_two = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block_stage_two = Feed(property_package=m.fs.stream_properties)

    surrogate_model_file_dict = {
        "D_11": "surrogate_models/rbf_pysmo_surrogate_d11_scaled.json",
        "D_12": "surrogate_models/rbf_pysmo_surrogate_d12_scaled.json",
        "D_21": "surrogate_models/rbf_pysmo_surrogate_d21_scaled.json",
        "D_22": "surrogate_models/rbf_pysmo_surrogate_d22_scaled.json",
        "alpha_1": "surrogate_models/rbf_pysmo_surrogate_alpha1.json",
        "alpha_2": "surrogate_models/rbf_pysmo_surrogate_alpha2.json",
    }

    # add the membranes
    m.fs.membrane_one = TwoSaltDiafiltration(
        property_package=m.fs.properties,
        NFE_module_length=10,
        NFE_membrane_thickness=5,
        charged_membrane=True,
        surrogate_model_files=surrogate_model_file_dict,
        diffusion_surrogate_scaling_factor=1e-07,
    )
    m.fs.membrane_two = TwoSaltDiafiltration(
        property_package=m.fs.properties,
        NFE_module_length=10,
        NFE_membrane_thickness=5,
        charged_membrane=True,
        surrogate_model_files=surrogate_model_file_dict,
        diffusion_surrogate_scaling_factor=1e-07,
    )

    # add product blocks for retentate and permeate
    m.fs.retentate_block_stage_one = Product(property_package=m.fs.stream_properties)
    m.fs.retentate_block_stage_two = Product(property_package=m.fs.stream_properties)
    m.fs.permeate_block_stage_two = Product(property_package=m.fs.stream_properties)

    # fix the degrees of freedom to their default values
    fix_variables(m)

    # add and connect flowsheet streams
    add_and_connect_streams(m)

    # check structural warnings
    dt = DiagnosticsToolbox(m)
    dt.assert_no_structural_warnings()

    # solve model
    solve_model(m)
    check_concentrations(m)

    # check numerical warnings
    dt.assert_no_numerical_warnings()

    # visualize the results
    plot_results(m)
    # plot_membrane_results(m)


def build_membrane_parameters(m):
    """
    Updates parameters needed in two salt diafiltration unit model if desired

    Args:
        m: Pyomo model
    """
    pass


def fix_variables(m):
    # fix degrees of freedom in the membranes
    m.fs.membrane_one.total_module_length.fix()
    m.fs.membrane_one.total_membrane_length.fix()
    m.fs.membrane_one.applied_pressure.fix()
    m.fs.membrane_two.total_module_length.fix()
    m.fs.membrane_two.total_membrane_length.fix()
    m.fs.membrane_two.applied_pressure.fix()

    # fix degrees of freedom in the flowsheet (feed to stage one)
    m.fs.membrane_one.feed_flow_volume.fix()
    m.fs.membrane_one.feed_conc_mol_comp[0, "Li"].fix()
    m.fs.membrane_one.feed_conc_mol_comp[0, "Co"].fix()
    m.fs.membrane_one.feed_conc_mol_comp[0, "Cl"].fix()

    m.fs.membrane_one.diafiltrate_flow_volume.fix()
    m.fs.membrane_one.diafiltrate_conc_mol_comp[0, "Li"].fix()
    m.fs.membrane_one.diafiltrate_conc_mol_comp[0, "Co"].fix()
    m.fs.membrane_one.diafiltrate_conc_mol_comp[0, "Cl"].fix()

    m.fs.membrane_two.diafiltrate_flow_volume.fix()
    m.fs.membrane_two.diafiltrate_conc_mol_comp[0, "Li"].fix()
    m.fs.membrane_two.diafiltrate_conc_mol_comp[0, "Co"].fix()
    m.fs.membrane_two.diafiltrate_conc_mol_comp[0, "Cl"].fix()

    # m.fs.membrane_two.diafiltrate_flow_volume.fix(
    #     value(m.fs.membrane_two.numerical_zero_tolerance)
    # )
    # m.fs.membrane_two.diafiltrate_conc_mol_comp[0, "Li"].fix(
    #     value(m.fs.membrane_two.numerical_zero_tolerance)
    # )
    # m.fs.membrane_two.diafiltrate_conc_mol_comp[0, "Co"].fix(
    #     value(m.fs.membrane_two.numerical_zero_tolerance)
    # )
    # m.fs.membrane_two.diafiltrate_conc_mol_comp[0, "Cl"].fix(
    #     value(m.fs.membrane_two.numerical_zero_tolerance)
    # )

    # m.fs.membrane_two.retentate_flow_volume[0, 0].unfix()
    # m.fs.membrane_two.retentate_conc_mol_comp[0, 0, "Li"].unfix()
    # m.fs.membrane_two.retentate_conc_mol_comp[0, 0, "Co"].unfix()

    # m.fs.membrane_two.retentate_flow_constraint = Constraint(
    #     expr=(
    #         m.fs.membrane_two.retentate_flow_volume[0, 0]
    #         == m.fs.membrane_one.permeate_flow_volume[0, 1]
    #     )
    # )
    # m.fs.membrane_two.retentate_lith_constraint = Constraint(
    #     expr=(
    #         m.fs.membrane_two.retentate_conc_mol_comp[0, 0, "Li"]
    #         == m.fs.membrane_one.permeate_conc_mol_comp[0, 1, "Li"]
    #     )
    # )
    # m.fs.membrane_two.retentate_cob_constraint = Constraint(
    #     expr=(
    #         m.fs.membrane_two.retentate_conc_mol_comp[0, 0, "Co"]
    #         == m.fs.membrane_one.permeate_conc_mol_comp[0, 1, "Co"]
    #     )
    # )

    # m.fs.membrane_two.fix_initial_values()


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
        destination=m.fs.retentate_block_stage_one.inlet,
    )
    m.fs.permeate_stream_stage_one = Arc(
        source=m.fs.membrane_one.permeate_outlet,
        destination=m.fs.membrane_two.feed_inlet,
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


def check_concentrations(m):
    for membrane in [m.fs.membrane_one, m.fs.membrane_two]:
        for x in membrane.dimensionless_module_length:
            for z in membrane.dimensionless_membrane_thickness:
                # skip check at x=0 as the concentration is expected to be 0 and the
                # diffusion coefficient calculation is not needed
                if x == 0:
                    pass
                elif not (50 < value(membrane.membrane_conc_mol_lithium[x, z]) < 200):
                    raise ValueError(
                        "WARNING: Membrane concentration for lithium ("
                        f"{value(membrane.membrane_conc_mol_lithium[x, z])} mM at "
                        f"x={x * value(membrane.total_module_length)} m and "
                        f"z={z * value(membrane.total_membrane_thickness)} m) is outside "
                        "of the valid range for the diffusion coefficient surrogate model "
                        "(50-200 mM). Consider re-training the surrogate model."
                    )
        if membrane.config.charged_membrane:
            for x in membrane.dimensionless_module_length:
                for z in membrane.dimensionless_membrane_thickness:
                    # skip check at x=0 as the concentration is expected to be 0 and the
                    # diffusion coefficient calculation is not needed
                    if x == 0:
                        pass
                    elif not (
                        80 < value(membrane.membrane_conc_mol_cobalt[x, z]) < 110
                    ):
                        raise ValueError(
                            "WARNING: Membrane concentration for cobalt ("
                            f"{value(membrane.membrane_conc_mol_cobalt[x, z])} mM at "
                            f"x={x * value(membrane.total_module_length)} m and "
                            f"z={z * value(membrane.total_membrane_thickness)} m) is outside "
                            "of the valid range for the diffusion coefficient surrogate model "
                            "(50-200 mM). Consider re-training the surrogate model."
                        )


def plot_results(m):
    """
    Plots concentration and flux variables across the length of the membrane module.

    Args:
        m: Pyomo model
    """
    # store values for x-coordinate
    x_axis_values = []

    # store values for concentration of lithium in the retentate
    conc_ret_lith = []
    # store values for concentration of lithium in the permeate
    conc_perm_lith = []
    # store values for concentration of cobalt in the retentate
    conc_ret_cob = []
    # store values for concentration of cobalt in the permeate
    conc_perm_cob = []

    # store values for water flux across membrane
    water_flux = []
    # store values for mol flux of lithium across membrane
    lithium_flux = []

    # store values for percent recovery
    percent_recovery = []

    # store values for lithium rejection
    lithium_rejection = []
    # store values for lithium solute passage
    lithium_sieving = []
    # store values for cobalt rejection
    cobalt_rejection = []
    # store values for cobalt solute passage
    cobalt_sieving = []

    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, dpi=75, figsize=(12, 10)
    )

    for membrane in [m.fs.membrane_one, m.fs.membrane_two]:
        if membrane == m.fs.membrane_one:
            mem = "Membrane 1"
        else:
            mem = "Membrane 2"
        for x_val in membrane.dimensionless_module_length:
            if x_val != 0:
                x_axis_values.append(x_val * value(membrane.total_module_length))
                conc_ret_lith.append(
                    value(membrane.retentate_conc_mol_comp[0, x_val, "Li"])
                )
                conc_perm_lith.append(
                    value(membrane.permeate_conc_mol_comp[0, x_val, "Li"])
                )
                conc_ret_cob.append(
                    value(membrane.retentate_conc_mol_comp[0, x_val, "Co"])
                )
                conc_perm_cob.append(
                    value(membrane.permeate_conc_mol_comp[0, x_val, "Co"])
                )

                water_flux.append(value(membrane.volume_flux_water[x_val]))
                lithium_flux.append(value(membrane.mol_flux_lithium[x_val]))

                lithium_rejection.append(
                    (
                        1
                        - (
                            value(membrane.permeate_conc_mol_comp[0, x_val, "Li"])
                            / value(membrane.retentate_conc_mol_comp[0, x_val, "Li"])
                        )
                    )
                    * 100
                )
                lithium_sieving.append(
                    (
                        value(membrane.permeate_conc_mol_comp[0, x_val, "Li"])
                        / value(membrane.retentate_conc_mol_comp[0, x_val, "Li"])
                    )
                )
                cobalt_rejection.append(
                    (
                        1
                        - (
                            value(membrane.permeate_conc_mol_comp[0, x_val, "Co"])
                            / value(membrane.retentate_conc_mol_comp[0, x_val, "Co"])
                        )
                    )
                    * 100
                )
                cobalt_sieving.append(
                    (
                        value(membrane.permeate_conc_mol_comp[0, x_val, "Co"])
                        / value(membrane.retentate_conc_mol_comp[0, x_val, "Co"])
                    )
                )

                percent_recovery.append(
                    (
                        value(membrane.permeate_flow_volume[0, x_val])
                        / value(membrane.feed_flow_volume[0])
                        * 100
                    )
                )

        ax1.plot(x_axis_values, conc_ret_lith, linewidth=2, label=f"Ret. ({mem})")
        ax1.plot(
            x_axis_values,
            conc_perm_lith,
            "--",
            linewidth=2,
            label=f"Perm. ({mem})",
        )

        ax2.plot(x_axis_values, conc_ret_cob, linewidth=2, label=f"Ret. ({mem})")
        ax2.plot(
            x_axis_values,
            conc_perm_cob,
            "--",
            linewidth=2,
            label=f"Perm. ({mem})",
        )

        ax3.plot(x_axis_values, water_flux, linewidth=2, label=f"{mem}")

        ax4.plot(x_axis_values, lithium_flux, linewidth=2, label=f"{mem}")

        ax5.plot(
            x_axis_values, lithium_rejection, linewidth=2, label=f"Lithium ({mem})"
        )
        ax5.plot(x_axis_values, cobalt_rejection, linewidth=2, label=f"Cobalt ({mem})")

        ax6.plot(x_axis_values, percent_recovery, linewidth=2, label=f"{mem}")

        x_axis_values = []
        conc_ret_lith = []
        conc_perm_lith = []
        conc_ret_cob = []
        conc_perm_cob = []
        water_flux = []
        lithium_flux = []
        percent_recovery = []
        lithium_rejection = []
        lithium_sieving = []
        cobalt_rejection = []
        cobalt_sieving = []

    ax1.set_ylabel(
        "Lithium Concentration\n(mol/m$^3$)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel(
        "Cobalt Concentration\n(mol/m$^3$)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_ylabel("Water Flux (m$^3$/m$^2$/h)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Lithium Molar Flux\n(mol/m$^2$/h)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Module Length (m)", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Solute Rejection (%)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Module Length (m)", fontsize=12, fontweight="bold")
    ax6.set_ylabel("Percent Recovery (%)", fontsize=12, fontweight="bold")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.legend(fontsize=12)
        ax.tick_params(direction="in", labelsize=10)

    plt.show()


def get_membrane_data(membrane):
    """
    Plots concentrations within the membrane.

    Args:
        m: Pyomo model
    """
    x_axis_values = []
    z_axis_values = []

    for x_val in membrane.dimensionless_module_length:
        if x_val != 0:
            x_axis_values.append(x_val * value(membrane.total_module_length))
    for z_val in membrane.dimensionless_membrane_thickness:
        z_axis_values.append(z_val * value(membrane.total_membrane_thickness) * 1e9)
    # store values for concentration of lithium in the membrane
    conc_mem_lith = []
    conc_mem_lith_dict = {}
    # store values for concentration of cobalt in the membrane
    conc_mem_cob = []
    conc_mem_cob_dict = {}

    for z_val in membrane.dimensionless_membrane_thickness:
        for x_val in membrane.dimensionless_module_length:
            if x_val != 0:
                conc_mem_lith.append(
                    value(membrane.membrane_conc_mol_lithium[x_val, z_val])
                )
                conc_mem_cob.append(
                    value(membrane.membrane_conc_mol_cobalt[x_val, z_val])
                )

        conc_mem_lith_dict[f"{z_val}"] = conc_mem_lith
        conc_mem_cob_dict[f"{z_val}"] = conc_mem_cob
        conc_mem_lith = []
        conc_mem_cob = []

    conc_mem_lith_df = DataFrame(index=x_axis_values, data=conc_mem_lith_dict)
    conc_mem_cob_df = DataFrame(index=x_axis_values, data=conc_mem_cob_dict)
    return (x_axis_values, z_axis_values, conc_mem_lith_df, conc_mem_cob_df)


def plot_membrane_results(m):
    (
        x_axis_values_one,
        z_axis_values_one,
        conc_mem_lith_df_one,
        conc_mem_cob_df_one,
    ) = get_membrane_data(m.fs.membrane_one)
    (
        x_axis_values_two,
        z_axis_values_two,
        conc_mem_lith_df_two,
        conc_mem_cob_df_two,
    ) = get_membrane_data(m.fs.membrane_two)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=125, figsize=(10, 10))

    lithium_plot_one = ax1.pcolor(
        z_axis_values_one, x_axis_values_one, conc_mem_lith_df_one, cmap="Reds"
    )
    ax1.set_title(
        "Lithium Concentration\n in Membrane One (mM)",
        fontsize=10,
        fontweight="bold",
    )
    fig.colorbar(lithium_plot_one, ax=ax1)

    lithium_plot_two = ax2.pcolor(
        z_axis_values_two, x_axis_values_two, conc_mem_lith_df_two, cmap="Reds"
    )
    ax2.set_title(
        "Lithium Concentration\n in Membrane Two (mM)",
        fontsize=10,
        fontweight="bold",
    )
    fig.colorbar(lithium_plot_two, ax=ax2)

    cobalt_plot_one = ax3.pcolor(
        z_axis_values_one, x_axis_values_one, conc_mem_cob_df_one, cmap="Blues"
    )
    ax3.set_title(
        "Cobalt Concentration\n in Membrane One (mM)", fontsize=10, fontweight="bold"
    )
    fig.colorbar(cobalt_plot_one, ax=ax3)

    cobalt_plot_two = ax4.pcolor(
        z_axis_values_two, x_axis_values_two, conc_mem_cob_df_two, cmap="Blues"
    )
    ax4.set_title(
        "Cobalt Concentration\n in Membrane Two (mM)", fontsize=10, fontweight="bold"
    )
    fig.colorbar(cobalt_plot_two, ax=ax4)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(direction="in", labelsize=10)
    for ax in [ax3, ax4]:
        ax.set_xlabel("Membrane Thickness (nm)", fontsize=10, fontweight="bold")
    for ax in [ax1, ax3]:
        ax.set_ylabel("Module Length (m)", fontsize=10, fontweight="bold")

    plt.show()


if __name__ == "__main__":
    main()
