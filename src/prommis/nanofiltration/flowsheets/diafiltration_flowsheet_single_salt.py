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

from prommis.nanofiltration.property_packages.diafiltration_single_salt_stream_properties import (
    DiafiltrationStreamParameter,
)
from prommis.nanofiltration.property_packages.diafiltration_single_salt_solute_properties import (
    SoluteParameter,
)
from prommis.nanofiltration.unit_models.diafiltration_single_salt import (
    SingleSaltDiafiltration,
)


def main():
    """
    Builds and solves flowsheet with two-salt diafiltration unit model.
    """
    # build flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.stream_properties = DiafiltrationStreamParameter()
    m.fs.properties = SoluteParameter(single_salt_system="aluminum_chloride")

    # update parameter inputs if desired
    build_membrane_parameters(m)

    # add feed blocks for feed and diafiltrate
    m.fs.feed_block = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block = Feed(property_package=m.fs.stream_properties)

    # add the membrane unit model
    m.fs.membrane = SingleSaltDiafiltration(
        property_package=m.fs.properties,
        NFE_module_length=20,
        NFE_membrane_thickness=10,
        charged_membrane=True,
    )

    # add product blocks for retentate and permeate
    m.fs.retentate_block = Product(property_package=m.fs.stream_properties)
    m.fs.permeate_block = Product(property_package=m.fs.stream_properties)

    # fix the degrees of freedom to their default values
    fix_variables(m)

    # add and connect flowsheet streams
    add_and_connect_streams(m)

    # check structural warnings
    dt = DiagnosticsToolbox(m)
    dt.assert_no_structural_warnings()

    # solve model
    solve_model(m)

    # check numerical warnings
    dt.assert_no_numerical_warnings()

    # visualize the results
    plot_results(m)
    plot_membrane_results(m)


def build_membrane_parameters(m):
    """
    Updates parameters needed in two salt diafiltration unit model if desired

    Args:
        m: Pyomo model
    """
    pass


def fix_variables(m):
    # fix degrees of freedom in the membrane
    m.fs.membrane.total_module_length.fix()
    m.fs.membrane.total_membrane_length.fix()
    # at a constant salt concentration in the feed, the ionic strength (this the osmostic pressure) changes
    if (
        m.fs.membrane.config.property_package.config.single_salt_system
        == "lithium_chloride"
    ):
        m.fs.membrane.applied_pressure.fix()
    elif (
        m.fs.membrane.config.property_package.config.single_salt_system
        == "cobalt_chloride"
    ):
        m.fs.membrane.applied_pressure.fix(25)
    elif (
        m.fs.membrane.config.property_package.config.single_salt_system
        == "aluminum_chloride"
    ):
        m.fs.membrane.applied_pressure.fix(35)

    # fix degrees of freedom in the flowsheet
    m.fs.membrane.feed_flow_volume.fix()
    m.fs.membrane.feed_conc_mol_comp[0, "cation"].fix()
    m.fs.membrane.feed_conc_mol_comp[0, "anion"].fix()

    m.fs.membrane.diafiltrate_flow_volume.fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "cation"].fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "anion"].fix()


def add_and_connect_streams(m):
    m.fs.feed_stream = Arc(
        source=m.fs.feed_block.outlet,
        destination=m.fs.membrane.feed_inlet,
    )
    m.fs.diafiltrate_stream = Arc(
        source=m.fs.diafiltrate_block.outlet,
        destination=m.fs.membrane.diafiltrate_inlet,
    )
    m.fs.retentate_stream = Arc(
        source=m.fs.membrane.retentate_outlet,
        destination=m.fs.retentate_block.inlet,
    )
    m.fs.permeate_stream = Arc(
        source=m.fs.membrane.permeate_outlet,
        destination=m.fs.permeate_block.inlet,
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


def plot_results(m):
    """
    Plots concentration and flux variables across the length of the membrane module.

    Args:
        m: Pyomo model
    """
    # store values for x-coordinate
    x_axis_values = []

    # store values for concentration of cation in the retentate
    conc_ret_cation = []
    # store values for concentration of cation in the permeate
    conc_perm_cation = []
    # store values for concentration of anion in the retentate
    conc_ret_anion = []
    # store values for concentration of anion in the permeate
    conc_perm_anion = []

    # store values for water flux across membrane
    water_flux = []
    # store values for mol flux of cation across membrane
    cation_flux = []
    # store values for mol flux of anion across membrane
    anion_flux = []

    # store values for percent recovery
    percent_recovery = []

    # store values for cation rejection
    cation_rejection = []
    # store values for cation solute passage
    cation_sieving = []
    # store values for anion rejection
    anion_rejection = []
    # store values for anion solute passage
    anion_sieving = []

    for x_val in m.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            x_axis_values.append(x_val * value(m.fs.membrane.total_module_length))
            conc_ret_cation.append(
                value(m.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation"])
            )
            conc_perm_cation.append(
                value(m.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation"])
            )
            conc_ret_anion.append(
                value(m.fs.membrane.retentate_conc_mol_comp[0, x_val, "anion"])
            )
            conc_perm_anion.append(
                value(m.fs.membrane.permeate_conc_mol_comp[0, x_val, "anion"])
            )

            water_flux.append(value(m.fs.membrane.volume_flux_water[x_val]))
            cation_flux.append(value(m.fs.membrane.mol_flux_cation[x_val]))
            anion_flux.append(value(m.fs.membrane.mol_flux_anion[x_val]))

            cation_rejection.append(
                (
                    1
                    - (
                        value(m.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation"])
                        / value(
                            m.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation"]
                        )
                    )
                )
                * 100
            )
            cation_sieving.append(
                (
                    value(m.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation"])
                    / value(m.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation"])
                )
            )

            anion_rejection.append(
                (
                    1
                    - (
                        value(m.fs.membrane.permeate_conc_mol_comp[0, x_val, "anion"])
                        / value(
                            m.fs.membrane.retentate_conc_mol_comp[0, x_val, "anion"]
                        )
                    )
                )
                * 100
            )
            anion_sieving.append(
                (
                    value(m.fs.membrane.permeate_conc_mol_comp[0, x_val, "anion"])
                    / value(m.fs.membrane.retentate_conc_mol_comp[0, x_val, "anion"])
                )
            )

            percent_recovery.append(
                (
                    value(m.fs.membrane.permeate_flow_volume[0, x_val])
                    / value(m.fs.membrane.retentate_flow_volume[0, 0])
                    * 100
                )
            )

    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, dpi=75, figsize=(12, 10)
    )

    ax1.plot(x_axis_values, conc_ret_cation, linewidth=2, label="retentate")
    ax1.plot(x_axis_values, conc_perm_cation, linewidth=2, label="permeate")
    ax1.set_ylabel(
        "Cation Concentration\n(mM)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.tick_params(direction="in", labelsize=10)
    ax1.legend(fontsize=12)

    ax2.plot(x_axis_values, conc_ret_anion, linewidth=2, label="retentate")
    ax2.plot(x_axis_values, conc_perm_anion, linewidth=2, label="permeate")
    ax2.set_ylabel(
        "Anion Concentration\n(mM)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.tick_params(direction="in", labelsize=10)
    ax2.legend(fontsize=12)

    ax3.plot(x_axis_values, water_flux, linewidth=2)
    ax3.set_ylabel("Water Flux (m$^3$/m$^2$/h)", fontsize=12, fontweight="bold")
    ax3.tick_params(direction="in", labelsize=10)

    ax4.plot(x_axis_values, cation_flux, linewidth=2)
    ax4.set_ylabel("Lithium Molar Flux\n(mol/m$^2$/h)", fontsize=12, fontweight="bold")
    ax4.tick_params(direction="in", labelsize=10)

    ax5.plot(x_axis_values, cation_rejection, linewidth=2, label="cation")
    ax5.set_xlabel("Module Length (m)", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Solute Rejection (%)", fontsize=12, fontweight="bold")
    ax5.tick_params(direction="in", labelsize=10)
    ax5.legend(fontsize=12)

    ax6.plot(x_axis_values, percent_recovery, linewidth=2)
    ax6.set_xlabel("Module Length (m)", fontsize=12, fontweight="bold")
    ax6.set_ylabel("Percent Recovery (%)", fontsize=12, fontweight="bold")
    ax6.tick_params(direction="in", labelsize=10)

    plt.suptitle(f"{m.fs.membrane.config.property_package.config.single_salt_system}")

    plt.show()


def plot_membrane_results(m):
    """
    Plots concentrations within the membrane.

    Args:
        m: Pyomo model
    """
    x_axis_values = []
    z_axis_values = []

    for x_val in m.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            x_axis_values.append(x_val * value(m.fs.membrane.total_module_length))
    for z_val in m.fs.membrane.dimensionless_membrane_thickness:
        z_axis_values.append(
            z_val * value(m.fs.membrane.total_membrane_thickness) * 1e9
        )
    # store values for concentration of cation in the membrane
    conc_mem_cation = []
    conc_mem_cation_dict = {}
    # store values for concentration of anion in the membrane
    conc_mem_anion = []
    conc_mem_anion_dict = {}

    for z_val in m.fs.membrane.dimensionless_membrane_thickness:
        for x_val in m.fs.membrane.dimensionless_module_length:
            if x_val != 0:
                conc_mem_cation.append(
                    value(m.fs.membrane.membrane_conc_mol_cation[x_val, z_val])
                )
                conc_mem_anion.append(
                    value(m.fs.membrane.membrane_conc_mol_anion[x_val, z_val])
                )

        conc_mem_cation_dict[f"{z_val}"] = conc_mem_cation
        conc_mem_anion_dict[f"{z_val}"] = conc_mem_anion
        conc_mem_cation = []
        conc_mem_anion = []

    conc_mem_cation_df = DataFrame(index=x_axis_values, data=conc_mem_cation_dict)
    conc_mem_anion_df = DataFrame(index=x_axis_values, data=conc_mem_anion_dict)

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=125, figsize=(10, 7))
    cation_plot = ax1.pcolor(
        z_axis_values, x_axis_values, conc_mem_cation_df, cmap="Reds"
    )
    ax1.set_xlabel("Membrane Thickness (nm)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax1.set_title(
        "Cation Concentration\n in Membrane (mM)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.tick_params(direction="in", labelsize=10)
    fig.colorbar(cation_plot, ax=ax1)

    anion_plot = ax2.pcolor(
        z_axis_values, x_axis_values, conc_mem_anion_df, cmap="Oranges"
    )
    ax2.set_xlabel("Membrane Thickness (nm)", fontsize=10, fontweight="bold")
    ax2.set_title(
        "Anion Concentration\n in Membrane (mM)",
        fontsize=10,
        fontweight="bold",
    )
    ax2.tick_params(direction="in", labelsize=10)
    fig.colorbar(anion_plot, ax=ax2)

    plt.suptitle(f"{m.fs.membrane.config.property_package.config.single_salt_system}")

    plt.show()


if __name__ == "__main__":
    main()
