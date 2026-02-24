#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Model comparison for multi-salt diafiltration.

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
import numpy as np
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
    # set default arguments
    anion_list = ["chloride"]
    inlet_flow_volume = {"feed": 12.5, "diafiltrate": 3.75}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    # single salt systems
    # lithium chloride
    m_li_cl = build_model(
        cation_list=["lithium"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"lithium": 245, "cobalt": 288, "chloride": 821},
            "diafiltrate": {"lithium": 14, "cobalt": 3, "chloride": 20},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # cobalt chloride
    m_co_cl = build_model(
        cation_list=["cobalt"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"cobalt": 288, "chloride": 576},
            "diafiltrate": {"cobalt": 3, "chloride": 6},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # aluminum chloride
    m_al_cl = build_model(
        cation_list=["aluminum"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"aluminum": 20, "chloride": 60},
            "diafiltrate": {"aluminum": 3, "chloride": 9},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # two salt systems
    # lithium chloride + cobalt chloride
    m_li_co_cl = build_model(
        cation_list=["lithium", "cobalt"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"lithium": 245, "cobalt": 288, "chloride": 821},
            "diafiltrate": {"lithium": 14, "cobalt": 3, "chloride": 20},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # lithium chloride + aluminum chloride
    m_li_al_cl = build_model(
        cation_list=["lithium", "aluminum"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"lithium": 245, "aluminum": 20, "chloride": 305},
            "diafiltrate": {"lithium": 14, "aluminum": 3, "chloride": 23},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # cobalt chloride + aluminum chloride
    m_co_al_cl = build_model(
        cation_list=["cobalt", "aluminum"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"cobalt": 288, "aluminum": 20, "chloride": 636},
            "diafiltrate": {"cobalt": 3, "aluminum": 3, "chloride": 15},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # m_li_co_al_cl = build_model(
    #     num_salts=3, salt_system="lithium_cobalt_aluminum_chloride"
    # )

    # solve models
    model_list = [m_li_cl, m_co_cl, m_al_cl, m_li_co_cl, m_li_al_cl, m_co_al_cl]
    for model in model_list:
        solve_model(model)

    # store results
    m_li_cl_results_dict = {}
    m_co_cl_results_dict = {}
    m_al_cl_results_dict = {}
    m_li_co_cl_results_dict = {}
    m_li_al_cl_results_dict = {}
    m_co_al_cl_results_dict = {}

    dict_names = [
        m_li_cl_results_dict,
        m_co_cl_results_dict,
        m_al_cl_results_dict,
        m_li_co_cl_results_dict,
        m_li_al_cl_results_dict,
        m_co_al_cl_results_dict,
    ]
    for i in range(len(dict_names)):
        dict_names[i] = extract_and_store_results(model_list[i])

    # plot individual results
    for results_dict in dict_names:
        plot_results_by_length(results_dict)
        plot_results_by_thickness(results_dict)

    # plot_relative_rejections_compact(
    #     m_li_cl, m_co_cl, m_al_cl, m_li_co_cl, m_li_al_cl, m_co_al_cl
    # )
    # plot_relative_rejections_by_component(
    #     m_li_cl, m_co_cl, m_al_cl, m_li_co_cl, m_li_al_cl, m_co_al_cl, m_li_co_al_cl
    # )
    # plot_rejection_versus_concentration(
    #     m_li_cl, m_co_cl, m_al_cl, m_li_co_cl, m_li_al_cl, m_co_al_cl, m_li_co_al_cl
    # )


def build_model(
    cation_list,
    anion_list,
    inlet_flow_volume,
    inlet_concentration,
    include_boundary_layer,
    NFE_module_length,
    NFE_boundary_layer_thickness,
    NFE_membrane_thickness,
):
    # build flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.stream_properties = MultiComponentDiafiltrationStreamParameter(
        cation_list=cation_list,
        anion_list=anion_list,
    )
    m.fs.properties = MultiComponentDiafiltrationSoluteParameter(
        cation_list=cation_list,
        anion_list=anion_list,
    )

    # add feed blocks for feed and diafiltrate
    m.fs.feed_block = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block = Feed(property_package=m.fs.stream_properties)

    # add the membrane unit model
    m.fs.membrane = MultiComponentDiafiltration(
        property_package=m.fs.properties,
        cation_list=cation_list,
        anion_list=anion_list,
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # add product blocks for retentate and permeate
    m.fs.retentate_block = Product(property_package=m.fs.stream_properties)
    m.fs.permeate_block = Product(property_package=m.fs.stream_properties)

    # fix the degrees of freedom to their default values
    m.fs.membrane.total_module_length.fix()
    m.fs.membrane.total_membrane_length.fix()
    m.fs.membrane.applied_pressure.fix()
    m.fs.membrane.feed_flow_volume.fix(inlet_flow_volume["feed"])
    m.fs.membrane.diafiltrate_flow_volume.fix(inlet_flow_volume["diafiltrate"])
    for t in m.fs.membrane.time:
        for j in m.fs.membrane.solutes:
            m.fs.membrane.feed_conc_mol_comp[t, j].fix(inlet_concentration["feed"][j])
            m.fs.membrane.diafiltrate_conc_mol_comp[t, j].fix(
                inlet_concentration["diafiltrate"][j]
            )

    # initialize membrane model
    initialized_membrane_model = m.fs.membrane.default_initializer()
    initialized_membrane_model.initialize(m.fs.membrane)

    # add and connect flowsheet streams
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

    # check structural warnings
    dt = DiagnosticsToolbox(m)
    dt.assert_no_structural_warnings()

    return m


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

    # check numerical warnings
    dt = DiagnosticsToolbox(m)
    dt.assert_no_numerical_warnings()

    return results


def extract_and_store_results(m):
    """
    Extracts relevant results and stores in dictionary

    Args:
        m: Pyomo model
    """
    # store values for x-coordinate (module length)
    x_axis_values = []

    # store values for z-coordinate (boundary layer)
    z_boundary_layer_values = []

    # store values for z-coordinate (membrane)
    z_membrane_values = []

    # store values for concentration in the retentate
    conc_ret_cation_1 = []
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_ret_cation_2 = []

    # store values for concentration at solution-membrane interface
    conc_int_cation_1 = []
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_int_cation_2 = []

    # store values for concentration in the permeate
    conc_perm_cation_1 = []
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_perm_cation_2 = []

    # store values for concentration in the boundary layer (2D)
    conc_bl_cation_1 = []
    conc_bl_cation_1_dict = {}
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_bl_cation_2 = []
        conc_bl_cation_2_dict = {}

    # store values for concentration in the membrane (2D)
    conc_mem_cation_1 = []
    conc_mem_cation_1_dict = {}
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_mem_cation_2 = []
        conc_mem_cation_2_dict = {}

    # store values for water flux across membrane
    water_flux = []

    # store values for mol flux across membrane
    cation_1_flux = []
    if len(m.fs.membrane.config.cation_list) > 1:
        cation_2_flux = []

    # store values for percent recovery
    percent_recovery = []

    # store values for rejection
    cation_1_rejection_observed = []
    cation_1_rejection_actual = []
    if len(m.fs.membrane.config.cation_list) > 1:
        cation_2_rejection_observed = []
        cation_2_rejection_actual = []

    for x_val in m.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            # x-coordinate
            x_axis_values.append(x_val * value(m.fs.membrane.total_module_length))

            # concentrations
            conc_ret_cation_1_val = value(
                m.fs.membrane.retentate_conc_mol_comp[
                    0, x_val, m.fs.membrane.config.cation_list[0]
                ]
            )
            conc_int_cation_1_val = value(
                m.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, m.fs.membrane.config.cation_list[0]
                ]
            )
            conc_perm_cation_1_val = value(
                m.fs.membrane.permeate_conc_mol_comp[
                    0, x_val, m.fs.membrane.config.cation_list[0]
                ]
            )

            conc_ret_cation_1.append(conc_ret_cation_1_val)
            conc_int_cation_1.append(conc_int_cation_1_val)
            conc_perm_cation_1.append(conc_perm_cation_1_val)

            if len(m.fs.membrane.config.cation_list) > 1:
                conc_ret_cation_2_val = value(
                    m.fs.membrane.retentate_conc_mol_comp[
                        0, x_val, m.fs.membrane.config.cation_list[1]
                    ]
                )
                conc_int_cation_2_val = value(
                    m.fs.membrane.boundary_layer_conc_mol_comp[
                        0, x_val, 1, m.fs.membrane.config.cation_list[1]
                    ]
                )
                conc_perm_cation_2_val = value(
                    m.fs.membrane.permeate_conc_mol_comp[
                        0, x_val, m.fs.membrane.config.cation_list[1]
                    ]
                )

                conc_ret_cation_2.append(conc_ret_cation_2_val)
                conc_int_cation_2.append(conc_int_cation_2_val)
                conc_perm_cation_2.append(conc_perm_cation_2_val)

            # flux
            water_flux.append(value(m.fs.membrane.volume_flux_water[0, x_val]))

            cation_1_flux.append(
                value(
                    m.fs.membrane.molar_ion_flux[
                        0, x_val, m.fs.membrane.config.cation_list[0]
                    ]
                )
            )
            if len(m.fs.membrane.config.cation_list) > 1:
                cation_2_flux.append(
                    value(
                        m.fs.membrane.molar_ion_flux[
                            0, x_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )
                )

            # rejection
            cation_1_rejection_observed.append(
                (1 - (conc_perm_cation_1_val / conc_ret_cation_1_val)) * 100
            )
            cation_1_rejection_actual.append(
                (1 - (conc_perm_cation_1_val / conc_int_cation_1_val)) * 100
            )
            if len(m.fs.membrane.config.cation_list) > 1:
                cation_2_rejection_observed.append(
                    (1 - (conc_perm_cation_2_val / conc_ret_cation_2_val)) * 100
                )
                cation_2_rejection_actual.append(
                    (1 - (conc_perm_cation_2_val / conc_int_cation_2_val)) * 100
                )

            # recovery
            percent_recovery.append(
                (
                    value(m.fs.membrane.permeate_flow_volume[0, x_val])
                    / (
                        value(m.fs.membrane.feed_flow_volume[0])
                        + value(m.fs.membrane.diafiltrate_flow_volume[0])
                    )
                    * 100
                )
            )

    # boundary layer
    for z_val in m.fs.membrane.dimensionless_boundary_layer_thickness:
        z_boundary_layer_values.append(
            z_val * value(m.fs.membrane.total_boundary_layer_thickness) * 1e6
        )
        for x_val in m.fs.membrane.dimensionless_module_length:
            if x_val != 0:
                conc_bl_cation_1_val = value(
                    m.fs.membrane.boundary_layer_conc_mol_comp[
                        0, x_val, z_val, m.fs.membrane.config.cation_list[0]
                    ]
                )
                conc_bl_cation_1.append(conc_bl_cation_1_val)
                if len(m.fs.membrane.config.cation_list) > 1:
                    conc_bl_cation_2_val = value(
                        m.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )
                    conc_bl_cation_2.append(conc_bl_cation_2_val)

        conc_bl_cation_1_dict[f"{z_val}"] = conc_bl_cation_1
        conc_bl_cation_1 = []
        if len(m.fs.membrane.config.cation_list) > 1:
            conc_bl_cation_2_dict[f"{z_val}"] = conc_bl_cation_2
            conc_bl_cation_2 = []

    # membrane
    for z_val in m.fs.membrane.dimensionless_membrane_thickness:
        z_membrane_values.append(
            z_val * value(m.fs.membrane.total_membrane_thickness) * 1e9
        )
        for x_val in m.fs.membrane.dimensionless_module_length:
            if x_val != 0:
                conc_mem_cation_1_val = value(
                    m.fs.membrane.membrane_conc_mol_comp[
                        0, x_val, z_val, m.fs.membrane.config.cation_list[0]
                    ]
                )
                conc_mem_cation_1.append(conc_mem_cation_1_val)
                if len(m.fs.membrane.config.cation_list) > 1:
                    conc_mem_cation_2_val = value(
                        m.fs.membrane.membrane_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )
                    conc_mem_cation_2.append(conc_mem_cation_2_val)

        conc_mem_cation_1_dict[f"{z_val}"] = conc_mem_cation_1
        conc_mem_cation_1 = []
        if len(m.fs.membrane.config.cation_list) > 1:
            conc_mem_cation_2_dict[f"{z_val}"] = conc_mem_cation_2
            conc_mem_cation_2 = []

    results_dict = {
        "cation_list": m.fs.membrane.config.cation_list,
        "cation_1": m.fs.membrane.config.cation_list[0],
        "x_values": x_axis_values,
        "z_bl_values": z_boundary_layer_values,
        "z_mem_values": z_membrane_values,
        "cation_1_retentate_concentration": conc_ret_cation_1,
        "cation_1_interface_concentration": conc_int_cation_1,
        "cation_1_permeate_concentration": conc_perm_cation_1,
        "cation_1_boundary_layer_concentration": conc_bl_cation_1_dict,
        "cation_1_membrane_concentration": conc_mem_cation_1_dict,
        "water_flux": water_flux,
        "cation_1_flux": cation_1_flux,
        "percent_recovery": percent_recovery,
        "cation_1_rejection_observed": cation_1_rejection_observed,
        "cation_1_rejection_actual": cation_1_rejection_actual,
    }
    if len(m.fs.membrane.config.cation_list) > 1:
        results_dict.update(
            {
                "cation_2": m.fs.membrane.config.cation_list[1],
                "cation_2_retentate_concentration": conc_ret_cation_2,
                "cation_2_interface_concentration": conc_int_cation_2,
                "cation_2_permeate_concentration": conc_perm_cation_2,
                "cation_2_boundary_layer_concentration": conc_bl_cation_2_dict,
                "cation_2_membrane_concentration": conc_mem_cation_2_dict,
                "cation_2_flux": cation_2_flux,
                "cation_2_rejection_observed": cation_2_rejection_observed,
                "cation_2_rejection_actual": cation_2_rejection_actual,
            }
        )

    return results_dict


def plot_results_by_length(results_dict):
    """
    Plots concentration and flux variables across the length of the membrane module.
    """
    cation_list = results_dict["cation_list"]
    cation_1 = results_dict["cation_1"]
    x_axis_values = results_dict["x_values"]
    conc_ret_cation_1 = results_dict["cation_1_retentate_concentration"]
    conc_int_cation_1 = results_dict["cation_1_interface_concentration"]
    conc_perm_cation_1 = results_dict["cation_1_permeate_concentration"]
    water_flux = results_dict["water_flux"]
    cation_1_flux = results_dict["cation_1_flux"]
    percent_recovery = results_dict["percent_recovery"]
    cation_1_rejection_observed = results_dict["cation_1_rejection_observed"]
    cation_1_rejection_actual = results_dict["cation_1_rejection_actual"]
    if len(cation_list) > 1:
        cation_2 = results_dict["cation_2"]
        conc_ret_cation_2 = results_dict["cation_2_retentate_concentration"]
        conc_int_cation_2 = results_dict["cation_2_interface_concentration"]
        conc_perm_cation_2 = results_dict["cation_2_permeate_concentration"]
        cation_2_flux = results_dict["cation_2_flux"]
        cation_2_rejection_observed = results_dict["cation_2_rejection_observed"]
        cation_2_rejection_actual = results_dict["cation_2_rejection_actual"]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, dpi=100, figsize=(12, 10)
    )

    ax1.plot(x_axis_values, conc_ret_cation_1, linewidth=2, label="retentate")
    ax1.plot(x_axis_values, conc_int_cation_1, linewidth=2, label="interface")
    ax1.plot(x_axis_values, conc_perm_cation_1, linewidth=2, label="permeate")
    ax1.set_ylabel(
        f"{cation_1.capitalize()} Concentration \n(mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.tick_params(direction="in", labelsize=10)
    ax1.legend()

    if len(cation_list) > 1:
        ax2.plot(x_axis_values, conc_ret_cation_2, linewidth=2, label="retentate")
        ax2.plot(x_axis_values, conc_int_cation_2, linewidth=2, label="interface")
        ax2.plot(x_axis_values, conc_perm_cation_2, linewidth=2, label="permeate")
        ax2.set_ylabel(
            f"{cation_2.capitalize()} Concentration \n(mol/m$^3$)",
            fontsize=10,
            fontweight="bold",
        )
        ax2.tick_params(direction="in", labelsize=10)
        ax2.legend()

    ax3.plot(x_axis_values, water_flux, linewidth=2)
    ax3.set_xlabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Water Flux (m$^3$/m$^2$/h)", fontsize=10, fontweight="bold")
    ax3.tick_params(direction="in", labelsize=10)

    ax4.plot(
        x_axis_values,
        cation_1_flux,
        linewidth=2,
        label=f"{cation_1.capitalize()}",
    )
    if len(cation_list) > 1:
        ax4.plot(
            x_axis_values,
            cation_2_flux,
            linewidth=2,
            label=f"{cation_2.capitalize()}",
        )
    ax4.set_xlabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Molar Flux (mol/m$^2$/h)", fontsize=10, fontweight="bold")
    ax4.tick_params(direction="in", labelsize=10)

    ax5.plot(
        x_axis_values,
        cation_1_rejection_observed,
        linewidth=2,
        label=f"{cation_1.capitalize()} (observed)",
    )
    ax5.plot(
        x_axis_values,
        cation_1_rejection_actual,
        linewidth=2,
        label=f"{cation_1.capitalize()} (actual)",
    )
    if len(cation_list) > 1:
        ax5.plot(
            x_axis_values,
            cation_2_rejection_observed,
            linewidth=2,
            label=f"{cation_2.capitalize()} (observed)",
        )
        ax5.plot(
            x_axis_values,
            cation_2_rejection_actual,
            linewidth=2,
            label=f"{cation_2.capitalize()} (actual)",
        )
    ax5.set_xlabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Solute Rejection (%)", fontsize=10, fontweight="bold")
    ax5.tick_params(direction="in", labelsize=10)
    ax5.legend()

    ax6.plot(x_axis_values, percent_recovery, linewidth=2)
    ax6.set_xlabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Percent Recovery (%)", fontsize=10, fontweight="bold")
    ax6.tick_params(direction="in", labelsize=10)

    plt.show()

    return fig


def plot_results_by_thickness(results_dict):
    """
    Plots concentrations within the boundary layer or membrane.
    """

    cation_list = results_dict["cation_list"]
    cation_1 = results_dict["cation_1"]
    x_axis_values = results_dict["x_values"]
    z_bl_axis_values = results_dict["z_bl_values"]
    z_mem_axis_values = results_dict["z_mem_values"]
    conc_bl_cation_1_dict = results_dict["cation_1_boundary_layer_concentration"]
    conc_mem_cation_1_dict = results_dict["cation_1_membrane_concentration"]
    if len(cation_list) > 1:
        cation_2 = results_dict["cation_2"]
        conc_bl_cation_2_dict = results_dict["cation_2_boundary_layer_concentration"]
        conc_mem_cation_2_dict = results_dict["cation_2_membrane_concentration"]

    conc_bl_cation_1_df = DataFrame(index=x_axis_values, data=conc_bl_cation_1_dict)
    conc_mem_cation_1_df = DataFrame(index=x_axis_values, data=conc_mem_cation_1_dict)
    if len(cation_list) > 1:
        conc_bl_cation_2_df = DataFrame(index=x_axis_values, data=conc_bl_cation_2_dict)
        conc_mem_cation_2_df = DataFrame(
            index=x_axis_values, data=conc_mem_cation_2_dict
        )

    fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=125, figsize=(15, 7))
    cation_1_plot_bl = ax1.pcolor(z_bl_axis_values, x_axis_values, conc_bl_cation_1_df)
    ax1.set_xlabel("Boundary Layer Thickness (um)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"{cation_1.capitalize()} Concentration\n in Boundary Layer (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.tick_params(direction="in", labelsize=10)
    fig1.colorbar(cation_1_plot_bl, ax=ax1)
    if len(cation_list) > 1:
        cation_2_plot_bl = ax2.pcolor(
            z_bl_axis_values, x_axis_values, conc_bl_cation_2_df
        )
        ax2.set_xlabel("Boundary Layer Thickness (um)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Module Length (m)", fontsize=10, fontweight="bold")
        ax2.set_title(
            f"{cation_2.capitalize()} Concentration\n in Boundary Layer (mol/m$^3$)",
            fontsize=10,
            fontweight="bold",
        )
        ax2.tick_params(direction="in", labelsize=10)
        fig1.colorbar(cation_2_plot_bl, ax=ax2)

    fig2, (ax3, ax4) = plt.subplots(1, 2, dpi=125, figsize=(15, 7))
    cation_1_plot_mem = ax3.pcolor(
        z_mem_axis_values, x_axis_values, conc_mem_cation_1_df
    )
    ax3.set_xlabel("Membrane Thickness (nm)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Module Length (m)", fontsize=10, fontweight="bold")
    ax3.set_title(
        f"{cation_1.capitalize()} Concentration\n in Membrane (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax3.tick_params(direction="in", labelsize=10)
    fig2.colorbar(cation_1_plot_mem, ax=ax3)

    if len(cation_list) > 1:
        cation_2_plot_mem = ax4.pcolor(
            z_mem_axis_values, x_axis_values, conc_mem_cation_2_df
        )
        ax4.set_xlabel("Membrane Thickness (nm)", fontsize=10, fontweight="bold")
        ax4.set_title(
            f"{cation_2.capitalize()} Concentration\n in Membrane (mol/m$^3$)",
            fontsize=10,
            fontweight="bold",
        )
        ax4.tick_params(direction="in", labelsize=10)
        fig2.colorbar(cation_2_plot_mem, ax=ax4)

    plt.show()

    # return fig


def plot_relative_rejections_compact(
    m_li_cl, m_co_cl, m_al_cl, m_li_co_cl, m_li_al_cl, m_co_al_cl
):
    """
    Plots relative solute rejection across the length of the membrane module.
    Rejections normalized to initial rejection (x=0).
    Compares models.

    Args:
        m: Pyomo model
    """
    # store values for x-coordinate
    x_axis_values = []

    # store values for rejection
    lithium_rejection_li = []
    lithium_rejection_li_co = []
    lithium_rejection_li_al = []

    cobalt_rejection_co = []
    cobalt_rejection_li_co = []
    cobalt_rejection_co_al = []

    aluminum_rejection_al = []
    aluminum_rejection_li_al = []
    aluminum_rejection_co_al = []

    for x_val in m_li_co_cl.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            x_axis_values.append(
                x_val
                * value(m_li_co_cl.fs.membrane.total_module_length)
                * value(m_li_co_cl.fs.membrane.total_membrane_length)
            )

            lithium_rej_li = (
                1
                - (
                    value(
                        m_li_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_li_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            cobalt_rej_co = (
                1
                - (
                    value(
                        m_co_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_co_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            aluminum_rej_al = (
                1
                - (
                    value(
                        m_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            lithium_rej_li_co = (
                1
                - (
                    value(
                        m_li_co_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_li_co_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            cobalt_rej_li_co = (
                1
                - (
                    value(
                        m_li_co_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_li_co_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                )
            ) * 100

            lithium_rej_li_al = (
                1
                - (
                    value(
                        m_li_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_li_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            aluminum_rej_li_al = (
                1
                - (
                    value(
                        m_li_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_li_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                )
            ) * 100

            cobalt_rej_co_al = (
                1
                - (
                    value(
                        m_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_co_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100
            aluminum_rej_co_al = (
                1
                - (
                    value(
                        m_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_co_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                )
            ) * 100

            lithium_rejection_li.append(lithium_rej_li)
            lithium_rejection_li_co.append(lithium_rej_li_co)
            lithium_rejection_li_al.append(lithium_rej_li_al)
            cobalt_rejection_co.append(cobalt_rej_co)
            cobalt_rejection_li_co.append(cobalt_rej_li_co)
            cobalt_rejection_co_al.append(cobalt_rej_co_al)
            aluminum_rejection_al.append(aluminum_rej_al)
            aluminum_rejection_li_al.append(aluminum_rej_li_al)
            aluminum_rejection_co_al.append(aluminum_rej_co_al)

    lithium_rejection_li_norm = [
        (i - lithium_rejection_li[0]) / abs(lithium_rejection_li[0]) * 100
        for i in lithium_rejection_li
    ]
    lithium_rejection_li_co_norm = [
        (i - lithium_rejection_li_co[0]) / abs(lithium_rejection_li_co[0]) * 100
        for i in lithium_rejection_li_co
    ]
    lithium_rejection_li_al_norm = [
        (i - lithium_rejection_li_al[0]) / abs(lithium_rejection_li_al[0]) * 100
        for i in lithium_rejection_li_al
    ]
    cobalt_rejection_co_norm = [
        (i - cobalt_rejection_co[0]) / abs(cobalt_rejection_co[0]) * 100
        for i in cobalt_rejection_co
    ]
    cobalt_rejection_li_co_norm = [
        (i - cobalt_rejection_li_co[0]) / abs(cobalt_rejection_li_co[0]) * 100
        for i in cobalt_rejection_li_co
    ]
    cobalt_rejection_co_al_norm = [
        (i - cobalt_rejection_co_al[0]) / abs(cobalt_rejection_co_al[0]) * 100
        for i in cobalt_rejection_co_al
    ]
    aluminum_rejection_al_norm = [
        (i - aluminum_rejection_al[0]) / abs(aluminum_rejection_al[0]) * 100
        for i in aluminum_rejection_al
    ]
    aluminum_rejection_li_al_norm = [
        (i - aluminum_rejection_li_al[0]) / abs(aluminum_rejection_li_al[0]) * 100
        for i in aluminum_rejection_li_al
    ]
    aluminum_rejection_co_al_norm = [
        (i - aluminum_rejection_co_al[0]) / abs(aluminum_rejection_co_al[0]) * 100
        for i in aluminum_rejection_co_al
    ]

    fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))

    ax1.plot(
        x_axis_values, lithium_rejection_li, "m-", linewidth=2
    )  # , label="Lithium (Li)")
    ax1.plot(
        x_axis_values, cobalt_rejection_co, "c-", linewidth=2
    )  # , label="Cobalt (Co)")
    ax1.plot(
        x_axis_values, aluminum_rejection_al, "g-", linewidth=2
    )  # , label="Aluminum (Al)")
    ax1.plot(
        x_axis_values, lithium_rejection_li_co, "m--", linewidth=2
    )  # , label="Lithium (Li-Co)")
    ax1.plot(
        x_axis_values, cobalt_rejection_li_co, "c--", linewidth=2
    )  # , label="Cobalt (Li-Co)")
    ax1.plot(
        x_axis_values, lithium_rejection_li_al, "m-.", linewidth=2
    )  # , label="Lithium (Li-Al)")
    ax1.plot(
        x_axis_values, aluminum_rejection_li_al, "g-.", linewidth=2
    )  # , label="Aluminum (Li-Al)")
    ax1.plot(
        x_axis_values, cobalt_rejection_co_al, "c:", linewidth=2
    )  # , label="Cobalt (Co-Al)")
    ax1.plot(
        x_axis_values, aluminum_rejection_co_al, "g:", linewidth=2
    )  # , label="Aluminum (Co-Al)")
    ax1.set_xlabel("Membrane Area (m$^2$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Solute Rejection (%)", fontsize=12, fontweight="bold")
    ax1.tick_params(direction="in", labelsize=10)
    # ax1.legend()

    ax2.plot(
        x_axis_values, lithium_rejection_li_norm, "m-", linewidth=2
    )  # , label="Lithium (Li)")
    ax2.plot(
        x_axis_values, cobalt_rejection_co_norm, "c-", linewidth=2
    )  # , label="Cobalt (Co)")
    ax2.plot(
        x_axis_values, aluminum_rejection_al_norm, "g-", linewidth=2
    )  # , label="Aluminum (Al)")
    ax2.plot(
        x_axis_values, lithium_rejection_li_co_norm, "m--", linewidth=2
    )  # , label="Lithium (Li-Co)")
    ax2.plot(
        x_axis_values, cobalt_rejection_li_co_norm, "c--", linewidth=2
    )  # , label="Cobalt (Li-Co)")
    ax2.plot(
        x_axis_values, lithium_rejection_li_al_norm, "m-.", linewidth=2
    )  # , label="Lithium (Li-Al)")
    ax2.plot(
        x_axis_values, aluminum_rejection_li_al_norm, "g-.", linewidth=2
    )  # , label="Aluminum (Li-Al)")
    ax2.plot(
        x_axis_values, cobalt_rejection_co_al_norm, "c:", linewidth=2
    )  # , label="Cobalt (Co-Al)")
    ax2.plot(
        x_axis_values, aluminum_rejection_co_al_norm, "g:", linewidth=2
    )  # , label="Aluminum (Co-Al)")
    ax2.set_xlabel("Membrane Area (m$^2$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel(
        "Percent Change in Solute Rejection (%)", fontsize=10, fontweight="bold"
    )
    ax2.tick_params(direction="in", top=True, right=True, labelsize=10)

    ax2.plot([0, 164], [0, 0], "k-", linewidth=0.5)

    # ax2.set_xlim(0, 164)
    # ax2.set_ylim(-12, 2)

    # ax2.set_ylim(-50, 2)

    # legend points
    # ax2.plot([],[], marker='None', linestyle='None', label="Solution (linestyle)")
    ax2.plot([], [], "k-", linewidth=2, label="Li-Co")
    ax2.plot([], [], "k--", linewidth=2, label="Li-Al")
    ax2.plot([], [], "k-.", linewidth=2, label="Co-Al")
    ax2.plot([], [], marker="None", linestyle="None", label="Solute (color)")
    ax2.plot([], [], "ms", markersize=8, label="Lithium")
    ax2.plot([], [], "cs", markersize=8, label="Cobalt")
    ax2.plot([], [], "gs", markersize=8, label="Aluminum")

    ax2.legend(
        loc="best", title="Solution (linestyle)"
    )  # , bbox_to_anchor=(0.43, 0.54))

    plt.tight_layout()

    plt.show()


def plot_relative_rejections_by_component(
    m_li_cl, m_co_cl, m_al_cl, m_li_co_cl, m_li_al_cl, m_co_al_cl, m_li_co_al_cl
):
    """
    Plots relative solute rejection across the length of the membrane module.
    Rejections normalized to initial rejection (x=0).
    Compares models.

    Args:
        m: Pyomo model
    """
    # store values for x-coordinate
    x_axis_values = []

    # store values for rejection
    observed_lithium_rejection_li = []
    observed_lithium_rejection_li_co = []
    observed_lithium_rejection_li_al = []
    observed_lithium_rejection_li_co_al = []

    observed_cobalt_rejection_co = []
    observed_cobalt_rejection_li_co = []
    observed_cobalt_rejection_co_al = []
    observed_cobalt_rejection_li_co_al = []

    observed_aluminum_rejection_al = []
    observed_aluminum_rejection_li_al = []
    observed_aluminum_rejection_co_al = []
    observed_aluminum_rejection_li_co_al = []

    actual_lithium_rejection_li = []
    actual_lithium_rejection_li_co = []
    actual_lithium_rejection_li_al = []
    actual_lithium_rejection_li_co_al = []

    actual_cobalt_rejection_co = []
    actual_cobalt_rejection_li_co = []
    actual_cobalt_rejection_co_al = []
    actual_cobalt_rejection_li_co_al = []

    actual_aluminum_rejection_al = []
    actual_aluminum_rejection_li_al = []
    actual_aluminum_rejection_co_al = []
    actual_aluminum_rejection_li_co_al = []

    for x_val in m_li_co_cl.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            x_axis_values.append(
                x_val
                * value(m_li_co_cl.fs.membrane.total_module_length)
                * value(m_li_co_cl.fs.membrane.total_membrane_length)
            )

            observed_lithium_rej_li = (
                1
                - (
                    value(
                        m_li_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_li_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            observed_cobalt_rej_co = (
                1
                - (
                    value(
                        m_co_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_co_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            observed_aluminum_rej_al = (
                1
                - (
                    value(
                        m_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            observed_lithium_rej_li_co = (
                1
                - (
                    value(
                        m_li_co_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_li_co_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            observed_cobalt_rej_li_co = (
                1
                - (
                    value(
                        m_li_co_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_li_co_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                )
            ) * 100

            observed_lithium_rej_li_al = (
                1
                - (
                    value(
                        m_li_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_li_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100

            observed_aluminum_rej_li_al = (
                1
                - (
                    value(
                        m_li_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_li_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                )
            ) * 100

            observed_cobalt_rej_co_al = (
                1
                - (
                    value(
                        m_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_co_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100
            observed_aluminum_rej_co_al = (
                1
                - (
                    value(
                        m_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_co_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                )
            ) * 100

            observed_lithium_rej_li_co_al = (
                1
                - (
                    value(
                        m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_li_co_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                )
            ) * 100
            observed_cobalt_rej_li_co_al = (
                1
                - (
                    value(
                        m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_li_co_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                )
            ) * 100
            observed_aluminum_rej_li_co_al = (
                1
                - (
                    value(
                        m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_3"
                        ]
                    )
                    / value(
                        m_li_co_al_cl.fs.membrane.retentate_conc_mol_comp[
                            0, x_val, "cation_3"
                        ]
                    )
                )
            ) * 100

            #############
            actual_lithium_rej_li = (
                1
                - (
                    value(
                        m_li_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_li_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_1"
                        ]
                    )
                )
            ) * 100

            actual_cobalt_rej_co = (
                1
                - (
                    value(
                        m_co_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_co_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_1"
                        ]
                    )
                )
            ) * 100

            actual_aluminum_rej_al = (
                1
                - (
                    value(
                        m_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
                    )
                    / value(
                        m_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_1"
                        ]
                    )
                )
            ) * 100

            actual_lithium_rej_li_co = (
                1
                - (
                    value(
                        m_li_co_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_li_co_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_1"
                        ]
                    )
                )
            ) * 100

            actual_cobalt_rej_li_co = (
                1
                - (
                    value(
                        m_li_co_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_li_co_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_2"
                        ]
                    )
                )
            ) * 100

            actual_lithium_rej_li_al = (
                1
                - (
                    value(
                        m_li_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_li_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_1"
                        ]
                    )
                )
            ) * 100

            actual_aluminum_rej_li_al = (
                1
                - (
                    value(
                        m_li_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_li_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_2"
                        ]
                    )
                )
            ) * 100

            actual_cobalt_rej_co_al = (
                1
                - (
                    value(
                        m_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_1"
                        ]
                    )
                )
            ) * 100
            actual_aluminum_rej_co_al = (
                1
                - (
                    value(
                        m_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_2"
                        ]
                    )
                )
            ) * 100

            actual_lithium_rej_li_co_al = (
                1
                - (
                    value(
                        m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_1"
                        ]
                    )
                    / value(
                        m_li_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_1"
                        ]
                    )
                )
            ) * 100
            actual_cobalt_rej_li_co_al = (
                1
                - (
                    value(
                        m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_2"
                        ]
                    )
                    / value(
                        m_li_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_2"
                        ]
                    )
                )
            ) * 100
            actual_aluminum_rej_li_co_al = (
                1
                - (
                    value(
                        m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[
                            0, x_val, "cation_3"
                        ]
                    )
                    / value(
                        m_li_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, 1, "cation_3"
                        ]
                    )
                )
            ) * 100

            observed_lithium_rejection_li.append(observed_lithium_rej_li)
            observed_lithium_rejection_li_co.append(observed_lithium_rej_li_co)
            observed_lithium_rejection_li_al.append(observed_lithium_rej_li_al)
            observed_lithium_rejection_li_co_al.append(observed_lithium_rej_li_co_al)
            observed_cobalt_rejection_co.append(observed_cobalt_rej_co)
            observed_cobalt_rejection_li_co.append(observed_cobalt_rej_li_co)
            observed_cobalt_rejection_co_al.append(observed_cobalt_rej_co_al)
            observed_cobalt_rejection_li_co_al.append(observed_cobalt_rej_li_co_al)
            observed_aluminum_rejection_al.append(observed_aluminum_rej_al)
            observed_aluminum_rejection_li_al.append(observed_aluminum_rej_li_al)
            observed_aluminum_rejection_co_al.append(observed_aluminum_rej_co_al)
            observed_aluminum_rejection_li_co_al.append(observed_aluminum_rej_li_co_al)

            actual_lithium_rejection_li.append(actual_lithium_rej_li)
            actual_lithium_rejection_li_co.append(actual_lithium_rej_li_co)
            actual_lithium_rejection_li_al.append(actual_lithium_rej_li_al)
            actual_lithium_rejection_li_co_al.append(actual_lithium_rej_li_co_al)
            actual_cobalt_rejection_co.append(actual_cobalt_rej_co)
            actual_cobalt_rejection_li_co.append(actual_cobalt_rej_li_co)
            actual_cobalt_rejection_co_al.append(actual_cobalt_rej_co_al)
            actual_cobalt_rejection_li_co_al.append(actual_cobalt_rej_li_co_al)
            actual_aluminum_rejection_al.append(actual_aluminum_rej_al)
            actual_aluminum_rejection_li_al.append(actual_aluminum_rej_li_al)
            actual_aluminum_rejection_co_al.append(actual_aluminum_rej_co_al)
            actual_aluminum_rejection_li_co_al.append(actual_aluminum_rej_li_co_al)

    observed_lithium_rejection_li_norm = [
        (i - observed_lithium_rejection_li[0])
        / abs(observed_lithium_rejection_li[0])
        * 100
        for i in observed_lithium_rejection_li
    ]
    observed_lithium_rejection_li_co_norm = [
        (i - observed_lithium_rejection_li_co[0])
        / abs(observed_lithium_rejection_li_co[0])
        * 100
        for i in observed_lithium_rejection_li_co
    ]
    observed_lithium_rejection_li_al_norm = [
        (i - observed_lithium_rejection_li_al[0])
        / abs(observed_lithium_rejection_li_al[0])
        * 100
        for i in observed_lithium_rejection_li_al
    ]
    observed_lithium_rejection_li_co_al_norm = [
        (i - observed_lithium_rejection_li_co_al[0])
        / abs(observed_lithium_rejection_li_co_al[0])
        * 100
        for i in observed_lithium_rejection_li_co_al
    ]
    observed_cobalt_rejection_co_norm = [
        (i - observed_cobalt_rejection_co[0])
        / abs(observed_cobalt_rejection_co[0])
        * 100
        for i in observed_cobalt_rejection_co
    ]
    observed_cobalt_rejection_li_co_norm = [
        (i - observed_cobalt_rejection_li_co[0])
        / abs(observed_cobalt_rejection_li_co[0])
        * 100
        for i in observed_cobalt_rejection_li_co
    ]
    observed_cobalt_rejection_co_al_norm = [
        (i - observed_cobalt_rejection_co_al[0])
        / abs(observed_cobalt_rejection_co_al[0])
        * 100
        for i in observed_cobalt_rejection_co_al
    ]
    observed_cobalt_rejection_li_co_al_norm = [
        (i - observed_cobalt_rejection_li_co_al[0])
        / abs(observed_cobalt_rejection_li_co_al[0])
        * 100
        for i in observed_cobalt_rejection_li_co_al
    ]
    observed_aluminum_rejection_al_norm = [
        (i - observed_aluminum_rejection_al[0])
        / abs(observed_aluminum_rejection_al[0])
        * 100
        for i in observed_aluminum_rejection_al
    ]
    observed_aluminum_rejection_li_al_norm = [
        (i - observed_aluminum_rejection_li_al[0])
        / abs(observed_aluminum_rejection_li_al[0])
        * 100
        for i in observed_aluminum_rejection_li_al
    ]
    observed_aluminum_rejection_co_al_norm = [
        (i - observed_aluminum_rejection_co_al[0])
        / abs(observed_aluminum_rejection_co_al[0])
        * 100
        for i in observed_aluminum_rejection_co_al
    ]
    observed_aluminum_rejection_li_co_al_norm = [
        (i - observed_aluminum_rejection_li_co_al[0])
        / abs(observed_aluminum_rejection_li_co_al[0])
        * 100
        for i in observed_aluminum_rejection_li_co_al
    ]

    #########
    actual_lithium_rejection_li_norm = [
        (i - actual_lithium_rejection_li[0]) / abs(actual_lithium_rejection_li[0]) * 100
        for i in actual_lithium_rejection_li
    ]
    actual_lithium_rejection_li_co_norm = [
        (i - actual_lithium_rejection_li_co[0])
        / abs(actual_lithium_rejection_li_co[0])
        * 100
        for i in actual_lithium_rejection_li_co
    ]
    actual_lithium_rejection_li_al_norm = [
        (i - actual_lithium_rejection_li_al[0])
        / abs(actual_lithium_rejection_li_al[0])
        * 100
        for i in actual_lithium_rejection_li_al
    ]
    actual_lithium_rejection_li_co_al_norm = [
        (i - actual_lithium_rejection_li_co_al[0])
        / abs(actual_lithium_rejection_li_co_al[0])
        * 100
        for i in actual_lithium_rejection_li_co_al
    ]
    actual_cobalt_rejection_co_norm = [
        (i - actual_cobalt_rejection_co[0]) / abs(actual_cobalt_rejection_co[0]) * 100
        for i in actual_cobalt_rejection_co
    ]
    actual_cobalt_rejection_li_co_norm = [
        (i - actual_cobalt_rejection_li_co[0])
        / abs(actual_cobalt_rejection_li_co[0])
        * 100
        for i in actual_cobalt_rejection_li_co
    ]
    actual_cobalt_rejection_co_al_norm = [
        (i - actual_cobalt_rejection_co_al[0])
        / abs(actual_cobalt_rejection_co_al[0])
        * 100
        for i in actual_cobalt_rejection_co_al
    ]
    actual_cobalt_rejection_li_co_al_norm = [
        (i - actual_cobalt_rejection_li_co_al[0])
        / abs(actual_cobalt_rejection_li_co_al[0])
        * 100
        for i in actual_cobalt_rejection_li_co_al
    ]
    actual_aluminum_rejection_al_norm = [
        (i - actual_aluminum_rejection_al[0])
        / abs(actual_aluminum_rejection_al[0])
        * 100
        for i in actual_aluminum_rejection_al
    ]
    actual_aluminum_rejection_li_al_norm = [
        (i - actual_aluminum_rejection_li_al[0])
        / abs(actual_aluminum_rejection_li_al[0])
        * 100
        for i in actual_aluminum_rejection_li_al
    ]
    actual_aluminum_rejection_co_al_norm = [
        (i - actual_aluminum_rejection_co_al[0])
        / abs(actual_aluminum_rejection_co_al[0])
        * 100
        for i in actual_aluminum_rejection_co_al
    ]
    actual_aluminum_rejection_li_co_al_norm = [
        (i - actual_aluminum_rejection_li_co_al[0])
        / abs(actual_aluminum_rejection_li_co_al[0])
        * 100
        for i in actual_aluminum_rejection_li_co_al
    ]

    fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))
    fig1.suptitle("Lithium Rejection")
    ax1.plot(
        x_axis_values, observed_lithium_rejection_li, "r-", alpha=0.25, linewidth=2
    )  # , label="Lithium (Li)")
    ax1.plot(
        x_axis_values, observed_lithium_rejection_li_co, "r--", alpha=0.25, linewidth=2
    )  # , label="Lithium (Li-Co)")
    ax1.plot(
        x_axis_values, observed_lithium_rejection_li_al, "r-.", alpha=0.25, linewidth=2
    )  # , label="Lithium (Li-Al)")
    ax1.plot(
        x_axis_values,
        observed_lithium_rejection_li_co_al,
        "r.-",
        alpha=0.25,
        linewidth=2,
    )  # , label="Lithium (Li-Co-Al)")
    ax1.plot(
        x_axis_values, actual_lithium_rejection_li, "r-", linewidth=2
    )  # , label="Lithium (Li)")
    ax1.plot(
        x_axis_values, actual_lithium_rejection_li_co, "r--", linewidth=2
    )  # , label="Lithium (Li-Co)")
    ax1.plot(
        x_axis_values, actual_lithium_rejection_li_al, "r-.", linewidth=2
    )  # , label="Lithium (Li-Al)")
    ax1.plot(
        x_axis_values, actual_lithium_rejection_li_co_al, "r.-", linewidth=2
    )  # , label="Lithium (Li-Co-Al)")

    ax1.set_xlabel("Membrane Area (m$^2$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Solute Rejection (%)", fontsize=12, fontweight="bold")
    ax1.tick_params(direction="in", labelsize=10)

    ax2.plot(
        x_axis_values, observed_lithium_rejection_li_norm, "r-", alpha=0.25, linewidth=2
    )  # , label="Lithium (Li)")
    ax2.plot(
        x_axis_values,
        observed_lithium_rejection_li_co_norm,
        "r--",
        alpha=0.25,
        linewidth=2,
    )  # , label="Lithium (Li-Co)")
    ax2.plot(
        x_axis_values,
        observed_lithium_rejection_li_al_norm,
        "r-.",
        alpha=0.25,
        linewidth=2,
    )  # , label="Lithium (Li-Al)")
    ax2.plot(
        x_axis_values,
        observed_lithium_rejection_li_co_al_norm,
        "r.-",
        alpha=0.25,
        linewidth=2,
    )  # , label="Lithium (Li-Co-Al)")
    ax2.plot(
        x_axis_values, actual_lithium_rejection_li_norm, "r-", linewidth=2
    )  # , label="Lithium (Li)")
    ax2.plot(
        x_axis_values, actual_lithium_rejection_li_co_norm, "r--", linewidth=2
    )  # , label="Lithium (Li-Co)")
    ax2.plot(
        x_axis_values, actual_lithium_rejection_li_al_norm, "r-.", linewidth=2
    )  # , label="Lithium (Li-Al)")
    ax2.plot(
        x_axis_values, actual_lithium_rejection_li_co_al_norm, "r.-", linewidth=2
    )  # , label="Lithium (Li-Co-Al)")

    ax2.set_xlabel("Membrane Area (m$^2$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel(
        "Percent Change in Solute Rejection (%)", fontsize=10, fontweight="bold"
    )
    ax2.tick_params(direction="in", top=True, right=True, labelsize=10)

    ax2.plot([0, 164], [0, 0], "k-", linewidth=0.5)
    ax2.set_xlim(0, 164)
    # ax1.set_ylim(0, 18)

    # legend points
    ax2.plot([], [], "k-", linewidth=2, label="LiCl")
    ax2.plot([], [], "k--", linewidth=2, label="LiCl + CoCl2")
    ax2.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl3")
    ax2.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl2 + AlCl3")
    ax2.plot([], [], marker="None", linestyle="None", label="Rejection (color)")
    ax2.plot([], [], "rs", alpha=0.25, markersize=8, label="Observed")
    ax2.plot([], [], "rs", markersize=8, label="Actual")
    ax2.legend(
        loc="best", title="Solution (linestyle)"
    )  # , bbox_to_anchor=(0.43, 0.54))

    fig2, (ax3, ax4) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))
    fig2.suptitle("Cobalt Rejection")
    ax3.plot(
        x_axis_values, observed_cobalt_rejection_co, "b-", alpha=0.25, linewidth=2
    )  # , label="Cobalt (Co)")
    ax3.plot(
        x_axis_values, observed_cobalt_rejection_li_co, "b--", alpha=0.25, linewidth=2
    )  # , label="Cobalt (Li-Co)")
    ax3.plot(
        x_axis_values, observed_cobalt_rejection_co_al, "b:", alpha=0.25, linewidth=2
    )  # , label="Cobalt (Co-Al)")
    ax3.plot(
        x_axis_values,
        observed_cobalt_rejection_li_co_al,
        "b.-",
        alpha=0.25,
        linewidth=2,
    )  # , label="Cobalt (Li-Co-Al)")
    ax3.plot(
        x_axis_values, actual_cobalt_rejection_co, "b-", linewidth=2
    )  # , label="Cobalt (Co)")
    ax3.plot(
        x_axis_values, actual_cobalt_rejection_li_co, "b--", linewidth=2
    )  # , label="Cobalt (Li-Co)")
    ax3.plot(
        x_axis_values, actual_cobalt_rejection_co_al, "b:", linewidth=2
    )  # , label="Cobalt (Co-Al)")
    ax3.plot(
        x_axis_values, actual_cobalt_rejection_li_co_al, "b.-", linewidth=2
    )  # , label="Cobalt (Li-Co-Al)")

    ax3.set_xlabel("Membrane Area (m$^2$)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Solute Rejection (%)", fontsize=12, fontweight="bold")
    ax3.tick_params(direction="in", labelsize=10)

    ax4.plot(
        x_axis_values, observed_cobalt_rejection_co_norm, "b-", alpha=0.25, linewidth=2
    )  # , label="Cobalt (Co)")
    ax4.plot(
        x_axis_values,
        observed_cobalt_rejection_li_co_norm,
        "b--",
        alpha=0.25,
        linewidth=2,
    )  # , label="Cobalt (Li-Co)")
    ax4.plot(
        x_axis_values,
        observed_cobalt_rejection_co_al_norm,
        "b:",
        alpha=0.25,
        linewidth=2,
    )  # , label="Cobalt (Co-Al)")
    ax4.plot(
        x_axis_values,
        observed_cobalt_rejection_li_co_al_norm,
        "b.-",
        alpha=0.25,
        linewidth=2,
    )  # , label="Cobalt (Li-Co-Al)")
    ax4.plot(
        x_axis_values, actual_cobalt_rejection_co_norm, "b-", linewidth=2
    )  # , label="Cobalt (Co)")
    ax4.plot(
        x_axis_values, actual_cobalt_rejection_li_co_norm, "b--", linewidth=2
    )  # , label="Cobalt (Li-Co)")
    ax4.plot(
        x_axis_values, actual_cobalt_rejection_co_al_norm, "b:", linewidth=2
    )  # , label="Cobalt (Co-Al)")
    ax4.plot(
        x_axis_values, actual_cobalt_rejection_li_co_al_norm, "b.-", linewidth=2
    )  # , label="Cobalt (Li-Co-Al)")

    ax4.set_xlabel("Membrane Area (m$^2$)", fontsize=10, fontweight="bold")
    ax4.set_ylabel(
        "Percent Change in Solute Rejection (%)", fontsize=10, fontweight="bold"
    )
    ax4.tick_params(direction="in", top=True, right=True, labelsize=10)

    ax4.plot([0, 164], [0, 0], "k-", linewidth=0.5)
    ax4.set_xlim(0, 164)
    # ax3.set_ylim(0, 18)

    # legend points
    ax4.plot([], [], "k-", linewidth=2, label="CoCl2")
    ax4.plot([], [], "k--", linewidth=2, label="LiCl + CoCl2")
    ax4.plot([], [], "k:", linewidth=2, label="CoCl2 + AlCl3")
    ax4.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl2 + AlCl3")
    ax4.plot([], [], marker="None", linestyle="None", label="Rejection (color)")
    ax4.plot([], [], "bs", alpha=0.25, markersize=8, label="Observed")
    ax4.plot([], [], "bs", markersize=8, label="Actual")
    ax4.legend(
        loc="best", title="Solution (linestyle)"
    )  # , bbox_to_anchor=(0.43, 0.54))

    fig3, (ax5, ax6) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))
    fig3.suptitle("Aluminum Rejection")
    ax5.plot(
        x_axis_values, observed_aluminum_rejection_al, "g-", alpha=0.25, linewidth=2
    )  # , label="Aluminum (Al)")
    ax5.plot(
        x_axis_values, observed_aluminum_rejection_li_al, "g-.", alpha=0.25, linewidth=2
    )  # , label="Aluminum (Li-Al)")
    ax5.plot(
        x_axis_values, observed_aluminum_rejection_co_al, "g:", alpha=0.25, linewidth=2
    )  # , label="Aluminum (Co-Al)")
    ax5.plot(
        x_axis_values,
        observed_aluminum_rejection_li_co_al,
        "g.-",
        alpha=0.25,
        linewidth=2,
    )  # , label="Aluminum Li-(Co-Al)")
    ax5.plot(
        x_axis_values, actual_aluminum_rejection_al, "g-", linewidth=2
    )  # , label="Aluminum (Al)")
    ax5.plot(
        x_axis_values, actual_aluminum_rejection_li_al, "g-.", linewidth=2
    )  # , label="Aluminum (Li-Al)")
    ax5.plot(
        x_axis_values, actual_aluminum_rejection_co_al, "g:", linewidth=2
    )  # , label="Aluminum (Co-Al)")
    ax5.plot(
        x_axis_values, actual_aluminum_rejection_li_co_al, "g.-", linewidth=2
    )  # , label="Aluminum Li-(Co-Al)")

    ax5.set_xlabel("Membrane Area (m$^2$)", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Solute Rejection (%)", fontsize=12, fontweight="bold")
    ax5.tick_params(direction="in", labelsize=10)

    ax6.plot(
        x_axis_values,
        observed_aluminum_rejection_al_norm,
        "g-",
        alpha=0.25,
        linewidth=2,
    )  # , label="Aluminum (Al)")
    ax6.plot(
        x_axis_values,
        observed_aluminum_rejection_li_al_norm,
        "g-.",
        alpha=0.25,
        linewidth=2,
    )  # , label="Aluminum (Li-Al)")
    ax6.plot(
        x_axis_values,
        observed_aluminum_rejection_co_al_norm,
        "g:",
        alpha=0.25,
        linewidth=2,
    )  # , label="Aluminum (Co-Al)")
    ax6.plot(
        x_axis_values,
        observed_aluminum_rejection_li_co_al_norm,
        "g.-",
        alpha=0.25,
        linewidth=2,
    )  # , label="Aluminum (Li-Co-Al)")
    ax6.plot(
        x_axis_values, actual_aluminum_rejection_al_norm, "g-", linewidth=2
    )  # , label="Aluminum (Al)")
    ax6.plot(
        x_axis_values, actual_aluminum_rejection_li_al_norm, "g-.", linewidth=2
    )  # , label="Aluminum (Li-Al)")
    ax6.plot(
        x_axis_values, actual_aluminum_rejection_co_al_norm, "g:", linewidth=2
    )  # , label="Aluminum (Co-Al)")
    ax6.plot(
        x_axis_values, actual_aluminum_rejection_li_co_al_norm, "g.-", linewidth=2
    )  # , label="Aluminum (Li-Co-Al)")

    ax6.set_xlabel("Membrane Area (m$^2$)", fontsize=10, fontweight="bold")
    ax6.set_ylabel(
        "Percent Change in Solute Rejection (%)", fontsize=10, fontweight="bold"
    )
    ax6.tick_params(direction="in", top=True, right=True, labelsize=10)

    ax6.plot([0, 164], [0, 0], "k-", linewidth=0.5)
    ax6.set_xlim(0, 164)
    # ax5.set_ylim(0, 18)

    # legend points
    ax6.plot([], [], "k-", linewidth=2, label="AlCl3")
    ax6.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl3")
    ax6.plot([], [], "k:", linewidth=2, label="CoCl2 + AlCl3")
    ax6.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl2 + AlCl3")
    ax6.plot([], [], marker="None", linestyle="None", label="Rejection (color)")
    ax6.plot([], [], "gs", alpha=0.25, markersize=8, label="Observed")
    ax6.plot([], [], "gs", markersize=8, label="Actual")
    ax6.legend(
        loc="best", title="Solution (linestyle)"
    )  # , bbox_to_anchor=(0.43, 0.54))

    plt.tight_layout()

    plt.show()


def plot_rejection_versus_concentration(
    m_li_cl, m_co_cl, m_al_cl, m_li_co_cl, m_li_al_cl, m_co_al_cl, m_li_co_al_cl
):
    """
    Plots rejection versus retentate-side concentration.
    """
    # store values for concentration
    lithium_concentration_li = []
    lithium_concentration_li_co = []
    lithium_concentration_li_al = []
    lithium_concentration_li_co_al = []

    cobalt_concentration_co = []
    cobalt_concentration_li_co = []
    cobalt_concentration_co_al = []
    cobalt_concentration_li_co_al = []

    aluminum_concentration_al = []
    aluminum_concentration_li_al = []
    aluminum_concentration_co_al = []
    aluminum_concentration_li_co_al = []

    # store values for rejection
    observed_lithium_rejection_li = []
    observed_lithium_rejection_li_co = []
    observed_lithium_rejection_li_al = []
    observed_lithium_rejection_li_co_al = []

    observed_cobalt_rejection_co = []
    observed_cobalt_rejection_li_co = []
    observed_cobalt_rejection_co_al = []
    observed_cobalt_rejection_li_co_al = []

    observed_aluminum_rejection_al = []
    observed_aluminum_rejection_li_al = []
    observed_aluminum_rejection_co_al = []
    observed_aluminum_rejection_li_co_al = []

    actual_lithium_rejection_li = []
    actual_lithium_rejection_li_co = []
    actual_lithium_rejection_li_al = []
    actual_lithium_rejection_li_co_al = []

    actual_cobalt_rejection_co = []
    actual_cobalt_rejection_li_co = []
    actual_cobalt_rejection_co_al = []
    actual_cobalt_rejection_li_co_al = []

    actual_aluminum_rejection_al = []
    actual_aluminum_rejection_li_al = []
    actual_aluminum_rejection_co_al = []
    actual_aluminum_rejection_li_co_al = []

    for x_val in m_li_cl.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            ########################################################################
            conc_ret_lith_li_cl = value(
                m_li_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_1"]
            )
            conc_int_lith_li_cl = value(
                m_li_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_1"
                ]
            )
            conc_perm_lith_li_cl = value(
                m_li_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
            )
            lithium_concentration_li.append(conc_int_lith_li_cl)
            observed_lithium_rejection_li.append(
                (1 - (conc_perm_lith_li_cl / conc_ret_lith_li_cl)) * 100
            )
            actual_lithium_rejection_li.append(
                (1 - (conc_perm_lith_li_cl / conc_int_lith_li_cl)) * 100
            )

            conc_ret_lith_li_co_cl = value(
                m_li_co_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_1"]
            )
            conc_int_lith_li_co_cl = value(
                m_li_co_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_1"
                ]
            )
            conc_perm_lith_li_co_cl = value(
                m_li_co_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
            )
            lithium_concentration_li_co.append(conc_int_lith_li_co_cl)
            observed_lithium_rejection_li_co.append(
                (1 - (conc_perm_lith_li_co_cl / conc_ret_lith_li_co_cl)) * 100
            )
            actual_lithium_rejection_li_co.append(
                (1 - (conc_perm_lith_li_co_cl / conc_int_lith_li_co_cl)) * 100
            )

            conc_ret_lith_li_al_cl = value(
                m_li_al_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_1"]
            )
            conc_int_lith_li_al_cl = value(
                m_li_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_1"
                ]
            )
            conc_perm_lith_li_al_cl = value(
                m_li_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
            )
            lithium_concentration_li_al.append(conc_int_lith_li_al_cl)
            observed_lithium_rejection_li_al.append(
                (1 - (conc_perm_lith_li_al_cl / conc_ret_lith_li_al_cl)) * 100
            )
            actual_lithium_rejection_li_al.append(
                (1 - (conc_perm_lith_li_al_cl / conc_int_lith_li_al_cl)) * 100
            )

            conc_ret_lith_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_1"]
            )
            conc_int_lith_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_1"
                ]
            )
            conc_perm_lith_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
            )
            lithium_concentration_li_co_al.append(conc_int_lith_li_co_al_cl)
            observed_lithium_rejection_li_co_al.append(
                (1 - (conc_perm_lith_li_co_al_cl / conc_ret_lith_li_co_al_cl)) * 100
            )
            actual_lithium_rejection_li_co_al.append(
                (1 - (conc_perm_lith_li_co_al_cl / conc_int_lith_li_co_al_cl)) * 100
            )

            ########################################################################
            conc_ret_cob_co_cl = value(
                m_co_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_1"]
            )
            conc_int_cob_co_cl = value(
                m_co_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_1"
                ]
            )
            conc_perm_cob_co_cl = value(
                m_co_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
            )
            cobalt_concentration_co.append(conc_int_cob_co_cl)
            observed_cobalt_rejection_co.append(
                (1 - (conc_perm_cob_co_cl / conc_ret_cob_co_cl)) * 100
            )
            actual_cobalt_rejection_co.append(
                (1 - (conc_perm_cob_co_cl / conc_int_cob_co_cl)) * 100
            )

            conc_ret_cob_li_co_cl = value(
                m_li_co_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_2"]
            )
            conc_int_cob_li_co_cl = value(
                m_li_co_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_2"
                ]
            )
            conc_perm_cob_li_co_cl = value(
                m_li_co_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_2"]
            )
            cobalt_concentration_li_co.append(conc_int_cob_li_co_cl)
            observed_cobalt_rejection_li_co.append(
                (1 - (conc_perm_cob_li_co_cl / conc_ret_cob_li_co_cl)) * 100
            )
            actual_cobalt_rejection_li_co.append(
                (1 - (conc_perm_cob_li_co_cl / conc_int_cob_li_co_cl)) * 100
            )

            conc_ret_cob_co_al_cl = value(
                m_co_al_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_1"]
            )
            conc_int_cob_co_al_cl = value(
                m_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_1"
                ]
            )
            conc_perm_cob_co_al_cl = value(
                m_co_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
            )
            cobalt_concentration_co_al.append(conc_int_cob_co_al_cl)
            observed_cobalt_rejection_co_al.append(
                (1 - (conc_perm_cob_co_al_cl / conc_ret_cob_co_al_cl)) * 100
            )
            actual_cobalt_rejection_co_al.append(
                (1 - (conc_perm_cob_co_al_cl / conc_int_cob_co_al_cl)) * 100
            )

            conc_ret_cob_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_2"]
            )
            conc_int_cob_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_2"
                ]
            )
            conc_perm_cob_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_2"]
            )
            cobalt_concentration_li_co_al.append(conc_int_cob_li_co_al_cl)
            observed_cobalt_rejection_li_co_al.append(
                (1 - (conc_perm_cob_li_co_al_cl / conc_ret_cob_li_co_al_cl)) * 100
            )
            actual_cobalt_rejection_li_co_al.append(
                (1 - (conc_perm_cob_li_co_al_cl / conc_int_cob_li_co_al_cl)) * 100
            )

            ########################################################################

            conc_ret_alum_al_cl = value(
                m_al_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_1"]
            )
            conc_int_alum_al_cl = value(
                m_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_1"
                ]
            )
            conc_perm_alum_al_cl = value(
                m_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_1"]
            )
            aluminum_concentration_al.append(conc_int_alum_al_cl)
            observed_aluminum_rejection_al.append(
                (1 - (conc_perm_alum_al_cl / conc_ret_alum_al_cl)) * 100
            )
            actual_aluminum_rejection_al.append(
                (1 - (conc_perm_alum_al_cl / conc_int_alum_al_cl)) * 100
            )

            conc_ret_alum_li_al_cl = value(
                m_li_al_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_2"]
            )
            conc_int_alum_li_al_cl = value(
                m_li_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_2"
                ]
            )
            conc_perm_alum_li_al_cl = value(
                m_li_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_2"]
            )
            aluminum_concentration_li_al.append(conc_int_alum_li_al_cl)
            observed_aluminum_rejection_li_al.append(
                (1 - (conc_perm_alum_li_al_cl / conc_ret_alum_li_al_cl)) * 100
            )
            actual_aluminum_rejection_li_al.append(
                (1 - (conc_perm_alum_li_al_cl / conc_int_alum_li_al_cl)) * 100
            )

            conc_ret_alum_co_al_cl = value(
                m_co_al_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_2"]
            )
            conc_int_alum_co_al_cl = value(
                m_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_2"
                ]
            )
            conc_perm_alum_co_al_cl = value(
                m_co_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_2"]
            )
            aluminum_concentration_co_al.append(conc_int_alum_co_al_cl)
            observed_aluminum_rejection_co_al.append(
                (1 - (conc_perm_alum_co_al_cl / conc_ret_alum_co_al_cl)) * 100
            )
            actual_aluminum_rejection_co_al.append(
                (1 - (conc_perm_alum_co_al_cl / conc_int_alum_co_al_cl)) * 100
            )

            conc_ret_alum_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.retentate_conc_mol_comp[0, x_val, "cation_3"]
            )
            conc_int_alum_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.boundary_layer_conc_mol_comp[
                    0, x_val, 1, "cation_3"
                ]
            )
            conc_perm_alum_li_co_al_cl = value(
                m_li_co_al_cl.fs.membrane.permeate_conc_mol_comp[0, x_val, "cation_3"]
            )
            aluminum_concentration_li_co_al.append(conc_int_alum_li_co_al_cl)
            observed_aluminum_rejection_li_co_al.append(
                (1 - (conc_perm_alum_li_co_al_cl / conc_ret_alum_li_co_al_cl)) * 100
            )
            actual_aluminum_rejection_li_co_al.append(
                (1 - (conc_perm_alum_li_co_al_cl / conc_int_alum_li_co_al_cl)) * 100
            )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=100, figsize=(10, 5))

    ax1.plot(
        lithium_concentration_li,
        observed_lithium_rejection_li,
        "r-",
        alpha=0.25,
        linewidth=2,
    )
    ax1.plot(
        lithium_concentration_li,
        actual_lithium_rejection_li,
        "r-",
        linewidth=2,
    )
    ax1.plot(
        lithium_concentration_li_co,
        observed_lithium_rejection_li_co,
        "r--",
        alpha=0.25,
        linewidth=2,
    )
    ax1.plot(
        lithium_concentration_li_co,
        actual_lithium_rejection_li_co,
        "r--",
        linewidth=2,
    )
    ax1.plot(
        lithium_concentration_li_al,
        observed_lithium_rejection_li_al,
        "r-.",
        alpha=0.25,
        linewidth=2,
    )
    ax1.plot(
        lithium_concentration_li_al,
        actual_lithium_rejection_li_al,
        "r-.",
        linewidth=2,
    )
    ax1.plot(
        lithium_concentration_li_co_al,
        observed_lithium_rejection_li_co_al,
        "r.-",
        alpha=0.25,
        linewidth=2,
    )
    ax1.plot(
        lithium_concentration_li_co_al,
        actual_lithium_rejection_li_co_al,
        "r.-",
        linewidth=2,
    )
    ax1.set_xlabel(
        "Lithium Concentration (Interface) (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_ylabel("Percent Rejection (%)", fontsize=10, fontweight="bold")
    ax1.tick_params(direction="in", labelsize=10)
    ax1.plot([], [], "k-", linewidth=2, label="LiCl")
    ax1.plot([], [], "k--", linewidth=2, label="LiCl + CoCl2")
    ax1.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl3")
    ax1.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl2 + AlCl3")
    ax1.plot([], [], marker="None", linestyle="None", label="Rejection (color)")
    ax1.plot([], [], "rs", alpha=0.25, markersize=8, label="Observed")
    ax1.plot([], [], "rs", markersize=8, label="Actual")
    ax1.legend(loc="best", title="Solution (linestyle)")

    ax2.plot(
        cobalt_concentration_co,
        observed_cobalt_rejection_co,
        "b-",
        alpha=0.25,
        linewidth=2,
    )
    ax2.plot(
        cobalt_concentration_co,
        actual_cobalt_rejection_co,
        "b-",
        linewidth=2,
    )
    ax2.plot(
        cobalt_concentration_li_co,
        observed_cobalt_rejection_li_co,
        "b--",
        alpha=0.25,
        linewidth=2,
    )
    ax2.plot(
        cobalt_concentration_li_co,
        actual_cobalt_rejection_li_co,
        "b--",
        linewidth=2,
    )
    ax2.plot(
        cobalt_concentration_co_al,
        observed_cobalt_rejection_co_al,
        "b:",
        alpha=0.25,
        linewidth=2,
    )
    ax2.plot(
        cobalt_concentration_co_al,
        actual_cobalt_rejection_co_al,
        "b:",
        linewidth=2,
    )
    ax2.plot(
        cobalt_concentration_li_co_al,
        observed_cobalt_rejection_li_co_al,
        "b.-",
        alpha=0.25,
        linewidth=2,
    )
    ax2.plot(
        cobalt_concentration_li_co_al,
        actual_cobalt_rejection_li_co_al,
        "b.-",
        linewidth=2,
    )
    ax2.set_xlabel(
        "Cobalt Concentration (Interface) (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax2.set_ylabel("Percent Rejection (%)", fontsize=10, fontweight="bold")
    ax2.tick_params(direction="in", labelsize=10)
    ax2.plot([], [], "k-", linewidth=2, label="CoCl2")
    ax2.plot([], [], "k--", linewidth=2, label="LiCl + CoCl2")
    ax2.plot([], [], "k:", linewidth=2, label="CoCl2 + AlCl3")
    ax2.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl2 + AlCl3")
    ax2.plot([], [], marker="None", linestyle="None", label="Rejection (color)")
    ax2.plot([], [], "bs", alpha=0.25, markersize=8, label="Observed")
    ax2.plot([], [], "bs", markersize=8, label="Actual")
    ax2.legend(loc="best", title="Solution (linestyle)")

    ax3.plot(
        aluminum_concentration_al,
        observed_aluminum_rejection_al,
        "g-",
        alpha=0.25,
        linewidth=2,
    )
    ax3.plot(
        aluminum_concentration_al,
        actual_aluminum_rejection_al,
        "g-",
        linewidth=2,
    )
    ax3.plot(
        aluminum_concentration_li_al,
        observed_aluminum_rejection_li_al,
        "g-.",
        alpha=0.25,
        linewidth=2,
    )
    ax3.plot(
        aluminum_concentration_li_al,
        actual_aluminum_rejection_li_al,
        "g-.",
        linewidth=2,
    )
    ax3.plot(
        aluminum_concentration_co_al,
        observed_aluminum_rejection_co_al,
        "g:",
        alpha=0.25,
        linewidth=2,
    )
    ax3.plot(
        aluminum_concentration_co_al,
        actual_aluminum_rejection_co_al,
        "g:",
        linewidth=2,
    )
    ax3.plot(
        aluminum_concentration_li_co_al,
        observed_aluminum_rejection_li_co_al,
        "g.-",
        alpha=0.25,
        linewidth=2,
    )
    ax3.plot(
        aluminum_concentration_li_co_al,
        actual_aluminum_rejection_li_co_al,
        "g.-",
        linewidth=2,
    )
    ax3.set_xlabel(
        "Aluminum Concentration (Interface) (mol/m$^3$)",
        fontsize=10,
        fontweight="bold",
    )
    ax3.set_ylabel("Percent Rejection (%)", fontsize=10, fontweight="bold")
    ax3.tick_params(direction="in", labelsize=10)

    ax3.plot([], [], "k-", linewidth=2, label="AlCl3")
    ax3.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl3")
    ax3.plot([], [], "k:", linewidth=2, label="CoCl2 + AlCl3")
    ax3.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl2 + AlCl3")
    ax3.plot([], [], marker="None", linestyle="None", label="Rejection (color)")
    ax3.plot([], [], "gs", alpha=0.25, markersize=8, label="Observed")
    ax3.plot([], [], "gs", markersize=8, label="Actual")
    ax3.legend(loc="best", title="Solution (linestyle)")

    plt.tight_layout()

    plt.show()


def plot_relative_flux():
    """
    Plots flux contributions for different systems.
    Compares two and three salt models.

    Args:
        m: Pyomo model
    """
    ionic_strength_list_2 = []
    li_pe_list_2 = []
    co_pe_list_2 = []
    # water_flux_list_2 = []

    lithium_pe_2 = []
    cobalt_pe_2 = []
    # water_flux_2 = []

    conc_list_2 = [
        [50, 100],
        [48, 96],
        [46, 92],
        [44, 88],
        [42, 84],
        [40, 80],
        [38, 76],
        [36, 72],
        [34, 68],
        [32, 64],
    ]

    m_two_salt = build_two_salt_model()
    results = solve_model(m_two_salt)
    # two_salt_model_checks(m_two_salt)

    for conc in conc_list_2:
        m_two_salt.fs.membrane.feed_conc_mol_comp[0, "Li"].fix(conc[0])
        m_two_salt.fs.membrane.feed_conc_mol_comp[0, "Co"].fix(conc[1])

        results = solve_model(m_two_salt)

        if results.solver.termination_condition == "optimal":
            dt = DiagnosticsToolbox(m_two_salt)
            dt.assert_no_numerical_warnings()

            for x in m_two_salt.fs.membrane.dimensionless_module_length:
                if x != 0:
                    # water_flux_2.append(value(m_two_salt.fs.membrane.volume_flux_water[x]))
                    for z in m_two_salt.fs.membrane.dimensionless_membrane_thickness:
                        lithium_pe_2.append(
                            value(m_two_salt.fs.membrane.peclet_number_lithium[x, z])
                        )
                        cobalt_pe_2.append(
                            value(m_two_salt.fs.membrane.peclet_number_cobalt[x, z])
                        )

        ionic_strength = calculate_ionic_strength_two_salt(m_two_salt)
        ionic_strength_list_2.append(ionic_strength)
        # print(lithium_pe_2)
        li_pe_list_2.append(np.average(lithium_pe_2))
        # print(cobalt_pe_2)
        co_pe_list_2.append(np.average(cobalt_pe_2))
        # water_flux_list_2.append(np.average(water_flux_2))

        lithium_pe_2 = []
        cobalt_pe_2 = []
        # water_flux_2 = []

    print(ionic_strength_list_2)
    print(li_pe_list_2)
    print(co_pe_list_2)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    # ax1.plot(ionic_strength_list_2, water_flux_list_2, '.')
    ax1.plot(
        ionic_strength_list_2,
        li_pe_list_2,
        "mx",
        markersize=6,
        linestyle="-",
        linewidth=0.7,
    )
    ax1.plot(
        ionic_strength_list_2,
        co_pe_list_2,
        "cx",
        markersize=6,
        linestyle="-",
        linewidth=0.7,
    )

    ionic_strength_list_3 = []
    li_pe_list_3 = []
    co_pe_list_3 = []
    al_pe_list_3 = []

    lithium_pe_3 = []
    cobalt_pe_3 = []
    aluminum_pe_3 = []
    # 10:20:2.5
    conc_list_3 = [
        [42, 84, 10.5],
        [40, 80, 10],
        [38, 76, 9.5],
        [36, 72, 9],
        [34, 68, 8.5],
        [32, 64, 8],
        [30, 60, 7.5],
        [28, 56, 7],
        [26, 52, 6.5],
    ]
    # initialize
    m_three_salt = build_three_salt_model()
    results = solve_model(m_three_salt)
    # three_salt_model_checks(m_three_salt)
    # m_three_salt.fs.membrane.diafiltrate_flow_volume.fix(1e-10)

    for conc in conc_list_3:
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Li"].fix(conc[0])
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Co"].fix(conc[1])
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Al"].fix(conc[2])

        results = solve_model(m_three_salt)
        if results.solver.termination_condition == "optimal":
            dt = DiagnosticsToolbox(m_two_salt)
            dt.assert_no_numerical_warnings()

            for x in m_three_salt.fs.membrane.dimensionless_module_length:
                if x != 0:
                    for z in m_three_salt.fs.membrane.dimensionless_membrane_thickness:
                        lithium_pe_3.append(
                            value(m_three_salt.fs.membrane.peclet_number_lithium[x, z])
                        )
                        cobalt_pe_3.append(
                            value(m_three_salt.fs.membrane.peclet_number_cobalt[x, z])
                        )
                        aluminum_pe_3.append(
                            value(m_three_salt.fs.membrane.peclet_number_aluminum[x, z])
                        )

        ionic_strength = calculate_ionic_strength_three_salt(m_three_salt)
        ionic_strength_list_3.append(ionic_strength)
        li_pe_list_3.append(np.average(lithium_pe_3))
        co_pe_list_3.append(np.average(cobalt_pe_3))
        al_pe_list_3.append(np.average(aluminum_pe_3))

        lithium_pe_3 = []
        cobalt_pe_3 = []
        aluminum_pe_3 = []

    ax1.plot(
        ionic_strength_list_3,
        li_pe_list_3,
        "m^",
        markersize=6,
        linestyle="-",
        linewidth=0.7,
    )
    ax1.plot(
        ionic_strength_list_3,
        co_pe_list_3,
        "c^",
        markersize=6,
        linestyle="-",
        linewidth=0.7,
    )
    ax1.plot(
        ionic_strength_list_3,
        al_pe_list_3,
        "g^",
        markersize=6,
        linestyle="-",
        linewidth=0.7,
    )

    ionic_strength_list_3 = []
    li_pe_list_3 = []
    co_pe_list_3 = []
    al_pe_list_3 = []

    lithium_pe_3 = []
    cobalt_pe_3 = []
    aluminum_pe_3 = []
    # 10:20:5
    conc_list_3 = [
        [38, 72, 18],
        [36, 70, 17.5],
        [34, 68, 17],
        [33, 66, 16.5],
        [32, 64, 16],
        [31, 62, 15.5],
        [30, 60, 15],
        [29, 58, 14.5],
        [28, 56, 14],
        [27, 54, 13.5],
        [26, 52, 13],
        [25, 50, 12.5],
        [24, 48, 12],
    ]
    # initialize
    m_three_salt = build_three_salt_model()
    results = solve_model(m_three_salt)

    for conc in conc_list_3:
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Li"].fix(conc[0])
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Co"].fix(conc[1])
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Al"].fix(conc[2])

        results = solve_model(m_three_salt)
        if results.solver.termination_condition == "optimal":
            dt = DiagnosticsToolbox(m_two_salt)
            dt.assert_no_numerical_warnings()

            for x in m_three_salt.fs.membrane.dimensionless_module_length:
                if x != 0:
                    for z in m_three_salt.fs.membrane.dimensionless_membrane_thickness:
                        lithium_pe_3.append(
                            value(m_three_salt.fs.membrane.peclet_number_lithium[x, z])
                        )
                        cobalt_pe_3.append(
                            value(m_three_salt.fs.membrane.peclet_number_cobalt[x, z])
                        )
                        aluminum_pe_3.append(
                            value(m_three_salt.fs.membrane.peclet_number_aluminum[x, z])
                        )

        ionic_strength = calculate_ionic_strength_three_salt(m_three_salt)
        ionic_strength_list_3.append(ionic_strength)
        li_pe_list_3.append(np.average(lithium_pe_3))
        co_pe_list_3.append(np.average(cobalt_pe_3))
        al_pe_list_3.append(np.average(aluminum_pe_3))

        lithium_pe_3 = []
        cobalt_pe_3 = []
        aluminum_pe_3 = []

    ax1.plot(
        ionic_strength_list_3,
        li_pe_list_3,
        "m*",
        markersize=7,
        linestyle="-",
        linewidth=0.7,
    )
    ax1.plot(
        ionic_strength_list_3,
        co_pe_list_3,
        "c*",
        markersize=7,
        linestyle="-",
        linewidth=0.7,
    )
    ax1.plot(
        ionic_strength_list_3,
        al_pe_list_3,
        "g*",
        markersize=7,
        linestyle="-",
        linewidth=0.7,
    )

    ax1.axhline(1, color="black", linewidth=1)

    # legend points
    # ax1.plot([],[], marker='None', linestyle='None', label="Solution (markerstyle)")
    ax1.plot([], [], "kx", markersize=6, label="10:20:0")
    # ax1.plot([], [], "ko", markersize=6, label="10:20:1")
    ax1.plot([], [], "k^", markersize=6, label="10:20:2.5")
    ax1.plot([], [], "k*", markersize=7, label="10:20:5")
    ax1.plot([], [], marker="None", linestyle="None", label="Solute (color)")
    ax1.plot([], [], "ms", markersize=8, label="Lithium")
    ax1.plot([], [], "cs", markersize=8, label="Cobalt")
    ax1.plot([], [], "gs", markersize=8, label="Aluminum")

    ax1.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Molar Ratio of\nLi:Co:Al (marker)",
    )
    ax1.set_xlabel("Ionic Strength of the Feed (mM)", fontsize=12, fontweight="bold")
    ax1.set_ylabel(
        "Convective:Diffusive\n& Electromigrative Flux", fontsize=12, fontweight="bold"
    )
    ax1.tick_params(direction="in", labelsize=10)

    plt.tight_layout()

    plt.show()


def plot_concentrations(m2, m3):
    """
    Plots permeate versus retentate concentrations for two and three salt models.

    Args:
        m2: two-salt Pyomo model
        m3: three-salt Pyomo model
    """

    # store values for lithium concentration
    retentate_lithium_conc_two_salt = []
    retentate_lithium_conc_three_salt = []
    permeate_lithium_conc_two_salt = []
    permeate_lithium_conc_three_salt = []
    # store values for cobalt concentration
    retentate_cobalt_conc_two_salt = []
    retentate_cobalt_conc_three_salt = []
    permeate_cobalt_conc_two_salt = []
    permeate_cobalt_conc_three_salt = []
    # store values for cobalt concentration
    retentate_aluminum_conc_three_salt = []
    permeate_aluminum_conc_three_salt = []

    for x_val in m2.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            retentate_lithium_conc_two_salt.append(
                value(m2.fs.membrane.retentate_conc_mol_comp[0, x_val, "Li"])
            )
            retentate_lithium_conc_three_salt.append(
                value(m3.fs.membrane.retentate_conc_mol_comp[0, x_val, "Li"])
            )
            permeate_lithium_conc_two_salt.append(
                value(m2.fs.membrane.permeate_conc_mol_comp[0, x_val, "Li"])
            )
            permeate_lithium_conc_three_salt.append(
                value(m3.fs.membrane.permeate_conc_mol_comp[0, x_val, "Li"])
            )
            retentate_cobalt_conc_two_salt.append(
                value(m2.fs.membrane.retentate_conc_mol_comp[0, x_val, "Co"])
            )
            retentate_cobalt_conc_three_salt.append(
                value(m3.fs.membrane.retentate_conc_mol_comp[0, x_val, "Co"])
            )
            permeate_cobalt_conc_two_salt.append(
                value(m2.fs.membrane.permeate_conc_mol_comp[0, x_val, "Co"])
            )
            permeate_cobalt_conc_three_salt.append(
                value(m3.fs.membrane.permeate_conc_mol_comp[0, x_val, "Co"])
            )
            retentate_aluminum_conc_three_salt.append(
                value(m3.fs.membrane.retentate_conc_mol_comp[0, x_val, "Al"])
            )
            permeate_aluminum_conc_three_salt.append(
                value(m3.fs.membrane.permeate_conc_mol_comp[0, x_val, "Al"])
            )

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=100, figsize=(15, 5))

    # fig1, ax2 = plt.subplots(1, 1, dpi=100, figsize=(5, 4))

    ax1.plot(
        retentate_lithium_conc_two_salt,
        permeate_lithium_conc_two_salt,
        "m-",
        linewidth=2,
    )  # , label="Lithium (Li-Co)")
    ax2.plot(
        retentate_cobalt_conc_two_salt, permeate_cobalt_conc_two_salt, "c-", linewidth=2
    )  # , label="Cobalt (Li-Co)")
    ax1.plot(
        retentate_lithium_conc_three_salt,
        permeate_lithium_conc_three_salt,
        "m--",
        linewidth=2,
    )  # , label="Lithium (Li-Co-Al)")
    ax2.plot(
        retentate_cobalt_conc_three_salt,
        permeate_cobalt_conc_three_salt,
        "c--",
        linewidth=2,
    )  # , label="Cobalt (Li-Co-Al)")
    ax3.plot(
        retentate_aluminum_conc_three_salt,
        permeate_aluminum_conc_three_salt,
        "g--",
        linewidth=2,
    )  # , label="Aluminum (Li-Co-Al)")

    lith_min = 149.5
    lith_max = 151
    ax1.plot([lith_min, lith_max], [lith_min, lith_max], "k-", linewidth=0.5)
    ax1.set_xlim(lith_min, lith_max)
    ax1.set_ylim(lith_min, lith_max)
    cob_min = 297
    cob_max = 302
    ax2.plot([cob_min, cob_max], [cob_min, cob_max], "k-", linewidth=0.5)
    ax2.set_xlim(cob_min, cob_max)
    ax2.set_ylim(cob_min, cob_max)
    al_min = 48
    al_max = 52
    ax3.plot([al_min, al_max], [al_min, al_max], "k-", linewidth=0.5)
    ax3.set_xlim(al_min, al_max)
    ax3.set_ylim(al_min, al_max)

    # legend points
    ax1.plot([], [], "m-", linewidth=2, label="Lithium (in Li-Co)")
    ax1.plot([], [], "m--", linewidth=2, label="Lithium (in Li-Co-Al)")
    ax2.plot([], [], "c-", linewidth=2, label="Cobalt (in Li-Co)")
    ax2.plot([], [], "c--", linewidth=2, label="Cobalt (in Li-Co-Al)")
    ax3.plot([], [], "g--", linewidth=2, label="Aluminum (in Li-Co-Al)")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Retentate Concentration (mM)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Permeate Concentration (mM)", fontsize=12, fontweight="bold")
        ax.tick_params(direction="in", labelsize=10)
        ax.legend(loc="upper left")

    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
