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
    Constraint,
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
    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5, "diafiltrate": 3.75}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    # single salt systems
    # lithium chloride
    m_li_cl = build_model(
        cation_list=["Li"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"Li": 245, "Cl": 245},
            "diafiltrate": {"Li": 14, "Cl": 14},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # cobalt chloride
    m_co_cl = build_model(
        cation_list=["Co"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"Co": 288, "Cl": 576},
            "diafiltrate": {"Co": 3, "Cl": 6},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # aluminum chloride
    m_al_cl = build_model(
        cation_list=["Al"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"Al": 20, "Cl": 60},
            "diafiltrate": {"Al": 3, "Cl": 9},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # two salt systems
    # lithium chloride + cobalt chloride
    m_li_co_cl = build_model(
        cation_list=["Li", "Co"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"Li": 245, "Co": 288, "Cl": 821},
            "diafiltrate": {"Li": 14, "Co": 3, "Cl": 20},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # lithium chloride + aluminum chloride
    m_li_al_cl = build_model(
        cation_list=["Li", "Al"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"Li": 245, "Al": 20, "Cl": 305},
            "diafiltrate": {"Li": 14, "Al": 3, "Cl": 23},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # cobalt chloride + aluminum chloride
    m_co_al_cl = build_model(
        cation_list=["Co", "Al"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"Co": 288, "Al": 20, "Cl": 636},
            "diafiltrate": {"Co": 3, "Al": 3, "Cl": 15},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # lithium chloride + cobalt chloride + aluminum chloride
    m_li_co_al_cl = build_model(
        cation_list=["Li", "Co", "Al"],
        anion_list=anion_list,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration={
            "feed": {"Li": 245, "Co": 288, "Al": 20, "Cl": 881},
            "diafiltrate": {"Li": 14, "Co": 3, "Al": 3, "Cl": 29},
        },
        include_boundary_layer=include_boundary_layer,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # solve models
    model_list = [
        m_li_cl,
        m_co_cl,
        m_al_cl,
        m_li_co_cl,
        m_li_al_cl,
        m_co_al_cl,
        m_li_co_al_cl,
    ]
    for model in model_list:
        solve_model(model)
        unfix_pressure(model)
        solve_model(model)

    # m_li_cl.fs.membrane.applied_pressure.display()
    # m_co_cl.fs.membrane.applied_pressure.display()
    # m_al_cl.fs.membrane.applied_pressure.display()
    # m_li_co_cl.fs.membrane.applied_pressure.display()
    # m_li_al_cl.fs.membrane.applied_pressure.display()
    # m_co_al_cl.fs.membrane.applied_pressure.display()
    # m_li_co_al_cl.fs.membrane.applied_pressure.display()

    # store results
    m_li_cl_results_dict = extract_and_store_results(m_li_cl)
    m_co_cl_results_dict = extract_and_store_results(m_co_cl)
    m_al_cl_results_dict = extract_and_store_results(m_al_cl)
    m_li_co_cl_results_dict = extract_and_store_results(m_li_co_cl)
    m_li_al_cl_results_dict = extract_and_store_results(m_li_al_cl)
    m_co_al_cl_results_dict = extract_and_store_results(m_co_al_cl)
    m_li_co_al_cl_results_dict = extract_and_store_results(m_li_co_al_cl)

    dict_list = [
        m_li_cl_results_dict,
        m_co_cl_results_dict,
        m_al_cl_results_dict,
        m_li_co_cl_results_dict,
        m_li_al_cl_results_dict,
        m_co_al_cl_results_dict,
    ]

    # plot individual results
    plot_individual = False
    if plot_individual:
        for results_dict in dict_list:
            plot_results_by_length(results_dict)
            plot_results_by_thickness(results_dict)

    # plot overall results
    plot_overall = True
    if plot_overall:
        # plot_rejection_versus_area(
        #     m_li_cl_results_dict,
        #     m_co_cl_results_dict,
        #     m_al_cl_results_dict,
        #     m_li_co_cl_results_dict,
        #     m_li_al_cl_results_dict,
        #     m_co_al_cl_results_dict,
        #     compact=True,
        # )
        # plot_rejection_versus_area(
        #     m_li_cl_results_dict,
        #     m_co_cl_results_dict,
        #     m_al_cl_results_dict,
        #     m_li_co_cl_results_dict,
        #     m_li_al_cl_results_dict,
        #     m_co_al_cl_results_dict,
        #     compact=False,
        # )
        # plot_rejection_versus_concentration(
        #     m_li_cl_results_dict,
        #     m_co_cl_results_dict,
        #     m_al_cl_results_dict,
        #     m_li_co_cl_results_dict,
        #     m_li_al_cl_results_dict,
        #     m_co_al_cl_results_dict,
        #     x_axis_conc="bulk",
        # )
        # # plot_rejection_versus_concentration(
        # #     m_li_cl_results_dict,
        # #     m_co_cl_results_dict,
        # #     m_al_cl_results_dict,
        # #     m_li_co_cl_results_dict,
        # #     m_li_al_cl_results_dict,
        # #     m_co_al_cl_results_dict,
        # #     x_axis_conc="interface",
        # # )
        # # plot_rejection_versus_concentration(
        # #     m_li_cl_results_dict,
        # #     m_co_cl_results_dict,
        # #     m_al_cl_results_dict,
        # #     m_li_co_cl_results_dict,
        # #     m_li_al_cl_results_dict,
        # #     m_co_al_cl_results_dict,
        # #     x_axis_conc="bulk-ionic-strength",
        # # )
        plot_rejection_versus_feed_ionic_strength(
            m_li_cl_results_dict,
            m_co_cl_results_dict,
            m_al_cl_results_dict,
            m_li_co_cl_results_dict,
            m_li_al_cl_results_dict,
            m_co_al_cl_results_dict,
            m_li_co_al_cl_results_dict,
        )
        # plot_flux_versus_length(
        #     m_li_cl_results_dict,
        #     # m_co_cl_results_dict,
        #     # m_al_cl_results_dict,
        #     m_li_co_cl_results_dict,
        #     m_li_al_cl_results_dict,
        #     # m_co_al_cl_results_dict,
        # )
        # plot_electric_potential_gradient(
        #     m_li_cl_results_dict,
        #     # m_co_cl_results_dict,
        #     # m_al_cl_results_dict,
        #     m_li_co_cl_results_dict,
        #     m_li_al_cl_results_dict,
        #     # m_co_al_cl_results_dict,
        # )

    plt.show()


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
    if len(cation_list) == 1:
        m.fs.membrane.applied_pressure.fix(5)
    else:
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


def unfix_pressure(m):
    m.fs.membrane.applied_pressure.unfix()

    def _water_flux_constraint(m):
        return m.fs.membrane.volume_flux_water[0, 0.1] == 0.08

    m.water_flux_constraint = Constraint(rule=_water_flux_constraint)


def extract_and_store_results(m):
    """
    Extracts relevant results and stores in dictionary

    Args:
        m: Pyomo model
    """
    # store values for x-coordinate (module length)
    x_axis_values = []
    x_axis_values_dimensionless = []
    membrane_area_values = []

    # store values for z-coordinate (boundary layer)
    z_boundary_layer_values = []

    # store values for z-coordinate (membrane)
    z_membrane_values = []

    # store values for concentration in the retentate
    conc_ret_anion = []
    conc_ret_cation_1 = []
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_ret_cation_2 = []
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_ret_cation_3 = []

    # store values for concentration at solution-membrane interface
    conc_int_cation_1 = []
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_int_cation_2 = []
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_int_cation_3 = []

    # store values for concentration in the permeate
    conc_perm_cation_1 = []
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_perm_cation_2 = []
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_perm_cation_3 = []

    # store values for concentration in the boundary layer (2D)
    conc_bl_cation_1_by_z = []
    conc_bl_cation_1_dict_by_z = {}
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_bl_cation_2_by_z = []
        conc_bl_cation_2_dict_by_z = {}
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_bl_cation_3_by_z = []
        conc_bl_cation_3_dict_by_z = {}

    conc_bl_cation_1_by_x = []
    conc_bl_cation_1_dict_by_x = {}
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_bl_cation_2_by_x = []
        conc_bl_cation_2_dict_by_x = {}
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_bl_cation_3_by_x = []
        conc_bl_cation_3_dict_by_x = {}

    # store values for concentration gradient in the boundary layer (2D)
    conc_grad_bl_cation_1_by_x = []
    conc_grad_bl_cation_1_dict_by_x = {}
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_grad_bl_cation_2_by_x = []
        conc_grad_bl_cation_2_dict_by_x = {}
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_grad_bl_cation_3_by_x = []
        conc_grad_bl_cation_3_dict_by_x = {}

    # store values for concentration in the membrane (2D)
    conc_mem_cation_1_by_z = []
    conc_mem_cation_1_dict_by_z = {}
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_mem_cation_2_by_z = []
        conc_mem_cation_2_dict_by_z = {}
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_mem_cation_3_by_z = []
        conc_mem_cation_3_dict_by_z = {}

    conc_mem_cation_1_by_x = []
    conc_mem_cation_1_dict_by_x = {}
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_mem_cation_2_by_x = []
        conc_mem_cation_2_dict_by_x = {}
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_mem_cation_3_by_x = []
        conc_mem_cation_3_dict_by_x = {}

    # store values for concentration gradient in the membrane (2D)
    conc_grad_mem_cation_1_by_x = []
    conc_grad_mem_cation_1_dict_by_x = {}
    if len(m.fs.membrane.config.cation_list) > 1:
        conc_grad_mem_cation_2_by_x = []
        conc_grad_mem_cation_2_dict_by_x = {}
    if len(m.fs.membrane.config.cation_list) > 2:
        conc_grad_mem_cation_3_by_x = []
        conc_grad_mem_cation_3_dict_by_x = {}

    # store values for water flux across membrane
    water_flux = []

    # store values for mol flux across membrane
    cation_1_flux = []
    if len(m.fs.membrane.config.cation_list) > 1:
        cation_2_flux = []
    if len(m.fs.membrane.config.cation_list) > 2:
        cation_3_flux = []

    # store values for percent recovery
    percent_recovery = []

    # store values for rejection
    cation_1_rejection_observed = []
    cation_1_rejection_actual = []
    if len(m.fs.membrane.config.cation_list) > 1:
        cation_2_rejection_observed = []
        cation_2_rejection_actual = []
    if len(m.fs.membrane.config.cation_list) > 2:
        cation_3_rejection_observed = []
        cation_3_rejection_actual = []

    for x_val in m.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            # x-coordinate
            x_axis_values.append(x_val * value(m.fs.membrane.total_module_length))
            x_axis_values_dimensionless.append(x_val)
            membrane_area_values.append(
                x_val
                * value(m.fs.membrane.total_module_length)
                * value(m.fs.membrane.total_membrane_length)
            )

            # concentrations
            conc_ret_anion_val = value(
                m.fs.membrane.retentate_conc_mol_comp[
                    0, x_val, m.fs.membrane.config.anion_list[0]
                ]
            )
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

            conc_ret_anion.append(conc_ret_anion_val)
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

            if len(m.fs.membrane.config.cation_list) > 2:
                conc_ret_cation_3_val = value(
                    m.fs.membrane.retentate_conc_mol_comp[
                        0, x_val, m.fs.membrane.config.cation_list[2]
                    ]
                )
                conc_int_cation_3_val = value(
                    m.fs.membrane.boundary_layer_conc_mol_comp[
                        0, x_val, 1, m.fs.membrane.config.cation_list[2]
                    ]
                )
                conc_perm_cation_3_val = value(
                    m.fs.membrane.permeate_conc_mol_comp[
                        0, x_val, m.fs.membrane.config.cation_list[2]
                    ]
                )

                conc_ret_cation_3.append(conc_ret_cation_3_val)
                conc_int_cation_3.append(conc_int_cation_3_val)
                conc_perm_cation_3.append(conc_perm_cation_3_val)

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
            if len(m.fs.membrane.config.cation_list) > 2:
                cation_3_flux.append(
                    value(
                        m.fs.membrane.molar_ion_flux[
                            0, x_val, m.fs.membrane.config.cation_list[2]
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
            if len(m.fs.membrane.config.cation_list) > 2:
                cation_3_rejection_observed.append(
                    (1 - (conc_perm_cation_3_val / conc_ret_cation_3_val)) * 100
                )
                cation_3_rejection_actual.append(
                    (1 - (conc_perm_cation_3_val / conc_int_cation_3_val)) * 100
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
                conc_bl_cation_1_val_by_z = value(
                    m.fs.membrane.boundary_layer_conc_mol_comp[
                        0, x_val, z_val, m.fs.membrane.config.cation_list[0]
                    ]
                )

                conc_bl_cation_1_by_z.append(conc_bl_cation_1_val_by_z)

                if len(m.fs.membrane.config.cation_list) > 1:
                    conc_bl_cation_2_val_by_z = value(
                        m.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )

                    conc_bl_cation_2_by_z.append(conc_bl_cation_2_val_by_z)
                if len(m.fs.membrane.config.cation_list) > 2:
                    conc_bl_cation_3_val_by_z = value(
                        m.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[2]
                        ]
                    )

                    conc_bl_cation_3_by_z.append(conc_bl_cation_3_val_by_z)

        conc_bl_cation_1_dict_by_z[f"{z_val}"] = conc_bl_cation_1_by_z
        conc_bl_cation_1_by_z = []
        if len(m.fs.membrane.config.cation_list) > 1:
            conc_bl_cation_2_dict_by_z[f"{z_val}"] = conc_bl_cation_2_by_z
            conc_bl_cation_2_by_z = []
        if len(m.fs.membrane.config.cation_list) > 2:
            conc_bl_cation_3_dict_by_z[f"{z_val}"] = conc_bl_cation_3_by_z
            conc_bl_cation_3_by_z = []

    for x_val in m.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            for z_val in m.fs.membrane.dimensionless_boundary_layer_thickness:
                conc_bl_cation_1_val_by_x = value(
                    m.fs.membrane.boundary_layer_conc_mol_comp[
                        0, x_val, z_val, m.fs.membrane.config.cation_list[0]
                    ]
                )
                conc_grad_bl_cation_1_val_by_x = value(
                    m.fs.membrane.d_boundary_layer_conc_mol_comp_dz[
                        0, x_val, z_val, m.fs.membrane.config.cation_list[0]
                    ]
                )

                conc_bl_cation_1_by_x.append(conc_bl_cation_1_val_by_x)
                conc_grad_bl_cation_1_by_x.append(conc_grad_bl_cation_1_val_by_x)

                if len(m.fs.membrane.config.cation_list) > 1:
                    conc_bl_cation_2_val_by_x = value(
                        m.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )
                    conc_grad_bl_cation_2_val_by_x = value(
                        m.fs.membrane.d_boundary_layer_conc_mol_comp_dz[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )

                    conc_bl_cation_2_by_x.append(conc_bl_cation_2_val_by_x)
                    conc_grad_bl_cation_2_by_x.append(conc_grad_bl_cation_2_val_by_x)

                if len(m.fs.membrane.config.cation_list) > 2:
                    conc_bl_cation_3_val_by_x = value(
                        m.fs.membrane.boundary_layer_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[2]
                        ]
                    )
                    conc_grad_bl_cation_3_val_by_x = value(
                        m.fs.membrane.d_boundary_layer_conc_mol_comp_dz[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[2]
                        ]
                    )

                    conc_bl_cation_3_by_x.append(conc_bl_cation_3_val_by_x)
                    conc_grad_bl_cation_3_by_x.append(conc_grad_bl_cation_3_val_by_x)

            conc_bl_cation_1_dict_by_x[f"{x_val}"] = conc_bl_cation_1_by_x
            conc_grad_bl_cation_1_dict_by_x[f"{x_val}"] = conc_grad_bl_cation_1_by_x
            conc_bl_cation_1_by_x = []
            conc_grad_bl_cation_1_by_x = []
            if len(m.fs.membrane.config.cation_list) > 1:
                conc_bl_cation_2_dict_by_x[f"{x_val}"] = conc_bl_cation_2_by_x
                conc_grad_bl_cation_2_dict_by_x[f"{x_val}"] = conc_grad_bl_cation_2_by_x
                conc_bl_cation_2_by_x = []
                conc_grad_bl_cation_2_by_x = []
            if len(m.fs.membrane.config.cation_list) > 2:
                conc_bl_cation_3_dict_by_x[f"{x_val}"] = conc_bl_cation_3_by_x
                conc_grad_bl_cation_3_dict_by_x[f"{x_val}"] = conc_grad_bl_cation_3_by_x
                conc_bl_cation_3_by_x = []
                conc_grad_bl_cation_3_by_x = []

    # membrane
    for z_val in m.fs.membrane.dimensionless_membrane_thickness:
        z_membrane_values.append(
            z_val * value(m.fs.membrane.total_membrane_thickness) * 1e9
        )
        for x_val in m.fs.membrane.dimensionless_module_length:
            if x_val != 0:
                conc_mem_cation_1_val_by_z = value(
                    m.fs.membrane.membrane_conc_mol_comp[
                        0, x_val, z_val, m.fs.membrane.config.cation_list[0]
                    ]
                )

                conc_mem_cation_1_by_z.append(conc_mem_cation_1_val_by_z)

                if len(m.fs.membrane.config.cation_list) > 1:
                    conc_mem_cation_2_val_by_z = value(
                        m.fs.membrane.membrane_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )

                    conc_mem_cation_2_by_z.append(conc_mem_cation_2_val_by_z)
                if len(m.fs.membrane.config.cation_list) > 2:
                    conc_mem_cation_3_val_by_z = value(
                        m.fs.membrane.membrane_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[2]
                        ]
                    )

                    conc_mem_cation_3_by_z.append(conc_mem_cation_3_val_by_z)

        conc_mem_cation_1_dict_by_z[f"{z_val}"] = conc_mem_cation_1_by_z
        conc_mem_cation_1_by_z = []
        if len(m.fs.membrane.config.cation_list) > 1:
            conc_mem_cation_2_dict_by_z[f"{z_val}"] = conc_mem_cation_2_by_z
            conc_mem_cation_2_by_z = []
        if len(m.fs.membrane.config.cation_list) > 2:
            conc_mem_cation_3_dict_by_z[f"{z_val}"] = conc_mem_cation_3_by_z
            conc_mem_cation_3_by_z = []

    for x_val in m.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            for z_val in m.fs.membrane.dimensionless_membrane_thickness:
                conc_mem_cation_1_val_by_x = value(
                    m.fs.membrane.membrane_conc_mol_comp[
                        0, x_val, z_val, m.fs.membrane.config.cation_list[0]
                    ]
                )
                conc_grad_mem_cation_1_val_by_x = value(
                    m.fs.membrane.d_membrane_conc_mol_comp_dz[
                        0, x_val, z_val, m.fs.membrane.config.cation_list[0]
                    ]
                )

                conc_mem_cation_1_by_x.append(conc_mem_cation_1_val_by_x)
                conc_grad_mem_cation_1_by_x.append(conc_grad_mem_cation_1_val_by_x)

                if len(m.fs.membrane.config.cation_list) > 1:
                    conc_mem_cation_2_val_by_x = value(
                        m.fs.membrane.membrane_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )
                    conc_grad_mem_cation_2_val_by_x = value(
                        m.fs.membrane.d_membrane_conc_mol_comp_dz[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[1]
                        ]
                    )

                    conc_mem_cation_2_by_x.append(conc_mem_cation_2_val_by_x)
                    conc_grad_mem_cation_2_by_x.append(conc_grad_mem_cation_2_val_by_x)

                if len(m.fs.membrane.config.cation_list) > 2:
                    conc_mem_cation_3_val_by_x = value(
                        m.fs.membrane.membrane_conc_mol_comp[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[2]
                        ]
                    )
                    conc_grad_mem_cation_3_val_by_x = value(
                        m.fs.membrane.d_membrane_conc_mol_comp_dz[
                            0, x_val, z_val, m.fs.membrane.config.cation_list[2]
                        ]
                    )

                    conc_mem_cation_3_by_x.append(conc_mem_cation_3_val_by_x)
                    conc_grad_mem_cation_3_by_x.append(conc_grad_mem_cation_3_val_by_x)

            conc_mem_cation_1_dict_by_x[f"{x_val}"] = conc_mem_cation_1_by_x
            conc_grad_mem_cation_1_dict_by_x[f"{x_val}"] = conc_grad_mem_cation_1_by_x
            conc_mem_cation_1_by_x = []
            conc_grad_mem_cation_1_by_x = []
            if len(m.fs.membrane.config.cation_list) > 1:
                conc_mem_cation_2_dict_by_x[f"{x_val}"] = conc_mem_cation_2_by_x
                conc_grad_mem_cation_2_dict_by_x[f"{x_val}"] = (
                    conc_grad_mem_cation_2_by_x
                )
                conc_mem_cation_2_by_x = []
                conc_grad_mem_cation_2_by_x = []
            if len(m.fs.membrane.config.cation_list) > 2:
                conc_mem_cation_3_dict_by_x[f"{x_val}"] = conc_mem_cation_3_by_x
                conc_grad_mem_cation_3_dict_by_x[f"{x_val}"] = (
                    conc_grad_mem_cation_3_by_x
                )
                conc_mem_cation_3_by_x = []
                conc_grad_mem_cation_3_by_x = []

    results_dict = {
        "cation_list": m.fs.membrane.config.cation_list,
        "cation_1": m.fs.membrane.config.cation_list[0],
        "feed_ionic_strength": value(m.fs.membrane.feed_ionic_strength[0]),
        "x_values": x_axis_values,
        "x_values_dimensionless": x_axis_values_dimensionless,
        "membrane_area_values": membrane_area_values,
        "z_bl_values": z_boundary_layer_values,
        "z_mem_values": z_membrane_values,
        "anion_retentate_concentration": conc_ret_anion,
        "cation_1_retentate_concentration": conc_ret_cation_1,
        "cation_1_interface_concentration": conc_int_cation_1,
        "cation_1_permeate_concentration": conc_perm_cation_1,
        "cation_1_boundary_layer_concentration_by_z": conc_bl_cation_1_dict_by_z,
        "cation_1_boundary_layer_concentration_by_x": conc_bl_cation_1_dict_by_x,
        "cation_1_boundary_layer_concentration_gradient_by_x": conc_grad_bl_cation_1_dict_by_x,
        "cation_1_membrane_concentration_by_z": conc_mem_cation_1_dict_by_z,
        "cation_1_membrane_concentration_by_x": conc_mem_cation_1_dict_by_x,
        "cation_1_membrane_concentration_gradient_by_x": conc_grad_mem_cation_1_dict_by_x,
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
                "cation_2_boundary_layer_concentration_by_z": conc_bl_cation_2_dict_by_z,
                "cation_2_boundary_layer_concentration_by_x": conc_bl_cation_2_dict_by_x,
                "cation_2_boundary_layer_concentration_gradient_by_x": conc_grad_bl_cation_2_dict_by_x,
                "cation_2_membrane_concentration_by_z": conc_mem_cation_2_dict_by_z,
                "cation_2_membrane_concentration_by_x": conc_mem_cation_2_dict_by_x,
                "cation_2_membrane_concentration_gradient_by_x": conc_grad_mem_cation_2_dict_by_x,
                "cation_2_flux": cation_2_flux,
                "cation_2_rejection_observed": cation_2_rejection_observed,
                "cation_2_rejection_actual": cation_2_rejection_actual,
            }
        )
    if len(m.fs.membrane.config.cation_list) > 2:
        results_dict.update(
            {
                "cation_3": m.fs.membrane.config.cation_list[2],
                "cation_3_retentate_concentration": conc_ret_cation_3,
                "cation_3_interface_concentration": conc_int_cation_3,
                "cation_3_permeate_concentration": conc_perm_cation_3,
                "cation_3_boundary_layer_concentration_by_z": conc_bl_cation_3_dict_by_z,
                "cation_3_boundary_layer_concentration_by_x": conc_bl_cation_3_dict_by_x,
                "cation_3_boundary_layer_concentration_gradient_by_x": conc_grad_bl_cation_3_dict_by_x,
                "cation_3_membrane_concentration_by_z": conc_mem_cation_3_dict_by_z,
                "cation_3_membrane_concentration_by_x": conc_mem_cation_3_dict_by_x,
                "cation_3_membrane_concentration_gradient_by_x": conc_grad_mem_cation_3_dict_by_x,
                "cation_3_flux": cation_3_flux,
                "cation_3_rejection_observed": cation_3_rejection_observed,
                "cation_3_rejection_actual": cation_3_rejection_actual,
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
        fontsize=14,
        fontweight="bold",
    )
    ax1.tick_params(direction="in", labelsize=14)
    ax1.legend()

    if len(cation_list) > 1:
        ax2.plot(x_axis_values, conc_ret_cation_2, linewidth=2, label="retentate")
        ax2.plot(x_axis_values, conc_int_cation_2, linewidth=2, label="interface")
        ax2.plot(x_axis_values, conc_perm_cation_2, linewidth=2, label="permeate")
        ax2.set_ylabel(
            f"{cation_2.capitalize()} Concentration \n(mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.tick_params(direction="in", labelsize=14)
        ax2.legend()

    ax3.plot(x_axis_values, water_flux, linewidth=2)
    ax3.set_xlabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Water Flux (m$^3$/m$^2$/h)", fontsize=14, fontweight="bold")
    ax3.tick_params(direction="in", labelsize=14)

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
    ax4.set_xlabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax4.set_ylabel("Molar Flux (mol/m$^2$/h)", fontsize=14, fontweight="bold")
    ax4.tick_params(direction="in", labelsize=14)
    ax4.legend()

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
    ax5.set_xlabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax5.set_ylabel("Solute Rejection (%)", fontsize=14, fontweight="bold")
    ax5.tick_params(direction="in", labelsize=14)
    ax5.legend()

    ax6.plot(x_axis_values, percent_recovery, linewidth=2)
    ax6.set_xlabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax6.set_ylabel("Percent Recovery (%)", fontsize=14, fontweight="bold")
    ax6.tick_params(direction="in", labelsize=14)

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
    conc_bl_cation_1_dict = results_dict["cation_1_boundary_layer_concentration_by_z"]
    conc_mem_cation_1_dict = results_dict["cation_1_membrane_concentration_by_z"]
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
    ax1.set_xlabel("Boundary Layer Thickness (um)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax1.set_title(
        f"{cation_1.capitalize()} Concentration\n in Boundary Layer (mol/m$^3$)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.tick_params(direction="in", labelsize=14)
    fig1.colorbar(cation_1_plot_bl, ax=ax1)
    if len(cation_list) > 1:
        cation_2_plot_bl = ax2.pcolor(
            z_bl_axis_values, x_axis_values, conc_bl_cation_2_df
        )
        ax2.set_xlabel("Boundary Layer Thickness (um)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Module Length (m)", fontsize=14, fontweight="bold")
        ax2.set_title(
            f"{cation_2.capitalize()} Concentration\n in Boundary Layer (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.tick_params(direction="in", labelsize=14)
        fig1.colorbar(cation_2_plot_bl, ax=ax2)

    fig2, (ax3, ax4) = plt.subplots(1, 2, dpi=125, figsize=(15, 7))
    cation_1_plot_mem = ax3.pcolor(
        z_mem_axis_values, x_axis_values, conc_mem_cation_1_df
    )
    ax3.set_xlabel("Membrane Thickness (nm)", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax3.set_title(
        f"{cation_1.capitalize()} Concentration\n in Membrane (mol/m$^3$)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.tick_params(direction="in", labelsize=14)
    fig2.colorbar(cation_1_plot_mem, ax=ax3)

    if len(cation_list) > 1:
        cation_2_plot_mem = ax4.pcolor(
            z_mem_axis_values, x_axis_values, conc_mem_cation_2_df
        )
        ax4.set_xlabel("Membrane Thickness (nm)", fontsize=14, fontweight="bold")
        ax4.set_title(
            f"{cation_2.capitalize()} Concentration\n in Membrane (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
        ax4.tick_params(direction="in", labelsize=14)
        fig2.colorbar(cation_2_plot_mem, ax=ax4)


def plot_rejection_versus_area(
    m_li_cl_results_dict,
    m_co_cl_results_dict,
    m_al_cl_results_dict,
    m_li_co_cl_results_dict,
    m_li_al_cl_results_dict,
    m_co_al_cl_results_dict,
    # m_li_co_al_cl_results_dict,
    compact=False,
):
    """
    Plots relative solute rejection across the length of the membrane module.
    Rejections normalized to initial rejection (x=0).
    Compares models.
    """
    membrane_area_values = m_li_cl_results_dict["membrane_area_values"]

    # lithium rejections
    observed_lithium_rejection_li = m_li_cl_results_dict["cation_1_rejection_observed"]
    actual_lithium_rejection_li = m_li_cl_results_dict["cation_1_rejection_actual"]
    observed_lithium_rejection_li_co = m_li_co_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_lithium_rejection_li_co = m_li_co_cl_results_dict[
        "cation_1_rejection_actual"
    ]
    observed_lithium_rejection_li_al = m_li_al_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_lithium_rejection_li_al = m_li_al_cl_results_dict[
        "cation_1_rejection_actual"
    ]
    # observed_lithium_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_1_rejection_observed"
    # ]
    # actual_lithium_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_1_rejection_actual"
    # ]

    # cobalt rejections
    observed_cobalt_rejection_co = m_co_cl_results_dict["cation_1_rejection_observed"]
    actual_cobalt_rejection_co = m_co_cl_results_dict["cation_1_rejection_actual"]
    observed_cobalt_rejection_li_co = m_li_co_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_cobalt_rejection_li_co = m_li_co_cl_results_dict["cation_2_rejection_actual"]
    observed_cobalt_rejection_co_al = m_co_al_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_cobalt_rejection_co_al = m_co_al_cl_results_dict["cation_1_rejection_actual"]
    # observed_cobalt_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_2_rejection_observed"
    # ]
    # actual_cobalt_rejection_li_co_al = m_li_co_al_cl_results_dict["cation_2_rejection_actual"]

    # aluminum rejections
    observed_aluminum_rejection_al = m_al_cl_results_dict["cation_1_rejection_observed"]
    actual_aluminum_rejection_al = m_al_cl_results_dict["cation_1_rejection_actual"]
    observed_aluminum_rejection_li_al = m_li_al_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_aluminum_rejection_li_al = m_li_al_cl_results_dict[
        "cation_2_rejection_actual"
    ]
    observed_aluminum_rejection_co_al = m_co_al_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_aluminum_rejection_co_al = m_co_al_cl_results_dict[
        "cation_2_rejection_actual"
    ]
    # observed_aluminum_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_3_rejection_observed"
    # ]
    # actual_aluminum_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_3_rejection_actual"
    # ]

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
    # observed_lithium_rejection_li_co_al_norm = [
    #     (i - observed_lithium_rejection_li_co_al[0])
    #     / abs(observed_lithium_rejection_li_co_al[0])
    #     * 100
    #     for i in observed_lithium_rejection_li_co_al
    # ]
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
    # actual_lithium_rejection_li_co_al_norm = [
    #     (i - actual_lithium_rejection_li_co_al[0])
    #     / abs(actual_lithium_rejection_li_co_al[0])
    #     * 100
    #     for i in actual_lithium_rejection_li_co_al
    # ]

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
    # observed_cobalt_rejection_li_co_al_norm = [
    #     (i - observed_cobalt_rejection_li_co_al[0])
    #     / abs(observed_cobalt_rejection_li_co_al[0])
    #     * 100
    #     for i in observed_cobalt_rejection_li_co_al
    # ]
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
    # actual_cobalt_rejection_li_o_al_norm = [
    #     (i - actual_cobalt_rejection_li_co_al[0])
    #     / abs(actual_cobalt_rejection_li_co_al[0])
    #     * 100
    #     for i in actual_cobalt_rejection_li_co_al
    # ]

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
    # observed_aluminum_rejection_li_co_al_norm = [
    #     (i - observed_aluminum_rejection_li_co_al[0])
    #     / abs(observed_aluminum_rejection_li_co_al[0])
    #     * 100
    #     for i in observed_aluminum_rejection_li_co_al
    # ]
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
    # actual_aluminum_rejection_li_co_al_norm = [
    #     (i - actual_aluminum_rejection_li_co_al[0])
    #     / abs(actual_aluminum_rejection_li_co_al[0])
    #     * 100
    #     for i in actual_aluminum_rejection_li_co_al
    # ]

    fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))
    if not compact:
        fig1.suptitle("Lithium Rejection")
        fig2, (ax3, ax4) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))
        fig2.suptitle("Cobalt Rejection")
        fig3, (ax5, ax6) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))
        fig3.suptitle("Aluminum Rejection")

        axis_cobalt = ax3
        axis_cobalt_norm = ax4
        axis_aluminum = ax5
        axis_aluminum_norm = ax6

        # legend points
        ax2.plot([], [], "k-", linewidth=2, label="LiCl")
        ax2.plot([], [], "k--", linewidth=2, label="LiCl + CoCl$_2$")
        ax2.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl$_3$")
        ax2.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl$_2$ + AlCl$_3$")
        ax2.plot(
            [], [], marker="None", linestyle="None", label="Rejection (transparency)"
        )
        ax2.plot([], [], "rs", alpha=0.25, markersize=8, label="Observed")
        ax2.plot([], [], "rs", markersize=8, label="Actual")

        ax4.plot([], [], "k-", linewidth=2, label="CoCl$_2$")
        ax4.plot([], [], "k--", linewidth=2, label="LiCl + CoCl$_2$")
        ax4.plot([], [], "k:", linewidth=2, label="CoCl$_2$ + AlCl$_3$")
        ax4.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl$_2$ + AlCl$_3$")
        ax4.plot(
            [], [], marker="None", linestyle="None", label="Rejection (transparency)"
        )
        ax4.plot([], [], "bs", alpha=0.25, markersize=8, label="Observed")
        ax4.plot([], [], "bs", markersize=8, label="Actual")

        ax6.plot([], [], "k-", linewidth=2, label="AlCl$_3$")
        ax6.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl$_3$")
        ax6.plot([], [], "k:", linewidth=2, label="CoCl$_2$ + AlCl$_3$")
        ax6.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl$_2$ + AlCl$_3$")
        ax6.plot(
            [], [], marker="None", linestyle="None", label="Rejection (transparency)"
        )
        ax6.plot([], [], "gs", alpha=0.25, markersize=8, label="Observed")
        ax6.plot([], [], "gs", markersize=8, label="Actual")

        for ax in [ax1, ax3, ax5]:
            ax.set_xlabel("Membrane Area (m$^2$)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Solute Rejection (%)", fontsize=14, fontweight="bold")
            ax.tick_params(direction="in", labelsize=14)
            ax.set_xlim(membrane_area_values[0], membrane_area_values[-1])

        for ax in [ax2, ax4, ax6]:
            ax.set_xlabel("Membrane Area (m$^2$)", fontsize=14, fontweight="bold")
            ax.set_ylabel(
                "Percent Change in Solute Rejection (%)", fontsize=14, fontweight="bold"
            )
            ax.tick_params(direction="in", top=True, right=True, labelsize=14)
            ax.legend(
                loc="best", title="Solution (linestyle)"
            )  # , bbox_to_anchor=(0.43, 0.54))

    else:
        axis_cobalt = ax1
        axis_cobalt_norm = ax2
        axis_aluminum = ax1
        axis_aluminum_norm = ax2

        ax1.set_xlabel("Membrane Area (m$^2$)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Solute Rejection (%)", fontsize=14, fontweight="bold")
        ax1.tick_params(direction="in", labelsize=14)
        ax1.set_xlim(membrane_area_values[0], membrane_area_values[-1])

        ax2.set_xlabel("Membrane Area (m$^2$)", fontsize=14, fontweight="bold")
        ax2.set_ylabel(
            "Percent Change in Solute Rejection (%)", fontsize=14, fontweight="bold"
        )
        ax2.tick_params(direction="in", top=True, right=True, labelsize=14)

        ax2.plot(
            [membrane_area_values[0], membrane_area_values[-1]],
            [0, 0],
            "k-",
            linewidth=0.5,
        )
        ax2.set_xlim(membrane_area_values[0], membrane_area_values[-1])

        # legend points
        # ax2.plot([],[], marker='None', linestyle='None', label="Solution (linestyle)")
        ax2.plot([], [], "k-", linewidth=2, label="Single Salt")
        ax2.plot([], [], "k--", linewidth=2, label="LiCl + CoCl$_2$")
        ax2.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl$_3$")
        ax2.plot([], [], "k:", linewidth=2, label="CoCl$_2$ + AlCl$_3$")
        ax2.plot(
            [], [], marker="None", linestyle="None", label="Rejection (transparency)"
        )
        ax2.plot([], [], "ks", alpha=0.25, markersize=8, label="Observed")
        ax2.plot([], [], "ks", markersize=8, label="Actual")
        ax2.plot([], [], marker="None", linestyle="None", label="Solute (color)")
        ax2.plot([], [], "rs", markersize=8, label="Lithium")
        ax2.plot([], [], "bs", markersize=8, label="Cobalt")
        ax2.plot([], [], "gs", markersize=8, label="Aluminum")

        ax2.legend(
            loc="best", title="Solution (linestyle)"
        )  # , bbox_to_anchor=(0.43, 0.54))

    ax1.plot(
        membrane_area_values,
        observed_lithium_rejection_li,
        "r-",
        linewidth=2,
        alpha=0.25,
    )  # , label="Lithium (Li)")
    ax1.plot(
        membrane_area_values, actual_lithium_rejection_li, "r-", linewidth=2
    )  # , label="Lithium (Li)")
    ax1.plot(
        membrane_area_values,
        observed_lithium_rejection_li_co,
        "r--",
        linewidth=2,
        alpha=0.25,
    )  # , label="Lithium (Li-Co)")
    ax1.plot(
        membrane_area_values, actual_lithium_rejection_li_co, "r--", linewidth=2
    )  # , label="Lithium (Li-Co)")
    ax1.plot(
        membrane_area_values,
        observed_lithium_rejection_li_al,
        "r-.",
        linewidth=2,
        alpha=0.25,
    )  # , label="Lithium (Li-Al)")
    ax1.plot(
        membrane_area_values, actual_lithium_rejection_li_al, "r-.", linewidth=2
    )  # , label="Lithium (Li-Al)")
    # ax1.plot(
    #     membrane_area_values,
    #     observed_lithium_rejection_li_co_al,
    #     "r.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )  # , label="Lithium (Li-Co-Al)")
    # ax1.plot(
    #     membrane_area_values, actual_lithium_rejection_li_co_al, "r.-", linewidth=2
    # )  # , label="Lithium (Li-Co-Al)")

    axis_cobalt.plot(
        membrane_area_values,
        observed_cobalt_rejection_co,
        "b-",
        linewidth=2,
        alpha=0.25,
    )  # , label="Cobalt (Co)")
    axis_cobalt.plot(
        membrane_area_values, actual_cobalt_rejection_co, "b-", linewidth=2
    )  # , label="Cobalt (Co)")
    axis_cobalt.plot(
        membrane_area_values,
        observed_cobalt_rejection_li_co,
        "b--",
        linewidth=2,
        alpha=0.25,
    )  # , label="Cobalt (Li-Co)")
    axis_cobalt.plot(
        membrane_area_values, actual_cobalt_rejection_li_co, "b--", linewidth=2
    )  # , label="Cobalt (Li-Co)")
    axis_cobalt.plot(
        membrane_area_values,
        observed_cobalt_rejection_co_al,
        "b:",
        linewidth=2,
        alpha=0.25,
    )  # , label="Cobalt (Co-Al)")
    axis_cobalt.plot(
        membrane_area_values, actual_cobalt_rejection_co_al, "b:", linewidth=2
    )  # , label="Cobalt (Co-Al)")
    # axis_cobalt.plot(
    #     membrane_area_values,
    #     observed_cobalt_rejection_li_co_al,
    #     "b.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )  # , label="Cobalt (Li-Co-Al)")
    # axis_cobalt.plot(
    #     membrane_area_values, actual_cobalt_rejection_li_co_al, "b.-", linewidth=2
    # )  # , label="Cobalt (Li-Co-Al)")

    axis_aluminum.plot(
        membrane_area_values,
        observed_aluminum_rejection_al,
        "g-",
        linewidth=2,
        alpha=0.25,
    )  # , label="Aluminum (Al)")
    axis_aluminum.plot(
        membrane_area_values, actual_aluminum_rejection_al, "g-", linewidth=2
    )  # , label="Aluminum (Al)")
    axis_aluminum.plot(
        membrane_area_values,
        observed_aluminum_rejection_li_al,
        "g-.",
        linewidth=2,
        alpha=0.25,
    )  # , label="Aluminum (Li-Al)")
    axis_aluminum.plot(
        membrane_area_values, actual_aluminum_rejection_li_al, "g-.", linewidth=2
    )  # , label="Aluminum (Li-Al)")
    axis_aluminum.plot(
        membrane_area_values,
        observed_aluminum_rejection_co_al,
        "g:",
        linewidth=2,
        alpha=0.25,
    )  # , label="Aluminum (Co-Al)")
    axis_aluminum.plot(
        membrane_area_values, actual_aluminum_rejection_co_al, "g:", linewidth=2
    )  # , label="Aluminum (Co-Al)")
    # axis_aluminum.plot(
    #     membrane_area_values,
    #     observed_aluminum_rejection_li_co_al,
    #     "g.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )  # , label="Aluminum (Li-Co-Al)")
    # axis_aluminum.plot(
    #     membrane_area_values, actual_aluminum_rejection_li_co_al, "g.-", linewidth=2
    # )  # , label="Aluminum (Li-(o-Al)")

    ax2.plot(
        membrane_area_values,
        observed_lithium_rejection_li_norm,
        "r-",
        linewidth=2,
        alpha=0.25,
    )  # , label="Lithium (Li)")
    ax2.plot(
        membrane_area_values, actual_lithium_rejection_li_norm, "r-", linewidth=2
    )  # , label="Lithium (Li)")
    ax2.plot(
        membrane_area_values,
        observed_lithium_rejection_li_co_norm,
        "r--",
        linewidth=2,
        alpha=0.25,
    )  # , label="Lithium (Li-Co)")
    ax2.plot(
        membrane_area_values, actual_lithium_rejection_li_co_norm, "r--", linewidth=2
    )  # , label="Lithium (Li-Co)")
    ax2.plot(
        membrane_area_values,
        observed_lithium_rejection_li_al_norm,
        "r-.",
        linewidth=2,
        alpha=0.25,
    )  # , label="Lithium (Li-Al)")
    ax2.plot(
        membrane_area_values, actual_lithium_rejection_li_al_norm, "r-.", linewidth=2
    )  # , label="Lithium (Li-Al)")
    # ax2.plot(
    #     membrane_area_values,
    #     observed_lithium_rejection_li_co_al_norm,
    #     "r.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )  # , label="Lithium (Li-Co-Al)")
    # ax2.plot(
    #     membrane_area_values, actual_lithium_rejection_li_co_al_norm, "r.-", linewidth=2
    # )  # , label="Lithium (Li-Co-Al)")

    axis_cobalt_norm.plot(
        membrane_area_values,
        observed_cobalt_rejection_co_norm,
        "b-",
        linewidth=2,
        alpha=0.25,
    )  # , label="Cobalt (Co)")
    axis_cobalt_norm.plot(
        membrane_area_values, actual_cobalt_rejection_co_norm, "b-", linewidth=2
    )  # , label="Cobalt (Co)")
    axis_cobalt_norm.plot(
        membrane_area_values,
        observed_cobalt_rejection_li_co_norm,
        "b--",
        linewidth=2,
        alpha=0.25,
    )  # , label="Cobalt (Li-Co)")
    axis_cobalt_norm.plot(
        membrane_area_values, actual_cobalt_rejection_li_co_norm, "b--", linewidth=2
    )  # , label="Cobalt (Li-Co)")
    axis_cobalt_norm.plot(
        membrane_area_values,
        observed_cobalt_rejection_co_al_norm,
        "b:",
        linewidth=2,
        alpha=0.25,
    )  # , label="Cobalt (Co-Al)")
    axis_cobalt_norm.plot(
        membrane_area_values, actual_cobalt_rejection_co_al_norm, "b:", linewidth=2
    )  # , label="Cobalt (Co-Al)")
    # axis_cobalt_norm.plot(
    #     membrane_area_values,
    #     observed_cobalt_rejection_li_co_al_norm,
    #     "b.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )  # , label="Cobalt (Li-Co-Al)")
    # axis_cobalt_norm.plot(
    #     membrane_area_values, actual_cobalt_rejection_li_co_al_norm, "b.-", linewidth=2
    # )  # , label="Cobalt (Li-Co-Al)")

    axis_aluminum_norm.plot(
        membrane_area_values,
        observed_aluminum_rejection_al_norm,
        "g-",
        linewidth=2,
        alpha=0.25,
    )  # , label="Aluminum (Al)")
    axis_aluminum_norm.plot(
        membrane_area_values, actual_aluminum_rejection_al_norm, "g-", linewidth=2
    )  # , label="Aluminum (Al)")
    axis_aluminum_norm.plot(
        membrane_area_values,
        observed_aluminum_rejection_li_al_norm,
        "g-.",
        linewidth=2,
        alpha=0.25,
    )  # , label="Aluminum (Li-Al)")
    axis_aluminum_norm.plot(
        membrane_area_values, actual_aluminum_rejection_li_al_norm, "g-.", linewidth=2
    )  # , label="Aluminum (Li-Al)")
    axis_aluminum_norm.plot(
        membrane_area_values,
        observed_aluminum_rejection_co_al_norm,
        "g:",
        linewidth=2,
        alpha=0.25,
    )  # , label="Aluminum (Co-Al)")
    axis_aluminum_norm.plot(
        membrane_area_values, actual_aluminum_rejection_co_al_norm, "g:", linewidth=2
    )  # , label="Aluminum (Co-Al)")
    # axis_aluminum_norm.plot(
    #     membrane_area_values,
    #     observed_aluminum_rejection_li_co_al_norm,
    #     "g.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )  # , label="Aluminum (Li-Co-Al)")
    # axis_aluminum_norm.plot(
    #     membrane_area_values, actual_aluminum_rejection_li_co_al_norm, "g.-", linewidth=2
    # )  # , label="Aluminum (Li-Co-Al)")

    plt.tight_layout()


def plot_rejection_versus_concentration(
    m_li_cl_results_dict,
    m_co_cl_results_dict,
    m_al_cl_results_dict,
    m_li_co_cl_results_dict,
    m_li_al_cl_results_dict,
    m_co_al_cl_results_dict,
    # m_li_co_al_cl_results_dict,
    x_axis_conc="bulk",
):
    """
    Plots rejection versus retentate-side concentration or ionic strength
    or interface concentration.
    """
    x_axis_values = m_li_cl_results_dict["x_values"]

    # concentrations
    lithium_bulk_concentration_li = m_li_cl_results_dict[
        "cation_1_retentate_concentration"
    ]
    lithium_bulk_concentration_li_co = m_li_co_cl_results_dict[
        "cation_1_retentate_concentration"
    ]
    lithium_bulk_concentration_li_al = m_li_al_cl_results_dict[
        "cation_1_retentate_concentration"
    ]
    # lithium_bulk_concentration_li_co_al = m_li_co_al_cl_results_dict["cation_1_retentate_concentration"]

    lithium_int_concentration_li = m_li_cl_results_dict[
        "cation_1_interface_concentration"
    ]
    lithium_int_concentration_li_co = m_li_co_cl_results_dict[
        "cation_1_interface_concentration"
    ]
    lithium_int_concentration_li_al = m_li_al_cl_results_dict[
        "cation_1_interface_concentration"
    ]
    # lithium_int_concentration_li_co_al = m_li_co_al_cl_results_dict["cation_1_interface_concentration"]

    cobalt_bulk_concentration_co = m_co_cl_results_dict[
        "cation_1_retentate_concentration"
    ]
    cobalt_bulk_concentration_li_co = m_li_co_cl_results_dict[
        "cation_2_retentate_concentration"
    ]
    cobalt_bulk_concentration_co_al = m_co_al_cl_results_dict[
        "cation_1_retentate_concentration"
    ]
    # cobalt_bulk_concentration_li_co_al = m_li_co_al_cl_results_dict["cation_2_retentate_concentration"]

    cobalt_int_concentration_co = m_co_cl_results_dict[
        "cation_1_interface_concentration"
    ]
    cobalt_int_concentration_li_co = m_li_co_cl_results_dict[
        "cation_2_interface_concentration"
    ]
    cobalt_int_concentration_co_al = m_co_al_cl_results_dict[
        "cation_1_interface_concentration"
    ]
    # cobalt_int_concentration_li_co_al = m_li_co_al_cl_results_dict["cation_2_interface_concentration"]

    aluminum_bulk_concentration_al = m_al_cl_results_dict[
        "cation_1_retentate_concentration"
    ]
    aluminum_bulk_concentration_li_al = m_li_al_cl_results_dict[
        "cation_2_retentate_concentration"
    ]
    aluminum_bulk_concentration_co_al = m_co_al_cl_results_dict[
        "cation_2_retentate_concentration"
    ]
    # aluminum_bulk_concentration_li_co_al = m_li_co_al_cl_results_dict["cation_3_retentate_concentration"]

    aluminum_int_concentration_al = m_al_cl_results_dict[
        "cation_1_interface_concentration"
    ]
    aluminum_int_concentration_li_al = m_li_al_cl_results_dict[
        "cation_2_interface_concentration"
    ]
    aluminum_int_concentration_co_al = m_co_al_cl_results_dict[
        "cation_2_interface_concentration"
    ]
    # aluminum_int_concentration_li_co_al = m_li_co_al_cl_results_dict["cation_3_interface_concentration"]

    anion_bulk_concentration_li = m_li_cl_results_dict["anion_retentate_concentration"]
    anion_bulk_concentration_co = m_co_cl_results_dict["anion_retentate_concentration"]
    anion_bulk_concentration_al = m_al_cl_results_dict["anion_retentate_concentration"]
    anion_bulk_concentration_li_co = m_li_co_cl_results_dict[
        "anion_retentate_concentration"
    ]
    anion_bulk_concentration_li_al = m_li_al_cl_results_dict[
        "anion_retentate_concentration"
    ]
    anion_bulk_concentration_co_al = m_co_al_cl_results_dict[
        "anion_retentate_concentration"
    ]
    # anion_bulk_concentration_li_co_al = m_li_co_al_cl_results_dict["anion_retentate_concentration"]

    # lithium rejections
    observed_lithium_rejection_li = m_li_cl_results_dict["cation_1_rejection_observed"]
    actual_lithium_rejection_li = m_li_cl_results_dict["cation_1_rejection_actual"]
    observed_lithium_rejection_li_co = m_li_co_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_lithium_rejection_li_co = m_li_co_cl_results_dict[
        "cation_1_rejection_actual"
    ]
    observed_lithium_rejection_li_al = m_li_al_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_lithium_rejection_li_al = m_li_al_cl_results_dict[
        "cation_1_rejection_actual"
    ]
    # observed_lithium_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_1_rejection_observed"
    # ]
    # actual_lithium_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_1_rejection_actual"
    # ]

    # cobalt rejections
    observed_cobalt_rejection_co = m_co_cl_results_dict["cation_1_rejection_observed"]
    actual_cobalt_rejection_co = m_co_cl_results_dict["cation_1_rejection_actual"]
    observed_cobalt_rejection_li_co = m_li_co_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_cobalt_rejection_li_co = m_li_co_cl_results_dict["cation_2_rejection_actual"]
    observed_cobalt_rejection_co_al = m_co_al_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_cobalt_rejection_co_al = m_co_al_cl_results_dict["cation_1_rejection_actual"]
    # observed_cobalt_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_2_rejection_observed"
    # ]
    # actual_cobalt_rejection_li_co_al = m_li_co_al_cl_results_dict["cation_2_rejection_actual"]

    # aluminum rejections
    observed_aluminum_rejection_al = m_al_cl_results_dict["cation_1_rejection_observed"]
    actual_aluminum_rejection_al = m_al_cl_results_dict["cation_1_rejection_actual"]
    observed_aluminum_rejection_li_al = m_li_al_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_aluminum_rejection_li_al = m_li_al_cl_results_dict[
        "cation_2_rejection_actual"
    ]
    observed_aluminum_rejection_co_al = m_co_al_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_aluminum_rejection_co_al = m_co_al_cl_results_dict[
        "cation_2_rejection_actual"
    ]
    # observed_aluminum_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_3_rejection_observed"
    # ]
    # actual_aluminum_rejection_li_co_al = m_li_co_al_cl_results_dict[
    #     "cation_3_rejection_actual"
    # ]
    # calculate bulk ionic strengths
    bulk_ionic_strength_li = [
        0.5
        * (
            (lithium_bulk_concentration_li[x] * (1) ** 2)
            + (anion_bulk_concentration_li[x] * (-1) ** 2)
        )
        for x in range(len(x_axis_values))
    ]
    bulk_ionic_strength_co = [
        0.5
        * (
            cobalt_bulk_concentration_co[x] * (2) ** 2
            + (anion_bulk_concentration_co[x] * (-1) ** 2)
        )
        for x in range(len(x_axis_values))
    ]
    bulk_ionic_strength_al = [
        0.5
        * (
            (aluminum_bulk_concentration_al[x] * (3) ** 2)
            + (anion_bulk_concentration_al[x] * (-1) ** 2)
        )
        for x in range(len(x_axis_values))
    ]
    bulk_ionic_strength_li_co = [
        0.5
        * (
            (lithium_bulk_concentration_li_co[x] * (1) ** 2)
            + (cobalt_bulk_concentration_li_co[x] * (2) ** 2)
            + (anion_bulk_concentration_li_co[x] * (-1) ** 2)
        )
        for x in range(len(x_axis_values))
    ]
    bulk_ionic_strength_li_al = [
        0.5
        * (
            (lithium_bulk_concentration_li_al[x] * (1) ** 2)
            + (aluminum_bulk_concentration_li_al[x] * (3) ** 2)
            + (anion_bulk_concentration_li_al[x] * (-1) ** 2)
        )
        for x in range(len(x_axis_values))
    ]
    bulk_ionic_strength_co_al = [
        0.5
        * (
            (cobalt_bulk_concentration_co_al[x] * (2) ** 2)
            + (aluminum_bulk_concentration_co_al[x] * (3) ** 2)
            + (anion_bulk_concentration_co_al[x] * (-1) ** 2)
        )
        for x in range(len(x_axis_values))
    ]

    if x_axis_conc == "bulk":
        lithium_concentration_li = lithium_bulk_concentration_li
        lithium_concentration_li_co = lithium_bulk_concentration_li_co
        lithium_concentration_li_al = lithium_bulk_concentration_li_al
        # lithium_concentration_li_co_al = lithium_bulk_concentration_li_co_al

        cobalt_concentration_co = cobalt_bulk_concentration_co
        cobalt_concentration_li_co = cobalt_bulk_concentration_li_co
        cobalt_concentration_co_al = cobalt_bulk_concentration_co_al
        # cobalt_concentration_li_co_al = cobalt_bulk_concentration_li_co_al

        aluminum_concentration_al = aluminum_bulk_concentration_al
        aluminum_concentration_li_al = aluminum_bulk_concentration_li_al
        aluminum_concentration_co_al = aluminum_bulk_concentration_co_al
        # aluminum_concentration_li_co_al = aluminum_bulk_concentration_li_co_al
    elif x_axis_conc == "interface":
        lithium_concentration_li = lithium_int_concentration_li
        lithium_concentration_li_co = lithium_int_concentration_li_co
        lithium_concentration_li_al = lithium_int_concentration_li_al
        # lithium_concentration_li_co_al = lithium_int_concentration_li_co_al

        cobalt_concentration_co = cobalt_int_concentration_co
        cobalt_concentration_li_co = cobalt_int_concentration_li_co
        cobalt_concentration_co_al = cobalt_int_concentration_co_al
        # cobalt_concentration_li_co_al = cobalt_int_concentration_li_co_al

        aluminum_concentration_al = aluminum_int_concentration_al
        aluminum_concentration_li_al = aluminum_int_concentration_li_al
        aluminum_concentration_co_al = aluminum_int_concentration_co_al
        # aluminum_concentration_li_co_al = aluminum_int_concentration_li_co_al
    elif x_axis_conc == "bulk-ionic-strength":
        lithium_concentration_li = bulk_ionic_strength_li
        lithium_concentration_li_co = bulk_ionic_strength_li_co
        lithium_concentration_li_al = bulk_ionic_strength_li_al

        cobalt_concentration_co = bulk_ionic_strength_co
        cobalt_concentration_li_co = bulk_ionic_strength_li_co
        cobalt_concentration_co_al = bulk_ionic_strength_co_al

        aluminum_concentration_al = bulk_ionic_strength_al
        aluminum_concentration_li_al = bulk_ionic_strength_li_al
        aluminum_concentration_co_al = bulk_ionic_strength_co_al

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=100, figsize=(15, 5))

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
    # ax1.plot(
    #     lithium_concentration_li_co_al,
    #     observed_lithium_rejection_li_co_al,
    #     "r.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )
    # ax1.plot(
    #     lithium_concentration_li_co_al,
    #     actual_lithium_rejection_li_co_al,
    #     "r.-",
    #     linewidth=2,
    # )
    if (x_axis_conc == "bulk") or (x_axis_conc == "interface"):
        ax1.set_xlabel(
            f"Lithium Concentration ({x_axis_conc.capitalize()}) (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
    elif x_axis_conc == "bulk-ionic-strength":
        ax1.set_xlabel(
            "Retentate-Side Ionic Strength (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
    ax1.set_title("Lithium Rejection")
    ax1.set_ylabel("Percent Rejection (%)", fontsize=14, fontweight="bold")
    ax1.tick_params(direction="in", labelsize=14)
    ax1.plot([], [], "k-", linewidth=2, label="LiCl")
    ax1.plot([], [], "k--", linewidth=2, label="LiCl + CoCl$_2$")
    ax1.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl$_3$")
    # ax1.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl$_2$ + AlCl$_3$")
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
    # ax2.plot(
    #     cobalt_concentration_li_co_al,
    #     observed_cobalt_rejection_li_co_al,
    #     "b.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )
    # ax2.plot(
    #     cobalt_concentration_li_co_al,
    #     actual_cobalt_rejection_li_co_al,
    #     "b.-",
    #     linewidth=2,
    # )
    if (x_axis_conc == "bulk") or (x_axis_conc == "interface"):
        ax2.set_xlabel(
            f"Cobalt Concentration ({x_axis_conc.capitalize()}) (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
    elif x_axis_conc == "bulk-ionic-strength":
        ax2.set_xlabel(
            "Retentate-Side Ionic Strength (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
    ax2.set_title("Cobalt Rejection")
    ax2.set_ylabel("Percent Rejection (%)", fontsize=14, fontweight="bold")
    ax2.tick_params(direction="in", labelsize=14)
    ax2.plot([], [], "k-", linewidth=2, label="CoCl$_2$")
    ax2.plot([], [], "k--", linewidth=2, label="LiCl + CoCl$_2$")
    ax2.plot([], [], "k:", linewidth=2, label="CoCl$_2$ + AlCl$_3$")
    # ax2.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl$_2$ + AlCl$_3$")
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
    # ax3.plot(
    #     aluminum_concentration_li_co_al,
    #     observed_aluminum_rejection_li_co_al,
    #     "g.-",
    #     alpha=0.25,
    #     linewidth=2,
    # )
    # ax3.plot(
    #     aluminum_concentration_li_co_al,
    #     actual_aluminum_rejection_li_co_al,
    #     "g.-",
    #     linewidth=2,
    # )
    if (x_axis_conc == "bulk") or (x_axis_conc == "interface"):
        ax3.set_xlabel(
            f"Aluminum Concentration ({x_axis_conc.capitalize()}) (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
    elif x_axis_conc == "bulk-ionic-strength":
        ax3.set_xlabel(
            "Retentate-Side Ionic Strength (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
    ax3.set_title("Aluminum Rejection")
    ax3.set_ylabel("Percent Rejection (%)", fontsize=14, fontweight="bold")
    ax3.tick_params(direction="in", labelsize=14)

    ax3.plot([], [], "k-", linewidth=2, label="AlCl$_3$")
    ax3.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl$_3$")
    ax3.plot([], [], "k:", linewidth=2, label="CoCl$_2$ + AlCl$_3$")
    # ax3.plot([], [], "k.-", linewidth=2, label="LiCl + CoCl$_2$ + AlCl$_3$")
    ax3.plot([], [], marker="None", linestyle="None", label="Rejection (color)")
    ax3.plot([], [], "gs", alpha=0.25, markersize=8, label="Observed")
    ax3.plot([], [], "gs", markersize=8, label="Actual")
    ax3.legend(loc="best", title="Solution (linestyle)")

    plt.tight_layout()


def plot_rejection_versus_feed_ionic_strength(
    m_li_cl_results_dict,
    m_co_cl_results_dict,
    m_al_cl_results_dict,
    m_li_co_cl_results_dict,
    m_li_al_cl_results_dict,
    m_co_al_cl_results_dict,
    m_li_co_al_cl_results_dict,
):
    """
    Plots rejection versus ionic strength of bulk fluid and feed.
    """
    x_axis_values = m_li_cl_results_dict["x_values"]

    # feed ionic strength
    feed_ionic_strength_li_val = m_li_cl_results_dict["feed_ionic_strength"]
    feed_ionic_strength_co_val = m_co_cl_results_dict["feed_ionic_strength"]
    feed_ionic_strength_al_val = m_al_cl_results_dict["feed_ionic_strength"]
    feed_ionic_strength_li_co_val = m_li_co_cl_results_dict["feed_ionic_strength"]
    feed_ionic_strength_li_al_val = m_li_al_cl_results_dict["feed_ionic_strength"]
    feed_ionic_strength_co_al_val = m_co_al_cl_results_dict["feed_ionic_strength"]
    feed_ionic_strength_li_co_al_val = m_li_co_al_cl_results_dict["feed_ionic_strength"]

    feed_ionic_strength_li = [
        feed_ionic_strength_li_val for i in range(len(x_axis_values))
    ]
    feed_ionic_strength_co = [
        feed_ionic_strength_co_val for i in range(len(x_axis_values))
    ]
    feed_ionic_strength_al = [
        feed_ionic_strength_al_val for i in range(len(x_axis_values))
    ]
    feed_ionic_strength_li_co = [
        feed_ionic_strength_li_co_val for i in range(len(x_axis_values))
    ]
    feed_ionic_strength_li_al = [
        feed_ionic_strength_li_al_val for i in range(len(x_axis_values))
    ]
    feed_ionic_strength_co_al = [
        feed_ionic_strength_co_al_val for i in range(len(x_axis_values))
    ]
    feed_ionic_strength_li_co_al = [
        feed_ionic_strength_li_co_al_val for i in range(len(x_axis_values))
    ]

    # lithium rejections
    observed_lithium_rejection_li = m_li_cl_results_dict["cation_1_rejection_observed"]
    actual_lithium_rejection_li = m_li_cl_results_dict["cation_1_rejection_actual"]
    observed_lithium_rejection_li_co = m_li_co_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_lithium_rejection_li_co = m_li_co_cl_results_dict[
        "cation_1_rejection_actual"
    ]
    observed_lithium_rejection_li_al = m_li_al_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_lithium_rejection_li_al = m_li_al_cl_results_dict[
        "cation_1_rejection_actual"
    ]
    observed_lithium_rejection_li_co_al = m_li_co_al_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_lithium_rejection_li_co_al = m_li_co_al_cl_results_dict[
        "cation_1_rejection_actual"
    ]

    # cobalt rejections
    observed_cobalt_rejection_co = m_co_cl_results_dict["cation_1_rejection_observed"]
    actual_cobalt_rejection_co = m_co_cl_results_dict["cation_1_rejection_actual"]
    observed_cobalt_rejection_li_co = m_li_co_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_cobalt_rejection_li_co = m_li_co_cl_results_dict["cation_2_rejection_actual"]
    observed_cobalt_rejection_co_al = m_co_al_cl_results_dict[
        "cation_1_rejection_observed"
    ]
    actual_cobalt_rejection_co_al = m_co_al_cl_results_dict["cation_1_rejection_actual"]
    observed_cobalt_rejection_li_co_al = m_li_co_al_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_cobalt_rejection_li_co_al = m_li_co_al_cl_results_dict[
        "cation_2_rejection_actual"
    ]

    # aluminum rejections
    observed_aluminum_rejection_al = m_al_cl_results_dict["cation_1_rejection_observed"]
    actual_aluminum_rejection_al = m_al_cl_results_dict["cation_1_rejection_actual"]
    observed_aluminum_rejection_li_al = m_li_al_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_aluminum_rejection_li_al = m_li_al_cl_results_dict[
        "cation_2_rejection_actual"
    ]
    observed_aluminum_rejection_co_al = m_co_al_cl_results_dict[
        "cation_2_rejection_observed"
    ]
    actual_aluminum_rejection_co_al = m_co_al_cl_results_dict[
        "cation_2_rejection_actual"
    ]
    observed_aluminum_rejection_li_co_al = m_li_co_al_cl_results_dict[
        "cation_3_rejection_observed"
    ]
    actual_aluminum_rejection_li_co_al = m_li_co_al_cl_results_dict[
        "cation_3_rejection_actual"
    ]

    fig, ax1 = plt.subplots(1, 1, dpi=100, figsize=(7, 6))

    # lithium
    ax1.plot(
        feed_ionic_strength_li,
        observed_lithium_rejection_li,
        "ro",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li,
        actual_lithium_rejection_li,
        "ro",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co,
        observed_lithium_rejection_li_co,
        "r*",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co,
        actual_lithium_rejection_li_co,
        "r*",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_al,
        observed_lithium_rejection_li_al,
        "r^",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_al,
        actual_lithium_rejection_li_al,
        "r^",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co_al,
        observed_lithium_rejection_li_co_al,
        "rD",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co_al,
        actual_lithium_rejection_li_co_al,
        "rD",
        markersize=10,
    )

    # cobalt
    ax1.plot(
        feed_ionic_strength_co,
        observed_cobalt_rejection_co,
        "bo",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_co,
        actual_cobalt_rejection_co,
        "bo",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co,
        observed_cobalt_rejection_li_co,
        "b*",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co,
        actual_cobalt_rejection_li_co,
        "b*",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_co_al,
        observed_cobalt_rejection_co_al,
        "bv",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_co_al,
        actual_cobalt_rejection_co_al,
        "bv",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co_al,
        observed_cobalt_rejection_li_co_al,
        "bD",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co_al,
        actual_cobalt_rejection_li_co_al,
        "bD",
        markersize=10,
    )

    # aluminum
    ax1.plot(
        feed_ionic_strength_al,
        observed_aluminum_rejection_al,
        "go",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_al,
        actual_aluminum_rejection_al,
        "go",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_al,
        observed_aluminum_rejection_li_al,
        "g^",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_al,
        actual_aluminum_rejection_li_al,
        "g^",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_co_al,
        observed_aluminum_rejection_co_al,
        "gv",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_co_al,
        actual_aluminum_rejection_co_al,
        "gv",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co_al,
        observed_aluminum_rejection_li_co_al,
        "gD",
        mfc="none",
        markersize=10,
    )
    ax1.plot(
        feed_ionic_strength_li_co_al,
        actual_aluminum_rejection_li_co_al,
        "gD",
        markersize=10,
    )

    ax1.axhline(0, color="black", linewidth=1)

    ax1.set_xlabel(
        "Inlet Feed Ionic Strength (mol/m$^3$)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Percent Rejection (%)", fontsize=14, fontweight="bold")
    ax1.tick_params(direction="in", labelsize=14)
    ax1.plot([], [], "ko", markersize=10, label="Single Salt")
    ax1.plot([], [], "k*", markersize=10, label="LiCl + CoCl$_2$")
    ax1.plot([], [], "k^", markersize=10, label="LiCl + AlCl$_3$")
    ax1.plot([], [], "kv", markersize=10, label="CoCl$_2$ + AlCl$_3$")
    ax1.plot([], [], "kD", markersize=10, label="LiCl + CoCl$_2$ + AlCl$_3$")
    ax1.plot([], [], marker="None", linestyle="None", label="Rejection (fill)")
    ax1.plot([], [], "ks", mfc="none", markersize=8, label="Observed")
    ax1.plot([], [], "ks", markersize=8, label="Actual")
    ax1.plot([], [], marker="None", linestyle="None", label="Solute (color)")
    ax1.plot([], [], "rs", markersize=8, label="Lithium")
    ax1.plot([], [], "bs", markersize=8, label="Cobalt")
    ax1.plot([], [], "gs", markersize=8, label="Aluminum")
    ax1.legend(loc="best", title="Solution (marker)", title_fontsize=10, fontsize=10)

    plt.tight_layout()


def plot_electric_potential_gradient(
    m_li_cl_results_dict,
    # m_co_cl_results_dict,
    # m_al_cl_results_dict,
    m_li_co_cl_results_dict,
    m_li_al_cl_results_dict,
    # m_co_al_cl_results_dict,
):
    """
    Plots electric potential gradient for different systems.
    # TODO add this calculation to the model
    """

    # global constants
    z_lithium = 1
    z_cobalt = 2
    z_aluminum = 3
    z_chloride = -1

    D_bl_lithium = 3.71e-6  # m2 / h
    D_bl_cobalt = 2.64e-6  # m2 / h
    D_bl_aluminum = 2.01e-6  # m2 / h
    D_bl_chloride = 7.31e-6  # m2 / h

    D_mem_lithium = 3.71e-6  # m2 / h
    D_mem_cobalt = 2.64e-6  # m2 / h
    D_mem_aluminum = 2.01e-6  # m2 / h
    D_mem_chloride = 7.31e-6  # m2 / h

    chi = -44  # mM

    R = 8.314  # J / K / mol
    F = 96485  # J / V / mol
    T = 298  # K

    x_axis_values = m_li_cl_results_dict["x_values_dimensionless"]
    z_bl_axis_values = m_li_cl_results_dict["z_bl_values"]
    z_mem_axis_values = m_li_cl_results_dict["z_mem_values"]

    # water flux
    water_flux_li = m_li_cl_results_dict["water_flux"]
    # water_flux_co = m_co_cl_results_dict["water_flux"]
    # water_flux_al = m_al_cl_results_dict["water_flux"]
    water_flux_li_co = m_li_co_cl_results_dict["water_flux"]
    water_flux_li_al = m_li_al_cl_results_dict["water_flux"]
    # water_flux_co_al = m_co_al_cl_results_dict["water_flux"]

    # boundary layer concentrations
    lithium_bl_concentration_li = m_li_cl_results_dict[
        "cation_1_boundary_layer_concentration_by_x"
    ]
    lithium_bl_concentration_li_co = m_li_co_cl_results_dict[
        "cation_1_boundary_layer_concentration_by_x"
    ]
    lithium_bl_concentration_li_al = m_li_al_cl_results_dict[
        "cation_1_boundary_layer_concentration_by_x"
    ]
    # cobalt_bl_concentration_co = m_co_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_by_x"
    # ]
    cobalt_bl_concentration_li_co = m_li_co_cl_results_dict[
        "cation_2_boundary_layer_concentration_by_x"
    ]
    # cobalt_bl_concentration_co_al = m_co_al_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_by_x"
    # ]
    # aluminum_bl_concentration_al = m_al_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_by_x"
    # ]
    aluminum_bl_concentration_li_al = m_li_al_cl_results_dict[
        "cation_2_boundary_layer_concentration_by_x"
    ]
    # aluminum_bl_concentration_co_al = m_co_al_cl_results_dict[
    #     "cation_2_boundary_layer_concentration_by_x"
    # ]

    # membrane concentrations
    lithium_mem_concentration_li = m_li_cl_results_dict[
        "cation_1_membrane_concentration_by_x"
    ]
    lithium_mem_concentration_li_co = m_li_co_cl_results_dict[
        "cation_1_membrane_concentration_by_x"
    ]
    lithium_mem_concentration_li_al = m_li_al_cl_results_dict[
        "cation_1_membrane_concentration_by_x"
    ]
    # cobalt_mem_concentration_co = m_co_cl_results_dict[
    #     "cation_1_membrane_concentration_by_x"
    # ]
    cobalt_mem_concentration_li_co = m_li_co_cl_results_dict[
        "cation_2_membrane_concentration_by_x"
    ]
    # cobalt_mem_concentration_co_al = m_co_al_cl_results_dict[
    #     "cation_1_membrane_concentration_by_x"
    # ]
    # aluminum_mem_concentration_al = m_al_cl_results_dict[
    #     "cation_1_membrane_concentration_by_x"
    # ]
    aluminum_mem_concentration_li_al = m_li_al_cl_results_dict[
        "cation_2_membrane_concentration_by_x"
    ]
    # aluminum_mem_concentration_co_al = m_co_al_cl_results_dict[
    #     "cation_2_membrane_concentration_by_x"
    # ]

    # boundary layer concentration gradients
    lithium_bl_concentration_gradient_li = m_li_cl_results_dict[
        "cation_1_boundary_layer_concentration_gradient_by_x"
    ]
    lithium_bl_concentration_gradient_li_co = m_li_co_cl_results_dict[
        "cation_1_boundary_layer_concentration_gradient_by_x"
    ]
    lithium_bl_concentration_gradient_li_al = m_li_al_cl_results_dict[
        "cation_1_boundary_layer_concentration_gradient_by_x"
    ]
    # cobalt_bl_concentration_gradient_co = m_co_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_gradient_by_x"
    # ]
    cobalt_bl_concentration_gradient_li_co = m_li_co_cl_results_dict[
        "cation_2_boundary_layer_concentration_gradient_by_x"
    ]
    # cobalt_bl_concentration_gradient_co_al = m_co_al_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_gradient_by_x"
    # ]
    # aluminum_bl_concentration_gradient_al = m_al_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_gradient_by_x"
    # ]
    aluminum_bl_concentration_gradient_li_al = m_li_al_cl_results_dict[
        "cation_2_boundary_layer_concentration_gradient_by_x"
    ]
    # aluminum_bl_concentration_gradient_co_al = m_co_al_cl_results_dict[
    #     "cation_2_boundary_layer_concentration_gradient_by_x"
    # ]

    # membrane concentrations
    lithium_mem_concentration_gradient_li = m_li_cl_results_dict[
        "cation_1_membrane_concentration_gradient_by_x"
    ]
    lithium_mem_concentration_gradient_li_co = m_li_co_cl_results_dict[
        "cation_1_membrane_concentration_gradient_by_x"
    ]
    lithium_mem_concentration_gradient_li_al = m_li_al_cl_results_dict[
        "cation_1_membrane_concentration_gradient_by_x"
    ]
    # cobalt_mem_concentration_gradient_co = m_co_cl_results_dict[
    #     "cation_1_membrane_concentration_gradient_by_x"
    # ]
    cobalt_mem_concentration_gradient_li_co = m_li_co_cl_results_dict[
        "cation_2_membrane_concentration_gradient_by_x"
    ]
    # cobalt_mem_concentration_gradient_co_al = m_co_al_cl_results_dict[
    #     "cation_1_membrane_concentration_gradient_by_x"
    # ]
    # aluminum_mem_concentration_gradient_al = m_al_cl_results_dict[
    #     "cation_1_membrane_concentration_gradient_by_x"
    # ]
    aluminum_mem_concentration_gradient_li_al = m_li_al_cl_results_dict[
        "cation_2_membrane_concentration_gradient_by_x"
    ]
    # aluminum_mem_concentration_gradient_co_al = m_co_al_cl_results_dict[
    #     "cation_2_membrane_concentration_gradient_by_x"
    # ]

    mem_del_phi_li = []
    mem_del_phi_li_co = []
    mem_del_phi_li_al = []

    mem_del_phi_li_dict = {}
    mem_del_phi_li_co_dict = {}
    mem_del_phi_li_al_dict = {}

    for z in range(len(z_mem_axis_values)):
        for x in x_axis_values:
            x_position = x_axis_values.index(x)

            mem_del_phi_li_val = (-R * T / F) * (
                z_lithium
                * D_mem_lithium
                * lithium_mem_concentration_li[str(x)][z]
                * (
                    (
                        (
                            (D_mem_lithium - D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_gradient_li[str(x)][z]
                        )
                        + (water_flux_li[x_position] * chi)
                    )
                    / (
                        (z_lithium * D_mem_lithium - z_chloride * D_mem_chloride)
                        * z_lithium
                        * lithium_mem_concentration_li[str(x)][z]
                    )
                )
            )
            mem_del_phi_li_co_val = (-R * T / F) * (
                z_lithium
                * D_mem_lithium
                * lithium_mem_concentration_li_co[str(x)][z]
                * (
                    (
                        (
                            (D_mem_lithium - D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_gradient_li_co[str(x)][z]
                        )
                        + (
                            (D_mem_cobalt - D_mem_chloride)
                            * z_cobalt
                            * cobalt_mem_concentration_gradient_li_co[str(x)][z]
                        )
                        + (water_flux_li_co[x_position] * chi)
                    )
                    / (
                        (
                            (z_lithium * D_mem_lithium - z_chloride * D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_li_co[str(x)][z]
                        )
                        + (
                            (z_cobalt * D_mem_cobalt - z_chloride * D_mem_chloride)
                            * z_cobalt
                            * cobalt_mem_concentration_li_co[str(x)][z]
                        )
                    )
                )
            )
            mem_del_phi_li_al_val = (-R * T / F) * (
                z_lithium
                * D_mem_lithium
                * lithium_mem_concentration_li_al[str(x)][z]
                * (
                    (
                        (
                            (D_mem_lithium - D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_gradient_li_al[str(x)][z]
                        )
                        + (
                            (D_mem_aluminum - D_mem_chloride)
                            * z_aluminum
                            * aluminum_mem_concentration_gradient_li_al[str(x)][z]
                        )
                        + (water_flux_li_al[x_position] * chi)
                    )
                    / (
                        (
                            (z_lithium * D_mem_lithium - z_chloride * D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_li_al[str(x)][z]
                        )
                        + (
                            (z_aluminum * D_mem_aluminum - z_chloride * D_mem_chloride)
                            * z_aluminum
                            * aluminum_mem_concentration_li_al[str(x)][z]
                        )
                    )
                )
            )

            mem_del_phi_li.append(mem_del_phi_li_val)
            mem_del_phi_li_co.append(mem_del_phi_li_co_val)
            mem_del_phi_li_al.append(mem_del_phi_li_al_val)

        mem_del_phi_li_dict[z] = mem_del_phi_li
        mem_del_phi_li_co_dict[z] = mem_del_phi_li_co
        mem_del_phi_li_al_dict[z] = mem_del_phi_li_al

        mem_del_phi_li = []
        mem_del_phi_li_co = []
        mem_del_phi_li_al = []

    mem_del_phi_li_dict_df = DataFrame(index=x_axis_values, data=mem_del_phi_li_dict)
    mem_del_phi_li_co_dict_df = DataFrame(
        index=x_axis_values, data=mem_del_phi_li_co_dict
    )
    mem_del_phi_li_al_dict_df = DataFrame(
        index=x_axis_values, data=mem_del_phi_li_al_dict
    )

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=125, figsize=(15, 7))
    mem_plot_li = ax1.pcolor(z_mem_axis_values, x_axis_values, mem_del_phi_li_dict_df)
    mem_plot_li_co = ax2.pcolor(
        z_mem_axis_values, x_axis_values, mem_del_phi_li_co_dict_df
    )
    mem_plot_li_al = ax3.pcolor(
        z_mem_axis_values, x_axis_values, mem_del_phi_li_al_dict_df
    )

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Membrane Thickness (nm)", fontsize=14, fontweight="bold")
        ax.tick_params(direction="in", labelsize=14)

    ax1.set_ylabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax1.set_title("LiCl")
    ax2.set_title("LiCl + CoCl$_2$")
    ax3.set_title("LiCl + AlCl$_3$")

    plt.suptitle(
        "Electric Potential Gradient in Membrane (V/m)",
        fontsize=14,
        fontweight="bold",
    )
    fig1.colorbar(mem_plot_li, ax=ax1)
    fig1.colorbar(mem_plot_li_co, ax=ax2)
    fig1.colorbar(mem_plot_li_al, ax=ax3)


def plot_flux_versus_length(
    m_li_cl_results_dict,
    # m_co_cl_results_dict,
    # m_al_cl_results_dict,
    m_li_co_cl_results_dict,
    m_li_al_cl_results_dict,
    # m_co_al_cl_results_dict,
):
    """
    Plots flux contributions for different systems.
    """
    # global constants
    z_lithium = 1
    z_cobalt = 2
    z_aluminum = 3
    z_chloride = -1

    D_bl_lithium = 3.71e-6  # m2 / h
    D_bl_cobalt = 2.64e-6  # m2 / h
    D_bl_aluminum = 2.01e-6  # m2 / h
    D_bl_chloride = 7.31e-6  # m2 / h

    D_mem_lithium = 3.71e-6  # m2 / h
    D_mem_cobalt = 2.64e-6  # m2 / h
    D_mem_aluminum = 2.01e-6  # m2 / h
    D_mem_chloride = 7.31e-6  # m2 / h

    chi = -44  # mM

    x_axis_values = m_li_cl_results_dict["x_values_dimensionless"]
    z_bl_axis_values = m_li_cl_results_dict["z_bl_values"]
    z_mem_axis_values = m_li_cl_results_dict["z_mem_values"]

    # water flux
    water_flux_li = m_li_cl_results_dict["water_flux"]
    # water_flux_co = m_co_cl_results_dict["water_flux"]
    # water_flux_al = m_al_cl_results_dict["water_flux"]
    water_flux_li_co = m_li_co_cl_results_dict["water_flux"]
    water_flux_li_al = m_li_al_cl_results_dict["water_flux"]
    # water_flux_co_al = m_co_al_cl_results_dict["water_flux"]

    # boundary layer concentrations
    lithium_bl_concentration_li = m_li_cl_results_dict[
        "cation_1_boundary_layer_concentration_by_x"
    ]
    lithium_bl_concentration_li_co = m_li_co_cl_results_dict[
        "cation_1_boundary_layer_concentration_by_x"
    ]
    lithium_bl_concentration_li_al = m_li_al_cl_results_dict[
        "cation_1_boundary_layer_concentration_by_x"
    ]
    # cobalt_bl_concentration_co = m_co_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_by_x"
    # ]
    cobalt_bl_concentration_li_co = m_li_co_cl_results_dict[
        "cation_2_boundary_layer_concentration_by_x"
    ]
    # cobalt_bl_concentration_co_al = m_co_al_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_by_x"
    # ]
    # aluminum_bl_concentration_al = m_al_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_by_x"
    # ]
    aluminum_bl_concentration_li_al = m_li_al_cl_results_dict[
        "cation_2_boundary_layer_concentration_by_x"
    ]
    # aluminum_bl_concentration_co_al = m_co_al_cl_results_dict[
    #     "cation_2_boundary_layer_concentration_by_x"
    # ]

    # membrane concentrations
    lithium_mem_concentration_li = m_li_cl_results_dict[
        "cation_1_membrane_concentration_by_x"
    ]
    lithium_mem_concentration_li_co = m_li_co_cl_results_dict[
        "cation_1_membrane_concentration_by_x"
    ]
    lithium_mem_concentration_li_al = m_li_al_cl_results_dict[
        "cation_1_membrane_concentration_by_x"
    ]
    # cobalt_mem_concentration_co = m_co_cl_results_dict[
    #     "cation_1_membrane_concentration_by_x"
    # ]
    cobalt_mem_concentration_li_co = m_li_co_cl_results_dict[
        "cation_2_membrane_concentration_by_x"
    ]
    # cobalt_mem_concentration_co_al = m_co_al_cl_results_dict[
    #     "cation_1_membrane_concentration_by_x"
    # ]
    # aluminum_mem_concentration_al = m_al_cl_results_dict[
    #     "cation_1_membrane_concentration_by_x"
    # ]
    aluminum_mem_concentration_li_al = m_li_al_cl_results_dict[
        "cation_2_membrane_concentration_by_x"
    ]
    # aluminum_mem_concentration_co_al = m_co_al_cl_results_dict[
    #     "cation_2_membrane_concentration_by_x"
    # ]

    # boundary layer concentration gradients
    lithium_bl_concentration_gradient_li = m_li_cl_results_dict[
        "cation_1_boundary_layer_concentration_gradient_by_x"
    ]
    lithium_bl_concentration_gradient_li_co = m_li_co_cl_results_dict[
        "cation_1_boundary_layer_concentration_gradient_by_x"
    ]
    lithium_bl_concentration_gradient_li_al = m_li_al_cl_results_dict[
        "cation_1_boundary_layer_concentration_gradient_by_x"
    ]
    # cobalt_bl_concentration_gradient_co = m_co_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_gradient_by_x"
    # ]
    cobalt_bl_concentration_gradient_li_co = m_li_co_cl_results_dict[
        "cation_2_boundary_layer_concentration_gradient_by_x"
    ]
    # cobalt_bl_concentration_gradient_co_al = m_co_al_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_gradient_by_x"
    # ]
    # aluminum_bl_concentration_gradient_al = m_al_cl_results_dict[
    #     "cation_1_boundary_layer_concentration_gradient_by_x"
    # ]
    aluminum_bl_concentration_gradient_li_al = m_li_al_cl_results_dict[
        "cation_2_boundary_layer_concentration_gradient_by_x"
    ]
    # aluminum_bl_concentration_gradient_co_al = m_co_al_cl_results_dict[
    #     "cation_2_boundary_layer_concentration_gradient_by_x"
    # ]

    # membrane concentrations
    lithium_mem_concentration_gradient_li = m_li_cl_results_dict[
        "cation_1_membrane_concentration_gradient_by_x"
    ]
    lithium_mem_concentration_gradient_li_co = m_li_co_cl_results_dict[
        "cation_1_membrane_concentration_gradient_by_x"
    ]
    lithium_mem_concentration_gradient_li_al = m_li_al_cl_results_dict[
        "cation_1_membrane_concentration_gradient_by_x"
    ]
    # cobalt_mem_concentration_gradient_co = m_co_cl_results_dict[
    #     "cation_1_membrane_concentration_gradient_by_x"
    # ]
    cobalt_mem_concentration_gradient_li_co = m_li_co_cl_results_dict[
        "cation_2_membrane_concentration_gradient_by_x"
    ]
    # cobalt_mem_concentration_gradient_co_al = m_co_al_cl_results_dict[
    #     "cation_1_membrane_concentration_gradient_by_x"
    # ]
    # aluminum_mem_concentration_gradient_al = m_al_cl_results_dict[
    #     "cation_1_membrane_concentration_gradient_by_x"
    # ]
    aluminum_mem_concentration_gradient_li_al = m_li_al_cl_results_dict[
        "cation_2_membrane_concentration_gradient_by_x"
    ]
    # aluminum_mem_concentration_gradient_co_al = m_co_al_cl_results_dict[
    #     "cation_2_membrane_concentration_gradient_by_x"
    # ]

    # boundary layer flux
    lithium_bl_convection_li = []
    lithium_bl_convection_li_co = []
    lithium_bl_convection_li_al = []
    lithium_bl_diffusion_li = []
    lithium_bl_diffusion_li_co = []
    lithium_bl_diffusion_li_al = []
    lithium_bl_electromigration_li = []
    lithium_bl_electromigration_li_co = []
    lithium_bl_electromigration_li_al = []

    lithium_bl_convection_li_dict = {}
    lithium_bl_convection_li_co_dict = {}
    lithium_bl_convection_li_al_dict = {}
    lithium_bl_diffusion_li_dict = {}
    lithium_bl_diffusion_li_co_dict = {}
    lithium_bl_diffusion_li_al_dict = {}
    lithium_bl_electromigration_li_dict = {}
    lithium_bl_electromigration_li_co_dict = {}
    lithium_bl_electromigration_li_al_dict = {}

    for x in x_axis_values:
        for z in range(len(z_bl_axis_values)):
            x_position = x_axis_values.index(x)

            lithium_bl_convection_li_val = (
                lithium_bl_concentration_li[str(x)][z] * water_flux_li[x_position]
            )
            lithium_bl_convection_li_co_val = (
                lithium_bl_concentration_li_co[str(x)][z] * water_flux_li_co[x_position]
            )
            lithium_bl_convection_li_al_val = (
                lithium_bl_concentration_li_al[str(x)][z] * water_flux_li_al[x_position]
            )

            lithium_bl_diffusion_li_val = (
                -lithium_bl_concentration_gradient_li[str(x)][z] * D_bl_lithium
            )
            lithium_bl_diffusion_li_co_val = (
                -lithium_bl_concentration_gradient_li_co[str(x)][z] * D_bl_lithium
            )
            lithium_bl_diffusion_li_al_val = (
                -lithium_bl_concentration_gradient_li_al[str(x)][z] * D_bl_lithium
            )

            lithium_bl_electromigration_li_val = (
                z_lithium
                * D_bl_lithium
                * lithium_bl_concentration_li[str(x)][z]
                * (
                    (
                        (D_bl_lithium - D_bl_chloride)
                        * z_lithium
                        * lithium_bl_concentration_gradient_li[str(x)][z]
                    )
                    / (
                        (z_lithium * D_bl_lithium - z_chloride * D_bl_chloride)
                        * z_lithium
                        * lithium_bl_concentration_li[str(x)][z]
                    )
                )
            )
            lithium_bl_electromigration_li_co_val = (
                z_lithium
                * D_bl_lithium
                * lithium_bl_concentration_li_co[str(x)][z]
                * (
                    (
                        (
                            (D_bl_lithium - D_bl_chloride)
                            * z_lithium
                            * lithium_bl_concentration_gradient_li_co[str(x)][z]
                        )
                        + (
                            (D_bl_cobalt - D_bl_chloride)
                            * z_cobalt
                            * cobalt_bl_concentration_gradient_li_co[str(x)][z]
                        )
                    )
                    / (
                        (
                            (z_lithium * D_bl_lithium - z_chloride * D_bl_chloride)
                            * z_lithium
                            * lithium_bl_concentration_li_co[str(x)][z]
                        )
                        + (
                            (z_cobalt * D_bl_cobalt - z_chloride * D_bl_chloride)
                            * z_cobalt
                            * cobalt_bl_concentration_li_co[str(x)][z]
                        )
                    )
                )
            )
            lithium_bl_electromigration_li_al_val = (
                z_lithium
                * D_bl_lithium
                * lithium_bl_concentration_li_al[str(x)][z]
                * (
                    (
                        (
                            (D_bl_lithium - D_bl_chloride)
                            * z_lithium
                            * lithium_bl_concentration_gradient_li_al[str(x)][z]
                        )
                        + (
                            (D_bl_aluminum - D_bl_chloride)
                            * z_aluminum
                            * aluminum_bl_concentration_gradient_li_al[str(x)][z]
                        )
                    )
                    / (
                        (
                            (z_lithium * D_bl_lithium - z_chloride * D_bl_chloride)
                            * z_lithium
                            * lithium_bl_concentration_li_al[str(x)][z]
                        )
                        + (
                            (z_aluminum * D_bl_aluminum - z_chloride * D_bl_chloride)
                            * z_aluminum
                            * aluminum_bl_concentration_li_al[str(x)][z]
                        )
                    )
                )
            )

            lithium_bl_convection_li.append(lithium_bl_convection_li_val)
            lithium_bl_convection_li_co.append(lithium_bl_convection_li_co_val)
            lithium_bl_convection_li_al.append(lithium_bl_convection_li_al_val)

            lithium_bl_diffusion_li.append(lithium_bl_diffusion_li_val)
            lithium_bl_diffusion_li_co.append(lithium_bl_diffusion_li_co_val)
            lithium_bl_diffusion_li_al.append(lithium_bl_diffusion_li_al_val)

            lithium_bl_electromigration_li.append(lithium_bl_electromigration_li_val)
            lithium_bl_electromigration_li_co.append(
                lithium_bl_electromigration_li_co_val
            )
            lithium_bl_electromigration_li_al.append(
                lithium_bl_electromigration_li_al_val
            )

        lithium_bl_convection_li_dict[x] = lithium_bl_convection_li
        lithium_bl_convection_li_co_dict[x] = lithium_bl_convection_li_co
        lithium_bl_convection_li_al_dict[x] = lithium_bl_convection_li_al

        lithium_bl_diffusion_li_dict[x] = lithium_bl_diffusion_li
        lithium_bl_diffusion_li_co_dict[x] = lithium_bl_diffusion_li_co
        lithium_bl_diffusion_li_al_dict[x] = lithium_bl_diffusion_li_al

        lithium_bl_electromigration_li_dict[x] = lithium_bl_electromigration_li
        lithium_bl_electromigration_li_co_dict[x] = lithium_bl_electromigration_li_co
        lithium_bl_electromigration_li_al_dict[x] = lithium_bl_electromigration_li_al

        lithium_bl_convection_li = []
        lithium_bl_convection_li_co = []
        lithium_bl_convection_li_al = []
        lithium_bl_diffusion_li = []
        lithium_bl_diffusion_li_co = []
        lithium_bl_diffusion_li_al = []
        lithium_bl_electromigration_li = []
        lithium_bl_electromigration_li_co = []
        lithium_bl_electromigration_li_al = []

    lithium_bl_convection_li_averaged = [
        sum(lithium_bl_convection_li_dict[k]) / len(lithium_bl_convection_li_dict[k])
        for k in lithium_bl_convection_li_dict.keys()
    ]
    lithium_bl_convection_li_co_averaged = [
        sum(lithium_bl_convection_li_co_dict[k])
        / len(lithium_bl_convection_li_co_dict[k])
        for k in lithium_bl_convection_li_co_dict.keys()
    ]
    lithium_bl_convection_li_al_averaged = [
        sum(lithium_bl_convection_li_al_dict[k])
        / len(lithium_bl_convection_li_al_dict[k])
        for k in lithium_bl_convection_li_al_dict.keys()
    ]

    lithium_bl_diffusion_li_averaged = [
        sum(lithium_bl_diffusion_li_dict[k]) / len(lithium_bl_diffusion_li_dict[k])
        for k in lithium_bl_diffusion_li_dict.keys()
    ]
    lithium_bl_diffusion_li_co_averaged = [
        sum(lithium_bl_diffusion_li_co_dict[k])
        / len(lithium_bl_diffusion_li_co_dict[k])
        for k in lithium_bl_diffusion_li_co_dict.keys()
    ]
    lithium_bl_diffusion_li_al_averaged = [
        sum(lithium_bl_diffusion_li_al_dict[k])
        / len(lithium_bl_diffusion_li_al_dict[k])
        for k in lithium_bl_diffusion_li_al_dict.keys()
    ]

    lithium_bl_electromigration_li_averaged = [
        sum(lithium_bl_electromigration_li_dict[k])
        / len(lithium_bl_electromigration_li_dict[k])
        for k in lithium_bl_electromigration_li_dict.keys()
    ]
    lithium_bl_electromigration_li_co_averaged = [
        sum(lithium_bl_electromigration_li_co_dict[k])
        / len(lithium_bl_electromigration_li_co_dict[k])
        for k in lithium_bl_electromigration_li_co_dict.keys()
    ]
    lithium_bl_electromigration_li_al_averaged = [
        sum(lithium_bl_electromigration_li_al_dict[k])
        / len(lithium_bl_electromigration_li_al_dict[k])
        for k in lithium_bl_electromigration_li_al_dict.keys()
    ]

    # membrane flux
    lithium_mem_convection_li = []
    lithium_mem_convection_li_co = []
    lithium_mem_convection_li_al = []
    lithium_mem_diffusion_li = []
    lithium_mem_diffusion_li_co = []
    lithium_mem_diffusion_li_al = []
    lithium_mem_electromigration_li = []
    lithium_mem_electromigration_li_co = []
    lithium_mem_electromigration_li_al = []

    lithium_mem_convection_li_dict = {}
    lithium_mem_convection_li_co_dict = {}
    lithium_mem_convection_li_al_dict = {}
    lithium_mem_diffusion_li_dict = {}
    lithium_mem_diffusion_li_co_dict = {}
    lithium_mem_diffusion_li_al_dict = {}
    lithium_mem_electromigration_li_dict = {}
    lithium_mem_electromigration_li_co_dict = {}
    lithium_mem_electromigration_li_al_dict = {}

    for x in x_axis_values:
        for z in range(len(z_mem_axis_values)):
            x_position = x_axis_values.index(x)

            lithium_mem_convection_li_val = (
                lithium_mem_concentration_li[str(x)][z] * water_flux_li[x_position]
            )
            lithium_mem_convection_li_co_val = (
                lithium_mem_concentration_li_co[str(x)][z]
                * water_flux_li_co[x_position]
            )
            lithium_mem_convection_li_al_val = (
                lithium_mem_concentration_li_al[str(x)][z]
                * water_flux_li_al[x_position]
            )

            lithium_mem_diffusion_li_val = (
                -lithium_mem_concentration_gradient_li[str(x)][z] * D_mem_lithium
            )
            lithium_mem_diffusion_li_co_val = (
                -lithium_mem_concentration_gradient_li_co[str(x)][z] * D_mem_lithium
            )
            lithium_mem_diffusion_li_al_val = (
                -lithium_mem_concentration_gradient_li_al[str(x)][z] * D_mem_lithium
            )

            lithium_mem_electromigration_li_val = (
                z_lithium
                * D_mem_lithium
                * lithium_mem_concentration_li[str(x)][z]
                * (
                    (
                        (
                            (D_mem_lithium - D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_gradient_li[str(x)][z]
                        )
                        + (water_flux_li[x_position] * chi)
                    )
                    / (
                        (z_lithium * D_mem_lithium - z_chloride * D_mem_chloride)
                        * z_lithium
                        * lithium_mem_concentration_li[str(x)][z]
                    )
                )
            )
            lithium_mem_electromigration_li_co_val = (
                z_lithium
                * D_mem_lithium
                * lithium_mem_concentration_li_co[str(x)][z]
                * (
                    (
                        (
                            (D_mem_lithium - D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_gradient_li_co[str(x)][z]
                        )
                        + (
                            (D_mem_cobalt - D_mem_chloride)
                            * z_cobalt
                            * cobalt_mem_concentration_gradient_li_co[str(x)][z]
                        )
                        + (water_flux_li_co[x_position] * chi)
                    )
                    / (
                        (
                            (z_lithium * D_mem_lithium - z_chloride * D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_li_co[str(x)][z]
                        )
                        + (
                            (z_cobalt * D_mem_cobalt - z_chloride * D_mem_chloride)
                            * z_cobalt
                            * cobalt_mem_concentration_li_co[str(x)][z]
                        )
                    )
                )
            )
            lithium_mem_electromigration_li_al_val = (
                z_lithium
                * D_mem_lithium
                * lithium_mem_concentration_li_al[str(x)][z]
                * (
                    (
                        (
                            (D_mem_lithium - D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_gradient_li_al[str(x)][z]
                        )
                        + (
                            (D_mem_aluminum - D_mem_chloride)
                            * z_aluminum
                            * aluminum_mem_concentration_gradient_li_al[str(x)][z]
                        )
                        + (water_flux_li_al[x_position] * chi)
                    )
                    / (
                        (
                            (z_lithium * D_mem_lithium - z_chloride * D_mem_chloride)
                            * z_lithium
                            * lithium_mem_concentration_li_al[str(x)][z]
                        )
                        + (
                            (z_aluminum * D_mem_aluminum - z_chloride * D_mem_chloride)
                            * z_aluminum
                            * aluminum_mem_concentration_li_al[str(x)][z]
                        )
                    )
                )
            )

            lithium_mem_convection_li.append(lithium_mem_convection_li_val)
            lithium_mem_convection_li_co.append(lithium_mem_convection_li_co_val)
            lithium_mem_convection_li_al.append(lithium_mem_convection_li_al_val)

            lithium_mem_diffusion_li.append(lithium_mem_diffusion_li_val)
            lithium_mem_diffusion_li_co.append(lithium_mem_diffusion_li_co_val)
            lithium_mem_diffusion_li_al.append(lithium_mem_diffusion_li_al_val)

            lithium_mem_electromigration_li.append(lithium_mem_electromigration_li_val)
            lithium_mem_electromigration_li_co.append(
                lithium_mem_electromigration_li_co_val
            )
            lithium_mem_electromigration_li_al.append(
                lithium_mem_electromigration_li_al_val
            )

        lithium_mem_convection_li_dict[x] = lithium_mem_convection_li
        lithium_mem_convection_li_co_dict[x] = lithium_mem_convection_li_co
        lithium_mem_convection_li_al_dict[x] = lithium_mem_convection_li_al

        lithium_mem_diffusion_li_dict[x] = lithium_mem_diffusion_li
        lithium_mem_diffusion_li_co_dict[x] = lithium_mem_diffusion_li_co
        lithium_mem_diffusion_li_al_dict[x] = lithium_mem_diffusion_li_al

        lithium_mem_electromigration_li_dict[x] = lithium_mem_electromigration_li
        lithium_mem_electromigration_li_co_dict[x] = lithium_mem_electromigration_li_co
        lithium_mem_electromigration_li_al_dict[x] = lithium_mem_electromigration_li_al

        lithium_mem_convection_li = []
        lithium_mem_convection_li_co = []
        lithium_mem_convection_li_al = []
        lithium_mem_diffusion_li = []
        lithium_mem_diffusion_li_co = []
        lithium_mem_diffusion_li_al = []
        lithium_mem_electromigration_li = []
        lithium_mem_electromigration_li_co = []
        lithium_mem_electromigration_li_al = []

    lithium_mem_convection_li_averaged = [
        sum(lithium_mem_convection_li_dict[k]) / len(lithium_mem_convection_li_dict[k])
        for k in lithium_mem_convection_li_dict.keys()
    ]
    lithium_mem_convection_li_co_averaged = [
        sum(lithium_mem_convection_li_co_dict[k])
        / len(lithium_mem_convection_li_co_dict[k])
        for k in lithium_mem_convection_li_co_dict.keys()
    ]
    lithium_mem_convection_li_al_averaged = [
        sum(lithium_mem_convection_li_al_dict[k])
        / len(lithium_mem_convection_li_al_dict[k])
        for k in lithium_mem_convection_li_al_dict.keys()
    ]

    lithium_mem_diffusion_li_averaged = [
        sum(lithium_mem_diffusion_li_dict[k]) / len(lithium_mem_diffusion_li_dict[k])
        for k in lithium_mem_diffusion_li_dict.keys()
    ]
    lithium_mem_diffusion_li_co_averaged = [
        sum(lithium_mem_diffusion_li_co_dict[k])
        / len(lithium_mem_diffusion_li_co_dict[k])
        for k in lithium_mem_diffusion_li_co_dict.keys()
    ]
    lithium_mem_diffusion_li_al_averaged = [
        sum(lithium_mem_diffusion_li_al_dict[k])
        / len(lithium_mem_diffusion_li_al_dict[k])
        for k in lithium_mem_diffusion_li_al_dict.keys()
    ]

    lithium_mem_electromigration_li_averaged = [
        sum(lithium_mem_electromigration_li_dict[k])
        / len(lithium_mem_electromigration_li_dict[k])
        for k in lithium_mem_electromigration_li_dict.keys()
    ]
    lithium_mem_electromigration_li_co_averaged = [
        sum(lithium_mem_electromigration_li_co_dict[k])
        / len(lithium_mem_electromigration_li_co_dict[k])
        for k in lithium_mem_electromigration_li_co_dict.keys()
    ]
    lithium_mem_electromigration_li_al_averaged = [
        sum(lithium_mem_electromigration_li_al_dict[k])
        / len(lithium_mem_electromigration_li_al_dict[k])
        for k in lithium_mem_electromigration_li_al_dict.keys()
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5))
    ax1.plot(x_axis_values, lithium_bl_convection_li_averaged, "r-", linewidth=2)
    ax1.plot(x_axis_values, lithium_bl_convection_li_co_averaged, "r--", linewidth=2)
    ax1.plot(x_axis_values, lithium_bl_convection_li_al_averaged, "r-.", linewidth=2)

    ax1.plot(x_axis_values, lithium_bl_diffusion_li_averaged, "b-", linewidth=2)
    ax1.plot(x_axis_values, lithium_bl_diffusion_li_co_averaged, "b--", linewidth=2)
    ax1.plot(x_axis_values, lithium_bl_diffusion_li_al_averaged, "b-.", linewidth=2)

    ax1.plot(x_axis_values, lithium_bl_electromigration_li_averaged, "g-", linewidth=2)
    ax1.plot(
        x_axis_values,
        lithium_bl_electromigration_li_co_averaged,
        "g--",
        linewidth=2,
    )
    ax1.plot(
        x_axis_values,
        lithium_bl_electromigration_li_al_averaged,
        "g-.",
        linewidth=2,
    )

    ax2.plot(x_axis_values, lithium_mem_convection_li_averaged, "r-", linewidth=2)
    ax2.plot(x_axis_values, lithium_mem_convection_li_co_averaged, "r--", linewidth=2)
    ax2.plot(x_axis_values, lithium_mem_convection_li_al_averaged, "r-.", linewidth=2)

    ax2.plot(x_axis_values, lithium_mem_diffusion_li_averaged, "b-", linewidth=2)
    ax2.plot(x_axis_values, lithium_mem_diffusion_li_co_averaged, "b--", linewidth=2)
    ax2.plot(x_axis_values, lithium_mem_diffusion_li_al_averaged, "b-.", linewidth=2)

    ax2.plot(x_axis_values, lithium_mem_electromigration_li_averaged, "g-", linewidth=2)
    ax2.plot(
        x_axis_values,
        lithium_mem_electromigration_li_co_averaged,
        "g--",
        linewidth=2,
    )
    ax2.plot(
        x_axis_values,
        lithium_mem_electromigration_li_al_averaged,
        "g-.",
        linewidth=2,
    )

    # legend points
    ax1.plot([], [], "k-", linewidth=2, label="LiCl")
    ax1.plot([], [], "k--", linewidth=2, label="LiCl + CoCl$_2$")
    ax1.plot([], [], "k-.", linewidth=2, label="LiCl + AlCl$_3$")
    ax1.plot([], [], marker="None", linestyle="None", label="Flux (color)")
    ax1.plot([], [], "rs", markersize=8, label="Convection")
    ax1.plot([], [], "bs", markersize=8, label="Diffusion")
    ax1.plot([], [], "gs", markersize=8, label="Electromigration")
    ax1.legend(loc="best", title="Solution (linestyle)")

    plt.suptitle("Lithium Flux Contributions (Averaged across Thicknesses)")
    ax1.set_title(
        "Boundary Layer",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Flux (mol m$^{-2}$ h$^{-1}$)", fontsize=14, fontweight="bold")
    ax1.tick_params(direction="in", top=True, right=True, labelsize=14)

    ax2.set_title(
        "Membrane",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Module Length (m)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Flux (mol m$^{-2}$ h$^{-1}$)", fontsize=14, fontweight="bold")
    ax2.tick_params(direction="in", top=True, right=True, labelsize=14)

    plt.tight_layout()


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
        ax.set_xlabel("Retentate Concentration (mM)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Permeate Concentration (mM)", fontsize=14, fontweight="bold")
        ax.tick_params(direction="in", labelsize=14)
        ax.legend(loc="upper left")

    # plt.tight_layout()


if __name__ == "__main__":
    main()
