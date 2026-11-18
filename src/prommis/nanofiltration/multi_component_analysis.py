#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Model comparison for multi-salt diafiltration.

Author: Molly Dougher
"""

from pathlib import Path

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
from idaes.core.util import to_json, from_json
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

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
    run_single_salt = False
    run_two_salt = False
    run_two_salt_ratios = False
    solve_and_save_models(
        water_flux=0.02,
        run_single_salt=run_single_salt,
        run_two_salt=run_two_salt,
        run_two_salt_ratios=run_two_salt_ratios,
    )

    save_figure_data(flux_values=False, flux_contrib=False, ratio=False)

    flux_plots_equimolar(save_figure=True)
    flux_plots_ratios(save_figure=True)

    plot_flux_contributions(save_figure=True)


def build_model(
    cation_list,
    inlet_concentration,
    default_args,
    NFE_args,
    run_single_salt=False,
    run_two_salt=False,
    run_two_salt_ratio=False,
    initialize_and_solve=True,
    fudge_factor_H_feed=1,
    fudge_factor_H_perm=1,
    water_flux=0.02,
    data_membrane_thickness=1e-7,  # default of 100 nm
    non_Donnan_partition_dict={},
    save=False,
    key=None,
    chloride_multiplier=None,
):
    anion_list, inlet_flow_volume, include_boundary_layer = default_args
    NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness = NFE_args
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
        non_Donnan_partition_dict=non_Donnan_partition_dict,
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
        total_membrane_thickness=data_membrane_thickness,
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
    # if len(cation_list) == 1:
    #     m.fs.membrane.applied_pressure.fix(5)
    # else:
    #     m.fs.membrane.applied_pressure.fix(10)
    m.fs.membrane.feed_flow_volume.fix(inlet_flow_volume["feed"])
    m.fs.membrane.diafiltrate_flow_volume.fix(inlet_flow_volume["diafiltrate"])
    for t in m.fs.membrane.time:
        for j in m.fs.membrane.solutes:
            m.fs.membrane.feed_conc_mol_comp[t, j].fix(inlet_concentration["feed"][j])
            m.fs.membrane.diafiltrate_conc_mol_comp[t, j].fix(
                inlet_concentration["diafiltrate"][j]
            )
    # initial pressure values may need to be tweaked to successfully initialize
    if len(cation_list) == 1:
        if value(m.fs.membrane.feed_ionic_strength[0]) < 35:
            m.fs.membrane.applied_pressure.fix(3)
        elif (value(m.fs.membrane.feed_ionic_strength[0]) >= 35) and (
            value(m.fs.membrane.feed_ionic_strength[0]) < 249
        ):
            m.fs.membrane.applied_pressure.fix(5)
        elif (value(m.fs.membrane.feed_ionic_strength[0]) >= 249) and (
            value(m.fs.membrane.feed_ionic_strength[0]) < 601
        ):
            m.fs.membrane.applied_pressure.fix(10)
        elif value(m.fs.membrane.feed_ionic_strength[0]) >= 601:
            m.fs.membrane.applied_pressure.fix(20)
    elif len(cation_list) > 1:
        if value(m.fs.membrane.feed_ionic_strength[0]) < 51:
            m.fs.membrane.applied_pressure.fix(5)
        elif (value(m.fs.membrane.feed_ionic_strength[0]) >= 51) and (
            value(m.fs.membrane.feed_ionic_strength[0]) < 199
        ):
            m.fs.membrane.applied_pressure.fix(10)
        elif (value(m.fs.membrane.feed_ionic_strength[0]) >= 99) and (
            value(m.fs.membrane.feed_ionic_strength[0]) < 299
        ):
            m.fs.membrane.applied_pressure.fix(15)
        elif value(m.fs.membrane.feed_ionic_strength[0]) >= 299:
            m.fs.membrane.applied_pressure.fix(20)

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

    if len(cation_list) == 2:
        if cation_list[1] == "La":
            m.fs.membrane.scaling_factor[m.fs.membrane.molar_ion_flux] = 1e2

    # initialize membrane model
    if initialize_and_solve:
        initialized_membrane_model = m.fs.membrane.default_initializer(
            multiplier_H_feed=fudge_factor_H_feed,
            multiplier_H_perm=fudge_factor_H_perm,
        )
        initialized_membrane_model.initialize(m.fs.membrane)

        solve_model(m)
        unfix_pressure(m, water_flux=water_flux)
        solve_model(m)

        # check numerical warnings
        dt = DiagnosticsToolbox(m)
        dt.assert_no_numerical_warnings()

        single_salt = run_single_salt
        two_salt = run_two_salt
        two_salt_ratio = run_two_salt_ratio

        key_name = "CONC"

        if save:
            if single_salt:
                fname = f"multi_component_case_studies/single_salt/{key_name}/{key_name}{key}_{cation_list[0]}Cl{chloride_multiplier}_{inlet_concentration['feed'][cation_list[0]]}mM"
            elif two_salt:
                fname = f"multi_component_case_studies/two_salt/{key_name}/{key_name}{key}_{cation_list[0]}{cation_list[1]}Cl{chloride_multiplier}_{inlet_concentration['feed'][cation_list[0]]}mM_{inlet_concentration['feed'][cation_list[1]]}mM"
            elif two_salt_ratio:
                fname = f"multi_component_case_studies/two_salt_ratio/RATIO{key}_{cation_list[0]}{cation_list[1]}Cl{chloride_multiplier}_{inlet_concentration['feed'][cation_list[0]]}mM_{inlet_concentration['feed'][cation_list[1]]}mM"
            to_json(m, fname=fname)

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

    return results


def unfix_pressure(m, water_flux=0.02):
    m.fs.membrane.applied_pressure.unfix()

    def _water_flux_constraint(m):
        return (
            sum(
                m.fs.membrane.volume_flux_water[0, x]
                for x in m.fs.membrane.dimensionless_module_length
                if x != 0
            )
            / (len(m.fs.membrane.dimensionless_module_length) - 1)
            == water_flux
        )

    m.water_flux_constraint = Constraint(rule=_water_flux_constraint)


def solve_and_save_models(
    water_flux=0.02,
    run_single_salt=True,
    run_two_salt=True,
    run_two_salt_ratios=True,
):
    # global variables
    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    diafiltrate = {"Na": 1e-10, "Ca": 1e-10, "La": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    CONC_key = [
        "010",
        "020",
        "030",
        "040",
        "050",
        "075",
        "100",
        "125",
        "150",
        "175",
        "200",
    ]

    if run_single_salt:
        feed = {
            "Na": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
            "Ca": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
            "La": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
        }
        key_list = CONC_key

        Dm_Cl = 2.03  # um2/s
        Dm_over_l_value = 60  # um/s
        l_um = Dm_Cl / Dm_over_l_value  # um
        l_m = l_um / 1e6  # m
        monovalent_phi_star_value = 0.75
        divalent_phi_star_value = 0.4
        trivalent_phi_star_value = 0.0001
        chloride_phi_star_value = 0.05

        for cation in feed.keys():
            if cation == "Na":
                chloride_multiplier = 1
                cation_phi_star_value = monovalent_phi_star_value
            elif cation == "Ca":
                chloride_multiplier = 2
                cation_phi_star_value = divalent_phi_star_value
            elif cation == "La":
                chloride_multiplier = 3
                cation_phi_star_value = trivalent_phi_star_value

            for concentration in feed[cation]:
                if concentration == 10:
                    model = build_model(
                        cation_list=[cation],
                        inlet_concentration={
                            "feed": {
                                cation: concentration,
                                "Cl": chloride_multiplier * concentration,
                            },
                            "diafiltrate": {
                                cation: diafiltrate[cation],
                                "Cl": 1e-10,
                            },
                        },
                        default_args=default_args,
                        NFE_args=NFE_args,
                        run_single_salt=run_single_salt,
                        run_two_salt=run_two_salt,
                        initialize_and_solve=True,
                        fudge_factor_H_feed=1.2,
                        fudge_factor_H_perm=1,
                        water_flux=water_flux,
                        data_membrane_thickness=l_m,
                        non_Donnan_partition_dict={
                            cation: cation_phi_star_value,
                            "Cl": chloride_phi_star_value,
                        },
                        save=True,
                        key=key_list[feed[cation].index(concentration)],
                        chloride_multiplier=chloride_multiplier,
                    )
                else:
                    # model.del_component(model.water_flux_constraint)
                    for t in model.fs.membrane.time:
                        for j in model.fs.membrane.solutes:
                            model.fs.membrane.feed_conc_mol_comp[t, j].fix(
                                concentration
                            )

                    # solve_model(model)
                    # unfix_pressure(model, water_flux=water_flux)
                    solve_model(model)

                    # check numerical warnings
                    dt = DiagnosticsToolbox(model)
                    dt.assert_no_numerical_warnings()

                    key_name = "CONC"
                    key = key_list[feed[cation].index(concentration)]

                    fname = f"multi_component_case_studies/single_salt/{key_name}/{key_name}{key}_{cation}Cl{chloride_multiplier}_{concentration}mM"
                    to_json(model, fname=fname)

    if run_two_salt:
        feed = {
            # high conc. at 1:1 ratios hit pressure limits for 20 lmh bar flux constraint
            "Na_Ca": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
            "Na_La": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
            "Ca_La": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
        }
        key_list = CONC_key

        Dm_Cl = 2.03  # um2/s
        Dm_over_l_value = 60  # um/s
        l_um = Dm_Cl / Dm_over_l_value  # um
        l_m = l_um / 1e6  # m
        monovalent_phi_star_value = 0.75
        divalent_phi_star_value = 0.4
        trivalent_phi_star_value = 0.0001
        chloride_phi_star_value = 0.05

        for salt in feed.keys():
            if salt == "Na_Ca":
                cation_1 = "Na"
                cation_2 = "Ca"
                chloride_multiplier = 3
                cation_1_phi_star_value = monovalent_phi_star_value
                cation_2_phi_star_value = divalent_phi_star_value
            elif salt == "Na_La":
                cation_1 = "Na"
                cation_2 = "La"
                chloride_multiplier = 4
                cation_1_phi_star_value = monovalent_phi_star_value
                cation_2_phi_star_value = trivalent_phi_star_value
            elif salt == "Ca_La":
                cation_1 = "Ca"
                cation_2 = "La"
                chloride_multiplier = 5
                cation_1_phi_star_value = divalent_phi_star_value
                cation_2_phi_star_value = trivalent_phi_star_value

            for concentration in feed[salt]:
                if concentration == 10:
                    model = build_model(
                        cation_list=[cation_1, cation_2],
                        inlet_concentration={
                            "feed": {
                                cation_1: concentration,
                                cation_2: concentration,
                                "Cl": chloride_multiplier * concentration,
                            },
                            "diafiltrate": {
                                cation_1: diafiltrate[cation_1],
                                cation_2: diafiltrate[cation_2],
                                "Cl": 1e-10,
                            },
                        },
                        default_args=default_args,
                        NFE_args=NFE_args,
                        run_single_salt=run_single_salt,
                        run_two_salt=run_two_salt,
                        initialize_and_solve=True,
                        fudge_factor_H_feed=1.2,
                        fudge_factor_H_perm=1,
                        water_flux=water_flux,
                        data_membrane_thickness=l_m,
                        non_Donnan_partition_dict={
                            cation_1: cation_1_phi_star_value,
                            cation_2: cation_2_phi_star_value,
                            "Cl": chloride_phi_star_value,
                        },
                        save=True,
                        key=key_list[feed[salt].index(concentration)],
                        chloride_multiplier=chloride_multiplier,
                    )
                else:
                    for t in model.fs.membrane.time:
                        for j in model.fs.membrane.solutes:
                            model.fs.membrane.feed_conc_mol_comp[t, j].fix(
                                concentration
                            )

                    solve_model(model)

                    # check numerical warnings
                    dt = DiagnosticsToolbox(model)
                    dt.assert_no_numerical_warnings()

                    key_name = "CONC"
                    key = key_list[feed[salt].index(concentration)]

                    fname = f"multi_component_case_studies/two_salt/{key_name}/{key_name}{key}_{cation_1}{cation_2}Cl{chloride_multiplier}_{concentration}mM_{concentration}mM"
                    to_json(model, fname=fname)

    if run_two_salt_ratios:
        cation_1_conc = [10, 20, 30]
        RATIO_key = ["001", "005", "010", "015", "020"]
        ratios = [0.1, 0.5, 1, 1.5, 2]

        Dm_Cl = 2.03  # um2/s
        Dm_over_l_value = 60  # um/s
        l_um = Dm_Cl / Dm_over_l_value  # um
        l_m = l_um / 1e6  # m
        monovalent_phi_star_value = 0.75
        divalent_phi_star_value = 0.4
        trivalent_phi_star_value = 0.0001
        chloride_phi_star_value = 0.05

        cation_1 = "Na"
        cation_2 = "Ca"
        cation_1_chloride_multiplier = 1
        cation_2_chloride_multiplier = 2
        total_chloride_multiplier = (
            cation_1_chloride_multiplier + cation_2_chloride_multiplier
        )
        cation_1_phi_star_value = monovalent_phi_star_value
        cation_2_phi_star_value = divalent_phi_star_value

        for cation_1_concentration in cation_1_conc:
            for ratio in ratios:
                if ratio == 0.1:
                    model = build_model(
                        cation_list=[cation_1, cation_2],
                        inlet_concentration={
                            "feed": {
                                cation_1: cation_1_concentration,
                                cation_2: ratio * cation_1_concentration,
                                "Cl": (
                                    (
                                        cation_1_chloride_multiplier
                                        * cation_1_concentration
                                    )
                                    + (
                                        cation_2_chloride_multiplier
                                        * ratio
                                        * cation_1_concentration
                                    )
                                ),
                            },
                            "diafiltrate": {
                                cation_1: diafiltrate[cation_1],
                                cation_2: diafiltrate[cation_2],
                                "Cl": 1e-10,
                            },
                        },
                        default_args=default_args,
                        NFE_args=NFE_args,
                        run_two_salt_ratio=True,
                        initialize_and_solve=True,
                        fudge_factor_H_feed=1.2,
                        fudge_factor_H_perm=1,
                        water_flux=water_flux,
                        data_membrane_thickness=l_m,
                        non_Donnan_partition_dict={
                            cation_1: cation_1_phi_star_value,
                            cation_2: cation_2_phi_star_value,
                            "Cl": chloride_phi_star_value,
                        },
                        save=True,
                        key=RATIO_key[ratios.index(ratio)],
                        chloride_multiplier=total_chloride_multiplier,
                    )
                else:
                    for t in model.fs.membrane.time:
                        model.fs.membrane.feed_conc_mol_comp[t, cation_2].fix(
                            ratio * cation_1_concentration
                        )
                        model.fs.membrane.feed_conc_mol_comp[t, "Cl"].fix(
                            (
                                (cation_1_chloride_multiplier * cation_1_concentration)
                                + (
                                    cation_2_chloride_multiplier
                                    * ratio
                                    * cation_1_concentration
                                )
                            )
                        )

                    solve_model(model)

                    # check numerical warnings
                    dt = DiagnosticsToolbox(model)
                    dt.assert_no_numerical_warnings()

                    key_name = "RATIO"
                    key = RATIO_key[ratios.index(ratio)]

                    fname = f"multi_component_case_studies/two_salt_ratio/RATIO{key}_{cation_1}{cation_2}Cl{total_chloride_multiplier}_{cation_1_concentration}mM_{ratio * cation_1_concentration}mM"
                    to_json(model, fname=fname)


def calculate_spread(list):

    return np.array(
        [
            [np.average(list) - min(list)],
            [max(list) - np.average(list)],
        ]
    )


def get_model_averages(model, solute):
    local_actual_sieving = []
    local_observed_sieving = []
    local_observed_rejection = []
    local_actual_rejection = []
    local_flux = []
    local_H_feed = []
    local_H_perm = []
    local_Donnan_potential_feed = []
    local_Donnan_potential_perm = []

    for t in model.fs.membrane.time:
        for x in model.fs.membrane.dimensionless_module_length:
            if x != 0:
                local_actual_sieving.append(
                    value(
                        model.fs.membrane.local_actual_sieving_coefficient[t, x, solute]
                    )
                )
                local_observed_sieving.append(
                    value(
                        model.fs.membrane.local_observed_sieving_coefficient[
                            t, x, solute
                        ]
                    )
                )
                local_observed_rejection.append(
                    value(
                        model.fs.membrane.local_observed_rejection_percent[t, x, solute]
                    )
                )
                local_actual_rejection.append(
                    value(
                        model.fs.membrane.local_actual_rejection_percent[t, x, solute]
                    )
                )
                local_flux.append(value(model.fs.membrane.molar_ion_flux[t, x, solute]))
                local_H_feed.append(
                    value(
                        model.fs.membrane.local_partition_coefficient_feed_side[
                            t, x, solute
                        ]
                    )
                )
                local_H_perm.append(
                    value(
                        model.fs.membrane.local_partition_coefficient_permeate_side[
                            t, x, solute
                        ]
                    )
                )
                local_Donnan_potential_feed.append(
                    value(model.fs.membrane.Donnan_potential_feed_side[t, x])
                )
                local_Donnan_potential_perm.append(
                    value(model.fs.membrane.Donnan_potential_permeate_side[t, x])
                )

    avg_local_actual_sieving = np.average(local_actual_sieving)
    spread_local_actual_sieving = calculate_spread(local_actual_sieving)

    avg_local_observed_sieving = np.average(local_observed_sieving)
    spread_local_observed_sieving = calculate_spread(local_observed_sieving)

    avg_local_observed_rejection = np.average(local_observed_rejection)
    spread_local_observed_rejection = calculate_spread(local_observed_rejection)

    avg_local_actual_rejection = np.average(local_actual_rejection)
    spread_local_actual_rejection = calculate_spread(local_actual_rejection)

    avg_local_flux = np.average(local_flux)
    spread_local_flux = calculate_spread(local_flux)

    avg_local_H_feed = np.average(local_H_feed)
    spread_local_H_feed = calculate_spread(local_H_feed)

    avg_local_H_perm = np.average(local_H_perm)
    spread_local_H_perm = calculate_spread(local_H_perm)

    avg_local_Donnan_potential_feed = np.average(local_Donnan_potential_feed)
    spread_local_Donnan_potential_feed = calculate_spread(local_Donnan_potential_feed)

    avg_local_Donnan_potential_perm = np.average(local_Donnan_potential_perm)
    spread_local_Donnan_potential_perm = calculate_spread(local_Donnan_potential_perm)

    info_dict = {
        "local_actual_sieving": {
            "avg": avg_local_actual_sieving,
            "spread": spread_local_actual_sieving,
        },
        "local_observed_sieving": {
            "avg": avg_local_observed_sieving,
            "spread": spread_local_observed_sieving,
        },
        "local_observed_rejection": {
            "avg": avg_local_observed_rejection,
            "spread": spread_local_observed_rejection,
        },
        "local_actual_rejection": {
            "avg": avg_local_actual_rejection,
            "spread": spread_local_actual_rejection,
        },
        "local_flux": {"avg": avg_local_flux, "spread": spread_local_flux},
        "local_H_feed": {"avg": avg_local_H_feed, "spread": spread_local_H_feed},
        "local_H_perm": {"avg": avg_local_H_perm, "spread": spread_local_H_perm},
        "local_Donnan_potential_feed": {
            "avg": avg_local_Donnan_potential_feed,
            "spread": spread_local_Donnan_potential_feed,
        },
        "local_Donnan_potential_perm": {
            "avg": avg_local_Donnan_potential_perm,
            "spread": spread_local_Donnan_potential_perm,
        },
    }

    return info_dict


def get_model_averages_flux(model, solute):
    flux = []

    convection_bl_by_x = []
    convection_bl_dict_by_x = {}
    diffusion_bl_by_x = []
    diffusion_bl_dict_by_x = {}
    electromigration_bl_by_x = []
    electromigration_bl_dict_by_x = {}

    convection_mem_by_x = []
    convection_mem_dict_by_x = {}
    diffusion_mem_by_x = []
    diffusion_mem_dict_by_x = {}
    electromigration_mem_by_x = []
    electromigration_mem_dict_by_x = {}

    for t in model.fs.membrane.time:
        for x in model.fs.membrane.dimensionless_module_length:
            if x != 0:
                flux.append(value(model.fs.membrane.molar_ion_flux[t, x, solute]))
                for z_bl in model.fs.membrane.dimensionless_boundary_layer_thickness:
                    if z_bl != 0:
                        convection_bl_by_x.append(
                            value(
                                model.fs.membrane.boundary_layer_convective_flux[
                                    0, x, z_bl, solute
                                ]
                            )
                        )
                        diffusion_bl_by_x.append(
                            value(
                                model.fs.membrane.boundary_layer_diffusive_flux[
                                    0, x, z_bl, solute
                                ]
                            )
                        )
                        electromigration_bl_by_x.append(
                            value(
                                model.fs.membrane.boundary_layer_electromigrative_flux[
                                    0, x, z_bl, solute
                                ]
                            )
                        )

                convection_bl_dict_by_x[f"{x}"] = convection_bl_by_x
                diffusion_bl_dict_by_x[f"{x}"] = diffusion_bl_by_x
                electromigration_bl_dict_by_x[f"{x}"] = electromigration_bl_by_x

                convection_bl_by_x = []
                diffusion_bl_by_x = []
                electromigration_bl_by_x = []

                for z_mem in model.fs.membrane.dimensionless_membrane_thickness:
                    if z_mem != 0:
                        convection_mem_by_x.append(
                            value(
                                model.fs.membrane.membrane_convective_flux[
                                    0, x, z_mem, solute
                                ]
                            )
                        )
                        diffusion_mem_by_x.append(
                            value(
                                model.fs.membrane.membrane_diffusive_flux[
                                    0, x, z_mem, solute
                                ]
                            )
                        )
                        electromigration_mem_by_x.append(
                            value(
                                model.fs.membrane.membrane_electromigrative_flux[
                                    0, x, z_mem, solute
                                ]
                            )
                        )

                convection_mem_dict_by_x[f"{x}"] = convection_mem_by_x
                diffusion_mem_dict_by_x[f"{x}"] = diffusion_mem_by_x
                electromigration_mem_dict_by_x[f"{x}"] = electromigration_mem_by_x

                convection_mem_by_x = []
                diffusion_mem_by_x = []
                electromigration_mem_by_x = []

    avg_flux = np.average(flux)
    spread_flux = calculate_spread(flux)

    bl_convection_averaged_over_z = [
        sum(convection_bl_dict_by_x[k]) / len(convection_bl_dict_by_x[k])
        for k in convection_bl_dict_by_x.keys()
    ]
    avg_bl_convection = np.average(bl_convection_averaged_over_z)
    spread_bl_convection = calculate_spread(bl_convection_averaged_over_z)
    avg_bl_convection_percent = (
        np.average(bl_convection_averaged_over_z) / np.average(flux) * 100
    )

    bl_diffusion_averaged_over_z = [
        sum(diffusion_bl_dict_by_x[k]) / len(diffusion_bl_dict_by_x[k])
        for k in diffusion_bl_dict_by_x.keys()
    ]
    avg_bl_diffusion = np.average(bl_diffusion_averaged_over_z)
    spread_bl_diffusion = calculate_spread(bl_diffusion_averaged_over_z)
    avg_bl_diffusion_percent = (
        np.average(bl_diffusion_averaged_over_z) / np.average(flux) * 100
    )

    bl_electromigration_averaged_over_z = [
        sum(electromigration_bl_dict_by_x[k]) / len(electromigration_bl_dict_by_x[k])
        for k in electromigration_bl_dict_by_x.keys()
    ]
    avg_bl_electromigration = np.average(bl_electromigration_averaged_over_z)
    spread_bl_electromigration = calculate_spread(bl_electromigration_averaged_over_z)
    avg_bl_electromigration_percent = (
        np.average(bl_electromigration_averaged_over_z) / np.average(flux) * 100
    )

    mem_convection_averaged_over_z = [
        sum(convection_mem_dict_by_x[k]) / len(convection_mem_dict_by_x[k])
        for k in convection_mem_dict_by_x.keys()
    ]
    avg_mem_convection = np.average(mem_convection_averaged_over_z)
    spread_mem_convection = calculate_spread(mem_convection_averaged_over_z)
    avg_mem_convection_percent = (
        np.average(mem_convection_averaged_over_z) / np.average(flux) * 100
    )

    mem_diffusion_averaged_over_z = [
        sum(diffusion_mem_dict_by_x[k]) / len(diffusion_mem_dict_by_x[k])
        for k in diffusion_mem_dict_by_x.keys()
    ]
    avg_mem_diffusion = np.average(mem_diffusion_averaged_over_z)
    spread_mem_diffusion = calculate_spread(mem_diffusion_averaged_over_z)
    avg_mem_diffusion_percent = (
        np.average(mem_diffusion_averaged_over_z) / np.average(flux) * 100
    )

    mem_electromigration_averaged_over_z = [
        sum(electromigration_mem_dict_by_x[k]) / len(electromigration_mem_dict_by_x[k])
        for k in electromigration_mem_dict_by_x.keys()
    ]
    avg_mem_electromigration = np.average(mem_electromigration_averaged_over_z)
    spread_mem_electromigration = calculate_spread(mem_electromigration_averaged_over_z)
    avg_mem_electromigration_percent = (
        np.average(mem_electromigration_averaged_over_z) / np.average(flux) * 100
    )

    info_dict = {
        "flux": {"avg": avg_flux, "spread": spread_flux},
        "bl_convection": {
            "avg": avg_bl_convection,
            "spread": spread_bl_convection,
            "percent": avg_bl_convection_percent,
        },
        "bl_diffusion": {
            "avg": avg_bl_diffusion,
            "spread": spread_bl_diffusion,
            "percent": avg_bl_diffusion_percent,
        },
        "bl_electromigration": {
            "avg": avg_bl_electromigration,
            "spread": spread_bl_electromigration,
            "percent": avg_bl_electromigration_percent,
        },
        "mem_convection": {
            "avg": avg_mem_convection,
            "spread": spread_mem_convection,
            "percent": avg_mem_convection_percent,
        },
        "mem_diffusion": {
            "avg": avg_mem_diffusion,
            "spread": spread_mem_diffusion,
            "percent": avg_mem_diffusion_percent,
        },
        "mem_electromigration": {
            "avg": avg_mem_electromigration,
            "spread": spread_mem_electromigration,
            "percent": avg_mem_electromigration_percent,
        },
    }

    return info_dict


def save_figure_data(flux_values=True, flux_contrib=True, ratio=True):
    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    model_folder_1 = Path("multi_component_case_studies/single_salt/CONC")
    # 46 characters (0-45) make up folder name before model name
    # multi_component_case_studies/single_salt/CONC/
    model_folder_2 = Path("multi_component_case_studies/two_salt/CONC")
    # 43 characters (0-42) make up folder name before model name
    # multi_component_case_studies/two_salt/CONC/

    case_study_list_1 = [file for file in model_folder_1.iterdir()]
    case_study_list_1.sort()
    case_study_list_2 = [file for file in model_folder_2.iterdir()]
    case_study_list_2.sort()

    if flux_values:
        case_studies = {
            "single": case_study_list_1,
            "two": case_study_list_2,
        }

        single_salt_data_dict = {
            "Na": {"x": [], "y": [], "y_err": []},
            "Ca": {"x": [], "y": [], "y_err": []},
            "La": {"x": [], "y": [], "y_err": []},
        }
        two_salt_data_dict = {
            "Na_Ca": {
                "Na": {"x": [], "y": [], "y_err": []},
                "Ca": {"x": [], "y": [], "y_err": []},
            },
            "Na_La": {
                "Na": {"x": [], "y": [], "y_err": []},
                "La": {"x": [], "y": [], "y_err": []},
            },
            "Ca_La": {
                "Ca": {"x": [], "y": [], "y_err": []},
                "La": {"x": [], "y": [], "y_err": []},
            },
        }

        for type, case_study_files in case_studies.items():
            for case_study in case_study_files:
                if type == "single":
                    cation = str(case_study)[54:56]
                    chloride_multiplier = float(str(case_study)[58])
                    concentration = float(50)  # mM
                    cation_list = [cation]
                    inlet_concentration = {
                        "feed": {
                            cation: concentration,
                            "Cl": chloride_multiplier * concentration,
                        },
                        "diafiltrate": {
                            cation: 1e-10,
                            "Cl": 1e-10,
                        },
                    }

                elif type == "two":
                    cation_1 = str(case_study)[51:53]
                    cation_2 = str(case_study)[53:55]
                    chloride_multiplier = float(str(case_study)[57])
                    concentration = float(50)  # mM
                    cation_list = [cation_1, cation_2]
                    inlet_concentration = {
                        "feed": {
                            cation_1: concentration,
                            cation_2: concentration,
                            "Cl": chloride_multiplier * concentration,
                        },
                        "diafiltrate": {
                            cation_1: 1e-10,
                            cation_2: 1e-10,
                            "Cl": 1e-10,
                        },
                    }

                    if cation_1 == "Na" and cation_2 == "Ca":
                        salt_key = "Na_Ca"
                    elif cation_1 == "Na" and cation_2 == "La":
                        salt_key = "Na_La"
                    elif cation_1 == "Ca" and cation_2 == "La":
                        salt_key = "Ca_La"

                model = build_model(
                    cation_list=cation_list,
                    inlet_concentration=inlet_concentration,
                    default_args=default_args,
                    NFE_args=NFE_args,
                    initialize_and_solve=False,
                )
                from_json(model, fname=case_study)

                for solute in model.fs.membrane.cations:
                    average_variable_dict = get_model_averages(model, solute)

                    x_value = value(
                        model.fs.membrane.retentate_conc_mol_comp[0, 0, solute]
                    )
                    y_data = average_variable_dict["local_flux"]["avg"]
                    y_err = average_variable_dict["local_flux"]["spread"]
                    y_err = y_err.tolist()

                    if type == "single":
                        single_salt_data_dict[solute]["x"].append(x_value)
                        single_salt_data_dict[solute]["y"].append(y_data)
                        single_salt_data_dict[solute]["y_err"].append(y_err)
                    elif type == "two":
                        two_salt_data_dict[salt_key][solute]["x"].append(x_value)
                        two_salt_data_dict[salt_key][solute]["y"].append(y_data)
                        two_salt_data_dict[salt_key][solute]["y_err"].append(y_err)

        with open(
            "figure_data/flux_versus_cation_concentration_single_salt_data.json", "w"
        ) as file:
            json.dump(single_salt_data_dict, file)
        with open(
            "figure_data/flux_versus_cation_concentration_two_salt_data.json", "w"
        ) as file:
            json.dump(two_salt_data_dict, file)

    if flux_contrib:
        case_studies = {
            "single": case_study_list_1,
        }

        sodium_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        cl_sodium_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        calcium_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        cl_calcium_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        lanthanum_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        cl_lanthanum_bl_avg = {
            "Convection": [],
            "Diffusion": [],
            "Electromigration": [],
        }

        sodium_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        cl_sodium_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        calcium_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        cl_calcium_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        lanthanum_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
        cl_lanthanum_mem_avg = {
            "Convection": [],
            "Diffusion": [],
            "Electromigration": [],
        }

        concentrations = [
            "10",
            "20",
            "30",
            "40",
            "50",
            "75",
            "100",
            "125",
            "150",
            "175",
            "200",
        ]

        x_values = concentrations

        for type, case_study_files in case_studies.items():
            for case_study in case_study_files:
                if type == "single":
                    cation = str(case_study)[54:56]
                    chloride_multiplier = float(str(case_study)[58])
                    concentration = float(50)  # mM
                    cation_list = [cation]
                    inlet_concentration = {
                        "feed": {
                            cation: concentration,
                            "Cl": chloride_multiplier * concentration,
                        },
                        "diafiltrate": {
                            cation: 1e-10,
                            "Cl": 1e-10,
                        },
                    }

                model = build_model(
                    cation_list=cation_list,
                    inlet_concentration=inlet_concentration,
                    default_args=default_args,
                    NFE_args=NFE_args,
                    initialize_and_solve=False,
                )
                from_json(model, fname=case_study)

                for solute in model.fs.membrane.solutes:
                    bl_dict_avg = {
                        "Convection": [],
                        "Diffusion": [],
                        "Electromigration": [],
                    }
                    mem_dict_avg = {
                        "Convection": [],
                        "Diffusion": [],
                        "Electromigration": [],
                    }

                    average_variable_dict = get_model_averages_flux(model, solute)

                    avg_bl_convection = average_variable_dict["bl_convection"]["avg"]
                    avg_bl_diffusion = average_variable_dict["bl_diffusion"]["avg"]
                    avg_bl_electromigration = average_variable_dict[
                        "bl_electromigration"
                    ]["avg"]
                    avg_mem_convection = average_variable_dict["mem_convection"]["avg"]
                    avg_mem_diffusion = average_variable_dict["mem_diffusion"]["avg"]
                    avg_mem_electromigration = average_variable_dict[
                        "mem_electromigration"
                    ]["avg"]

                    if solute == "Na":
                        bl_dict_avg = sodium_bl_avg
                        mem_dict_avg = sodium_mem_avg
                    elif solute == "Ca":
                        bl_dict_avg = calcium_bl_avg
                        mem_dict_avg = calcium_mem_avg
                    elif solute == "La":
                        bl_dict_avg = lanthanum_bl_avg
                        mem_dict_avg = lanthanum_mem_avg
                    elif solute == "Cl":
                        if model.fs.membrane.cations.at(1) == "Na":
                            bl_dict_avg = cl_sodium_bl_avg
                            mem_dict_avg = cl_sodium_mem_avg
                        elif model.fs.membrane.cations.at(1) == "Ca":
                            bl_dict_avg = cl_calcium_bl_avg
                            mem_dict_avg = cl_calcium_mem_avg
                        elif model.fs.membrane.cations.at(1) == "La":
                            bl_dict_avg = cl_lanthanum_bl_avg
                            mem_dict_avg = cl_lanthanum_mem_avg

                    bl_dict_avg["Convection"].append(avg_bl_convection)
                    bl_dict_avg["Diffusion"].append(avg_bl_diffusion)
                    bl_dict_avg["Electromigration"].append(avg_bl_electromigration)

                    mem_dict_avg["Convection"].append(avg_mem_convection)
                    mem_dict_avg["Diffusion"].append(avg_mem_diffusion)
                    mem_dict_avg["Electromigration"].append(avg_mem_electromigration)

        sodium_bl_data = sodium_bl_avg
        cl_sodium_bl_data = cl_sodium_bl_avg
        # calcium_bl_data = calcium_bl_avg
        # cl_calcium_bl_data = cl_calcium_bl_avg
        # lanthanum_bl_data = lanthanum_bl_avg
        # cl_lanthanum_bl_data = cl_lanthanum_bl_avg

        sodium_mem_data = sodium_mem_avg
        cl_sodium_mem_data = cl_sodium_mem_avg
        # calcium_mem_data = calcium_mem_avg
        # cl_calcium_mem_data = cl_calcium_mem_avg
        # lanthanum_mem_data = lanthanum_mem_avg
        # cl_lanthanum_mem_data = cl_lanthanum_mem_avg

        sodium_bl_df = pd.DataFrame(sodium_bl_data, index=x_values)
        cl_sodium_bl_df = pd.DataFrame(cl_sodium_bl_data, index=x_values)
        # calcium_bl_df = pd.DataFrame(calcium_bl_data, index=x_values)
        # cl_calcium_bl_df = pd.DataFrame(cl_calcium_bl_data, index=x_values)
        # lanthanum_bl_df = pd.DataFrame(lanthanum_bl_data, index=x_values)
        # cl_lanthanum_bl_df = pd.DataFrame(cl_lanthanum_bl_data, index=x_values)

        sodium_mem_df = pd.DataFrame(sodium_mem_data, index=x_values)
        cl_sodium_mem_df = pd.DataFrame(cl_sodium_mem_data, index=x_values)
        # calcium_mem_df = pd.DataFrame(calcium_mem_data, index=x_values)
        # cl_calcium_mem_df = pd.DataFrame(cl_calcium_mem_data, index=x_values)
        # lanthanum_mem_df = pd.DataFrame(lanthanum_mem_data, index=x_values)
        # cl_lanthanum_mem_df = pd.DataFrame(cl_lanthanum_mem_data, index=x_values)

        sodium_bl_df.to_json(
            f"figure_data/total_flux_breakdown_single_salt_cation_Na_bl"
        )
        cl_sodium_bl_df.to_json(
            f"figure_data/total_flux_breakdown_single_salt_anion_NaCl_bl"
        )

        sodium_mem_df.to_json(
            f"figure_data/total_flux_breakdown_single_salt_cation_Na_mem"
        )
        cl_sodium_mem_df.to_json(
            f"figure_data/total_flux_breakdown_single_salt_anion_NaCl_mem"
        )

    if ratio:
        RATIO_key = ["001", "005", "010", "015", "020"]
        ratios = [0.1, 0.5, 1, 1.5, 2]

        model_folder_2 = Path("multi_component_case_studies/two_salt_ratio")
        # 44 characters (0-43) make up folder name before model name
        # multi_component_case_studies/two_salt_ratio/

        case_study_list_2 = [file for file in model_folder_2.iterdir()]
        case_study_list_2.sort()

        Na_Ca_data_dict = {
            "Na_10mM": {
                "Na": {"x": [], "y": [], "y_err": []},
                "Ca": {"x": [], "y": [], "y_err": []},
            },
            "Na_20mM": {
                "Na": {"x": [], "y": [], "y_err": []},
                "Ca": {"x": [], "y": [], "y_err": []},
            },
            "Na_30mM": {
                "Na": {"x": [], "y": [], "y_err": []},
                "Ca": {"x": [], "y": [], "y_err": []},
            },
        }

        for case_study in case_study_list_2:
            ratio_key = str(case_study)[49:52]
            ratio = ratios[RATIO_key.index(ratio_key)]
            cation_1 = str(case_study)[53:55]
            cation_2 = str(case_study)[55:57]
            cation_1_conc = str(case_study)[61:63]

            concentration = float(50)  # mM
            cation_list = [cation_1, cation_2]
            inlet_concentration = {
                "feed": {
                    cation_1: concentration,
                    cation_2: concentration,
                    "Cl": concentration,
                },
                "diafiltrate": {
                    cation_1: 1e-10,
                    cation_2: 1e-10,
                    "Cl": 1e-10,
                },
            }

            # if cation_1 == "Na" and cation_2 == "Ca":
            #     salt_key = "Na_Ca"
            # elif cation_1 == "Na" and cation_2 == "La":
            #     salt_key = "Na_La"
            # elif cation_1 == "Ca" and cation_2 == "La":
            #     salt_key = "Ca_La"

            model = build_model(
                cation_list=cation_list,
                inlet_concentration=inlet_concentration,
                default_args=default_args,
                NFE_args=NFE_args,
                initialize_and_solve=False,
            )
            from_json(model, fname=case_study)

            for solute in model.fs.membrane.cations:
                average_variable_dict = get_model_averages(model, solute)

                x_value = ratio
                y_data = average_variable_dict["local_flux"]["avg"]
                y_err = average_variable_dict["local_flux"]["spread"]
                y_err = y_err.tolist()

                Na_Ca_data_dict[f"Na_{cation_1_conc}mM"][solute]["x"].append(x_value)
                Na_Ca_data_dict[f"Na_{cation_1_conc}mM"][solute]["y"].append(y_data)
                Na_Ca_data_dict[f"Na_{cation_1_conc}mM"][solute]["y_err"].append(y_err)

        with open(
            "figure_data/flux_versus_cation_concentration_two_salt_ratio_data.json", "w"
        ) as file:
            json.dump(Na_Ca_data_dict, file)


def flux_plots_equimolar(save_figure=True):
    markersize = 10
    fontsize = 14
    alpha = 0.6
    linestyle = "--"
    linewidth = 1

    fig1, (ax1a, ax1b, ax1c) = plt.subplots(
        1, 3, dpi=100, figsize=(12, 4), constrained_layout=True
    )

    ax1a.set_xlabel(
        "Sodium Feed Conc. (mM)",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax1b.set_xlabel(
        "Calcium Feed Conc. (mM)",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax1c.set_xlabel(
        "Lanthanum Feed Conc. (mM)",
        fontsize=fontsize,
        fontweight="bold",
    )

    ax1a.set_ylabel(
        "Cation Flux (mol m$\mathbf{^{-3}}$ h$\mathbf{^{-1}}$)",
        fontsize=fontsize,
        fontweight="bold",
    )

    # color blind friendly
    tol_bright_hex = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]
    na_color = tol_bright_hex[0]
    ca_color = tol_bright_hex[1]
    la_color = tol_bright_hex[2]
    na_ca_color = tol_bright_hex[3]
    na_la_color = tol_bright_hex[4]
    ca_la_color = tol_bright_hex[5]

    cation_legend_dict = {
        "sodium": {
            "marker": "o",
            "ax": ax1a,
            "salt_colors": {
                "NaCl": na_color,
                "NaCl + CaCl$_2$": na_ca_color,
                "NaCl + LaCl$_3$": na_la_color,
            },
        },
        "calcium": {
            "marker": "v",
            "ax": ax1b,
            "salt_colors": {
                "CaCl$_2$": ca_color,
                "NaCl + CaCl$_2$": na_ca_color,
                "CaCl$_2$ + LaCl$_3$": ca_la_color,
            },
        },
        "lanthanum": {
            "marker": "^",
            "ax": ax1c,
            "salt_colors": {
                "LaCl$_3$": la_color,
                "NaCl + LaCl$_3$": na_la_color,
                "CaCl$_2$ + LaCl$_3$": ca_la_color,
            },
        },
    }

    for cation_dict in cation_legend_dict.values():
        for salt, color in cation_dict["salt_colors"].items():
            cation_dict["ax"].plot(
                [],
                [],
                color=color,
                marker=cation_dict["marker"],
                markersize=markersize,
                linestyle="None",
                label=salt,
            )

    # for ax in fig1.axes:
    ax1a.legend(
        loc="best",
        fontsize=fontsize - 2,
        title="Sodium in:",
        title_fontsize=fontsize - 2,
    )
    ax1b.legend(
        loc="best",
        fontsize=fontsize - 2,
        title="Calcium in:",
        title_fontsize=fontsize - 2,
    )
    ax1c.legend(
        loc="best",
        fontsize=fontsize - 2,
        title="Lanthanum in:",
        title_fontsize=fontsize - 2,
    )

    for ax in fig1.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)

    with open(
        "figure_data/flux_versus_cation_concentration_single_salt_data.json", "r"
    ) as file:
        single_salt_data_dict = json.load(file)
    with open(
        "figure_data/flux_versus_cation_concentration_two_salt_data.json", "r"
    ) as file:
        two_salt_data_dict = json.load(file)

    case_study_data = {
        "single": single_salt_data_dict,
        "two": two_salt_data_dict,
    }

    for type, data_dict in case_study_data.items():
        if type == "single":
            for cation in data_dict.keys():
                if cation == "Na":
                    marker = "o"
                    ax_flux = ax1a
                    color = na_color
                elif cation == "Ca":
                    marker = "v"
                    ax_flux = ax1b
                    color = ca_color
                elif cation == "La":
                    marker = "^"
                    ax_flux = ax1c
                    color = la_color

                y_err = [np.array(i) for i in data_dict[cation]["y_err"]]

                ax_flux.plot(
                    data_dict[cation]["x"],
                    data_dict[cation]["y"],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=color,
                    alpha=alpha,
                    markersize=markersize,
                )
                ax_flux.errorbar(
                    data_dict[cation]["x"],
                    data_dict[cation]["y"],
                    yerr=np.squeeze(y_err).T,
                    ecolor=color,
                    capsize=3,
                    linestyle="None",
                )

        elif type == "two":
            for salt, salt_data in data_dict.items():
                for cation in salt_data.keys():
                    if salt == "Na_Ca":
                        color = na_ca_color
                    elif salt == "Na_La":
                        color = na_la_color
                    elif salt == "Ca_La":
                        color = ca_la_color

                    if cation == "Na":
                        marker = "o"
                        ax_flux = ax1a
                    elif cation == "Ca":
                        marker = "v"
                        ax_flux = ax1b
                    elif cation == "La":
                        marker = "^"
                        ax_flux = ax1c

                    y_err = [np.array(i) for i in salt_data[cation]["y_err"]]

                    ax_flux.plot(
                        salt_data[cation]["x"],
                        salt_data[cation]["y"],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        color=color,
                        alpha=alpha,
                        markersize=markersize,
                    )
                    ax_flux.errorbar(
                        salt_data[cation]["x"],
                        salt_data[cation]["y"],
                        yerr=np.squeeze(y_err).T,
                        ecolor=color,
                        capsize=3,
                        linestyle="None",
                    )

    if save_figure:
        plt.savefig(f"figures/flux_versus_cation_concentration.png", dpi=600)


def flux_plots_ratios(save_figure=True):
    # TODO: add a horizontal line representing the single salt cation flux?
    # TODO: shade left and right of R=1?
    # TODO: plot Na:cation ratio on x axis, plot both Ca and La at a fixed Na concentration? optional panels for more Na conc. values
    markersize = 10
    fontsize = 14
    alpha = 0.6
    linestyle = "--"
    linewidth = 1

    fig1, (ax1a, ax1b) = plt.subplots(
        2, 1, dpi=100, figsize=(4, 9), constrained_layout=True
    )

    ax1b.set_xlabel(
        "Na:Ca Feed Conc. Ratio",
        fontsize=fontsize,
        fontweight="bold",
    )

    ax1a.set_ylabel(
        "Na Flux (mol m$\mathbf{^{-3}}$ h$\mathbf{^{-1}}$)",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax1b.set_ylabel(
        "Ca Flux (mol m$\mathbf{^{-3}}$ h$\mathbf{^{-1}}$)",
        fontsize=fontsize,
        fontweight="bold",
    )

    # color blind friendly
    tol_bright_hex = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]
    na_10mM_color = tol_bright_hex[0]
    na_20mM_color = tol_bright_hex[1]
    na_30mM_color = tol_bright_hex[2]
    # na_ca_color = tol_bright_hex[3]
    # na_la_color = tol_bright_hex[4]
    # ca_la_color = tol_bright_hex[5]

    legend_dict = {
        "marker": "o",
        "ax": ax1b,
        "salt_colors": {
            "10 mM": na_10mM_color,
            "20 mM": na_20mM_color,
            "30 mM": na_30mM_color,
        },
    }

    for salt, color in legend_dict["salt_colors"].items():
        legend_dict["ax"].plot(
            [],
            [],
            color=color,
            marker=legend_dict["marker"],
            markersize=markersize,
            linestyle="None",
            label=salt,
        )

    # for ax in fig1.axes:
    ax1b.legend(
        loc="best",
        fontsize=fontsize - 2,
        title="Na Conc.:",
        title_fontsize=fontsize - 2,
    )

    for ax in fig1.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)

    with open(
        "figure_data/flux_versus_cation_concentration_two_salt_ratio_data.json", "r"
    ) as file:
        Na_Ca_data_dict = json.load(file)

    for conc_1_info, salt_data in Na_Ca_data_dict.items():
        for cation in salt_data.keys():
            marker = "o"

            if conc_1_info == "Na_10mM":
                color = na_10mM_color
            elif conc_1_info == "Na_20mM":
                color = na_20mM_color
            elif conc_1_info == "Na_30mM":
                color = na_30mM_color

            if cation == "Na":
                ax_flux = ax1a
            elif cation == "Ca":
                ax_flux = ax1b

            y_err = [np.array(i) for i in salt_data[cation]["y_err"]]

            ax_flux.plot(
                salt_data[cation]["x"],
                salt_data[cation]["y"],
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
                markersize=markersize,
            )
            ax_flux.errorbar(
                salt_data[cation]["x"],
                salt_data[cation]["y"],
                yerr=np.squeeze(y_err).T,
                ecolor=color,
                capsize=3,
                linestyle="None",
            )

    if save_figure:
        plt.savefig(f"figures/flux_versus_cation_ratio.png", dpi=600)


def plot_flux_contributions(save_figure=True):
    """
    Plots flux contributions for different systems.
    """
    markersize = 10
    fontsize = 14

    # fig1, ((ax1a, ax1b), (ax2a, ax2b), (ax3a, ax3b)) = plt.subplots(
    #     3, 2, figsize=(12, 12), dpi=100, constrained_layout=True
    # )
    # fig2, ((ax4a, ax4b), (ax5a, ax5b), (ax6a, ax6b)) = plt.subplots(
    #     3, 2, figsize=(12, 12), dpi=100, constrained_layout=True
    # )
    fig1, (ax1a, ax1b) = plt.subplots(
        1, 2, figsize=(12, 4), dpi=100, constrained_layout=True, sharey=True
    )
    fig2, (ax4a, ax4b) = plt.subplots(
        1, 2, figsize=(12, 4), dpi=100, constrained_layout=True
    )

    tol_bright_hex = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]
    conv_color = tol_bright_hex[0]
    diff_color = tol_bright_hex[1]
    elec_color = tol_bright_hex[2]

    # concentrations = ["30", "40", "50", "75", "100", "125", "150", "175", "200"]
    concentrations = [
        "10",
        "20",
        "30",
        "40",
        "50",
        "75",
        "100",
        "125",
        "150",
        "175",
        "200",
    ]

    x_values = concentrations
    x_label = "Sodium Feed Concentration (mM)"

    color_list = [conv_color, diff_color, elec_color]
    x = np.arange(len(x_values))

    sodium_bl_df = pd.read_json(
        "figure_data/total_flux_breakdown_single_salt_cation_Na_bl"
    )
    cl_sodium_bl_df = pd.read_json(
        "figure_data/total_flux_breakdown_single_salt_anion_NaCl_bl"
    )
    sodium_mem_df = pd.read_json(
        "figure_data/total_flux_breakdown_single_salt_cation_Na_mem"
    )
    cl_sodium_mem_df = pd.read_json(
        "figure_data/total_flux_breakdown_single_salt_anion_NaCl_mem"
    )

    sodium_bl_df.plot(
        ax=ax1a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    cl_sodium_bl_df.plot(
        ax=ax4a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    # calcium_bl_df.plot(
    #     ax=ax2a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    # )
    # cl_calcium_bl_df.plot(
    #     ax=ax5a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    # )
    # lanthanum_bl_df.plot(
    #     ax=ax3a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    # )
    # cl_lanthanum_bl_df.plot(
    #     ax=ax6a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    # )

    sodium_mem_df.plot(
        ax=ax1b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    cl_sodium_mem_df.plot(
        ax=ax4b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    # calcium_mem_df.plot(
    #     ax=ax2b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    # )
    # cl_calcium_mem_df.plot(
    #     ax=ax5b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    # )
    # lanthanum_mem_df.plot(
    #     ax=ax3b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    # )
    # cl_lanthanum_mem_df.plot(
    #     ax=ax6b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    # )

    bar_width = 0.5
    x_positions = ax1a.get_xticks()
    x_min = x_positions - (bar_width / 2)
    x_max = x_positions + (bar_width / 2)

    totals_sodium_bl = sodium_bl_df.sum(axis=1)
    totals_sodium_mem = sodium_mem_df.sum(axis=1)
    totals_cl_sodium_bl = cl_sodium_bl_df.sum(axis=1)
    totals_cl_sodium_mem = cl_sodium_mem_df.sum(axis=1)

    ax1a.hlines(totals_sodium_bl, x_min, x_max, colors="black", linewidth=3)
    ax1b.hlines(totals_sodium_mem, x_min, x_max, colors="black", linewidth=3)
    ax4a.hlines(totals_cl_sodium_bl, x_min, x_max, colors="black", linewidth=3)
    ax4b.hlines(totals_cl_sodium_mem, x_min, x_max, colors="black", linewidth=3)

    # totals_calcium_bl = calcium_bl_df.sum(axis=1)
    # totals_calcium_mem = calcium_mem_df.sum(axis=1)
    # totals_cl_calcium_bl = cl_calcium_bl_df.sum(axis=1)
    # totals_cl_calcium_mem = cl_calcium_mem_df.sum(axis=1)

    # ax2a.hlines(totals_calcium_bl, x_min, x_max, colors="black", linewidth=3)
    # ax2b.hlines(totals_calcium_mem, x_min, x_max, colors="black", linewidth=3)
    # ax5a.hlines(totals_cl_calcium_bl, x_min, x_max, colors="black", linewidth=3)
    # ax5b.hlines(totals_cl_calcium_mem, x_min, x_max, colors="black", linewidth=3)

    # totals_lanthanum_bl = lanthanum_bl_df.sum(axis=1)
    # totals_lanthanum_mem = lanthanum_mem_df.sum(axis=1)
    # totals_cl_lanthanum_bl = cl_lanthanum_bl_df.sum(axis=1)
    # totals_cl_lanthanum_mem = cl_lanthanum_mem_df.sum(axis=1)

    # ax3a.hlines(totals_lanthanum_bl, x_min, x_max, colors="black", linewidth=3)
    # ax3b.hlines(totals_lanthanum_mem, x_min, x_max, colors="black", linewidth=3)
    # ax6a.hlines(totals_cl_lanthanum_bl, x_min, x_max, colors="black", linewidth=3)
    # ax6b.hlines(totals_cl_lanthanum_mem, x_min, x_max, colors="black", linewidth=3)

    for ax in fig1.axes:
        ax.set_xticks(x)
        ax.set_xticklabels(x_values)
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.axhline(0, color="black", linewidth=1.5)
    for ax in fig2.axes:
        ax.set_xticks(x)
        ax.set_xticklabels(x_values)
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.axhline(0, color="black", linewidth=1.5)

    # for ax in [ax1a, ax1b]:
    #     ax.set_ylim(-6, 17)
    # for ax in [ax4a, ax4b]:
    #     ax.set_ylim(-8, 19)

    # fig1.suptitle(
    #     "Average Cation Flux Breakdown (Excluding x,z=0)",
    #     fontsize=fontsize + 2,
    #     fontweight="bold",
    # )
    # fig2.suptitle(
    #     "Average Anion Flux Breakdown (Excluding x,z=0)",
    #     fontsize=fontsize + 2,
    #     fontweight="bold",
    # )

    na_label = "Sodium Flux (mol m$\mathbf{^{-2}}$ h$\mathbf{^{-1}}$)"
    ca_label = "Calcium Flux (mol m$\mathbf{^{-2}}$ h$\mathbf{^{-1}}$)"
    la_label = "Lanthanum Flux (mol m$\mathbf{^{-2}}$ h$\mathbf{^{-1}}$)"
    cl_label = "Chloride Flux (mol m$\mathbf{^{-2}}$ h$\mathbf{^{-1}}$)"

    ax1a.set_ylabel(
        na_label,
        fontsize=fontsize,
        fontweight="bold",
    )
    # ax2a.set_ylabel(
    #     ca_label,
    #     fontsize=fontsize,
    #     fontweight="bold",
    # )
    # ax3a.set_ylabel(
    #     la_label,
    #     fontsize=fontsize,
    #     fontweight="bold",
    # )
    for ax in [ax4a]:  # , ax5a, ax6a]:
        ax.set_ylabel(
            cl_label,
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax1a, ax4a]:
        ax.set_title(
            "Boundary Layer",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax1b, ax4b]:
        ax.set_title(
            "Membrane",
            fontsize=fontsize,
            fontweight="bold",
        )
    # for ax in [ax3a, ax3b, ax6a, ax6b]:
    for ax in [ax1a, ax1b, ax4a, ax4b]:
        ax.set_xlabel(
            x_label,
            fontsize=fontsize,
            fontweight="bold",
        )
    # for ax in [ax1a, ax1b, ax4a, ax4b]:
    #     ax.text(
    #         0.5,
    #         0.95,
    #         "NaCl",
    #         ha="center",
    #         va="center",
    #         transform=ax.transAxes,
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )
    # for ax in [ax2a, ax2b, ax5a, ax5b]:
    #     ax.text(
    #         0.5,
    #         0.95,
    #         "CaCl$_2$",
    #         ha="center",
    #         va="center",
    #         transform=ax.transAxes,
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )
    # for ax in [ax3a, ax3b, ax6a, ax6b]:
    #     ax.text(
    #         0.5,
    #         0.95,
    #         "LaCl$_3$",
    #         ha="center",
    #         va="center",
    #         transform=ax.transAxes,
    #         fontsize=fontsize,
    #         fontweight="bold",
    # )
    # ax1a.plot([],[], "k-", linewidth=3, label="Total Flux")
    ax1a.legend(fontsize=fontsize - 2)
    ax4a.legend(fontsize=fontsize - 2)

    if save_figure:
        fig1.savefig(f"figures/total_flux_breakdown_single_salt_cation.png", dpi=600)
        fig2.savefig(f"figures/total_flux_breakdown_single_salt_anion.png", dpi=600)


if __name__ == "__main__":
    main()
