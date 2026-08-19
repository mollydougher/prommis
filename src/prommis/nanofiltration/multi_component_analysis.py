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

from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
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
from idaes.core.util.exceptions import InitializationError
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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
    set_IS = True
    run_data = False
    run_single_salt = False
    run_two_salt = False
    run_three_salt = False
    run_salt_ratio = False
    solve_and_save_models(
        water_flux=0.02,
        run_data=run_data,
        run_single_salt=run_single_salt,
        run_two_salt=run_two_salt,
        run_three_salt=run_three_salt,
        run_salt_ratio=run_salt_ratio,
        set_IS=set_IS,
    )

    # data_comparison_plots(save_figure=True)

    # rejection_plots_equimolar(x_axis="ionic_strength", sieving=False, save_figure=True)
    rejection_plots_equimolar(x_axis="ionic_strength", sieving=True, save_figure=True)
    # rejection_plots_equimolar(
    #     x_axis="cation_concentration", sieving=True, save_figure=True
    # )
    h_plots_equimolar(x_axis="ionic_strength", inset=True, save_figure=True)

    # combined_plots_vary_salt_ratio(save_figure=True)
    # plot_only_rejections(save_figure=True)

    # plot_flux_contributions(x_axis="ionic_strength", percent=False, save_figure=True)
    # plot_flux_contributions(x_axis="ionic_strength", percent=True, save_figure=True)

    # plot_Donnan_potentials(total_h=True, sieving=True)
    plt.show()


def build_model(
    cation_list,
    inlet_concentration,
    default_args,
    NFE_args,
    initialize_and_solve=True,
    water_flux=0.02,
    data_membrane_thickness=1e-7,  # default of 100 nm
    non_Donnan_partition_dict={},
    save=False,
    chloride_phi_star_key=None,
    Dm_over_l_key=None,
    cation_phi_star_key=None,
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
    # TODO: initial pressure values may need to be tweaked
    if len(cation_list) == 1:
        m.fs.membrane.applied_pressure.fix(5)
    elif len(cation_list) > 1:
        if value(m.fs.membrane.total_feed_ionic_strength[0]) < 99:
            m.fs.membrane.applied_pressure.fix(5)
        elif (value(m.fs.membrane.total_feed_ionic_strength[0]) >= 99) and (
            value(m.fs.membrane.total_feed_ionic_strength[0]) < 199
        ):
            m.fs.membrane.applied_pressure.fix(15)
        elif value(m.fs.membrane.total_feed_ionic_strength[0]) >= 199:
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

    # initialize membrane model
    if initialize_and_solve:
        # if ("La" in cation_list) or ("Al" in cation_list):
        #     # guess larger H_permeate
        #     if inlet_concentration["feed"][cation_list[0]] <= 8:
        #         H_feed_guesses = np.arange(3.2, 6, 0.2)
        #         H_permeate_guesses = np.arange(10, 17, 0.5)
        #     else:
        #         H_feed_guesses = np.arange(0.2, 3.5, 0.1)
        #         H_permeate_guesses = np.arange(2, 18.5, 0.5)
        # else:
        #     H_feed_guesses = np.arange(0.2, 3.5, 0.1)
        #     H_permeate_guesses = np.arange(0.2, 3.5, 0.1)

        # H_guesses = np.column_stack((H_feed_guesses, H_permeate_guesses))
        # H_guesses = np.flip(H_guesses, axis=0)

        # for H_feed_guess, H_permeate_guess in H_guesses:
        #     try:
        initialized_membrane_model = m.fs.membrane.default_initializer(
            # H_feed_guess=H_feed_guess, H_permeate_guess=H_permeate_guess
        )
        initialized_membrane_model.initialize(m.fs.membrane)

        solve_model(m)
        unfix_pressure(m, water_flux=water_flux)
        solve_model(m)

        full_sensitivity = False
        data = False
        single_salt = False
        two_salt = True

        key_name = "IS"

        if save:
            if full_sensitivity:
                fname = f"multi_component_case_studies/DATA_comparison/Cl_phi_{chloride_phi_star_key}/{Dm_over_l_key}umpers/cation_phi_{cation_phi_star_key}/{cation_list[0]}_{inlet_concentration['feed'][cation_list[0]]}mM"
            elif data:
                fname = f"multi_component_case_studies/DATA_comparison/{cation_list[0]}_{inlet_concentration['feed'][cation_list[0]]}mM"
            elif single_salt:
                fname = f"multi_component_case_studies/single_salt/{key_name}/{key_name}{key}_{cation_list[0]}Cl{chloride_multiplier}_{inlet_concentration['feed'][cation_list[0]]}mM"
            elif two_salt:
                fname = f"multi_component_case_studies/two_salt/{key_name}/{key_name}{key}_{cation_list[0]}{cation_list[1]}Cl{chloride_multiplier}_{inlet_concentration['feed'][cation_list[0]]}mM_{inlet_concentration['feed'][cation_list[1]]}mM"
            to_json(m, fname=fname)

            #     break
            # except (
            #     InitializationError,
            #     NoFeasibleSolutionError,
            #     RuntimeError,
            # ):
            #     continue

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
    run_data=True,
    run_single_salt=True,
    run_two_salt=True,
    run_three_salt=True,
    run_salt_ratio=True,
    set_IS=False,  # set concentration
):
    # global variables
    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    diafiltrate = {"Li": 1e-10, "Co": 1e-10, "Al": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    if run_data:
        # set concentrations
        feed = {
            "Na": [
                # 9.6199,
                30.5114,
                43.1343,
                54.5286,
                64.7312,
                74.1947,
                83.0378,
                91.3303,
                98.4631,
                105.1806,
            ],
            "Ca": [
                # 3.0588,
                10.4631,
                15.2044,
                19.6055,
                23.6953,
                27.5362,
                31.0213,
                34.3589,
                37.4959,
                40.3890,
            ],
            "La": [
                # 1.4129,
                4.8072,
                7.0910,
                9.2925,
                11.3234,
                13.5457,
                15.5382,
                17.4718,
                19.3323,
                21.1177,
            ],
        }

        # set average flux
        flux = {
            "Na": [
                # 0.018,
                0.033,
                0.031,
                0.030,
                0.028,
                0.026,
                0.025,
                0.026,
                0.023,
                0.023,
            ],
            "Ca": [
                # 0.015,
                0.030,
                0.026,
                0.026,
                0.023,
                0.022,
                0.021,
                0.019,
                0.019,
                0.018,
            ],
            "La": [
                # 0.016,
                0.030,
                0.029,
                0.026,
                0.022,
                0.024,
                0.020,
                0.019,
                0.018,
                0.016,
            ],
        }

        Dm_Cl = 2.03  # um2/s

        full_sensitivity = False

        if full_sensitivity:
            Dm_over_l_sensitivity = [80, 70, 60, 50, 40]  # um/s
            Dm_over_l_sensitivity_keys = ["80", "70", "60", "50", "40"]  # um/s
            # Dm_over_l_sensitivity = [80, 70]  # um/s
            # Dm_over_l_sensitivity_keys = ["80", "70"]  # um/s
            # Dm_over_l_sensitivity = [60, 50, 40]  # um/s
            # Dm_over_l_sensitivity_keys = ["60", "50", "40"]  # um/s

            # Na
            monovalent_phi_star_sensitivity = [
                0.7,
                0.65,
                0.6,
                0.55,
                0.5,
                0.45,
                0.4,
                0.35,
                0.3,
                0.25,
                0.2,
            ]
            monovalent_phi_star_sensitivity_keys = [
                "0700",
                "0650",
                "0600",
                "0550",
                "0500",
                "0450",
                "0400",
                "0350",
                "0300",
                "0250",
                "0200",
            ]
            # Ca
            divalent_phi_star_sensitivity = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
            divalent_phi_star_sensitivity_keys = [
                "0500",
                "0450",
                "0400",
                "0350",
                "0300",
                "0250",
                "0200",
                "0150",
            ]
            # La
            trivalent_phi_star_sensitivity = [
                0.25,
                0.2,
                0.15,
                0.1,
                0.05,
                0.01,
                0.005,
                0.001,
                0.0005,
            ]
            trivalent_phi_star_sensitivity_keys = [
                "0250",
                "0200",
                "0150",
                "0100",
                "0050",
                "0010",
                "0005",
                "0001",
                "0000",
            ]

            chloride_phi_star_sensitivity = [0.1, 0.05]
            chloride_phi_star_sensitivity_keys = ["010", "005"]

            for chloride_phi_star in chloride_phi_star_sensitivity:
                for Dm_over_l in Dm_over_l_sensitivity:
                    for cation in feed.keys():
                        l_um = Dm_Cl / Dm_over_l  # um
                        l_m = l_um / 1e6  # m

                        if cation == "Na":
                            chloride_multiplier = 1
                            cation_phi_star_sensitivity = (
                                monovalent_phi_star_sensitivity
                            )
                            cation_phi_star_sensitivity_keys = (
                                monovalent_phi_star_sensitivity_keys
                            )
                        elif cation == "Ca":
                            chloride_multiplier = 2
                            cation_phi_star_sensitivity = divalent_phi_star_sensitivity
                            cation_phi_star_sensitivity_keys = (
                                divalent_phi_star_sensitivity_keys
                            )
                        elif cation == "La":
                            chloride_multiplier = 3
                            cation_phi_star_sensitivity = trivalent_phi_star_sensitivity
                            cation_phi_star_sensitivity_keys = (
                                trivalent_phi_star_sensitivity_keys
                            )

                        for cation_phi_star in cation_phi_star_sensitivity:
                            for concentration in feed[cation]:

                                model = build_model(
                                    cation_list=[cation],
                                    inlet_concentration={
                                        "feed": {
                                            cation: concentration,
                                            "Cl": chloride_multiplier * concentration,
                                        },
                                        "diafiltrate": {
                                            cation: 1e-10,
                                            "Cl": chloride_multiplier * 1e-10,
                                        },
                                    },
                                    default_args=default_args,
                                    NFE_args=NFE_args,
                                    initialize_and_solve=True,
                                    water_flux=flux[cation][
                                        feed[cation].index(concentration)
                                    ],
                                    data_membrane_thickness=l_m,
                                    non_Donnan_partition_dict={
                                        cation: cation_phi_star,
                                        "Cl": chloride_phi_star,
                                    },
                                    save=True,
                                    chloride_phi_star_key=chloride_phi_star_sensitivity_keys[
                                        chloride_phi_star_sensitivity.index(
                                            chloride_phi_star
                                        )
                                    ],
                                    Dm_over_l_key=Dm_over_l_sensitivity_keys[
                                        Dm_over_l_sensitivity.index(Dm_over_l)
                                    ],
                                    cation_phi_star_key=cation_phi_star_sensitivity_keys[
                                        cation_phi_star_sensitivity.index(
                                            cation_phi_star
                                        )
                                    ],
                                )

        else:
            Dm_over_l_value = 40  # um/s
            monovalent_phi_star_value = 0.7
            divalent_phi_star_value = 0.3
            trivalent_phi_star_value = 0.0005
            chloride_phi_star_value = 0.1

            for cation in feed.keys():
                l_um = Dm_Cl / Dm_over_l_value  # um
                l_m = l_um / 1e6  # m

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
                    model = build_model(
                        cation_list=[cation],
                        inlet_concentration={
                            "feed": {
                                cation: concentration,
                                "Cl": chloride_multiplier * concentration,
                            },
                            "diafiltrate": {
                                cation: 1e-10,
                                "Cl": chloride_multiplier * 1e-10,
                            },
                        },
                        default_args=default_args,
                        NFE_args=NFE_args,
                        initialize_and_solve=True,
                        water_flux=flux[cation][feed[cation].index(concentration)],
                        data_membrane_thickness=l_m,
                        non_Donnan_partition_dict={
                            cation: cation_phi_star_value,
                            "Cl": chloride_phi_star_value,
                        },
                        save=True,
                    )

    IS_key = ["025", "050", "075", "100", "150", "200", "400", "600", "800"]
    # CONC_key = ["025", "050", "075", "100", "150", "200", "250", "300"]
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
        if set_IS:
            feed = {
                "Li": [25, 50, 75, 100, 150, 200, 400, 600, 800],
                "Co": [8.334, 16.667, 25, 33.334, 50, 66.667, 133.334, 200, 266.667],
                "Al": [4.167, 8.334, 12.5, 16.667, 25, 33.334, 66.667, 100, 133.334],
            }
            key_list = IS_key
        else:
            feed = {
                # "Li": [25, 50, 75, 100, 150, 200, 250, 300],
                # "Co": [25, 50, 75, 100, 150, 200, 250, 300],
                # "Al": [25, 50, 75, 100, 150, 200, 250, 300],
                "Li": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
                "Co": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
                "Al": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
                # TODO debug 10mM and all Al systems with new initialization
            }
            key_list = CONC_key

        Dm_Cl = 2.03  # um2/s
        Dm_over_l_value = 40  # um/s
        l_um = Dm_Cl / Dm_over_l_value  # um
        l_m = l_um / 1e6  # m
        monovalent_phi_star_value = 0.7
        divalent_phi_star_value = 0.3
        trivalent_phi_star_value = 0.0005
        chloride_phi_star_value = 0.1

        for cation in feed.keys():
            if cation == "Li":
                chloride_multiplier = 1
                cation_phi_star_value = monovalent_phi_star_value
            elif cation == "Co":
                chloride_multiplier = 2
                cation_phi_star_value = divalent_phi_star_value
            elif cation == "Al":
                chloride_multiplier = 3
                cation_phi_star_value = trivalent_phi_star_value

            for concentration in feed[cation]:
                model = build_model(
                    cation_list=[cation],
                    inlet_concentration={
                        "feed": {
                            cation: concentration,
                            "Cl": chloride_multiplier * concentration,
                        },
                        "diafiltrate": {
                            cation: diafiltrate[cation],
                            "Cl": chloride_multiplier * 1e-10,
                        },
                    },
                    default_args=default_args,
                    NFE_args=NFE_args,
                    initialize_and_solve=True,
                    water_flux=0.02,
                    data_membrane_thickness=l_m,
                    non_Donnan_partition_dict={
                        cation: cation_phi_star_value,
                        "Cl": chloride_phi_star_value,
                    },
                    save=True,
                    key=key_list[feed[cation].index(concentration)],
                    chloride_multiplier=chloride_multiplier,
                )

    if run_two_salt:

        if set_IS:
            feed = {
                "Li_Co": [6.25, 12.5, 18.75, 25, 37.5, 50, 100, 150, 200],
                # TODO: debug Li/Al and Co/Al systems
                # "Li_Al": [
                #     3.571,
                #     7.143,
                #     10.7145,
                #     14.286,
                #     21.429,
                #     28.572,
                #     57.143,
                #     85.715,
                #     114.286,
                # ],
                # "Co_Al": [
                #     2.778,
                #     5.556,
                #     8.334,
                #     11.112,
                #     16.667,
                #     22.223,
                #     44.445,
                #     66.667,
                #     88.889,
                # ],
            }
            key_list = IS_key
        else:
            feed = {
                # "Li_Co": [25],
                # "Li_Co": [25, 50, 75, 100, 150, 200, 250, 300],
                # "Li_Al": [25, 50, 75, 100, 150, 200, 250, 300],
                # "Co_Al": [25, 50, 75, 100, 150, 200, 250, 300],
                "Li_Co": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
                "Li_Al": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
                "Co_Al": [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],
            }
            key_list = CONC_key

        Dm_Cl = 2.03  # um2/s
        Dm_over_l_value = 40  # um/s
        l_um = Dm_Cl / Dm_over_l_value  # um
        l_m = l_um / 1e6  # m
        monovalent_phi_star_value = 0.7
        divalent_phi_star_value = 0.3
        trivalent_phi_star_value = 0.0005
        chloride_phi_star_value = 0.1

        for salt in feed.keys():
            if salt == "Li_Co":
                cation_1 = "Li"
                cation_2 = "Co"
                chloride_multiplier = 3
                cation_1_phi_star_value = monovalent_phi_star_value
                cation_2_phi_star_value = divalent_phi_star_value
            elif salt == "Li_Al":
                cation_1 = "Li"
                cation_2 = "Al"
                chloride_multiplier = 4
                cation_1_phi_star_value = monovalent_phi_star_value
                cation_2_phi_star_value = trivalent_phi_star_value
            elif salt == "Co_Al":
                cation_1 = "Co"
                cation_2 = "Al"
                chloride_multiplier = 5
                cation_1_phi_star_value = divalent_phi_star_value
                cation_2_phi_star_value = trivalent_phi_star_value

            for concentration in feed[salt]:
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
                            "Cl": chloride_multiplier * 1e-10,
                        },
                    },
                    default_args=default_args,
                    NFE_args=NFE_args,
                    initialize_and_solve=True,
                    water_flux=0.02,
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

    if run_three_salt:
        if set_IS:
            feed = {
                "Li_Co_Al": [5, 7.5, 10, 15, 20, 40, 60, 80],
                # "Li_Co_Al": [5], # TODO: debug this system
            }
        else:
            feed = {
                "Li_Co_Al": [25, 50, 75, 100, 150, 200, 250, 300],
            }

        H_feed_guesses = np.arange(0.5, 2.6, 0.1)
        H_permeate_guesses = np.arange(0.5, 2.6, 0.1)
        # add in non equal guesses
        H_feed_guesses = np.append(
            H_feed_guesses,
            1,
        )
        H_permeate_guesses = np.append(H_permeate_guesses, 2)
        H_guesses = np.column_stack((H_feed_guesses, H_permeate_guesses))

        for salt in feed.keys():
            if salt == "Li_Co_Al":
                cation_1 = "Li"
                cation_2 = "Co"
                cation_3 = "Al"
                chloride_multiplier = 6

            for concentration in feed[salt]:
                # start with the low H's at higher concentrations
                if concentration >= 400:
                    H_guesses = H_guesses
                # start with the high H's at lower concentrations
                else:
                    H_guesses = np.flip(H_guesses, axis=0)
                for H_feed_guess, H_permeate_guess in H_guesses:
                    try:
                        model = build_model(
                            cation_list=[cation_1, cation_2, cation_3],
                            inlet_concentration={
                                "feed": {
                                    cation_1: concentration,
                                    cation_2: concentration,
                                    cation_3: concentration,
                                    "Cl": chloride_multiplier * concentration,
                                },
                                "diafiltrate": {
                                    cation_1: diafiltrate[cation_1],
                                    cation_2: diafiltrate[cation_2],
                                    cation_3: diafiltrate[cation_3],
                                    "Cl": 1e-10,
                                },
                            },
                            default_args=default_args,
                            H_feed_guess=H_feed_guess,
                            H_permeate_guess=H_permeate_guess,
                            NFE_args=NFE_args,
                            initialize=True,
                            rpore=6e-10,
                            data_membrane_thickness=5e-8,
                        )
                        solve_model(model)
                        unfix_pressure(model, water_flux=water_flux)
                        solve_model(model)
                        if set_IS:
                            fname = f"multi_component_case_studies/three_salt/IS/IS{IS_key[feed[salt].index(concentration)]}_{cation_1}{cation_2}{cation_3}Cl{chloride_multiplier}_{concentration}mM_{concentration}mM_{concentration}mM"
                        else:
                            fname = f"multi_component_case_studies/three_salt/CONC/CONC{CONC_key[feed[salt].index(concentration)]}_{cation_1}{cation_2}{cation_3}Cl{chloride_multiplier}_{concentration}mM_{concentration}mM_{concentration}mM"
                        to_json(model, fname=fname)
                        break
                    except (InitializationError, NoFeasibleSolutionError):
                        continue

    if run_salt_ratio:
        # cation 2 multiplier order
        cation_2_multiplier = [3, 2, 1, 0.5, 0.3333]

        # feed = {ionic_strength: {salt: [cation_1_conc]}}
        feed = {
            "050_mM": {
                "Li_Co": [5, 7.143, 12.5, 20, 25],
                "Li_Al": [2.632, 3.846, 7.143, 12.5, 16.667],
                "Co_Al": [2.381, 3.333, 5.556, 8.333, 10],
            },
            "075_mM": {
                "Li_Co": [7.5, 10.714, 18.75, 30, 37.5],
                "Li_Al": [3.947, 5.769, 10.714, 18.75, 25],
                "Co_Al": [3.571, 5, 8.333, 12.5, 15],
            },
            "100_mM": {
                "Li_Co": [10, 14.286, 25, 40, 50],
                "Li_Al": [5.263, 7.692, 14.286, 25, 33.333],
                "Co_Al": [4.762, 6.667, 11.111, 16.667, 20],
            },
            "150_mM": {
                "Li_Co": [15, 21.429, 37.5, 60, 75],
                "Li_Al": [7.895, 11.538, 21.429, 37.5, 50],
                "Co_Al": [7.143, 10, 16.667, 25, 30],
            },
            "200_mM": {
                "Li_Co": [20, 28.571, 50, 80, 100],
                "Li_Al": [10.526, 15.385, 28.571, 50, 66.667],
                "Co_Al": [9.524, 13.333, 22.222, 33.333, 40],
            },
            "400_mM": {
                "Li_Co": [40, 57.143, 100, 160, 200],
                "Li_Al": [21.053, 30.769, 57.143, 100, 133.333],
                "Co_Al": [19.048, 26.667, 44.444, 66.667, 80],
            },
        }

        H_feed_guesses = np.arange(0.5, 2.6, 0.1)
        H_permeate_guesses = np.arange(0.5, 2.6, 0.1)
        # add in non equal guesses
        H_feed_guesses = np.append(
            H_feed_guesses,
            1,
        )
        H_permeate_guesses = np.append(H_permeate_guesses, 2)
        H_guesses = np.column_stack((H_feed_guesses, H_permeate_guesses))

        for ionic_strength, conc_dict in feed.items():
            for salt in conc_dict.keys():
                if salt == "Li_Co":
                    cation_1 = "Li"
                    cation_2 = "Co"
                    z_1 = 1
                    z_2 = 2
                    chloride_multiplier = 3
                elif salt == "Li_Al":
                    cation_1 = "Li"
                    cation_2 = "Al"
                    z_1 = 1
                    z_2 = 3
                    chloride_multiplier = 4
                elif salt == "Co_Al":
                    cation_1 = "Co"
                    cation_2 = "Al"
                    z_1 = 2
                    z_2 = 3
                    chloride_multiplier = 5

                for cation_1_concentration in conc_dict[salt]:
                    multiplier_index = conc_dict[salt].index(cation_1_concentration)
                    cation_2_concentration = (
                        cation_2_multiplier[multiplier_index] * cation_1_concentration
                    )

                    # start with the low H's at higher concentrations
                    if cation_1_concentration >= 400:
                        H_guesses = H_guesses
                    # start with the high H's at lower concentrations
                    else:
                        H_guesses = np.flip(H_guesses, axis=0)
                    for H_feed_guess, H_permeate_guess in H_guesses:
                        try:
                            model = build_model(
                                cation_list=[cation_1, cation_2],
                                inlet_concentration={
                                    "feed": {
                                        cation_1: cation_1_concentration,
                                        cation_2: cation_2_concentration,
                                        "Cl": z_1 * cation_1_concentration
                                        + z_2 * cation_2_concentration,
                                    },
                                    "diafiltrate": {
                                        cation_1: diafiltrate[cation_1],
                                        cation_2: diafiltrate[cation_2],
                                        "Cl": 1e-10,
                                    },
                                },
                                default_args=default_args,
                                H_feed_guess=H_feed_guess,
                                H_permeate_guess=H_permeate_guess,
                                NFE_args=NFE_args,
                                initialize=True,
                            )

                            solve_model(model)
                            unfix_pressure(model, water_flux=water_flux)
                            solve_model(model)

                            to_json(
                                model,
                                fname=f"multi_component_case_studies/vary_salt_ratio/{ionic_strength}/{cation_2_multiplier[multiplier_index]}_{cation_1}{cation_2}Cl{chloride_multiplier}_{cation_1_concentration}mM_{cation_2_concentration}mM",
                            )

                            break

                        except (InitializationError, NoFeasibleSolutionError):
                            continue


def calculate_spread(list):
    return np.array(
        [
            [np.average(list) - min(list)],
            [max(list) - np.average(list)],
        ]
    )


def get_model_averages(model, solute):
    actual_sieving = []
    observed_sieving = []
    observed_rejection = []
    actual_rejection = []
    flux = []
    H_feed = []
    H_perm = []
    Donnan_potential_feed = []
    Donnan_potential_perm = []

    for t in model.fs.membrane.time:
        for x in model.fs.membrane.dimensionless_module_length:
            if x != 0:
                actual_sieving.append(
                    value(model.fs.membrane.actual_sieving_coefficient[t, x, solute])
                )
                observed_sieving.append(
                    value(model.fs.membrane.observed_sieving_coefficient[t, x, solute])
                )
                observed_rejection.append(
                    value(model.fs.membrane.observed_rejection_percent[t, x, solute])
                )
                actual_rejection.append(
                    value(model.fs.membrane.actual_rejection_percent[t, x, solute])
                )
                flux.append(value(model.fs.membrane.molar_ion_flux[t, x, solute]))
                H_feed.append(
                    value(
                        model.fs.membrane.overall_partition_coefficient_feed_side[
                            t, x, solute
                        ]
                    )
                )
                H_perm.append(
                    value(
                        model.fs.membrane.overall_partition_coefficient_permeate_side[
                            t, x, solute
                        ]
                    )
                )
                Donnan_potential_feed.append(
                    value(model.fs.membrane.Donnan_potential_feed_side[t, x])
                )
                Donnan_potential_perm.append(
                    value(model.fs.membrane.Donnan_potential_permeate_side[t, x])
                )

    avg_actual_sieving = np.average(actual_sieving)
    spread_actual_sieving = calculate_spread(actual_sieving)

    avg_observed_sieving = np.average(observed_sieving)
    spread_observed_sieving = calculate_spread(observed_sieving)

    avg_observed_rejection = np.average(observed_rejection)
    spread_observed_rejection = calculate_spread(observed_rejection)

    avg_actual_rejection = np.average(actual_rejection)
    spread_actual_rejection = calculate_spread(actual_rejection)

    avg_flux = np.average(flux)
    spread_flux = calculate_spread(flux)

    avg_H_feed = np.average(H_feed)
    spread_H_feed = calculate_spread(H_feed)

    avg_H_perm = np.average(H_perm)
    spread_H_perm = calculate_spread(H_perm)

    avg_Donnan_potential_feed = np.average(Donnan_potential_feed)
    spread_Donnan_potential_feed = calculate_spread(Donnan_potential_feed)

    avg_Donnan_potential_perm = np.average(Donnan_potential_perm)
    spread_Donnan_potential_perm = calculate_spread(Donnan_potential_perm)

    info_dict = {
        "actual_sieving": {
            "avg": avg_actual_sieving,
            "spread": spread_actual_sieving,
        },
        "observed_sieving": {
            "avg": avg_observed_sieving,
            "spread": spread_observed_sieving,
        },
        # "observed_rejection": {"avg": avg_observed_rejection, "spread": spread_observed_rejection},
        "actual_rejection": {
            "avg": avg_actual_rejection,
            "spread": spread_actual_rejection,
        },
        "flux": {"avg": avg_flux, "spread": spread_flux},
        "H_feed": {"avg": avg_H_feed, "spread": spread_H_feed},
        "H_perm": {"avg": avg_H_perm, "spread": spread_H_perm},
        "Donnan_potential_feed": {
            "avg": avg_Donnan_potential_feed,
            "spread": spread_Donnan_potential_feed,
        },
        "Donnan_potential_perm": {
            "avg": avg_Donnan_potential_perm,
            "spread": spread_Donnan_potential_perm,
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


def data_comparison_plots(save_figure=True):
    markersize = 10
    fontsize = 14

    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    fig1, (ax1a, ax1b, ax1c) = plt.subplots(
        1, 3, dpi=75, figsize=(15, 5), constrained_layout=True, sharey=True
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

    S_predicted_Na_color = tol_bright_hex[0]
    S_predicted_Ca_color = tol_bright_hex[1]
    S_predicted_La_color = tol_bright_hex[2]
    # S_predicted_10_color = tol_bright_hex[3]
    # S_predicted_05_color = tol_bright_hex[4]
    S_measured_color = "black"

    for ax in [ax1a]:
        ax.set_title(
            "Sodium",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.set_xlabel(
            "Feed Concentration (mM)",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.set_ylabel(
            "Observed Sieving Coefficient", fontsize=fontsize, fontweight="bold"
        )
        ax.plot(
            [],
            [],
            color=S_predicted_Na_color,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="predicted",
        )
    for ax in [ax1b]:
        ax.set_title(
            "Calcium",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.set_xlabel(
            "Feed Concentration (mM)",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.plot(
            [],
            [],
            color=S_predicted_Ca_color,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="predicted",
        )
    for ax in [ax1c]:
        ax.set_title(
            "Lanthanum",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.set_xlabel(
            "Feed Concentration (mM)",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.plot(
            [],
            [],
            color=S_predicted_La_color,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="predicted",
        )
    for ax in fig1.axes:
        ax.plot(
            [],
            [],
            color=S_measured_color,
            marker="s",
            markersize=markersize,
            linestyle="None",
            label="measured",
        )
        ax.legend(loc="upper left", fontsize=fontsize - 2)
        ax.tick_params(
            direction="in", top=True, right=True, labelsize=fontsize - 2, labelleft=True
        )

    model_folder = Path(f"multi_component_case_studies/DATA_comparison/")
    # 45 characters (0-44) make up folder name before model name
    # multi_component_case_studies/DATA_comparison/

    case_study_list = [file for file in model_folder.iterdir()]

    Dm_over_l = 60  # um/s
    Dm_Cl = 2.03  # um2/s
    chloride_phi_star = 0.1

    for case_study_file in case_study_list:
        cation = str(case_study_file)[45:47]
        concentration = float(50)  # mM
        cation_list = [cation]
        if cation == "Na":
            chloride_multiplier = 1
            cation_phi_star = 0.4
            color = S_predicted_Na_color
            ax = ax1a
        if cation == "Ca":
            chloride_multiplier = 2
            cation_phi_star = 0.3
            color = S_predicted_Ca_color
            ax = ax1b
        if cation == "La":
            chloride_multiplier = 3
            cation_phi_star = 0.005
            color = S_predicted_La_color
            ax = ax1c
        inlet_concentration = {
            "feed": {
                cation: concentration,
                "Cl": chloride_multiplier * concentration,
            },
            "diafiltrate": {
                cation: 1e-10,
                "Cl": chloride_multiplier * 1e-10,
            },
        }

        l_um = Dm_Cl / Dm_over_l  # um
        l_m = l_um / 1e6  # m

        non_Donnan_partition_dict = {
            cation: cation_phi_star,
            "Cl": chloride_phi_star,
        }

        model = build_model(
            cation_list=cation_list,
            inlet_concentration=inlet_concentration,
            default_args=default_args,
            NFE_args=NFE_args,
            initialize_and_solve=False,
            data_membrane_thickness=l_m,
            non_Donnan_partition_dict=non_Donnan_partition_dict,
            save=False,
        )
        from_json(model, fname=case_study_file)

        average_variable_dict = get_model_averages(model, cation)

        x_value_predicted = value(
            model.fs.membrane.retentate_conc_mol_comp[0, 0, cation]
        )
        y_obs_data_predicted = average_variable_dict["observed_sieving"]["avg"]
        y_obs_err_predicted = average_variable_dict["observed_sieving"]["spread"]

        alpha = 1
        marker = "o"

        ax.errorbar(
            x_value_predicted,
            y_obs_data_predicted,
            yerr=y_obs_err_predicted,
            ecolor="black",
            capsize=3,
        )
        ax.plot(
            x_value_predicted,
            y_obs_data_predicted,
            color=color,
            marker=marker,
            alpha=alpha,
            markersize=markersize,
        )

    NF270_MC5_07_23_24_NaCl = {
        "conc_feed": [
            # 9.6199,
            30.5114,
            43.1343,
            54.5286,
            64.7312,
            74.1947,
            83.0378,
            91.3303,
            98.4631,
            105.1806,
        ],
        "sieving_obs": [
            # 0.1875,
            0.2296,
            0.3566,
            0.4416,
            0.5049,
            0.5511,
            0.5910,
            0.6251,
            0.6356,
            0.6680,
        ],
        "sieving_obs_error": [
            # 0.00312,
            0.00099,
            0.00070,
            0.00056,
            0.00047,
            0.00041,
            0.00037,
            0.00033,
            0.00031,
            0.00029,
        ],
    }

    ax1a.plot(
        NF270_MC5_07_23_24_NaCl["conc_feed"],
        NF270_MC5_07_23_24_NaCl["sieving_obs"],
        color=S_measured_color,
        marker="s",
        linestyle="None",
        alpha=alpha,
        markersize=markersize,
    )
    ax1a.errorbar(
        NF270_MC5_07_23_24_NaCl["conc_feed"],
        NF270_MC5_07_23_24_NaCl["sieving_obs"],
        yerr=NF270_MC5_07_23_24_NaCl["sieving_obs_error"],
        ecolor="black",
        capsize=3,
        linestyle="None",
    )

    NF270_MC3_07_11_24_SCaCl2 = {
        "conc_feed": [
            # 3.0588,
            10.4631,
            15.2044,
            19.6055,
            23.6953,
            27.5362,
            31.0213,
            34.3589,
            37.4959,
            40.3890,
        ],
        "sieving_obs": [
            # 0.1893,
            0.2077,
            0.3179,
            0.3572,
            0.3761,
            0.3927,
            0.4157,
            0.4313,
            0.4388,
            0.4500,
        ],
        "sieving_obs_error": [
            # 0.00982,
            0.00287,
            0.00198,
            0.00154,
            0.00127,
            0.00110,
            0.00098,
            0.00088,
            0.00081,
            0.00075,
        ],
    }

    ax1b.plot(
        NF270_MC3_07_11_24_SCaCl2["conc_feed"],
        NF270_MC3_07_11_24_SCaCl2["sieving_obs"],
        color=S_measured_color,
        marker="s",
        linestyle="None",
        alpha=alpha,
        markersize=markersize,
    )
    ax1b.errorbar(
        NF270_MC3_07_11_24_SCaCl2["conc_feed"],
        NF270_MC3_07_11_24_SCaCl2["sieving_obs"],
        yerr=NF270_MC3_07_11_24_SCaCl2["sieving_obs_error"],
        ecolor="black",
        capsize=3,
        linestyle="None",
    )

    NF270_MC2_05_21_24_LaCl3 = {
        "conc_feed": [
            # 1.4129,
            4.8072,
            7.0910,
            9.2925,
            11.3234,
            13.5457,
            15.5382,
            17.4718,
            19.3323,
            21.1177,
        ],
        "sieving_obs": [
            # 0.0429,
            0.0240,
            0.0286,
            0.0312,
            0.0335,
            0.0343,
            0.0358,
            0.0377,
            0.0391,
            0.0419,
        ],
        "sieving_obs_error": [
            # 0.02123,
            0.00624,
            0.00423,
            0.00323,
            0.00265,
            0.00221,
            0.00193,
            0.00172,
            0.00155,
            0.00142,
        ],
    }
    ax1c.plot(
        NF270_MC2_05_21_24_LaCl3["conc_feed"],
        NF270_MC2_05_21_24_LaCl3["sieving_obs"],
        color=S_measured_color,
        marker="s",
        linestyle="None",
        alpha=alpha,
        markersize=markersize,
    )
    ax1c.errorbar(
        NF270_MC2_05_21_24_LaCl3["conc_feed"],
        NF270_MC2_05_21_24_LaCl3["sieving_obs"],
        yerr=NF270_MC2_05_21_24_LaCl3["sieving_obs_error"],
        ecolor="black",
        capsize=3,
        linestyle="None",
    )

    if save_figure:
        fig1.savefig("sieving_data_comparison_single_salts.png", dpi=600)


def rejection_plots_equimolar(x_axis="ionic_strength", sieving=True, save_figure=True):
    markersize = 10
    fontsize = 14

    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    fig1, (
        ax1a,
        ax1b,
        ax1c,  # cation rejection
        # (ax2a, ax2b, ax2c),  # anion rejection
    ) = plt.subplots(1, 3, dpi=90, figsize=(13, 4), constrained_layout=True)

    if x_axis == "ionic_strength":
        for ax in [ax1a, ax1b, ax1c]:
            ax.set_xlabel(
                "Inlet Feed Ionic Strength (mM)",
                fontsize=fontsize,
                fontweight="bold",
            )
    else:
        ax1a.set_xlabel(
            "+1 Cation Concentration\nin Feed (mM)",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax1b.set_xlabel(
            "+2 Cation Concentration\nin Feed (mM)",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax1c.set_xlabel(
            "+3 Cation Concentration\nin Feed (mM)",
            fontsize=fontsize,
            fontweight="bold",
        )
    # elif x_axis == "cation_concentration":
    #     ax2a.set_xlabel(
    #         "Lithium Feed Concentration\n(mol/m$\mathbf{^3}$)",
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )
    #     ax2b.set_xlabel(
    #         "Cobalt Feed Concentration\n(mol/m$\mathbf{^3}$)",
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )
    #     ax2c.set_xlabel(
    #         "Aluminum Feed Concentration\n(mol/m$\mathbf{^3}$)",
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )

    # ax1a.set_title("+1 Cation", fontsize=fontsize, fontweight="bold")
    # ax1b.set_title("+2 Cation", fontsize=fontsize, fontweight="bold")
    # ax1c.set_title("+3 Cation", fontsize=fontsize, fontweight="bold")

    if sieving:
        ax1a.set_ylabel(
            "Observed Cation\nSieving Coefficient", fontsize=fontsize, fontweight="bold"
        )
        # ax2a.set_ylabel(
        #     "Observed Anionn\nSieving Coefficient", fontsize=fontsize, fontweight="bold"
        # )
    else:
        ax1a.set_ylabel(
            "Actual Cation Rejection (%)", fontsize=fontsize, fontweight="bold"
        )
        # ax2a.set_ylabel(
        #     "Actual Anion Rejection (%)", fontsize=fontsize, fontweight="bold"
        # )

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
    li_color = tol_bright_hex[0]
    co_color = tol_bright_hex[1]
    al_color = tol_bright_hex[2]
    li_co_color = tol_bright_hex[3]
    li_al_color = tol_bright_hex[4]
    co_al_color = tol_bright_hex[5]
    li_co_al_color = tol_bright_hex[6]

    legend_dict = {
        "lithium": {
            "marker": "o",
            "ax": ax1a,
            "salt_colors": {
                "single salt": li_color,
                "two salt: +1 & +2": li_co_color,
                "two salt: +1 & +3": li_al_color,
                # "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
        "cobalt": {
            "marker": "v",
            "ax": ax1b,
            "salt_colors": {
                "single salt": co_color,
                "two salt: +1 & +2": li_co_color,
                "two salt: +2 & +3": co_al_color,
                # "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
        "aluminum": {
            "marker": "^",
            "ax": ax1c,
            "salt_colors": {
                "single salt": al_color,
                "two salt: +1 & +3": li_al_color,
                "two salt: +2 & +3": co_al_color,
                # "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
    }

    # ax2a.plot([], [], "ko", markersize=markersize, label="Li")
    # ax2a.plot([], [], "k*", markersize=markersize, label="Cl")
    # ax2b.plot([], [], "kv", markersize=markersize, label="Co")
    # ax2b.plot([], [], "k*", markersize=markersize, label="Cl")
    # ax2c.plot([], [], "k^", markersize=markersize, label="Al")
    # ax2c.plot([], [], "k*", markersize=markersize, label="Cl")
    # ax2a.plot([], [], marker="None", linestyle="None", label="Solution (color)")
    # ax2b.plot([], [], marker="None", linestyle="None", label="Solution (color)")
    # ax2c.plot([], [], marker="None", linestyle="None", label="Solution (color)")
    for cation_dict in legend_dict.values():
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
        # bbox_to_anchor=(0.85, -0.3),
        title="+1 Cation in:",
        title_fontsize=fontsize - 2,
    )
    ax1b.legend(
        loc="best",
        fontsize=fontsize - 2,
        # bbox_to_anchor=(0.85, -0.3),
        title="+2 Cation in:",
        title_fontsize=fontsize - 2,
    )
    ax1c.legend(
        loc="best",
        fontsize=fontsize - 2,
        # bbox_to_anchor=(0.85, -0.3),
        title="+3 Cation in:",
        title_fontsize=fontsize - 2,
    )

    for ax in fig1.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        if x_axis == "ionic_strength":
            ax.set_xlim(0, 900)

    if x_axis == "ionic_strength":
        model_folder_1 = Path("multi_component_case_studies/single_salt/IS")
        # 44 characters (0-43) make up folder name before model name
        # multi_component_case_studies/single_salt/IS/
        model_folder_2 = Path("multi_component_case_studies/two_salt/IS")
        # 41 characters (0-40) make up folder name before model name
        # multi_component_case_studies/two_salt/IS/
        # model_folder_3 = Path("multi_component_case_studies/three_salt/IS")
        # 43 characters (0-42) make up folder name before model name
        # multi_component_case_studies/three_salt/IS/
    else:
        model_folder_1 = Path("multi_component_case_studies/single_salt/CONC")
        # 46 characters (0-45) make up folder name before model name
        # multi_component_case_studies/single_salt/CONC/
        model_folder_2 = Path("multi_component_case_studies/two_salt/CONC")
        # 43 characters (0-42) make up folder name before model name
        # multi_component_case_studies/two_salt/CONC/
        # model_folder_3 = Path("multi_component_case_studies/three_salt/CONC")
        # 45 characters (0-44) make up folder name before model name
        # multi_component_case_studies/three_salt/CONC/

    case_study_list_1 = [file for file in model_folder_1.iterdir()]
    case_study_list_2 = [file for file in model_folder_2.iterdir()]
    # case_study_list_3 = [file for file in model_folder_3.iterdir()]
    case_studies = {
        "single": case_study_list_1,
        "two": case_study_list_2,
        # "three": case_study_list_3,
    }

    for type, case_study_files in case_studies.items():
        for case_study in case_study_files:
            if type == "single":
                if x_axis == "ionic_strength":
                    cation = str(case_study)[50:52]
                    chloride_multiplier = float(str(case_study)[54])
                else:
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
                if x_axis == "ionic_strength":
                    cation_1 = str(case_study)[47:49]
                    cation_2 = str(case_study)[49:51]
                    chloride_multiplier = float(str(case_study)[53])
                else:
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
            elif type == "three":
                if x_axis == "ionic_strength":
                    cation_1 = str(case_study)[49:51]
                    cation_2 = str(case_study)[51:53]
                    cation_3 = str(case_study)[53:55]
                    chloride_multiplier = float(str(case_study)[57])
                else:
                    cation_1 = str(case_study)[51:53]
                    cation_2 = str(case_study)[53:55]
                    cation_3 = str(case_study)[55:57]
                    chloride_multiplier = float(str(case_study)[59])
                concentration = float(50)  # mM
                cation_list = [cation_1, cation_2, cation_3]
                inlet_concentration = {
                    "feed": {
                        cation_1: concentration,
                        cation_2: concentration,
                        cation_3: concentration,
                        "Cl": chloride_multiplier * concentration,
                    },
                    "diafiltrate": {
                        cation_1: 1e-10,
                        cation_2: 1e-10,
                        cation_3: 1e-10,
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

            for solute in model.fs.membrane.cations:
                average_variable_dict = get_model_averages(model, solute)

                if x_axis == "ionic_strength":
                    x_value = value(model.fs.membrane.total_feed_ionic_strength[0])
                elif x_axis == "cation_concentration":
                    x_value = value(
                        model.fs.membrane.retentate_conc_mol_comp[0, 0, solute]
                    )

                alpha = 1

                if solute == "Li":
                    marker = "o"
                    ax_rej = ax1a
                    if type == "single":
                        color = li_color
                    elif type == "two":
                        if cation_2 == "Co":
                            color = li_co_color
                        elif cation_2 == "Al":
                            color = li_al_color

                elif solute == "Co":
                    marker = "v"
                    ax_rej = ax1b
                    if type == "single":
                        color = co_color
                    elif type == "two":
                        if cation_1 == "Li":
                            color = li_co_color
                        elif cation_2 == "Al":
                            color = co_al_color

                elif solute == "Al":
                    marker = "^"
                    ax_rej = ax1c
                    if type == "single":
                        color = al_color
                    elif type == "two":
                        if cation_1 == "Li":
                            color = li_al_color
                        elif cation_1 == "Co":
                            color = co_al_color

                # elif solute == "Cl":
                #     marker = "*"

                #     if type == "single":
                #         if model.fs.membrane.cations.at(1) == "Li":
                #             color = li_color
                #             ax_rej = ax2a
                #         elif model.fs.membrane.cations.at(1) == "Co":
                #             color = co_color
                #             ax_rej = ax2b
                #         elif model.fs.membrane.cations.at(1) == "Al":
                #             color = al_color
                #             ax_rej = ax2c
                #     elif type == "two":
                #         if cation_1 == "Li" and cation_2 == "Co":
                #             color = li_co_color
                #             ax_rej = [ax2a, ax2b]
                #         elif cation_1 == "Li" and cation_2 == "Al":
                #             color = li_al_color
                #             ax_rej = [ax2a, ax2c]
                #         elif cation_1 == "Co" and cation_2 == "Al":
                #             color = co_al_color
                #             ax_rej = [ax2b, ax2c]
                #     elif type == "three":
                #         ax_rej = [ax2a, ax2b, ax2c]

                if type == "three":
                    color = li_co_al_color

                if sieving:
                    y_data = average_variable_dict["observed_sieving"]["avg"]
                    y_err = average_variable_dict["observed_sieving"]["spread"]
                else:
                    y_data = average_variable_dict["actual_rejection"]["avg"]
                    y_err = average_variable_dict["actual_rejection"]["spread"]

                if isinstance(ax_rej, list):
                    for ax in ax_rej:
                        ax.plot(
                            x_value,
                            y_data,
                            color=color,
                            marker=marker,
                            alpha=alpha,
                            markersize=markersize,
                        )
                        ax.errorbar(
                            x_value,
                            y_data,
                            yerr=y_err,
                            ecolor=color,
                            capsize=3,
                        )
                else:
                    ax_rej.plot(
                        x_value,
                        y_data,
                        color=color,
                        marker=marker,
                        alpha=alpha,
                        markersize=markersize,
                    )
                    ax_rej.errorbar(
                        x_value,
                        y_data,
                        yerr=y_err,
                        ecolor=color,
                        capsize=3,
                    )
    if sieving == True:
        y_axis = "sieving"
    else:
        y_axis = "rejection"
    if save_figure:
        plt.savefig(f"{y_axis}_versus_{x_axis}.png", dpi=600)


def h_plots_equimolar(x_axis="ionic_strength", inset=True, save_figure=True):
    markersize = 10
    fontsize = 14

    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    fig1, (
        ax3a,
        ax3b,
        ax3c,  # H_feed
        # (ax4a, ax4b, ax4c, ax4d),  # H_permeate
    ) = plt.subplots(1, 3, dpi=90, figsize=(15, 5), constrained_layout=True)

    if x_axis == "ionic_strength":
        for ax in fig1.axes:
            ax.set_xlabel(
                "Inlet Feed Ionic Strength (mM)",
                fontsize=fontsize,
                fontweight="bold",
            )
    # elif x_axis == "cation_concentration":
    #     ax4a.set_xlabel(
    #         "Lithium Feed Concentration\n(mol/m$\mathbf{^3}$)",
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )
    #     ax4b.set_xlabel(
    #         "Cobalt Feed Concentration\n(mol/m$\mathbf{^3}$)",
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )
    #     ax4c.set_xlabel(
    #         "Aluminum Feed Concentration\n(mol/m$\mathbf{^3}$)",
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )
    #     ax4d.set_xlabel(
    #         "Cation Feed Concentration\n(mol/m$\mathbf{^3}$)",
    #         fontsize=fontsize,
    #         fontweight="bold",
    #     )

    # ax3a.set_title("+1 Cation", fontsize=fontsize, fontweight="bold")
    # ax3b.set_title("+2 Cation", fontsize=fontsize, fontweight="bold")
    # ax3c.set_title("+3 Cation", fontsize=fontsize, fontweight="bold")
    # ax3d.set_title("Chloride", fontsize=fontsize, fontweight="bold")

    ax3a.set_ylabel(
        "$\mathbf{c_{membrane}/c_{interface}}$", fontsize=fontsize, fontweight="bold"
    )
    # ax4a.set_ylabel(
    #     "$\mathbf{c_{membrane}/c_{permeate}}$", fontsize=fontsize, fontweight="bold"
    # )

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
    li_color = tol_bright_hex[0]
    co_color = tol_bright_hex[1]
    al_color = tol_bright_hex[2]
    li_co_color = tol_bright_hex[3]
    li_al_color = tol_bright_hex[4]
    co_al_color = tol_bright_hex[5]
    li_co_al_color = tol_bright_hex[6]

    legend_dict = {
        "lithium": {
            "marker": "o",
            "ax": ax3a,
            "salt_colors": {
                "single salt": li_color,
                "two salt: +1 & +2": li_co_color,
                "two salt: +1 & +3": li_al_color,
                # "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
        "cobalt": {
            "marker": "v",
            "ax": ax3b,
            "salt_colors": {
                "single salt": co_color,
                "two salt: +1 & +2": li_co_color,
                "two salt: +2 & +3": co_al_color,
                # "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
        "aluminum": {
            "marker": "^",
            "ax": ax3c,
            "salt_colors": {
                "single salt": al_color,
                "two salt: +1 & +3": li_al_color,
                "two salt: +2 & +3": co_al_color,
                # "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
    }

    for cation_dict in legend_dict.values():
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

    ax3a.legend(
        loc="best",
        fontsize=fontsize - 2,
        # bbox_to_anchor=(0.85, -0.3),
        title="+1 Cation in:",
        title_fontsize=fontsize - 2,
    )
    ax3b.legend(
        loc="best",
        fontsize=fontsize - 2,
        # bbox_to_anchor=(0.85, -0.3),
        title="+2 Cation in:",
        title_fontsize=fontsize - 2,
    )
    ax3c.legend(
        loc="best",
        fontsize=fontsize - 2,
        # bbox_to_anchor=(0.85, -0.3),
        title="+3 Cation in:",
        title_fontsize=fontsize - 2,
    )

    for ax in fig1.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        if x_axis == "ionic_strength":
            ax.set_xlim(0, 900)

    # if inset:
    #     # create inset axes for overlapping points
    #     inax3a = ax3a.inset_axes([0.4, 0.35, 0.55, 0.6])
    #     inax3b = ax3b.inset_axes([0.4, 0.35, 0.55, 0.6])
    #     inax3c = ax3c.inset_axes([0.4, 0.35, 0.55, 0.6])
    #     # inax3d = ax3d.inset_axes([0.1, 0.65, 0.45, 0.4])

    #     inax4a = ax4a.inset_axes([0.4, 0.35, 0.55, 0.6])
    #     inax4b = ax4b.inset_axes([0.4, 0.35, 0.55, 0.6])
    #     inax4c = ax4c.inset_axes([0.4, 0.35, 0.55, 0.6])
    #     # inax4d = ax4d.inset_axes([0.1, 0.65, 0.45, 0.4])

    #     inset_axes_dict = {
    #         ax3a: inax3a,
    #         ax3b: inax3b,
    #         ax3c: inax3c,
    #         # ax3d: inax3d,
    #         ax4a: inax4a,
    #         ax4b: inax4b,
    #         ax4c: inax4c,
    #         # ax4d: inax4d,
    #     }

    #     for ax, inax in inset_axes_dict.items():
    #         inax.set_xlim(25, 225)
    #         inax.tick_params(direction="in", top=True, right=True, labelsize=10)
    #         ax.indicate_inset_zoom(inax, edgecolor="black")

    #     inax3a.set_ylim(0.2, 0.57)
    #     inax3b.set_ylim(0.2, 1)
    #     inax3c.set_ylim(0.2, 1.2)
    #     # inax3d.set_ylim(0.005, 0.03)

    #     inax4a.set_ylim(0.15, 1)
    #     inax4b.set_ylim(0.4, 3.5)
    #     inax4c.set_ylim(0.5, 3.1)
    #     # inax4d.set_ylim(0.001, 0.02)

    if x_axis == "ionic_strength":
        model_folder_1 = Path("multi_component_case_studies/single_salt/IS")
        # 44 characters (0-43) make up folder name before model name
        # multi_component_case_studies/single_salt/IS/
        model_folder_2 = Path("multi_component_case_studies/two_salt/IS")
        # 41 characters (0-40) make up folder name before model name
        # multi_component_case_studies/two_salt/IS/
        model_folder_3 = Path("multi_component_case_studies/three_salt/IS")
        # 43 characters (0-42) make up folder name before model name
        # multi_component_case_studies/three_salt/IS/
    else:
        model_folder_1 = Path("multi_component_case_studies/single_salt/CONC")
        # 46 characters (0-45) make up folder name before model name
        # multi_component_case_studies/single_salt/CONC/
        model_folder_2 = Path("multi_component_case_studies/two_salt/CONC")
        # 43 characters (0-42) make up folder name before model name
        # multi_component_case_studies/two_salt/CONC/
        model_folder_3 = Path("multi_component_case_studies/three_salt/CONC")
        # 45 characters (0-44) make up folder name before model name
        # multi_component_case_studies/three_salt/CONC/

    case_study_list_1 = [file for file in model_folder_1.iterdir()]
    case_study_list_2 = [file for file in model_folder_2.iterdir()]
    case_study_list_3 = [file for file in model_folder_3.iterdir()]
    case_studies = {
        "single": case_study_list_1,
        "two": case_study_list_2,
        "three": case_study_list_3,
    }

    for type, case_study_files in case_studies.items():
        for case_study in case_study_files:
            if type == "single":
                if x_axis == "ionic_strength":
                    cation = str(case_study)[50:52]
                    chloride_multiplier = float(str(case_study)[54])
                else:
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
                if x_axis == "ionic_strength":
                    cation_1 = str(case_study)[47:49]
                    cation_2 = str(case_study)[49:51]
                    chloride_multiplier = float(str(case_study)[53])
                else:
                    cation_1 = str(case_study)[49:51]
                    cation_2 = str(case_study)[51:53]
                    chloride_multiplier = float(str(case_study)[55])
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
            elif type == "three":
                if x_axis == "ionic_strength":
                    cation_1 = str(case_study)[49:51]
                    cation_2 = str(case_study)[51:53]
                    cation_3 = str(case_study)[53:55]
                    chloride_multiplier = float(str(case_study)[57])
                else:
                    cation_1 = str(case_study)[51:53]
                    cation_2 = str(case_study)[53:55]
                    cation_3 = str(case_study)[55:57]
                    chloride_multiplier = float(str(case_study)[59])
                concentration = float(50)  # mM
                cation_list = [cation_1, cation_2, cation_3]
                inlet_concentration = {
                    "feed": {
                        cation_1: concentration,
                        cation_2: concentration,
                        cation_3: concentration,
                        "Cl": chloride_multiplier * concentration,
                    },
                    "diafiltrate": {
                        cation_1: 1e-10,
                        cation_2: 1e-10,
                        cation_3: 1e-10,
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

            for solute in model.fs.membrane.cations:
                average_variable_dict = get_model_averages(model, solute)

                if x_axis == "ionic_strength":
                    x_value = value(model.fs.membrane.total_feed_ionic_strength[0])
                elif x_axis == "cation_concentration":
                    x_value = value(
                        model.fs.membrane.retentate_conc_mol_comp[0, 0, solute]
                    )

                alpha = 1

                if solute == "Li":
                    marker = "o"
                    ax_hfeed = ax3a
                    # ax_hperm = ax4a
                    # if inset:
                    #     inax_hfeed = inax3a
                    #     inax_hperm = inax4a
                    if type == "single":
                        color = li_color
                    elif type == "two":
                        if cation_2 == "Co":
                            color = li_co_color
                        elif cation_2 == "Al":
                            color = li_al_color

                elif solute == "Co":
                    marker = "v"
                    ax_hfeed = ax3b
                    # ax_hperm = ax4b
                    # if inset:
                    #     inax_hfeed = inax3b
                    #     inax_hperm = inax4b
                    if type == "single":
                        color = co_color
                    elif type == "two":
                        if cation_1 == "Li":
                            color = li_co_color
                        elif cation_2 == "Al":
                            color = co_al_color

                elif solute == "Al":
                    marker = "^"
                    ax_hfeed = ax3c
                    # ax_hperm = ax4c
                    # if inset:
                    #     inax_hfeed = inax3c
                    #     inax_hperm = inax4c
                    if type == "single":
                        color = al_color
                    elif type == "two":
                        if cation_1 == "Li":
                            color = li_al_color
                        elif cation_1 == "Co":
                            color = co_al_color

                elif solute == "Cl":
                    marker = "*"
                    # ax_hfeed = ax3d
                    # ax_hperm = ax4d
                    # if inset:
                    # inax_hfeed = inax3d
                    # inax_hperm = inax4d
                    if type == "single":
                        if model.fs.membrane.cations.at(1) == "Li":
                            color = li_color
                        elif model.fs.membrane.cations.at(1) == "Co":
                            color = co_color
                        elif model.fs.membrane.cations.at(1) == "Al":
                            color = al_color
                    elif type == "two":
                        if cation_1 == "Li" and cation_2 == "Co":
                            color = li_co_color
                        elif cation_1 == "Li" and cation_2 == "Al":
                            color = li_al_color
                        elif cation_1 == "Co" and cation_2 == "Al":
                            color = co_al_color

                if type == "three":
                    color = li_co_al_color

                dict = {
                    "H_feed": [ax_hfeed]  # , inax_hfeed],
                    # "H_perm": [ax_hperm, inax_hperm],
                }
                for h, ax_list in dict.items():
                    ax_list[0].plot(
                        x_value,
                        average_variable_dict[h]["avg"],
                        color=color,
                        marker=marker,
                        alpha=alpha,
                        markersize=markersize,
                    )
                    ax_list[0].errorbar(
                        x_value,
                        average_variable_dict[h]["avg"],
                        yerr=average_variable_dict[h]["spread"],
                        ecolor=color,
                        capsize=4,
                    )

                    # if inset:
                    #     ax_list[1].plot(
                    #         x_value,
                    #         average_variable_dict[h]["avg"],
                    #         color=color,
                    #         marker=marker,
                    #         alpha=alpha,
                    #         markersize=markersize,
                    #     )
                    #     ax_list[1].errorbar(
                    #         x_value,
                    #         average_variable_dict[h]["avg"],
                    #         yerr=average_variable_dict[h]["spread"],
                    #         ecolor=color,
                    #         capsize=4,
                    #     )

    if save_figure:
        plt.savefig(f"partition_versus_{x_axis}.png", dpi=600)


def combined_plots_vary_salt_ratio(inset=True, save_figure=True):
    """
    Plots rejection versus ionic strength of bulk fluid and feed.
    """
    markersize = 8
    fontsize = 14

    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [
        NFE_module_length,
        NFE_boundary_layer_thickness,
        NFE_membrane_thickness,
    ]

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
    color_033 = tol_bright_hex[0]
    color_05 = tol_bright_hex[1]
    color_1 = tol_bright_hex[2]
    color_2 = tol_bright_hex[3]
    color_3 = tol_bright_hex[4]

    # each plot is a salt solution
    fig1, (
        (ax1a, ax1b, ax1c),  # rejection
        (ax2a, ax2b, ax2c),  # solute flux
        (ax3a, ax3b, ax3c),  # H_feed
        (ax4a, ax4b, ax4c),  # H_permeate
    ) = plt.subplots(4, 3, dpi=75, figsize=(12, 12), constrained_layout=True)
    fig1.suptitle("LiCl + CoCl$_2$", fontsize=fontsize, fontweight="bold")
    fig2, (
        (ax5a, ax5b, ax5c),  # rejection
        (ax6a, ax6b, ax6c),  # solute flux
        (ax7a, ax7b, ax7c),  # H_feed
        (ax8a, ax8b, ax8c),  # H_permeate
    ) = plt.subplots(4, 3, dpi=75, figsize=(12, 12), constrained_layout=True)
    fig2.suptitle("LiCl + AlCl$_3$", fontsize=fontsize, fontweight="bold")
    fig3, (
        (ax9a, ax9b, ax9c),  # rejection
        (ax10a, ax10b, ax10c),  # solute flux
        (ax11a, ax11b, ax11c),  # H_feed
        (ax12a, ax12b, ax12c),  # H_permeate
    ) = plt.subplots(4, 3, dpi=75, figsize=(12, 12), constrained_layout=True)
    fig3.suptitle("CoCl$_2$ + AlCl$_3$", fontsize=fontsize, fontweight="bold")

    x_axis_labels = [
        ax4a,
        ax4b,
        ax4c,
        ax8a,
        ax8b,
        ax8c,
        ax12a,
        ax12b,
        ax12c,
    ]

    for ax in x_axis_labels:
        ax.set_xlabel(
            "Inlet Feed Ionic Strength\n(mol/m$^3$)",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax1a, ax5a, ax9a]:
        ax.plot(
            [],
            [],
            color=color_033,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="0.33",
        )
        ax.plot(
            [],
            [],
            color=color_05,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="0.5",
        )
        ax.plot(
            [],
            [],
            color=color_1,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="1",
        )
        ax.plot(
            [],
            [],
            color=color_2,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="2",
        )
        ax.plot(
            [],
            [],
            color=color_3,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="3",
        )
        ax.legend(loc="best", fontsize=fontsize)
        # ax.legend(loc="best", fontsize=fontsize, bbox_to_anchor=(0.85, -0.5))

    for ax in [ax1a, ax5a]:
        ax.set_title("Lithium", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel("Actual Ion\nRejection (%)", fontsize=fontsize, fontweight="bold")
    for ax in [ax1b, ax9a]:
        ax.set_title("Cobalt", fontsize=fontsize, fontweight="bold")
    for ax in [ax5b, ax9b]:
        ax.set_title("Aluminum", fontsize=fontsize, fontweight="bold")
    for ax in [ax1c, ax5c, ax9c]:
        ax.set_title("Chloride", fontsize=fontsize, fontweight="bold")
    for ax in [ax2a, ax6a, ax10a]:
        ax.set_ylabel("Ion Flux (mol/m$^2$/h)", fontsize=fontsize, fontweight="bold")
    for ax in [ax3a, ax7a, ax11a]:
        ax.set_ylabel(
            "$c_{membrane}/c_{interface}$", fontsize=fontsize, fontweight="bold"
        )
    for ax in [ax4a, ax8a, ax12a]:
        ax.set_ylabel(
            "$c_{membrane}/c_{permeate}$", fontsize=fontsize, fontweight="bold"
        )

    IS_min = 25
    IS_max = 225
    for ax in fig1.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.set_xlim(IS_min, IS_max)
    for ax in fig2.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.set_xlim(IS_min, IS_max)
    for ax in fig3.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.set_xlim(IS_min, IS_max)

    # if inset:
    #     # create inset axes for overlapping points
    #     # inax2a = ax2a.inset_axes([0.1, 0.65, 0.45, 0.4])
    #     # inax2b = ax2b.inset_axes([0.1, 0.65, 0.45, 0.4])
    #     # inax2c = ax2c.inset_axes([0.1, 0.65, 0.45, 0.4])
    #     # inax2d = ax2d.inset_axes([0.1, 0.65, 0.45, 0.4])

    #     inax3a = ax3a.inset_axes([0.45, 0.4, 0.45, 0.5])
    #     inax3b = ax3b.inset_axes([0.45, 0.4, 0.45, 0.5])
    #     inax3c = ax3c.inset_axes([0.45, 0.4, 0.45, 0.5])
    #     # inax3d = ax3d.inset_axes([0.1, 0.65, 0.45, 0.4])

    #     inax4a = ax4a.inset_axes([0.45, 0.4, 0.45, 0.5])
    #     inax4b = ax4b.inset_axes([0.45, 0.4, 0.45, 0.5])
    #     inax4c = ax4c.inset_axes([0.45, 0.4, 0.45, 0.5])
    #     # inax4d = ax4d.inset_axes([0.1, 0.65, 0.45, 0.4])

    #     inset_axes_dict = {
    #         # ax2a: inax2a,
    #         # ax2b: inax2b,
    #         # ax2c: inax2c,
    #         # ax2d: inax2d,
    #         ax3a: inax3a,
    #         ax3b: inax3b,
    #         ax3c: inax3c,
    #         # ax3d: inax3d,
    #         ax4a: inax4a,
    #         ax4b: inax4b,
    #         ax4c: inax4c,
    #         # ax4d: inax4d,
    #     }

    #     for ax, inax in inset_axes_dict.items():
    #         inax.set_xlim(25, 225)
    #         inax.tick_params(direction="in", top=True, right=True, labelsize=10)
    #         ax.indicate_inset_zoom(inax, edgecolor="black")

    #     # inax2a.set_ylim(0, 3)
    #     # inax2b.set_ylim(0, 0.9)
    #     # inax2c.set_ylim(0, 0.35)
    #     # inax2d.set_ylim(0, 3)

    #     inax3a.set_ylim(0.18, 0.57)
    #     inax3b.set_ylim(0.2, 1.42)
    #     inax3c.set_ylim(0.18, 1.2)
    #     # inax3d.set_ylim(0.005, 0.03)

    #     inax4a.set_ylim(0.2, 1)
    #     inax4b.set_ylim(0.4, 3.5)
    #     inax4c.set_ylim(0.5, 3.1)
    #     # inax4d.set_ylim(0.001, 0.02)

    ionic_strengths = ["050_mM", "075_mM", "100_mM", "150_mM", "200_mM"]  # , "400_mM"]

    for ionic_strength in ionic_strengths:
        model_folder = Path(
            f"multi_component_case_studies/vary_salt_ratio/{ionic_strength}"
        )
        # 52 characters (0-51) make up folder name before model name
        # multi_component_case_studies/vary_salt_ratio/100_mM/

        for case_study_file in model_folder.iterdir():
            if str(case_study_file)[52:55] == "0.5":
                ratio = str(case_study_file)[52:55]
                cation_1 = str(case_study_file)[56:58]
                cation_2 = str(case_study_file)[58:60]
                chloride_multiplier = float(str(case_study_file)[62])
            elif str(case_study_file)[52:55] == "0.3":
                ratio = str(case_study_file)[52:56]
                cation_1 = str(case_study_file)[59:61]
                cation_2 = str(case_study_file)[61:63]
                chloride_multiplier = float(str(case_study_file)[65])
            else:
                ratio = str(case_study_file)[52]
                cation_1 = str(case_study_file)[54:56]
                cation_2 = str(case_study_file)[56:58]
                chloride_multiplier = float(str(case_study_file)[60])
            concentration = float(50)  # mM
            model = build_model(
                cation_list=[cation_1, cation_2],
                inlet_concentration={
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
                },
                default_args=default_args,
                H_feed_guess=1,
                H_permeate_guess=1,
                NFE_args=NFE_args,
                initialize=False,
            )
            from_json(model, fname=case_study_file)

            feed_ionic_strength_val = value(
                model.fs.membrane.total_feed_ionic_strength[0]
            )

            for solute in model.fs.membrane.solutes:
                observed_rejection = []
                actual_rejection = []
                flux = []
                H_feed = []
                H_perm = []
                bl_convection_by_x = []
                bl_convection_dict_by_x = {}
                bl_diffusion_by_x = []
                bl_diffusion_dict_by_x = {}
                bl_electromigration_by_x = []
                bl_electromigration_dict_by_x = {}
                mem_convection_by_x = []
                mem_convection_dict_by_x = {}
                mem_diffusion_by_x = []
                mem_diffusion_dict_by_x = {}
                mem_electromigration_by_x = []
                mem_electromigration_dict_by_x = {}

                for t in model.fs.membrane.time:
                    for x in model.fs.membrane.dimensionless_module_length:
                        if x != 0:
                            observed_rejection.append(
                                value(
                                    model.fs.membrane.observed_rejection_percent[
                                        t, x, solute
                                    ]
                                )
                            )
                            actual_rejection.append(
                                value(
                                    model.fs.membrane.actual_rejection_percent[
                                        t, x, solute
                                    ]
                                )
                            )
                            flux.append(
                                value(model.fs.membrane.molar_ion_flux[t, x, solute])
                            )
                            H_feed.append(
                                value(
                                    model.fs.membrane.overall_partition_coefficient_feed_side[
                                        t, x, solute
                                    ]
                                )
                            )
                            H_perm.append(
                                value(
                                    model.fs.membrane.overall_partition_coefficient_permeate_side[
                                        t, x, solute
                                    ]
                                )
                            )
                            for (
                                z_bl
                            ) in (
                                model.fs.membrane.dimensionless_boundary_layer_thickness
                            ):
                                bl_convection_by_x.append(
                                    value(
                                        model.fs.membrane.boundary_layer_convective_flux[
                                            0, x, z_bl, solute
                                        ]
                                    )
                                )
                                bl_diffusion_by_x.append(
                                    value(
                                        model.fs.membrane.boundary_layer_diffusive_flux[
                                            0, x, z_bl, solute
                                        ]
                                    )
                                )
                                bl_electromigration_by_x.append(
                                    value(
                                        model.fs.membrane.boundary_layer_electromigrative_flux[
                                            0, x, z_bl, solute
                                        ]
                                    )
                                )
                            bl_convection_dict_by_x[f"{x}"] = bl_convection_by_x
                            bl_diffusion_dict_by_x[f"{x}"] = bl_diffusion_by_x
                            bl_electromigration_dict_by_x[f"{x}"] = (
                                bl_electromigration_by_x
                            )
                            bl_convection_by_x = []
                            bl_diffusion_by_x = []
                            bl_electromigration_by_x = []

                            for (
                                z_mem
                            ) in model.fs.membrane.dimensionless_membrane_thickness:
                                mem_convection_by_x.append(
                                    value(
                                        model.fs.membrane.membrane_convective_flux[
                                            0, x, z_mem, solute
                                        ]
                                    )
                                )
                                mem_diffusion_by_x.append(
                                    value(
                                        model.fs.membrane.membrane_diffusive_flux[
                                            0, x, z_mem, solute
                                        ]
                                    )
                                )
                                mem_electromigration_by_x.append(
                                    value(
                                        model.fs.membrane.membrane_electromigrative_flux[
                                            0, x, z_mem, solute
                                        ]
                                    )
                                )
                            mem_convection_dict_by_x[f"{x}"] = mem_convection_by_x
                            mem_diffusion_dict_by_x[f"{x}"] = mem_diffusion_by_x
                            mem_electromigration_dict_by_x[f"{x}"] = (
                                mem_electromigration_by_x
                            )
                            mem_convection_by_x = []
                            mem_diffusion_by_x = []
                            mem_electromigration_by_x = []

                avg_observed_rejection = np.average(observed_rejection)
                spread_observed_rejection = calculate_spread(observed_rejection)

                avg_actual_rejection = np.average(actual_rejection)
                spread_actual_rejection = calculate_spread(actual_rejection)

                avg_flux = np.average(flux)
                spread_flux = calculate_spread(flux)

                avg_H_feed = np.average(H_feed)
                spread_H_feed = calculate_spread(H_feed)

                avg_H_perm = np.average(H_perm)
                spread_H_perm = calculate_spread(H_perm)

                bl_convection_averaged_over_z = [
                    sum(bl_convection_dict_by_x[k]) / len(bl_convection_dict_by_x[k])
                    for k in bl_convection_dict_by_x.keys()
                ]
                avg_bl_convection = np.average(bl_convection_averaged_over_z)
                spread_bl_convection = calculate_spread(bl_convection_averaged_over_z)

                bl_diffusion_averaged_over_z = [
                    sum(bl_diffusion_dict_by_x[k]) / len(bl_diffusion_dict_by_x[k])
                    for k in bl_diffusion_dict_by_x.keys()
                ]
                avg_bl_diffusion = np.average(bl_diffusion_averaged_over_z)
                spread_bl_diffusion = calculate_spread(bl_diffusion_averaged_over_z)

                bl_electromigration_averaged_over_z = [
                    sum(bl_electromigration_dict_by_x[k])
                    / len(bl_electromigration_dict_by_x[k])
                    for k in bl_electromigration_dict_by_x.keys()
                ]
                avg_bl_electromigration = np.average(
                    bl_electromigration_averaged_over_z
                )
                spread_bl_electromigration = calculate_spread(
                    bl_electromigration_averaged_over_z
                )

                mem_convection_averaged_over_z = [
                    sum(mem_convection_dict_by_x[k]) / len(mem_convection_dict_by_x[k])
                    for k in mem_convection_dict_by_x.keys()
                ]
                avg_mem_convection = np.average(mem_convection_averaged_over_z)
                spread_mem_convection = calculate_spread(mem_convection_averaged_over_z)

                mem_diffusion_averaged_over_z = [
                    sum(mem_diffusion_dict_by_x[k]) / len(mem_diffusion_dict_by_x[k])
                    for k in mem_diffusion_dict_by_x.keys()
                ]
                avg_mem_diffusion = np.average(mem_diffusion_averaged_over_z)
                spread_mem_diffusion = calculate_spread(mem_diffusion_averaged_over_z)

                mem_electromigration_averaged_over_z = [
                    sum(mem_electromigration_dict_by_x[k])
                    / len(mem_electromigration_dict_by_x[k])
                    for k in mem_electromigration_dict_by_x.keys()
                ]
                avg_mem_electromigration = np.average(
                    mem_electromigration_averaged_over_z
                )
                spread_mem_electromigration = calculate_spread(
                    mem_electromigration_averaged_over_z
                )

                alpha = 1
                marker = "o"

                if ratio == "0.33":
                    color = color_033
                elif ratio == "0.5":
                    color = color_05
                elif ratio == "1":
                    color = color_1
                elif ratio == "2":
                    color = color_2
                elif ratio == "3":
                    color = color_3

                if cation_1 == "Li" and cation_2 == "Co":
                    rejection_axes = [ax1a, ax1b, ax1c]
                    flux_axes = [ax2a, ax2b, ax2c]
                    hfeed_axes = [ax3a, ax3b, ax3c]
                    hperm_axes = [ax4a, ax4b, ax4c]
                elif cation_1 == "Li" and cation_2 == "Al":
                    rejection_axes = [ax5a, ax5b, ax5c]
                    flux_axes = [ax6a, ax6b, ax6c]
                    hfeed_axes = [ax7a, ax7b, ax7c]
                    hperm_axes = [ax8a, ax8b, ax8c]
                elif cation_1 == "Co" and cation_2 == "Al":
                    rejection_axes = [ax9a, ax9b, ax9c]
                    flux_axes = [ax10a, ax10b, ax10c]
                    hfeed_axes = [ax11a, ax11b, ax11c]
                    hperm_axes = [ax12a, ax12b, ax12c]

                if solute == cation_1:
                    ax_rej = rejection_axes[0]
                    ax_flux = flux_axes[0]
                    ax_hfeed = hfeed_axes[0]
                    ax_hperm = hperm_axes[0]
                elif solute == cation_2:
                    ax_rej = rejection_axes[1]
                    ax_flux = flux_axes[1]
                    ax_hfeed = hfeed_axes[1]
                    ax_hperm = hperm_axes[1]
                elif solute == "Cl":
                    ax_rej = rejection_axes[2]
                    ax_flux = flux_axes[2]
                    ax_hfeed = hfeed_axes[2]
                    ax_hperm = hperm_axes[2]

                ax_rej.plot(
                    feed_ionic_strength_val,
                    avg_actual_rejection,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    markersize=markersize,
                )
                ax_rej.errorbar(
                    feed_ionic_strength_val,
                    avg_actual_rejection,
                    yerr=spread_actual_rejection,
                    ecolor="grey",
                    capsize=4,
                )

                ax_flux.plot(
                    feed_ionic_strength_val,
                    avg_flux,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    markersize=markersize,
                )
                ax_flux.errorbar(
                    feed_ionic_strength_val,
                    avg_flux,
                    yerr=spread_flux,
                    ecolor="grey",
                    capsize=4,
                )

                ax_hfeed.plot(
                    feed_ionic_strength_val,
                    avg_H_feed,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    markersize=markersize,
                )
                ax_hfeed.errorbar(
                    feed_ionic_strength_val,
                    avg_H_feed,
                    yerr=spread_H_feed,
                    ecolor="grey",
                    capsize=4,
                )

                ax_hperm.plot(
                    feed_ionic_strength_val,
                    avg_H_perm,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    markersize=markersize,
                )
                ax_hperm.errorbar(
                    feed_ionic_strength_val,
                    avg_H_perm,
                    yerr=spread_H_perm,
                    ecolor="grey",
                    capsize=4,
                )
                # if inset:
                #     # inax_flux.plot(
                #     #     feed_ionic_strength_val,
                #     #     avg_flux,
                #     #     color=color,
                #     #     marker=marker,
                #     #     alpha=alpha,
                #     #     markersize=markersize,
                #     # )
                #     # inax_flux.errorbar(
                #     #     feed_ionic_strength_val,
                #     #     avg_flux,
                #     #     yerr=spread_flux,
                #     #     ecolor="grey",
                #     #     capsize=4,
                #     # )
                #     inax_hfeed.plot(
                #         feed_ionic_strength_val,
                #         avg_H_feed,
                #         color=color,
                #         marker=marker,
                #         alpha=alpha,
                #         markersize=markersize,
                #     )
                #     inax_hfeed.errorbar(
                #         feed_ionic_strength_val,
                #         avg_H_feed,
                #         yerr=spread_H_feed,
                #         ecolor="grey",
                #         capsize=4,
                #     )
                #     inax_hperm.plot(
                #         feed_ionic_strength_val,
                #         avg_H_perm,
                #         color=color,
                #         marker=marker,
                #         alpha=alpha,
                #         markersize=markersize,
                #     )
                #     inax_hperm.errorbar(
                #         feed_ionic_strength_val,
                #         avg_H_perm,
                #         yerr=spread_H_perm,
                #         ecolor="grey",
                #         capsize=4,
                #     )

    if save_figure:
        fig1.savefig("Li_Co_rejection_flux_partition_vary_salt_ratio.png", dpi=600)
        fig2.savefig("Li_Al_rejection_flux_partition_vary_salt_ratio.png", dpi=600)
        fig3.savefig("Co_Al_rejection_flux_partition_vary_salt_ratio.png", dpi=600)


def plot_only_rejections(save_figure=True):
    """
    Plots rejection versus ionic strength of bulk fluid and feed.
    """
    markersize = 8
    fontsize = 14

    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [
        NFE_module_length,
        NFE_boundary_layer_thickness,
        NFE_membrane_thickness,
    ]

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
    color_033 = tol_bright_hex[0]
    color_05 = tol_bright_hex[1]
    color_1 = tol_bright_hex[2]
    color_2 = tol_bright_hex[3]
    color_3 = tol_bright_hex[4]

    # each plot is a salt solution
    fig1, (ax1a, ax1b, ax1c) = plt.subplots(
        1, 3, dpi=75, figsize=(15, 5), constrained_layout=True
    )
    fig1.suptitle("LiCl + CoCl$_2$", fontsize=fontsize, fontweight="bold")
    fig2, (ax5a, ax5b, ax5c) = plt.subplots(
        1, 3, dpi=75, figsize=(15, 5), constrained_layout=True
    )
    fig2.suptitle("LiCl + AlCl$_3$", fontsize=fontsize, fontweight="bold")
    fig3, (ax9a, ax9b, ax9c) = plt.subplots(
        1, 3, dpi=75, figsize=(15, 5), constrained_layout=True
    )
    fig3.suptitle("CoCl$_2$ + AlCl$_3$", fontsize=fontsize, fontweight="bold")

    for ax in [ax1a, ax5a, ax9a]:
        ax.plot(
            [],
            [],
            color=color_033,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="0.33",
        )
        ax.plot(
            [],
            [],
            color=color_05,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="0.5",
        )
        ax.plot(
            [],
            [],
            color=color_1,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="1",
        )
        ax.plot(
            [],
            [],
            color=color_2,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="2",
        )
        ax.plot(
            [],
            [],
            color=color_3,
            marker="o",
            markersize=markersize,
            linestyle="None",
            label="3",
        )
        ax.legend(loc="best", fontsize=fontsize)
        # ax.legend(loc="best", fontsize=fontsize, bbox_to_anchor=(0.85, -0.5))

    for ax in [ax1a, ax5a]:
        ax.set_title("Lithium", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel("Actual Ion\nRejection (%)", fontsize=fontsize, fontweight="bold")
    for ax in [ax1b, ax9a]:
        ax.set_title("Cobalt", fontsize=fontsize, fontweight="bold")
    for ax in [ax5b, ax9b]:
        ax.set_title("Aluminum", fontsize=fontsize, fontweight="bold")
    for ax in [ax1c, ax5c, ax9c]:
        ax.set_title("Chloride", fontsize=fontsize, fontweight="bold")

    IS_min = 25
    IS_max = 425
    for ax in fig1.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.set_xlim(IS_min, IS_max)
        ax.set_xlabel(
            "Inlet Feed Ionic Strength\n(mol/m$^3$)",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in fig2.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.set_xlim(IS_min, IS_max)
        ax.set_xlabel(
            "Inlet Feed Ionic Strength\n(mol/m$^3$)",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in fig3.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.set_xlim(IS_min, IS_max)
        ax.set_xlabel(
            "Inlet Feed Ionic Strength\n(mol/m$^3$)",
            fontsize=fontsize,
            fontweight="bold",
        )

    ionic_strengths = ["050_mM", "075_mM", "100_mM", "150_mM", "200_mM", "400_mM"]

    for ionic_strength in ionic_strengths:
        model_folder = Path(
            f"multi_component_case_studies/vary_salt_ratio/{ionic_strength}"
        )
        # 52 characters (0-51) make up folder name before model name
        # multi_component_case_studies/vary_salt_ratio/100_mM/

        for case_study_file in model_folder.iterdir():
            if str(case_study_file)[52:55] == "0.5":
                ratio = str(case_study_file)[52:55]
                cation_1 = str(case_study_file)[56:58]
                cation_2 = str(case_study_file)[58:60]
                chloride_multiplier = float(str(case_study_file)[62])
            elif str(case_study_file)[52:55] == "0.3":
                ratio = str(case_study_file)[52:56]
                cation_1 = str(case_study_file)[59:61]
                cation_2 = str(case_study_file)[61:63]
                chloride_multiplier = float(str(case_study_file)[65])
            else:
                ratio = str(case_study_file)[52]
                cation_1 = str(case_study_file)[54:56]
                cation_2 = str(case_study_file)[56:58]
                chloride_multiplier = float(str(case_study_file)[60])
            concentration = float(50)  # mM
            model = build_model(
                cation_list=[cation_1, cation_2],
                inlet_concentration={
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
                },
                default_args=default_args,
                H_feed_guess=1,
                H_permeate_guess=1,
                NFE_args=NFE_args,
                initialize=False,
            )
            from_json(model, fname=case_study_file)

            feed_ionic_strength_val = value(
                model.fs.membrane.total_feed_ionic_strength[0]
            )

            for solute in model.fs.membrane.solutes:
                actual_rejection = []

                for t in model.fs.membrane.time:
                    for x in model.fs.membrane.dimensionless_module_length:
                        if x != 0:
                            actual_rejection.append(
                                value(
                                    model.fs.membrane.actual_rejection_percent[
                                        t, x, solute
                                    ]
                                )
                            )

                avg_actual_rejection = np.average(actual_rejection)
                spread_actual_rejection = calculate_spread(actual_rejection)

                alpha = 1
                marker = "o"

                if ratio == "0.33":
                    color = color_033
                elif ratio == "0.5":
                    color = color_05
                elif ratio == "1":
                    color = color_1
                elif ratio == "2":
                    color = color_2
                elif ratio == "3":
                    color = color_3

                if cation_1 == "Li" and cation_2 == "Co":
                    rejection_axes = [ax1a, ax1b, ax1c]
                elif cation_1 == "Li" and cation_2 == "Al":
                    rejection_axes = [ax5a, ax5b, ax5c]
                elif cation_1 == "Co" and cation_2 == "Al":
                    rejection_axes = [ax9a, ax9b, ax9c]

                if solute == cation_1:
                    ax_rej = rejection_axes[0]
                elif solute == cation_2:
                    ax_rej = rejection_axes[1]
                elif solute == "Cl":
                    ax_rej = rejection_axes[2]

                ax_rej.plot(
                    feed_ionic_strength_val,
                    avg_actual_rejection,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    markersize=markersize,
                )
                ax_rej.errorbar(
                    feed_ionic_strength_val,
                    avg_actual_rejection,
                    yerr=spread_actual_rejection,
                    ecolor="grey",
                    capsize=4,
                )

    if save_figure:
        fig1.savefig("Li_Co_rejection_vary_salt_ratio.png", dpi=600)
        fig2.savefig("Li_Al_rejection_vary_salt_ratio.png", dpi=600)
        fig3.savefig("Co_Al_rejection_vary_salt_ratio.png", dpi=600)


def plot_flux_contributions(x_axis="ionic_strength", percent=False, save_figure=True):
    """
    Plots flux contributions for different systems.
    """
    markersize = 10
    fontsize = 14

    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]
    diafiltrate = {"Li": 1e-10, "Co": 1e-10, "Al": 1e-10}

    fig1, ((ax1a, ax1b), (ax2a, ax2b), (ax3a, ax3b)) = plt.subplots(
        3, 2, figsize=(12, 12), dpi=100, constrained_layout=True
    )
    fig2, ((ax4a, ax4b), (ax5a, ax5b), (ax6a, ax6b)) = plt.subplots(
        3, 2, figsize=(12, 12), dpi=100, constrained_layout=True
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

    if x_axis == "ionic_strength":
        model_folder_1 = Path("multi_component_case_studies/single_salt/IS")
        # 44 characters (0-43) make up folder name before model name
        # multi_component_case_studies/single_salt/IS/
        model_folder_2 = Path("multi_component_case_studies/two_salt/IS")
        # 41 characters (0-40) make up folder name before model name
        # multi_component_case_studies/two_salt/IS/
        model_folder_3 = Path("multi_component_case_studies/three_salt/IS")
        # 43 characters (0-42) make up folder name before model name
        # multi_component_case_studies/three_salt/IS/
    else:
        model_folder_1 = Path("multi_component_case_studies/single_salt/CONC")
        # 46 characters (0-45) make up folder name before model name
        # multi_component_case_studies/single_salt/CONC/
        model_folder_2 = Path("multi_component_case_studies/two_salt/CONC")
        # 43 characters (0-42) make up folder name before model name
        # multi_component_case_studies/two_salt/CONC/
        model_folder_3 = Path("multi_component_case_studies/three_salt/CONC")
        # 45 characters (0-44) make up folder name before model name
        # multi_component_case_studies/three_salt/CONC/

    case_study_list_1 = []
    for file in model_folder_1.iterdir():
        case_study_list_1.append(file)
    case_study_list_1.sort()
    case_study_list_2 = [file for file in model_folder_2.iterdir()]
    case_study_list_3 = [file for file in model_folder_3.iterdir()]
    case_studies = {
        "single": case_study_list_1,
        # "two": case_study_list_2.sort(),
        # "three": case_study_list_3.sort(),
    }

    lithium_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_lithium_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cobalt_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_cobalt_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    aluminum_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_aluminum_bl_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}

    lithium_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_lithium_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cobalt_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_cobalt_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    aluminum_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_aluminum_mem_avg = {"Convection": [], "Diffusion": [], "Electromigration": []}

    lithium_bl_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_lithium_bl_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cobalt_bl_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_cobalt_bl_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    aluminum_bl_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_aluminum_bl_per = {"Convection": [], "Diffusion": [], "Electromigration": []}

    lithium_mem_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_lithium_mem_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cobalt_mem_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_cobalt_mem_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    aluminum_mem_per = {"Convection": [], "Diffusion": [], "Electromigration": []}
    cl_aluminum_mem_per = {"Convection": [], "Diffusion": [], "Electromigration": []}

    for type, case_study_files in case_studies.items():
        for case_study in case_study_files:
            if type == "single":
                if x_axis == "ionic_strength":
                    cation = str(case_study)[50:52]
                    chloride_multiplier = float(str(case_study)[54])
                else:
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
            # elif type == "two":
            #     if x_axis == "ionic_strength":
            #         cation_1 = str(case_study)[41:43]
            #         cation_2 = str(case_study)[43:45]
            #         chloride_multiplier = float(str(case_study)[47])
            #     else:
            #         cation_1 = str(case_study)[43:45]
            #         cation_2 = str(case_study)[45:47]
            #         chloride_multiplier = float(str(case_study)[49])
            #     concentration = float(50)  # mM
            #     cation_list = [cation_1, cation_2]
            #     inlet_concentration = {
            #         "feed": {
            #             cation_1: concentration,
            #             cation_2: concentration,
            #             "Cl": chloride_multiplier * concentration,
            #         },
            #         "diafiltrate": {
            #             cation_1: 1e-10,
            #             cation_2: 1e-10,
            #             "Cl": 1e-10,
            #         },
            #     }
            # elif type == "three":
            #     if x_axis == "ionic_strength":
            #         cation_1 = str(case_study)[43:45]
            #         cation_2 = str(case_study)[45:47]
            #         cation_3 = str(case_study)[47:49]
            #         chloride_multiplier = float(str(case_study)[51])
            #     else:
            #         cation_1 = str(case_study)[45:47]
            #         cation_2 = str(case_study)[47:49]
            #         cation_3 = str(case_study)[49:51]
            #         chloride_multiplier = float(str(case_study)[53])
            #     concentration = float(50)  # mM
            #     cation_list = [cation_1, cation_2, cation_3]
            #     inlet_concentration = {
            #         "feed": {
            #             cation_1: concentration,
            #             cation_2: concentration,
            #             cation_3: concentration,
            #             "Cl": chloride_multiplier * concentration,
            #         },
            #         "diafiltrate": {
            #             cation_1: 1e-10,
            #             cation_2: 1e-10,
            #             cation_3: 1e-10,
            #             "Cl": 1e-10,
            #         },
            #     }

            model = build_model(
                cation_list=cation_list,
                inlet_concentration=inlet_concentration,
                default_args=default_args,
                H_feed_guess=1,
                H_permeate_guess=1,
                NFE_args=NFE_args,
                initialize=False,
            )
            from_json(model, fname=case_study)

            for solute in model.fs.membrane.solutes:
                average_variable_dict = get_model_averages_flux(model, solute)

                avg_bl_convection = average_variable_dict["bl_convection"]["avg"]
                avg_bl_diffusion = average_variable_dict["bl_diffusion"]["avg"]
                avg_bl_electromigration = average_variable_dict["bl_electromigration"][
                    "avg"
                ]
                avg_mem_convection = average_variable_dict["mem_convection"]["avg"]
                avg_mem_diffusion = average_variable_dict["mem_diffusion"]["avg"]
                avg_mem_electromigration = average_variable_dict[
                    "mem_electromigration"
                ]["avg"]

                spread_bl_convection = average_variable_dict["bl_convection"]["spread"]
                spread_bl_diffusion = average_variable_dict["bl_diffusion"]["spread"]
                spread_bl_electromigration = average_variable_dict[
                    "bl_electromigration"
                ]["spread"]
                spread_mem_convection = average_variable_dict["mem_convection"][
                    "spread"
                ]
                spread_mem_diffusion = average_variable_dict["mem_diffusion"]["spread"]
                spread_mem_electromigration = average_variable_dict[
                    "mem_electromigration"
                ]["spread"]

                percent_bl_convection = average_variable_dict["bl_convection"][
                    "percent"
                ]
                percent_bl_diffusion = average_variable_dict["bl_diffusion"]["percent"]
                percent_bl_electromigration = average_variable_dict[
                    "bl_electromigration"
                ]["percent"]
                percent_mem_convection = average_variable_dict["mem_convection"][
                    "percent"
                ]
                percent_mem_diffusion = average_variable_dict["mem_diffusion"][
                    "percent"
                ]
                percent_mem_electromigration = average_variable_dict[
                    "mem_electromigration"
                ]["percent"]

                if solute == "Li":
                    bl_dict_avg = lithium_bl_avg
                    bl_dict_per = lithium_bl_per
                    mem_dict_avg = lithium_mem_avg
                    mem_dict_per = lithium_mem_per
                elif solute == "Co":
                    bl_dict_avg = cobalt_bl_avg
                    bl_dict_per = cobalt_bl_per
                    mem_dict_avg = cobalt_mem_avg
                    mem_dict_per = cobalt_mem_per
                elif solute == "Al":
                    bl_dict_avg = aluminum_bl_avg
                    bl_dict_per = aluminum_bl_per
                    mem_dict_avg = aluminum_mem_avg
                    mem_dict_per = aluminum_mem_per
                elif solute == "Cl":
                    if model.fs.membrane.cations.at(1) == "Li":
                        bl_dict_avg = cl_lithium_bl_avg
                        bl_dict_per = cl_lithium_bl_per
                        mem_dict_avg = cl_lithium_mem_avg
                        mem_dict_per = cl_lithium_mem_per
                    elif model.fs.membrane.cations.at(1) == "Co":
                        bl_dict_avg = cl_cobalt_bl_avg
                        bl_dict_per = cl_cobalt_bl_per
                        mem_dict_avg = cl_cobalt_mem_avg
                        mem_dict_per = cl_cobalt_mem_per
                    elif model.fs.membrane.cations.at(1) == "Al":
                        bl_dict_avg = cl_aluminum_bl_avg
                        bl_dict_per = cl_aluminum_bl_per
                        mem_dict_avg = cl_aluminum_mem_avg
                        mem_dict_per = cl_aluminum_mem_per

                bl_dict_avg["Convection"].append(avg_bl_convection)
                bl_dict_avg["Diffusion"].append(avg_bl_diffusion)
                bl_dict_avg["Electromigration"].append(avg_bl_electromigration)

                bl_dict_per["Convection"].append(percent_bl_convection)
                bl_dict_per["Diffusion"].append(percent_bl_diffusion)
                bl_dict_per["Electromigration"].append(percent_bl_electromigration)

                mem_dict_avg["Convection"].append(avg_mem_convection)
                mem_dict_avg["Diffusion"].append(avg_mem_diffusion)
                mem_dict_avg["Electromigration"].append(avg_mem_electromigration)

                mem_dict_per["Convection"].append(percent_mem_convection)
                mem_dict_per["Diffusion"].append(percent_mem_diffusion)
                mem_dict_per["Electromigration"].append(percent_mem_electromigration)

    ionic_strengths = ["50", "75", "100", "150", "200", "400", "600", "800"]
    concentrations = ["25", "50", "75", "100", "150", "200", "250", "300"]
    if x_axis == "ionic_strength":
        x_values = ionic_strengths
        x_label = "Inlet Feed Ionic Strength (mM)"
    else:
        x_values = concentrations
        x_label = "Cation Feed Concentration (mol/m$\mathbf{^3}$)"

    color_list = [conv_color, diff_color, elec_color]
    x = np.arange(len(x_values))

    aluminum_bl_avg["Convection"].insert(0, 0)
    aluminum_bl_avg["Diffusion"].insert(0, 0)
    aluminum_bl_avg["Electromigration"].insert(0, 0)
    cl_aluminum_bl_avg["Convection"].insert(0, 0)
    cl_aluminum_bl_avg["Diffusion"].insert(0, 0)
    cl_aluminum_bl_avg["Electromigration"].insert(0, 0)

    aluminum_mem_avg["Convection"].insert(0, 0)
    aluminum_mem_avg["Diffusion"].insert(0, 0)
    aluminum_mem_avg["Electromigration"].insert(0, 0)
    cl_aluminum_mem_avg["Convection"].insert(0, 0)
    cl_aluminum_mem_avg["Diffusion"].insert(0, 0)
    cl_aluminum_mem_avg["Electromigration"].insert(0, 0)

    if percent:
        lithium_bl_data = lithium_bl_per
        cl_lithium_bl_data = cl_lithium_bl_per
        cobalt_bl_data = cobalt_bl_per
        cl_cobalt_bl_data = cl_cobalt_bl_per
        aluminum_bl_data = aluminum_bl_per
        cl_aluminum_bl_data = cl_aluminum_bl_per

        lithium_mem_data = lithium_mem_per
        cl_lithium_mem_data = cl_lithium_mem_per
        cobalt_mem_data = cobalt_mem_per
        cl_cobalt_mem_data = cl_cobalt_mem_per
        aluminum_mem_data = aluminum_mem_per
        cl_aluminum_mem_data = cl_aluminum_mem_per
    else:
        lithium_bl_data = lithium_bl_avg
        cl_lithium_bl_data = cl_lithium_bl_avg
        cobalt_bl_data = cobalt_bl_avg
        cl_cobalt_bl_data = cl_cobalt_bl_avg
        aluminum_bl_data = aluminum_bl_avg
        cl_aluminum_bl_data = cl_aluminum_bl_avg

        lithium_mem_data = lithium_mem_avg
        cl_lithium_mem_data = cl_lithium_mem_avg
        cobalt_mem_data = cobalt_mem_avg
        cl_cobalt_mem_data = cl_cobalt_mem_avg
        aluminum_mem_data = aluminum_mem_avg
        cl_aluminum_mem_data = cl_aluminum_mem_avg

    lithium_bl_df = pd.DataFrame(lithium_bl_data, index=x_values)
    cl_lithium_bl_df = pd.DataFrame(cl_lithium_bl_data, index=x_values)
    cobalt_bl_df = pd.DataFrame(cobalt_bl_data, index=x_values)
    cl_cobalt_bl_df = pd.DataFrame(cl_cobalt_bl_data, index=x_values)
    aluminum_bl_df = pd.DataFrame(aluminum_bl_data, index=x_values)
    cl_aluminum_bl_df = pd.DataFrame(cl_aluminum_bl_data, index=x_values)

    lithium_mem_df = pd.DataFrame(lithium_mem_data, index=x_values)
    cl_lithium_mem_df = pd.DataFrame(cl_lithium_mem_data, index=x_values)
    cobalt_mem_df = pd.DataFrame(cobalt_mem_data, index=x_values)
    cl_cobalt_mem_df = pd.DataFrame(cl_cobalt_mem_data, index=x_values)
    aluminum_mem_df = pd.DataFrame(aluminum_mem_data, index=x_values)
    cl_aluminum_mem_df = pd.DataFrame(cl_aluminum_mem_data, index=x_values)

    lithium_bl_df.plot(ax=ax1a, kind="bar", stacked=True, color=color_list, rot=0)
    cl_lithium_bl_df.plot(ax=ax4a, kind="bar", stacked=True, color=color_list, rot=0)
    cobalt_bl_df.plot(
        ax=ax2a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    cl_cobalt_bl_df.plot(
        ax=ax5a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    aluminum_bl_df.plot(
        ax=ax3a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    cl_aluminum_bl_df.plot(
        ax=ax6a, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )

    lithium_mem_df.plot(
        ax=ax1b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    cl_lithium_mem_df.plot(
        ax=ax4b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    cobalt_mem_df.plot(
        ax=ax2b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    cl_cobalt_mem_df.plot(
        ax=ax5b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    aluminum_mem_df.plot(
        ax=ax3b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )
    cl_aluminum_mem_df.plot(
        ax=ax6b, kind="bar", stacked=True, color=color_list, rot=0, legend=False
    )

    bar_width = 0.5
    x_positions = ax1a.get_xticks()
    x_min = x_positions - (bar_width / 2)
    x_max = x_positions + (bar_width / 2)

    totals_lithium_bl = lithium_bl_df.sum(axis=1)
    totals_lithium_mem = lithium_mem_df.sum(axis=1)
    totals_cl_lithium_bl = cl_lithium_bl_df.sum(axis=1)
    totals_cl_lithium_mem = cl_lithium_mem_df.sum(axis=1)

    ax1a.hlines(totals_lithium_bl, x_min, x_max, colors="black", linewidth=3)
    ax1b.hlines(totals_lithium_mem, x_min, x_max, colors="black", linewidth=3)
    ax4a.hlines(totals_cl_lithium_bl, x_min, x_max, colors="black", linewidth=3)
    ax4b.hlines(totals_cl_lithium_mem, x_min, x_max, colors="black", linewidth=3)

    totals_cobalt_bl = cobalt_bl_df.sum(axis=1)
    totals_cobalt_mem = cobalt_mem_df.sum(axis=1)
    totals_cl_cobalt_bl = cl_cobalt_bl_df.sum(axis=1)
    totals_cl_cobalt_mem = cl_cobalt_mem_df.sum(axis=1)

    ax2a.hlines(totals_cobalt_bl, x_min, x_max, colors="black", linewidth=3)
    ax2b.hlines(totals_cobalt_mem, x_min, x_max, colors="black", linewidth=3)
    ax5a.hlines(totals_cl_cobalt_bl, x_min, x_max, colors="black", linewidth=3)
    ax5b.hlines(totals_cl_cobalt_mem, x_min, x_max, colors="black", linewidth=3)

    totals_aluminum_bl = aluminum_bl_df.sum(axis=1)
    totals_aluminum_mem = aluminum_mem_df.sum(axis=1)
    totals_cl_aluminum_bl = cl_aluminum_bl_df.sum(axis=1)
    totals_cl_aluminum_mem = cl_aluminum_mem_df.sum(axis=1)

    ax3a.hlines(totals_aluminum_bl, x_min, x_max, colors="black", linewidth=3)
    ax3b.hlines(totals_aluminum_mem, x_min, x_max, colors="black", linewidth=3)
    ax6a.hlines(totals_cl_aluminum_bl, x_min, x_max, colors="black", linewidth=3)
    ax6b.hlines(totals_cl_aluminum_mem, x_min, x_max, colors="black", linewidth=3)

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

    for ax in [ax1a, ax1b]:
        ax.set_ylim(-6, 17)
    for ax in [ax4a, ax4b]:
        ax.set_ylim(-8, 19)

    fig1.suptitle(
        "Average Cation Flux Breakdown (Excluding x,z=0)",
        fontsize=fontsize + 2,
        fontweight="bold",
    )
    fig2.suptitle(
        "Average Anion Flux Breakdown (Excluding x,z=0)",
        fontsize=fontsize + 2,
        fontweight="bold",
    )
    if percent:
        li_label = "Contribution to Lithium Flux (%)"
        co_label = "Contribution to Cobalt Flux (%)"
        al_label = "Contribution to ALuminum Flux (%)"
        cl_label = "Contribution to Chloride Flux (%)"
    else:
        li_label = "+1 Cation Flux (mol m$\mathbf{^{-2}}$ h$\mathbf{^{-1}}$)"
        co_label = "Cobalt Flux (mol m$\mathbf{^{-2}}$ h$\mathbf{^{-1}}$)"
        al_label = "Aluminum Flux (mol m$\mathbf{^{-2}}$ h$\mathbf{^{-1}}$)"
        cl_label = "Chloride Flux (mol m$\mathbf{^{-2}}$ h$\mathbf{^{-1}}$)"

    ax1a.set_ylabel(
        li_label,
        fontsize=fontsize,
        fontweight="bold",
    )
    ax2a.set_ylabel(
        co_label,
        fontsize=fontsize,
        fontweight="bold",
    )
    ax3a.set_ylabel(
        al_label,
        fontsize=fontsize,
        fontweight="bold",
    )
    for ax in [ax4a, ax5a, ax6a]:
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
    for ax in [ax3a, ax3b, ax6a, ax6b]:
        ax.set_xlabel(
            x_label,
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax1a, ax1b, ax4a, ax4b]:
        ax.text(
            0.5,
            0.95,
            "LiCl",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax2a, ax2b, ax5a, ax5b]:
        ax.text(
            0.5,
            0.95,
            "CoCl$_2$",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax3a, ax3b, ax6a, ax6b]:
        ax.text(
            0.5,
            0.95,
            "AlCl$_3$",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
        )
    # ax1a.plot([],[], "k-", linewidth=3, label="Total Flux")
    # ax1a.legend(fontsize=fontsize-2)

    if save_figure:
        if percent:
            fig1.savefig(f"percent_flux_breakdown_single_salt_cation.png", dpi=600)
            fig2.savefig(f"percent_flux_breakdown_single_salt_anion.png", dpi=600)
        else:
            fig1.savefig(f"total_flux_breakdown_single_salt_cation.png", dpi=600)
            fig2.savefig(f"total_flux_breakdown_single_salt_anion.png", dpi=600)


def plot_Donnan_potentials(
    x_axis="ionic_strength", save_figure=True, total_h=False, sieving=False
):
    markersize = 10
    fontsize = 14

    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    fig1, ((ax1a, ax1b), (ax2a, ax2b)) = plt.subplots(
        2, 2, dpi=90, figsize=(10, 10), constrained_layout=True
    )

    if total_h:
        fig2, (ax3a, ax3b) = plt.subplots(
            1, 2, dpi=90, figsize=(10, 5), constrained_layout=True
        )
    if sieving:
        fig3, ax4 = plt.subplots(1, 1, dpi=90, figsize=(5, 5), constrained_layout=True)

    if x_axis == "ionic_strength":
        x_label = "Inlet Feed Ionic Strength (mM)"
    elif x_axis == "cation_concentration":
        x_label = "Cation Concentration in Feed (mM)"

    for ax in [ax2a, ax2b]:
        ax.set_xlabel(
            x_label,
            fontsize=fontsize,
            fontweight="bold",
        )
    if total_h:
        for ax in [ax3a, ax3b]:
            ax.set_xlabel(
                x_label,
                fontsize=fontsize,
                fontweight="bold",
            )
    if sieving:
        for ax in [ax4]:
            ax.set_xlabel(
                x_label,
                fontsize=fontsize,
                fontweight="bold",
            )

    ax1a.set_ylabel(
        "Dimensionless Donnan Potential",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax2a.set_ylabel(
        "$z_i$ * Dimensionless Donnan Potential",
        fontsize=fontsize,
        fontweight="bold",
    )

    ax1a.set_title(
        "Feed Side",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax1b.set_title(
        "Permeate Side",
        fontsize=fontsize,
        fontweight="bold",
    )

    if total_h:
        ax3a.set_ylabel(
            "Overall Partitioning Coefficient",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax3a.set_title(
            "Feed Side",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax3b.set_title(
            "Permeate Side",
            fontsize=fontsize,
            fontweight="bold",
        )
    if sieving:
        ax4.set_ylabel(
            "Observed Sieving Coefficient (Average)",
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
    li_color = tol_bright_hex[0]
    co_color = tol_bright_hex[1]
    al_color = tol_bright_hex[2]

    legend_dict = {
        "lithium": {
            "marker": "o",
            "color": li_color,
        },
        "cobalt": {
            "marker": "v",
            "color": co_color,
        },
        "aluminum": {
            "marker": "^",
            "color": al_color,
        },
    }
    for solute, cation_dict in legend_dict.items():
        ax1a.plot(
            [],
            [],
            color=cation_dict["color"],
            marker=cation_dict["marker"],
            markersize=markersize,
            linestyle="None",
            label=solute,
        )
        if total_h:
            ax3a.plot(
                [],
                [],
                color=cation_dict["color"],
                marker=cation_dict["marker"],
                markersize=markersize,
                linestyle="None",
                label=solute,
            )
        if sieving:
            ax4.plot(
                [],
                [],
                color=cation_dict["color"],
                marker=cation_dict["marker"],
                markersize=markersize,
                linestyle="None",
                label=solute,
            )

    ax1a.legend(
        loc="best",
        fontsize=fontsize - 2,
    )
    for ax in fig1.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)

    if total_h:
        ax3a.legend(
            loc="best",
            fontsize=fontsize - 2,
        )
        for ax in fig2.axes:
            ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
    if sieving:
        ax4.legend(
            loc="best",
            fontsize=fontsize - 2,
        )
        for ax in fig3.axes:
            ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)

    if x_axis == "ionic_strength":
        model_folder_1 = Path("multi_component_case_studies/single_salt/IS")
        # 44 characters (0-43) make up folder name before model name
        # multi_component_case_studies/single_salt/IS/
    else:
        model_folder_1 = Path("multi_component_case_studies/single_salt/CONC")
        # 46 characters (0-45) make up folder name before model name
        # multi_component_case_studies/single_salt/CONC/

    case_study_list_1 = [file for file in model_folder_1.iterdir()]

    case_study_list_1.sort()

    case_studies = {
        "single": case_study_list_1,
    }

    y_ax1a_li = []
    y_err_ax1a_li = []
    y_ax1b_li = []
    y_err_ax1b_li = []
    y_ax2a_li = []
    y_err_ax2a_li = []
    y_ax2b_li = []
    y_err_ax2b_li = []

    y_ax1a_co = []
    y_err_ax1a_co = []
    y_ax1b_co = []
    y_err_ax1b_co = []
    y_ax2a_co = []
    y_err_ax2a_co = []
    y_ax2b_co = []
    y_err_ax2b_co = []

    y_ax1a_al = []
    y_err_ax1a_al = []
    y_ax1b_al = []
    y_err_ax1b_al = []
    y_ax2a_al = []
    y_err_ax2a_al = []
    y_ax2b_al = []
    y_err_ax2b_al = []

    if total_h:
        y_ax3a_li = []
        y_ax3b_li = []
        y_ax3a_co = []
        y_ax3b_co = []
        y_ax3a_al = []
        y_ax3b_al = []

    if sieving:
        y_ax4_li = []
        y_ax4_co = []
        y_ax4_al = []

    for type, case_study_files in case_studies.items():
        for case_study in case_study_files:
            if type == "single":
                if x_axis == "ionic_strength":
                    cation = str(case_study)[50:52]
                    chloride_multiplier = float(str(case_study)[54])
                else:
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

            for solute in model.fs.membrane.cations:
                average_variable_dict = get_model_averages(model, solute)

                # if x_axis == "ionic_strength":
                #     x_value = value(model.fs.membrane.total_feed_ionic_strength[0])
                # elif x_axis == "cation_concentration":
                #     x_value = value(
                #         model.fs.membrane.retentate_conc_mol_comp[0, 0, solute]
                #     )

                alpha = 1

                if solute == "Li":
                    # marker = "o"
                    # color = li_color
                    y_ax1a = y_ax1a_li
                    y_err_ax1a = y_err_ax1a_li
                    y_ax1b = y_ax1b_li
                    y_err_ax1b = y_err_ax1b_li
                    y_ax2a = y_ax2a_li
                    y_err_ax2a = y_err_ax2a_li
                    y_ax2b = y_ax2b_li
                    y_err_ax2b = y_err_ax2b_li

                    if total_h:
                        y_ax3a = y_ax3a_li
                        y_ax3b = y_ax3b_li

                    if sieving:
                        y_ax4 = y_ax4_li

                elif solute == "Co":
                    # marker = "v"
                    # color = co_color
                    y_ax1a = y_ax1a_co
                    y_err_ax1a = y_err_ax1a_co
                    y_ax1b = y_ax1b_co
                    y_err_ax1b = y_err_ax1b_co
                    y_ax2a = y_ax2a_co
                    y_err_ax2a = y_err_ax2a_co
                    y_ax2b = y_ax2b_co
                    y_err_ax2b = y_err_ax2b_co

                    if total_h:
                        y_ax3a = y_ax3a_co
                        y_ax3b = y_ax3b_co

                    if sieving:
                        y_ax4 = y_ax4_co

                elif solute == "Al":
                    # marker = "^"
                    # color = al_color
                    y_ax1a = y_ax1a_al
                    y_err_ax1a = y_err_ax1a_al
                    y_ax1b = y_ax1b_al
                    y_err_ax1b = y_err_ax1b_al
                    y_ax2a = y_ax2a_al
                    y_err_ax2a = y_err_ax2a_al
                    y_ax2b = y_ax2b_al
                    y_err_ax2b = y_err_ax2b_al

                    if total_h:
                        y_ax3a = y_ax3a_al
                        y_ax3b = y_ax3b_al

                    if total_h:
                        y_ax4 = y_ax4_al

                y_ax1a.append(average_variable_dict["Donnan_potential_feed"]["avg"])
                y_err_ax1a.append(
                    average_variable_dict["Donnan_potential_feed"]["spread"]
                )

                y_ax1b.append(average_variable_dict["Donnan_potential_perm"]["avg"])
                y_err_ax1b.append(
                    average_variable_dict["Donnan_potential_perm"]["spread"]
                )

                y_ax2a.append(
                    -value(model.fs.membrane.config.property_package.charge[solute])
                    * average_variable_dict["Donnan_potential_feed"]["avg"]
                )
                y_err_ax2a.append(
                    value(model.fs.membrane.config.property_package.charge[solute])
                    * average_variable_dict["Donnan_potential_feed"]["spread"]
                )

                y_ax2b.append(
                    -value(model.fs.membrane.config.property_package.charge[solute])
                    * average_variable_dict["Donnan_potential_perm"]["avg"]
                )
                y_err_ax2b.append(
                    value(model.fs.membrane.config.property_package.charge[solute])
                    * average_variable_dict["Donnan_potential_perm"]["spread"]
                )

                if total_h:
                    y_ax3a.append(average_variable_dict["H_feed"]["avg"])
                    y_ax3b.append(average_variable_dict["H_perm"]["avg"])

                if sieving:
                    y_ax4.append(average_variable_dict["observed_sieving"]["avg"])

    data_dict_li = {
        ax1a: [y_ax1a_li, y_err_ax1a_li],
        ax1b: [y_ax1b_li, y_err_ax1b_li],
        ax2a: [y_ax2a_li, y_err_ax2a_li],
        ax2b: [y_ax2b_li, y_err_ax2b_li],
    }
    data_dict_co = {
        ax1a: [y_ax1a_co, y_err_ax1a_co],
        ax1b: [y_ax1b_co, y_err_ax1b_co],
        ax2a: [y_ax2a_co, y_err_ax2a_co],
        ax2b: [y_ax2b_co, y_err_ax2b_co],
    }
    data_dict_al = {
        ax1a: [y_ax1a_al, y_err_ax1a_al],
        ax1b: [y_ax1b_al, y_err_ax1b_al],
        ax2a: [y_ax2a_al, y_err_ax2a_al],
        ax2b: [y_ax2b_al, y_err_ax2b_al],
    }

    if total_h:
        h_dict_li = {
            ax3a: y_ax3a_li,
            ax3b: y_ax3b_li,
        }
        h_dict_co = {
            ax3a: y_ax3a_co,
            ax3b: y_ax3b_co,
        }
        h_dict_al = {
            ax3a: y_ax3a_al,
            ax3b: y_ax3b_al,
        }

    x_li = [25, 50, 75, 100, 150, 200, 400, 600, 800]
    x_co = [25, 50, 75, 100, 150, 200, 400, 600, 800]
    x_al = [25, 50, 75, 100, 150, 200, 400, 600, 800]

    for ax, y in data_dict_li.items():
        ax.plot(
            x_li,
            y[0],
            color=li_color,
            marker="o",
            linestyle="None",
            alpha=alpha,
            markersize=markersize,
        )
        # ax.errorbar(
        #     x_li,
        #     y[0],
        #     yerr=y[1],
        #     ecolor=li_color,
        #     capsize=3,
        # )

    for ax, y in data_dict_co.items():
        ax.plot(
            x_co,
            y[0],
            color=co_color,
            marker="v",
            linestyle="None",
            alpha=alpha,
            markersize=markersize,
        )
        # ax.errorbar(
        #     x_co,
        #     y[0],
        #     yerr=y[1],
        #     ecolor=co_color,
        #     capsize=3,
        # )

    for ax, y in data_dict_al.items():
        ax.plot(
            x_al,
            y[0],
            color=al_color,
            marker="^",
            linestyle="None",
            alpha=alpha,
            markersize=markersize,
        )
        # ax.errorbar(
        #     x_al,
        #     y[0],
        #     yerr=y[1],
        #     ecolor=al_color,
        #     capsize=3,
        # )

    if total_h:
        for ax, y in h_dict_li.items():
            ax.plot(
                x_li,
                y,
                color=li_color,
                marker="o",
                linestyle="None",
                alpha=alpha,
                markersize=markersize,
            )
        for ax, y in h_dict_co.items():
            ax.plot(
                x_co,
                y,
                color=co_color,
                marker="v",
                linestyle="None",
                alpha=alpha,
                markersize=markersize,
            )
        for ax, y in h_dict_al.items():
            ax.plot(
                x_al,
                y,
                color=al_color,
                marker="^",
                linestyle="None",
                alpha=alpha,
                markersize=markersize,
            )

    if sieving:
        ax4.plot(
            x_li,
            y_ax4_li,
            color=li_color,
            marker="o",
            linestyle="None",
            alpha=alpha,
            markersize=markersize,
        )
        ax4.plot(
            x_co,
            y_ax4_co,
            color=co_color,
            marker="v",
            linestyle="None",
            alpha=alpha,
            markersize=markersize,
        )
        ax4.plot(
            x_al,
            y_ax4_al,
            color=al_color,
            marker="^",
            linestyle="None",
            alpha=alpha,
            markersize=markersize,
        )

    # add trendlines
    def exp_fit(x, a, b, c):
        # y = a exp(bx) + c
        return a * np.exp(b * x) + c

    coeffs_ax2a_li, covs_ax2a_li = curve_fit(
        exp_fit, np.array(x_li), np.array(y_ax2a_li), p0=[3, -0.1, -0.9]
    )
    a_ax2a_li, b_ax2a_li, c_ax2a_li = coeffs_ax2a_li
    eqn_ax2a_li = exp_fit(np.array(x_li), a_ax2a_li, b_ax2a_li, c_ax2a_li)
    ax2a.plot(np.array(x_li), eqn_ax2a_li, color=li_color, linestyle="--")

    coeffs_ax2a_co, covs_ax2a_co = curve_fit(
        exp_fit, np.array(x_co), np.array(y_ax2a_co), p0=[3, -0.1, -0.9]
    )
    a_ax2a_co, b_ax2a_co, c_ax2a_co = coeffs_ax2a_co
    eqn_ax2a_co = exp_fit(np.array(x_co), a_ax2a_co, b_ax2a_co, c_ax2a_co)
    ax2a.plot(np.array(x_co), eqn_ax2a_co, color=co_color, linestyle="--")

    coeffs_ax2a_al, covs_ax2a_al = curve_fit(
        exp_fit, np.array(x_al), np.array(y_ax2a_al), p0=[3, -0.1, -0.9]
    )
    a_ax2a_al, b_ax2a_al, c_ax2a_al = coeffs_ax2a_al
    eqn_ax2a_al = exp_fit(np.array(x_al), a_ax2a_al, b_ax2a_al, c_ax2a_al)
    ax2a.plot(np.array(x_al), eqn_ax2a_al, color=al_color, linestyle="--")

    coeffs_ax2b_li, covs_ax2b_li = curve_fit(
        exp_fit, np.array(x_li), np.array(y_ax2b_li), p0=[3, -0.1, -0.9]
    )
    a_ax2b_li, b_ax2b_li, c_ax2b_li = coeffs_ax2b_li
    eqn_ax2b_li = exp_fit(np.array(x_li), a_ax2b_li, b_ax2b_li, c_ax2b_li)
    ax2b.plot(np.array(x_li), eqn_ax2b_li, color=li_color, linestyle="--")

    coeffs_ax2b_co, covs_ax2b_co = curve_fit(
        exp_fit, np.array(x_co), np.array(y_ax2b_co), p0=[3, -0.1, -0.9]
    )
    a_ax2b_co, b_ax2b_co, c_ax2b_co = coeffs_ax2b_co
    eqn_ax2b_co = exp_fit(np.array(x_co), a_ax2b_co, b_ax2b_co, c_ax2b_co)
    ax2b.plot(np.array(x_co), eqn_ax2b_co, color=co_color, linestyle="--")

    coeffs_ax2b_al, covs_ax2b_al = curve_fit(
        exp_fit, np.array(x_al), np.array(y_ax2b_al), p0=[3, -0.1, -0.9]
    )
    a_ax2b_al, b_ax2b_al, c_ax2b_al = coeffs_ax2b_al
    eqn_ax2b_al = exp_fit(np.array(x_al), a_ax2b_al, b_ax2b_al, c_ax2b_al)
    ax2b.plot(np.array(x_al), eqn_ax2b_al, color=al_color, linestyle="--")

    print(
        f"Feed side, +1: $z_i \Delta \Phi^D$ = {a_ax2a_li:.2g} * e^({b_ax2a_li:.2g} * x) + {c_ax2a_li:.2g}"
    )
    print(
        f"Feed side, +2: $z_i \Delta \Phi^D$ = {a_ax2a_co:.2g} * e^({b_ax2a_co:.2g} * x) + {c_ax2a_co:.2g}"
    )
    print(
        f"Feed side, +3: $z_i \Delta \Phi^D$ = {a_ax2a_al:.2g} * e^({b_ax2a_al:.2g} * x) + {c_ax2a_al:.2g}"
    )

    print(
        f"Permeate side, +1: $z_i \Delta \Phi^D$ = {a_ax2b_li:.2g} * e^({b_ax2b_li:.2g} * x) + {c_ax2b_li:.2g}"
    )
    print(
        f"Permeate side, +2: $z_i \Delta \Phi^D$ = {a_ax2b_co:.2g} * e^({b_ax2b_co:.2g} * x) + {c_ax2b_co:.2g}"
    )
    print(
        f"Permeate side, +3: $z_i \Delta \Phi^D$ = {a_ax2b_al:.2g} * e^({b_ax2b_al:.2g} * x) + {c_ax2b_al:.2g}"
    )

    if total_h:
        coeffs_ax3a_li, covs_ax3a_li = curve_fit(
            exp_fit, np.array(x_li), np.array(y_ax3a_li), p0=[3, -0.1, 0.3]
        )
        a_ax3a_li, b_ax3a_li, c_ax3a_li = coeffs_ax3a_li
        eqn_ax3a_li = exp_fit(np.array(x_li), a_ax3a_li, b_ax3a_li, c_ax3a_li)
        ax3a.plot(np.array(x_li), eqn_ax3a_li, color=li_color, linestyle="--")

        coeffs_ax3a_co, covs_ax3a_co = curve_fit(
            exp_fit, np.array(x_co), np.array(y_ax3a_co), p0=[3, -0.1, 0.3]
        )
        a_ax3a_co, b_ax3a_co, c_ax3a_co = coeffs_ax3a_co
        eqn_ax3a_co = exp_fit(np.array(x_co), a_ax3a_co, b_ax3a_co, c_ax3a_co)
        ax3a.plot(np.array(x_co), eqn_ax3a_co, color=co_color, linestyle="--")

        coeffs_ax3a_al, covs_ax3a_al = curve_fit(
            exp_fit, np.array(x_al), np.array(y_ax3a_al), p0=[3, -0.1, 0.3]
        )
        a_ax3a_al, b_ax3a_al, c_ax3a_al = coeffs_ax3a_al
        eqn_ax3a_al = exp_fit(np.array(x_al), a_ax3a_al, b_ax3a_al, c_ax3a_al)
        ax3a.plot(np.array(x_al), eqn_ax3a_al, color=al_color, linestyle="--")

        coeffs_ax3b_li, covs_ax3b_li = curve_fit(
            exp_fit, np.array(x_li), np.array(y_ax3b_li), p0=[3, -0.1, 0.3]
        )
        a_ax3b_li, b_ax3b_li, c_ax3b_li = coeffs_ax3b_li
        eqn_ax3b_li = exp_fit(np.array(x_li), a_ax3b_li, b_ax3b_li, c_ax3b_li)
        ax3b.plot(np.array(x_li), eqn_ax3b_li, color=li_color, linestyle="--")

        coeffs_ax3b_co, covs_ax3b_co = curve_fit(
            exp_fit, np.array(x_co), np.array(y_ax3b_co), p0=[3, -0.1, 0.3]
        )
        a_ax3b_co, b_ax3b_co, c_ax3b_co = coeffs_ax3b_co
        eqn_ax3b_co = exp_fit(np.array(x_co), a_ax3b_co, b_ax3b_co, c_ax3b_co)
        ax3b.plot(np.array(x_co), eqn_ax3b_co, color=co_color, linestyle="--")

        coeffs_ax3b_al, covs_ax3b_al = curve_fit(
            exp_fit, np.array(x_al), np.array(y_ax3b_al), p0=[3, -0.1, 0.3]
        )
        a_ax3b_al, b_ax3b_al, c_ax3b_al = coeffs_ax3b_al
        eqn_ax3b_al = exp_fit(np.array(x_al), a_ax3b_al, b_ax3b_al, c_ax3b_al)
        ax3b.plot(np.array(x_al), eqn_ax3b_al, color=al_color, linestyle="--")

        print(
            f"Feed side, +1: $H_i$ = {a_ax3a_li:.2g} * e^({b_ax3a_li:.2g} * x) + {c_ax3a_li:.2g}"
        )
        print(
            f"Feed side, +2: $H_i$ = {a_ax3a_co:.2g} * e^({b_ax3a_co:.2g} * x) + {c_ax3a_co:.2g}"
        )
        print(
            f"Feed side, +3: $H_i$ = {a_ax3a_al:.2g} * e^({b_ax3a_al:.2g} * x) + {c_ax3a_al:.2g}"
        )

        print(
            f"Permeate side, +1: $H_i$ = {a_ax3b_li:.2g} * e^({b_ax3b_li:.2g} * x) + {c_ax3b_li:.2g}"
        )
        print(
            f"Permeate side, +2: $H_i$ = {a_ax3b_co:.2g} * e^({b_ax3b_co:.2g} * x) + {c_ax3b_co:.2g}"
        )
        print(
            f"Permeate side, +3: $H_i$ = {a_ax3b_al:.2g} * e^({b_ax3b_al:.2g} * x) + {c_ax3b_al:.2g}"
        )

        ax3a.annotate(
            f"$y = {a_ax3a_li:.2g} \exp({b_ax3a_li:.2g} x) + {c_ax3a_li:.2g}$",
            xy=(200, 2),
            fontsize=fontsize - 2,
            color=li_color,
            alpha=alpha,
        )
        ax3a.annotate(
            f"$y = {a_ax3a_co:.2g} \exp({b_ax3a_co:.2g} x) + {c_ax3a_co:.2g}$",
            xy=(200, 1.5),
            fontsize=fontsize - 2,
            color=co_color,
            alpha=alpha,
        )
        ax3a.annotate(
            f"$y = {a_ax3a_al:.2g} \exp({b_ax3a_al:.2g} x) + {c_ax3a_al:.2g}$",
            xy=(200, 1),
            fontsize=fontsize - 2,
            color=al_color,
            alpha=alpha,
        )
        ax3b.annotate(
            f"$y = {a_ax3b_li:.2g} \exp({b_ax3b_li:.2g} x) + {c_ax3b_li:.2g}$",
            xy=(200, 48),
            fontsize=fontsize - 2,
            color=li_color,
            alpha=alpha,
        )
        ax3b.annotate(
            f"$y = {a_ax3b_co:.2g} \exp({b_ax3b_co:.2g} x) + {c_ax3b_co:.2g}$",
            xy=(200, 33),
            fontsize=fontsize - 2,
            color=co_color,
            alpha=alpha,
        )
        ax3b.annotate(
            f"$y = {a_ax3b_al:.2g} \exp({b_ax3b_al:.2g} x) + {c_ax3b_al:.2g}$",
            xy=(200, 21),
            fontsize=fontsize - 2,
            color=al_color,
            alpha=alpha,
        )

    if sieving:
        coeffs_ax4_li, covs_ax4_li = curve_fit(
            exp_fit, np.array(x_li), np.array(y_ax4_li), p0=[-3, -0.1, 0.3]
        )
        a_ax4_li, b_ax4_li, c_ax4_li = coeffs_ax4_li
        eqn_ax4_li = exp_fit(np.array(x_li), a_ax4_li, b_ax4_li, c_ax4_li)
        ax4.plot(np.array(x_li), eqn_ax4_li, color=li_color, linestyle="--")

        coeffs_ax4_co, covs_ax4_co = curve_fit(
            exp_fit, np.array(x_co), np.array(y_ax4_co), p0=[-3, -0.1, 0.3]
        )
        a_ax4_co, b_ax4_co, c_ax4_co = coeffs_ax4_co
        eqn_ax4_co = exp_fit(np.array(x_co), a_ax4_co, b_ax4_co, c_ax4_co)
        ax4.plot(np.array(x_co), eqn_ax4_co, color=co_color, linestyle="--")

        coeffs_ax4_al, covs_ax4_al = curve_fit(
            exp_fit, np.array(x_al), np.array(y_ax4_al), p0=[-3, -0.1, 0.3]
        )
        a_ax4_al, b_ax4_al, c_ax4_al = coeffs_ax4_al
        eqn_ax4_al = exp_fit(np.array(x_al), a_ax4_al, b_ax4_al, c_ax4_al)
        ax4.plot(np.array(x_al), eqn_ax4_al, color=al_color, linestyle="--")

        print(
            f"Observed, +1: $S_i$ = {a_ax4_li:.2g} * e^({b_ax4_li:.2g} * x) + {c_ax4_li:.2g}"
        )
        print(
            f"Observed, +2: $S_i$ = {a_ax4_co:.2g} * e^({b_ax4_co:.2g} * x) + {c_ax4_co:.2g}"
        )
        print(
            f"Observed,+3: $S_i$ = {a_ax4_al:.2g} * e^({b_ax4_al:.2g} * x) + {c_ax4_al:.2g}"
        )

        ax4.annotate(
            f"$y = {a_ax4_li:.2g} \exp({b_ax4_li:.2g} x) + {c_ax4_li:.2g}$",
            xy=(200, 0.63),
            fontsize=fontsize - 2,
            color=li_color,
            alpha=alpha,
        )
        ax4.annotate(
            f"$y = {a_ax4_co:.2g} \exp({b_ax4_co:.2g} x) + {c_ax4_co:.2g}$",
            xy=(200, 0.45),
            fontsize=fontsize - 2,
            color=co_color,
            alpha=alpha,
        )
        ax4.annotate(
            f"$y = {a_ax4_al:.2g} \exp({b_ax4_al:.2g} x) + {c_ax4_al:.2g}$",
            xy=(200, 0.15),
            fontsize=fontsize - 2,
            color=al_color,
            alpha=alpha,
        )

    if save_figure:
        fig1.savefig(f"Donnan_poentials_versus_{x_axis}.png", dpi=600)
        if total_h:
            fig2.savefig(f"overall_H_versus_{x_axis}_regressed.png")
        if sieving:
            fig3.savefig(f"observed_sieving_versus_{x_axis}_regressed.png")


if __name__ == "__main__":
    main()
