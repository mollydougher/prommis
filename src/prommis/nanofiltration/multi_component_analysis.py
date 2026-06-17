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
from idaes.core.util.exceptions import InitializationError
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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
    run_three_salt = False
    run_salt_ratio = False
    solve_and_save_models(
        water_flux=0.02,
        run_single_salt=run_single_salt,
        run_two_salt=run_two_salt,
        run_three_salt=run_three_salt,
        run_salt_ratio=run_salt_ratio,
    )

    plot_together = True
    if plot_together:
        combined_plots_equimolar(
            x_axis="cation_concentration", inset=False, save_figure=False
        )
        # combined_plots_vary_salt_ratio(save_figure=True)
        # plot_only_rejections(save_figure=True)
    else:
        single_salt_plots()
        two_salt_plots()
        plot_flux_contributions(percentages=True)

    plt.show()


def build_model(
    cation_list,
    inlet_concentration,
    default_args,
    H_feed_guess,
    H_permeate_guess,
    NFE_args,
    initialize=True,
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

    # m.fs.membrane.total_feed_ionic_strength.display()
    # fix the degrees of freedom to their default values
    m.fs.membrane.total_module_length.fix()
    m.fs.membrane.total_membrane_length.fix()
    if len(cation_list) == 1:
        m.fs.membrane.applied_pressure.fix(5)
    # elif value(m.fs.membrane.total_feed_ionic_strength[0]) >= 100:
    #     m.fs.membrane.applied_pressure.fix(10)
    else:
        m.fs.membrane.applied_pressure.fix(10)
    m.fs.membrane.feed_flow_volume.fix(inlet_flow_volume["feed"])
    m.fs.membrane.diafiltrate_flow_volume.fix(inlet_flow_volume["diafiltrate"])
    for t in m.fs.membrane.time:
        for j in m.fs.membrane.solutes:
            m.fs.membrane.feed_conc_mol_comp[t, j].fix(inlet_concentration["feed"][j])
            m.fs.membrane.diafiltrate_conc_mol_comp[t, j].fix(
                inlet_concentration["diafiltrate"][j]
            )

    # initialize membrane model
    if initialize:
        initialized_membrane_model = m.fs.membrane.default_initializer(
            H_feed_guess=H_feed_guess, H_permeate_guess=H_permeate_guess
        )
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


def unfix_pressure(m, water_flux=0.02):
    m.fs.membrane.applied_pressure.unfix()

    def _water_flux_constraint(m):
        return (
            m.fs.membrane.volume_flux_water[
                0, m.fs.membrane.dimensionless_module_length.at(2)
            ]
            == water_flux
        )

    m.water_flux_constraint = Constraint(rule=_water_flux_constraint)


def solve_and_save_models(
    water_flux=0.02,
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

    IS_key = ["050", "075", "100", "150", "200", "400", "600", "800"]
    CONC_key = ["025", "050", "075", "100", "150", "200", "250", "300"]

    if run_single_salt:
        if set_IS:
            feed = {
                "Li": [50, 75, 100, 150, 200, 400, 600, 800],
                "Co": [16.667, 25, 33.334, 50, 66.667, 133.334, 200, 266.667],
                "Al": [8.334, 12.5, 16.667, 25, 33.334, 66.667, 100, 133.334],
            }
        else:
            feed = {
                "Li": [25, 50, 75, 100, 150, 200, 250, 300],
                "Co": [25, 50, 75, 100, 150, 200, 250, 300],
                "Al": [25, 50, 75, 100, 150, 200, 250, 300],
            }

        H_feed_guesses = np.arange(0.5, 2.1, 0.1)
        H_permeate_guesses = np.arange(0.5, 2.1, 0.1)
        # add in non equal guesses needed for some systems
        H_feed_guesses = np.append(H_feed_guesses, 1)
        H_permeate_guesses = np.append(H_permeate_guesses, 2)
        H_guesses = np.column_stack((H_feed_guesses, H_permeate_guesses))

        for cation in feed.keys():
            if cation == "Li":
                chloride_multiplier = 1
            elif cation == "Co":
                chloride_multiplier = 2
            elif cation == "Al":
                chloride_multiplier = 3

            for concentration in feed[cation]:
                # start with the low H's at higher concentrations
                if concentration >= 200:
                    H_guesses = H_guesses
                # start with the high H's at lower concentrations
                else:
                    H_guesses = np.flip(H_guesses, axis=0)
                for H_feed_guess, H_permeate_guess in H_guesses:
                    try:
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
                            H_feed_guess=H_feed_guess,
                            H_permeate_guess=H_permeate_guess,
                            NFE_args=NFE_args,
                            initialize=True,
                        )
                        solve_model(model)
                        unfix_pressure(model, water_flux=water_flux)
                        solve_model(model)
                        if set_IS:
                            fname = f"multi_component_case_studies/single_salt/IS/IS{IS_key[feed[cation].index(concentration)]}_{cation}Cl{chloride_multiplier}_{concentration}mM"
                        else:
                            fname = f"multi_component_case_studies/single_salt/CONC/CONC{CONC_key[feed[cation].index(concentration)]}_{cation}Cl{chloride_multiplier}_{concentration}mM"
                        to_json(model, fname=fname)
                        break
                    except:
                        continue
    if run_two_salt:

        feed = {
            "Li_Co": [12.5, 18.75, 25, 37.5, 50, 100, 150, 200],
            # "Li_Al": [7.143], # TODO: debug this system
            "Li_Al": [7.143, 10.7145, 14.286, 21.429, 28.572, 57.143, 85.715, 114.286],
            # "Co_Al": [5.556],# TODO: debug this system
            "Co_Al": [5.556, 8.334, 11.112, 16.667, 22.223, 44.445, 66.667, 88.889],
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
            if salt == "Li_Co":
                cation_1 = "Li"
                cation_2 = "Co"
                chloride_multiplier = 3
            elif salt == "Li_Al":
                cation_1 = "Li"
                cation_2 = "Al"
                chloride_multiplier = 4
            elif salt == "Co_Al":
                cation_1 = "Co"
                cation_2 = "Al"
                chloride_multiplier = 5

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
                            fname=f"multi_component_case_studies/two_salt/{cation_1}{cation_2}Cl{chloride_multiplier}_{concentration}mM_{concentration}mM",
                        )
                        break
                    except:
                        continue

    if run_three_salt:

        feed = {
            "Li_Co_Al": [5, 7.5, 10, 15, 20, 40, 60, 80],
            # "Li_Co_Al": [5], # TODO: debug this system
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
                        )
                        solve_model(model)
                        unfix_pressure(model, water_flux=water_flux)
                        solve_model(model)
                        to_json(
                            model,
                            fname=f"multi_component_case_studies/three_salt/{cation_1}{cation_2}{cation_3}Cl{chloride_multiplier}_{concentration}mM_{concentration}mM_{concentration}mM",
                        )
                        break
                    except:
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

                        except InitializationError:
                            continue


def calculate_spread(list):
    return np.array(
        [
            [np.average(list) - min(list)],
            [max(list) - np.average(list)],
        ]
    )


def single_salt_plots():
    """
    Plots rejection versus ionic strength of bulk fluid and feed.
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
    NFE_args = [
        NFE_module_length,
        NFE_boundary_layer_thickness,
        NFE_membrane_thickness,
    ]

    fig1, ((ax1a, ax1b), (ax1c, ax1d)) = plt.subplots(2, 2, dpi=75, figsize=(15, 12))
    fig1.suptitle(
        "Cations in Single Salt Systems", fontsize=fontsize, fontweight="bold"
    )
    # fig1.tight_layout()

    fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, dpi=75, figsize=(15, 12))
    fig2.suptitle("Anion in Single Salt Systems", fontsize=fontsize, fontweight="bold")
    # fig2.tight_layout()

    for ax in [ax1c, ax1d, ax2c, ax2d]:
        ax.set_xlabel(
            "Inlet Feed Ionic Strength (mol/m$^3$)",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax1a, ax2a]:
        ax.set_title("Solute Rejection", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel("Rejection (%)", fontsize=fontsize, fontweight="bold")
        ax.plot([], [], "ro", markersize=markersize, label="LiCl")
        ax.plot([], [], "bo", markersize=markersize, label="CoCl$_2$")
        ax.plot([], [], "go", markersize=markersize, label="AlCl$_2$")
        ax.plot([], [], marker="None", linestyle="None", label="Rejection (fill)")
        ax.plot([], [], "ko", mfc="none", markersize=markersize, label="Observed")
        ax.plot([], [], "ko", markersize=markersize, label="Actual")
    for ax in [ax1b, ax2b]:
        ax.set_title("Solute Flux", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel("Flux (mol/m$^2$/h)", fontsize=fontsize, fontweight="bold")
    for ax in [ax1c, ax2c]:
        ax.set_title(
            "Feed-Side Partition Coefficient", fontsize=fontsize, fontweight="bold"
        )
        ax.set_ylabel(
            "$c_{membrane}/c_{interface}$", fontsize=fontsize, fontweight="bold"
        )
    for ax in [ax1d, ax2d]:
        ax.set_title(
            "Permeate-Side Partition Coefficient", fontsize=fontsize, fontweight="bold"
        )
        ax.set_ylabel(
            "$c_{membrane}/c_{permeate}$", fontsize=fontsize, fontweight="bold"
        )
    for ax in [ax1b, ax1c, ax1d, ax2b, ax2c, ax2d]:
        ax.plot([], [], "ro", markersize=markersize, label="LiCl")
        ax.plot([], [], "bo", markersize=markersize, label="CoCl$_2$")
        ax.plot([], [], "go", markersize=markersize, label="AlCl$_2$")
    for ax in [ax1a, ax1b, ax1c, ax1d, ax2a, ax2b, ax2c, ax2d]:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        if ax == ax1a or ax == ax2a:
            ax.legend(loc="best", fontsize=fontsize, ncol=2)
        else:
            ax.legend(loc="best", fontsize=fontsize)

    fig3, ((ax3a, ax3b), (ax3c, ax3d), (ax3e, ax3f)) = plt.subplots(
        3, 2, dpi=75, figsize=(18, 15)
    )
    fig3.suptitle(
        "Average Cation Flux Contributions", fontsize=fontsize, fontweight="bold"
    )
    # fig3.tight_layout()

    fig4, ((ax4a, ax4b), (ax4c, ax4d), (ax4e, ax4f)) = plt.subplots(
        3, 2, dpi=75, figsize=(18, 15)
    )
    fig4.suptitle(
        "Average Anion Flux Contributions", fontsize=fontsize, fontweight="bold"
    )
    # fig4.tight_layout()

    ax3a.set_ylabel(
        "Lithium Flux\n(mol m$^{-2}$ h$^{-1}$)", fontsize=fontsize, fontweight="bold"
    )
    ax3c.set_ylabel(
        "Cobalt Flux\n(mol m$^{-2}$ h$^{-1}$)", fontsize=fontsize, fontweight="bold"
    )
    ax3e.set_ylabel(
        "Aluminum Flux\n(mol m$^{-2}$ h$^{-1}$)", fontsize=fontsize, fontweight="bold"
    )
    ax4a.set_ylabel(
        "Chloride Flux in LiCl\n(mol m$^{-2}$ h$^{-1}$)",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax4c.set_ylabel(
        "Chloride Flux in CoCl$_2$\n(mol m$^{-2}$ h$^{-1}$)",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax4e.set_ylabel(
        "Chloride Flux in AlCl$_3$\n(mol m$^{-2}$ h$^{-1}$)",
        fontsize=fontsize,
        fontweight="bold",
    )
    for ax in [ax3a, ax4a]:
        ax.set_title("Boundary Layer", fontsize=fontsize, fontweight="bold")
    for ax in [ax3b, ax4b]:
        ax.set_title("Membrane", fontsize=fontsize, fontweight="bold")
    for ax in [ax3e, ax3f, ax4e, ax4f]:
        ax.set_xlabel(
            "Inlet Feed Ionic Strength (mol/m$^3$)",
            fontsize=14,
            fontweight="bold",
        )
    for ax in [ax3a, ax3b, ax3c, ax3d, ax3e, ax3f, ax4a, ax4b, ax4c, ax4d, ax4e, ax4f]:
        ax.plot([], [], "ro", markersize=8, label="Convection")
        ax.plot([], [], "bo", markersize=8, label="Diffusion")
        ax.plot([], [], "go", markersize=8, label="Electromigration")
        ax.tick_params(direction="in", top=True, right=True, labelsize=14)
        ax.axhline(0, color="black", linewidth=1.5)
        ax.legend(loc="best", fontsize=fontsize)

    model_folder = Path("multi_component_case_studies/single_salt")
    # 41 characters (0-40) make up folder name before model name
    # multi_component_case_studies/single_salt/

    for case_study_file in model_folder.iterdir():
        cation = str(case_study_file)[47:49]
        chloride_multiplier = float(str(case_study_file)[51])
        concentration = float(50)  # mM
        model = build_model(
            cation_list=[cation],
            inlet_concentration={
                "feed": {
                    cation: concentration,
                    "Cl": chloride_multiplier * concentration,
                },
                "diafiltrate": {
                    cation: 1e-10,
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

        feed_ionic_strength_val = value(model.fs.membrane.total_feed_ionic_strength[0])

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
                                model.fs.membrane.actual_rejection_percent[t, x, solute]
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
                        ) in model.fs.membrane.dimensionless_boundary_layer_thickness:
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
                        bl_electromigration_dict_by_x[f"{x}"] = bl_electromigration_by_x
                        bl_convection_by_x = []
                        bl_diffusion_by_x = []
                        bl_electromigration_by_x = []

                        for z_mem in model.fs.membrane.dimensionless_membrane_thickness:
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
            avg_bl_electromigration = np.average(bl_electromigration_averaged_over_z)
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
            avg_mem_electromigration = np.average(mem_electromigration_averaged_over_z)
            spread_mem_electromigration = calculate_spread(
                mem_electromigration_averaged_over_z
            )

            alpha = 1
            marker = "o"

            if solute != "Cl":
                ax_rej = ax1a
                ax_flux = ax1b
                ax_hfeed = ax1c
                ax_hperm = ax1d

            if solute == "Li":
                color = "red"
                ax_bl_flux = ax3a
                ax_mem_flux = ax3b
            elif solute == "Co":
                color = diff_color
                ax_bl_flux = ax3c
                ax_mem_flux = ax3d
            elif solute == "Al":
                color = elec_color
                ax_bl_flux = ax3e
                ax_mem_flux = ax3f
            elif solute == "Cl":
                ax_rej = ax2a
                ax_flux = ax2b
                ax_hfeed = ax2c
                ax_hperm = ax2d

                if model.fs.membrane.cations.at(1) == "Li":
                    color = "red"
                    ax_bl_flux = ax4a
                    ax_mem_flux = ax4b
                elif model.fs.membrane.cations.at(1) == "Co":
                    color = diff_color
                    ax_bl_flux = ax4c
                    ax_mem_flux = ax4d
                elif model.fs.membrane.cations.at(1) == "Al":
                    color = elec_color
                    ax_bl_flux = ax4e
                    ax_mem_flux = ax4f

            ax_rej.plot(
                feed_ionic_strength_val,
                avg_observed_rejection,
                color=color,
                marker=marker,
                alpha=alpha,
                mfc="none",
                markersize=markersize,
            )
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
                avg_observed_rejection,
                yerr=spread_observed_rejection,
                ecolor="grey",
                capsize=4,
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

            ax_bl_flux.plot(
                feed_ionic_strength_val,
                avg_bl_convection,
                "ro",
                alpha=alpha,
                markersize=markersize,
            )
            ax_bl_flux.errorbar(
                feed_ionic_strength_val,
                avg_bl_convection,
                yerr=spread_bl_convection,
                ecolor="grey",
                capsize=4,
            )
            ax_bl_flux.plot(
                feed_ionic_strength_val,
                avg_bl_diffusion,
                "bo",
                alpha=alpha,
                markersize=markersize,
            )
            # TODO: debug negative yerr exception
            # ax_bl_flux.errorbar(
            #     feed_ionic_strength_val,
            #     avg_bl_diffusion,
            #     yerr=spread_bl_diffusion,
            #     ecolor="grey",
            #     capsize=4,
            # )
            ax_bl_flux.plot(
                feed_ionic_strength_val,
                avg_bl_electromigration,
                "go",
                alpha=alpha,
                markersize=markersize,
            )
            ax_bl_flux.errorbar(
                feed_ionic_strength_val,
                avg_bl_electromigration,
                yerr=spread_bl_electromigration,
                ecolor="grey",
                capsize=4,
            )
            # flux check
            # TODO: debug mismatch of fluxes
            ax_bl_flux.plot(
                feed_ionic_strength_val,
                (avg_bl_convection + avg_bl_diffusion + avg_bl_electromigration),
                "ms",
                mfc="None",
                mew=3,
                alpha=alpha,
                markersize=markersize,
            )

            ax_mem_flux.plot(
                feed_ionic_strength_val,
                avg_mem_convection,
                "ro",
                alpha=alpha,
                markersize=markersize,
            )
            ax_mem_flux.errorbar(
                feed_ionic_strength_val,
                avg_mem_convection,
                yerr=spread_mem_convection,
                ecolor="grey",
                capsize=4,
            )
            ax_mem_flux.plot(
                feed_ionic_strength_val,
                avg_mem_diffusion,
                "bo",
                alpha=alpha,
                markersize=markersize,
            )
            # TODO: debug negative yerr exception
            # ax_mem_flux.errorbar(
            #     feed_ionic_strength_val,
            #     avg_mem_diffusion,
            #     yerr=spread_mem_diffusion,
            #     ecolor="grey",
            #     capsize=4,
            # )
            ax_mem_flux.plot(
                feed_ionic_strength_val,
                avg_mem_electromigration,
                "go",
                alpha=alpha,
                markersize=markersize,
            )
            ax_mem_flux.errorbar(
                feed_ionic_strength_val,
                avg_mem_electromigration,
                yerr=spread_mem_electromigration,
                ecolor="grey",
                capsize=4,
            )
            # flux check
            # TODO: debug mismatch of fluxes
            ax_mem_flux.plot(
                feed_ionic_strength_val,
                (avg_mem_convection + avg_mem_diffusion + avg_mem_electromigration),
                "ms",
                mfc="None",
                mew=3,
                alpha=alpha,
                markersize=markersize,
            )

            for ax in [ax_bl_flux, ax_mem_flux]:
                ax.plot(
                    feed_ionic_strength_val,
                    avg_flux,
                    "ks",
                    alpha=alpha,
                    markersize=markersize,
                )
                ax.errorbar(
                    feed_ionic_strength_val,
                    avg_flux,
                    yerr=spread_flux,
                    ecolor="grey",
                    capsize=4,
                )


def two_salt_plots():
    """
    Plots rejection versus ionic strength of bulk fluid and feed.
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
    NFE_args = [
        NFE_module_length,
        NFE_boundary_layer_thickness,
        NFE_membrane_thickness,
    ]

    fig1, ((ax1a, ax1b), (ax1c, ax1d)) = plt.subplots(2, 2, dpi=75, figsize=(15, 12))
    fig1.suptitle(
        "Lithium Chloride + Cobalt Chloride", fontsize=fontsize, fontweight="bold"
    )
    # fig1.tight_layout()

    fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, dpi=75, figsize=(15, 12))
    fig2.suptitle(
        "Lithium Chloride + Aluminum Chloride", fontsize=fontsize, fontweight="bold"
    )
    # fig2.tight_layout()

    fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2, dpi=75, figsize=(15, 12))
    fig3.suptitle(
        "Cobalt Chloride + Aluminum Chloride", fontsize=fontsize, fontweight="bold"
    )
    # fig3.tight_layout()

    for ax in [ax1c, ax1d, ax2c, ax2d, ax3c, ax3d]:
        ax.set_xlabel(
            "Inlet Feed Ionic Strength (mol/m$^3$)",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax1a, ax2a, ax3a]:
        ax.set_title("Solute Rejection", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel("Rejection (%)", fontsize=fontsize, fontweight="bold")
        if ax == ax1a:
            ax.plot([], [], "ro", markersize=markersize, label="Li")
            ax.plot([], [], "bo", markersize=markersize, label="Co")
        elif ax == ax2a:
            ax.plot([], [], "ro", markersize=markersize, label="Li")
            ax.plot([], [], "go", markersize=markersize, label="Al")
        elif ax == ax3a:
            ax.plot([], [], "bo", markersize=markersize, label="Co")
            ax.plot([], [], "go", markersize=markersize, label="Al")
        ax.plot(
            [],
            [],
            color="orange",
            marker="o",
            linestyle="None",
            markersize=markersize,
            label="Cl",
        )
        ax.plot([], [], marker="None", linestyle="None", label="Rejection (fill)")
        ax.plot([], [], "ko", mfc="none", markersize=markersize, label="Observed")
        ax.plot([], [], "ko", markersize=markersize, label="Actual")
    for ax in [ax1b, ax2b, ax3b]:
        ax.set_title("Solute Flux", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel("Flux (mol/m$^2$/h)", fontsize=fontsize, fontweight="bold")
    for ax in [ax1c, ax2c, ax3c]:
        ax.set_title(
            "Feed-Side Partition Coefficient", fontsize=fontsize, fontweight="bold"
        )
        ax.set_ylabel(
            "$c_{membrane}/c_{interface}$", fontsize=fontsize, fontweight="bold"
        )
    for ax in [ax1d, ax2d, ax3d]:
        ax.set_title(
            "Permeate-Side Partition Coefficient", fontsize=fontsize, fontweight="bold"
        )
        ax.set_ylabel(
            "$c_{membrane}/c_{permeate}$", fontsize=fontsize, fontweight="bold"
        )
    for ax in [ax1b, ax1c, ax1d]:
        ax.plot([], [], "ro", markersize=markersize, label="Li")
        ax.plot([], [], "bo", markersize=markersize, label="Co")
        ax.plot(
            [],
            [],
            color="orange",
            marker="o",
            linestyle="None",
            markersize=markersize,
            label="Cl",
        )
    for ax in [ax2b, ax2c, ax2d]:
        ax.plot([], [], "ro", markersize=markersize, label="Li")
        ax.plot([], [], "go", markersize=markersize, label="Al")
        ax.plot(
            [],
            [],
            color="orange",
            marker="o",
            linestyle="None",
            markersize=markersize,
            label="Cl",
        )
    for ax in [ax3b, ax3c, ax3d]:
        ax.plot([], [], "bo", markersize=markersize, label="Co")
        ax.plot([], [], "go", markersize=markersize, label="Al")
        ax.plot(
            [],
            [],
            color="orange",
            marker="o",
            linestyle="None",
            markersize=markersize,
            label="Cl",
        )
    for ax in [ax1a, ax1b, ax1c, ax1d, ax2a, ax2b, ax2c, ax2d, ax3a, ax3b, ax3c, ax3d]:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        if ax == ax1a or ax == ax2a or ax == ax3a:
            ax.legend(loc="best", fontsize=fontsize, ncol=2)
        else:
            ax.legend(loc="best", fontsize=fontsize)

    model_folder = Path("multi_component_case_studies/two_salt")
    # 38 characters (0-37) make up folder name before model name
    # multi_component_case_studies/two_salt/

    for case_study_file in model_folder.iterdir():
        cation_1 = str(case_study_file)[38:40]
        cation_2 = str(case_study_file)[40:42]
        chloride_multiplier = float(str(case_study_file)[44])
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

        feed_ionic_strength_val = value(model.fs.membrane.total_feed_ionic_strength[0])

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
                                model.fs.membrane.actual_rejection_percent[t, x, solute]
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
                        ) in model.fs.membrane.dimensionless_boundary_layer_thickness:
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
                        bl_electromigration_dict_by_x[f"{x}"] = bl_electromigration_by_x
                        bl_convection_by_x = []
                        bl_diffusion_by_x = []
                        bl_electromigration_by_x = []

                        for z_mem in model.fs.membrane.dimensionless_membrane_thickness:
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
            avg_bl_electromigration = np.average(bl_electromigration_averaged_over_z)
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
            avg_mem_electromigration = np.average(mem_electromigration_averaged_over_z)
            spread_mem_electromigration = calculate_spread(
                mem_electromigration_averaged_over_z
            )

            alpha = 1
            marker = "o"

            if cation_1 == "Li" and cation_2 == "Co":
                ax_rej = ax1a
                ax_flux = ax1b
                ax_hfeed = ax1c
                ax_hperm = ax1d
            elif cation_1 == "Li" and cation_2 == "Al":
                ax_rej = ax2a
                ax_flux = ax2b
                ax_hfeed = ax2c
                ax_hperm = ax2d
            elif cation_1 == "Co" and cation_2 == "Al":
                ax_rej = ax3a
                ax_flux = ax3b
                ax_hfeed = ax3c
                ax_hperm = ax3d

            if solute == "Li":
                color = "red"
            elif solute == "Co":
                color = diff_color
            elif solute == "Al":
                color = elec_color
            elif solute == "Cl":
                color = "orange"

            ax_rej.plot(
                feed_ionic_strength_val,
                avg_observed_rejection,
                color=color,
                marker=marker,
                alpha=alpha,
                mfc="none",
                markersize=markersize,
            )
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
                avg_observed_rejection,
                yerr=spread_observed_rejection,
                ecolor="grey",
                capsize=4,
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


def get_model_averages(model, solute):
    observed_rejection = []
    actual_rejection = []
    flux = []
    H_feed = []
    H_perm = []

    for t in model.fs.membrane.time:
        for x in model.fs.membrane.dimensionless_module_length:
            if x != 0:
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

    info_dict = {
        # "observed_rejection": {"avg": avg_observed_rejection, "spread": spread_observed_rejection},
        "actual_rejection": {
            "avg": avg_actual_rejection,
            "spread": spread_actual_rejection,
        },
        "flux": {"avg": avg_flux, "spread": spread_flux},
        "H_feed": {"avg": avg_H_feed, "spread": spread_H_feed},
        "H_perm": {"avg": avg_H_perm, "spread": spread_H_perm},
    }

    return info_dict


def combined_plots_equimolar(x_axis="ionic_strength", inset=True, save_figure=True):
    """
    Plots rejection versus ionic strength of bulk fluid and feed.
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
    NFE_args = [
        NFE_module_length,
        NFE_boundary_layer_thickness,
        NFE_membrane_thickness,
    ]

    fig1, (
        (ax1a, ax1b, ax1c, ax1d),  # rejection
        (ax2a, ax2b, ax2c, ax2d),  # solute flux
        (ax3a, ax3b, ax3c, ax3d),  # H_feed
        (ax4a, ax4b, ax4c, ax4d),  # H_permeate
    ) = plt.subplots(4, 4, dpi=50, figsize=(18, 18), constrained_layout=True)

    if x_axis == "ionic_strength":
        for ax in [ax4a, ax4b, ax4c, ax4d]:
            ax.set_xlabel(
                "Inlet Feed Ionic Strength\n(mol/m$\mathbf{^3}$)",
                fontsize=fontsize,
                fontweight="bold",
            )
    elif x_axis == "cation_concentration":
        ax4a.set_xlabel(
            "Lithium Feed Concentration\n(mol/m$\mathbf{^3}$)",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax4b.set_xlabel(
            "Cobalt Feed Concentration\n(mol/m$\mathbf{^3}$)",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax4c.set_xlabel(
            "Aluminum Feed Concentration\n(mol/m$\mathbf{^3}$)",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax4d.set_xlabel(
            "Cation Feed Concentration\n(mol/m$\mathbf{^3}$)",
            fontsize=fontsize,
            fontweight="bold",
        )

    ax1a.set_title("Lithium", fontsize=fontsize, fontweight="bold")
    ax1b.set_title("Cobalt", fontsize=fontsize, fontweight="bold")
    ax1c.set_title("Aluminum", fontsize=fontsize, fontweight="bold")
    ax1d.set_title("Chloride", fontsize=fontsize, fontweight="bold")

    ax1a.set_ylabel("Actual Ion\nRejection (%)", fontsize=fontsize, fontweight="bold")
    # for ax in [ax1a, ax1b, ax1c, ax1d]:
    #     ax.set_ylim(15, 75)
    ax2a.set_ylabel("Ion Flux (mol/m$^2$/h)", fontsize=fontsize, fontweight="bold")
    # for ax in [ax2a, ax2b, ax2c, ax2d]:
    #     ax.set_ylim(0, 15)
    ax3a.set_ylabel(
        "$c_{membrane}/c_{interface}$", fontsize=fontsize, fontweight="bold"
    )
    # for ax in [ax3a, ax3b, ax3c]:
    #     ax.set_ylim(0, 1.1)
    # for ax in [ax4a, ax4b, ax4c]:
    #     ax.set_ylim(0, 4.1)
    # for ax in [ax3d, ax4d]:
    #     ax.set_ylim(0, 0.07)
    ax4a.set_ylabel("$c_{membrane}/c_{permeate}$", fontsize=fontsize, fontweight="bold")

    # continuous cmap
    # cmap = plt.colormaps["viridis"]
    # cmap_loc = np.linspace(0, 1.0, 6)
    # li_color = cmap(cmap_loc[0])
    # co_color = cmap(cmap_loc[3])
    # al_color = cmap(cmap_loc[2])
    # li_co_color = cmap(cmap_loc[1])
    # li_al_color = cmap(cmap_loc[4])
    # co_al_color = cmap(cmap_loc[5])

    # qualitative cmap
    # cmap = plt.colormaps["Dark2"]
    # li_color = cmap(0)
    # co_color = cmap(1)
    # al_color = cmap(2)
    # li_co_color = cmap(3)
    # li_al_color = cmap(4)
    # co_al_color = cmap(5)

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
            "ax": ax4a,
            "salt_colors": {
                "LiCl": li_color,
                "LiCl + CoCl$_2$": li_co_color,
                "LiCl + AlCl$_3$": li_al_color,
                "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
        "cobalt": {
            "marker": "v",
            "ax": ax4b,
            "salt_colors": {
                "CoCl$_2$": co_color,
                "LiCl + CoCl$_2$": li_co_color,
                "CoCl$_2$ + AlCl$_3$": co_al_color,
                "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
        "aluminum": {
            "marker": "^",
            "ax": ax4c,
            "salt_colors": {
                "AlCl$_3$": al_color,
                "LiCl + AlCl$_3$": li_al_color,
                "CoCl$_2$ + AlCl$_3$": co_al_color,
                "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
            },
        },
        "chloride": {
            "marker": "*",
            "ax": ax4d,
            "salt_colors": {
                "LiCl": li_color,
                "CoCl$_2$": co_color,
                "AlCl$_3$": al_color,
                "LiCl + CoCl$_2$": li_co_color,
                "LiCl + AlCl$_3$": li_al_color,
                "CoCl$_2$ + AlCl$_3$": co_al_color,
                "LiCl + CoCl$_2$ + AlCl$_3$": li_co_al_color,
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

    for ax in [ax4a, ax4b, ax4c]:
        ax.legend(loc="best", fontsize=fontsize, bbox_to_anchor=(0.85, -0.3))
    ax4d.legend(loc="best", fontsize=fontsize, bbox_to_anchor=(1.1, -0.3), ncol=2)

    for ax in fig1.axes:
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        if x_axis == "ionic_strength":
            ax.set_xlim(0, 900)

    if inset:
        # create inset axes for overlapping points
        # inax2a = ax2a.inset_axes([0.1, 0.65, 0.45, 0.4])
        # inax2b = ax2b.inset_axes([0.1, 0.65, 0.45, 0.4])
        # inax2c = ax2c.inset_axes([0.1, 0.65, 0.45, 0.4])
        # inax2d = ax2d.inset_axes([0.1, 0.65, 0.45, 0.4])

        inax3a = ax3a.inset_axes([0.45, 0.4, 0.45, 0.5])
        inax3b = ax3b.inset_axes([0.45, 0.4, 0.45, 0.5])
        inax3c = ax3c.inset_axes([0.45, 0.4, 0.45, 0.5])
        # inax3d = ax3d.inset_axes([0.1, 0.65, 0.45, 0.4])

        inax4a = ax4a.inset_axes([0.45, 0.4, 0.45, 0.5])
        inax4b = ax4b.inset_axes([0.45, 0.4, 0.45, 0.5])
        inax4c = ax4c.inset_axes([0.45, 0.4, 0.45, 0.5])
        # inax4d = ax4d.inset_axes([0.1, 0.65, 0.45, 0.4])

        inset_axes_dict = {
            # ax2a: inax2a,
            # ax2b: inax2b,
            # ax2c: inax2c,
            # ax2d: inax2d,
            ax3a: inax3a,
            ax3b: inax3b,
            ax3c: inax3c,
            # ax3d: inax3d,
            ax4a: inax4a,
            ax4b: inax4b,
            ax4c: inax4c,
            # ax4d: inax4d,
        }

        for ax, inax in inset_axes_dict.items():
            inax.set_xlim(25, 225)
            inax.tick_params(direction="in", top=True, right=True, labelsize=10)
            ax.indicate_inset_zoom(inax, edgecolor="black")

        # inax2a.set_ylim(0, 3)
        # inax2b.set_ylim(0, 0.9)
        # inax2c.set_ylim(0, 0.35)
        # inax2d.set_ylim(0, 3)

        inax3a.set_ylim(0.18, 0.57)
        inax3b.set_ylim(0.2, 1.42)
        inax3c.set_ylim(0.18, 1.2)
        # inax3d.set_ylim(0.005, 0.03)

        inax4a.set_ylim(0.2, 1)
        inax4b.set_ylim(0.4, 3.5)
        inax4c.set_ylim(0.5, 3.1)
        # inax4d.set_ylim(0.001, 0.02)

    if x_axis == "ionic_strength":
        model_folder_1 = Path("multi_component_case_studies/single_salt/IS")
        # 44 characters (0-43) make up folder name before model name
        # multi_component_case_studies/single_salt/IS/
        model_folder_2 = Path("multi_component_case_studies/two_salt")
        # 38 characters (0-37) make up folder name before model name
        # multi_component_case_studies/two_salt/
        model_folder_3 = Path("multi_component_case_studies/three_salt")
        # 40 characters (0-39) make up folder name before model name
        # multi_component_case_studies/three_salt/

        case_study_list_1 = [file for file in model_folder_1.iterdir()]
        case_study_list_2 = [file for file in model_folder_2.iterdir()]
        case_study_list_3 = [file for file in model_folder_3.iterdir()]
        case_studies = {
            "single": case_study_list_1,
            "two": case_study_list_2,
            "three": case_study_list_3,
        }
    else:
        model_folder_1 = Path("multi_component_case_studies/single_salt/CONC")
        # 46 characters (0-45) make up folder name before model name
        # multi_component_case_studies/single_salt/CONC/

        case_study_list_1 = [file for file in model_folder_1.iterdir()]
        case_studies = {
            "single": case_study_list_1,
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
                cation_1 = str(case_study)[38:40]
                cation_2 = str(case_study)[40:42]
                chloride_multiplier = float(str(case_study)[44])
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
                cation_1 = str(case_study)[40:42]
                cation_2 = str(case_study)[42:44]
                cation_3 = str(case_study)[44:46]
                chloride_multiplier = float(str(case_study)[48])
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
                H_feed_guess=1,
                H_permeate_guess=1,
                NFE_args=NFE_args,
                initialize=False,
            )
            from_json(model, fname=case_study)

            for solute in model.fs.membrane.solutes:
                average_variable_dict = get_model_averages(model, solute)

                if x_axis == "ionic_strength":
                    x_value = feed_ionic_strength_val = value(
                        model.fs.membrane.total_feed_ionic_strength[0]
                    )
                elif x_axis == "cation_concentration":
                    x_value = value(
                        model.fs.membrane.retentate_conc_mol_comp[0, 0, solute]
                    )

                alpha = 1

                if solute == "Li":
                    marker = "o"
                    ax_rej = ax1a
                    ax_flux = ax2a
                    ax_hfeed = ax3a
                    ax_hperm = ax4a
                    if inset:
                        # inax_flux = inax2a
                        inax_hfeed = inax3a
                        inax_hperm = inax4a
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
                    ax_flux = ax2b
                    ax_hfeed = ax3b
                    ax_hperm = ax4b
                    if inset:
                        # inax_flux = inax2b
                        inax_hfeed = inax3b
                        inax_hperm = inax4b
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
                    ax_flux = ax2c
                    ax_hfeed = ax3c
                    ax_hperm = ax4c
                    if inset:
                        # inax_flux = inax2c
                        inax_hfeed = inax3c
                        inax_hperm = inax4c
                    if type == "single":
                        color = al_color
                    elif type == "two":
                        if cation_1 == "Li":
                            color = li_al_color
                        elif cation_1 == "Co":
                            color = co_al_color

                elif solute == "Cl":
                    marker = "*"
                    ax_rej = ax1d
                    ax_flux = ax2d
                    ax_hfeed = ax3d
                    ax_hperm = ax4d
                    # if inset:
                    # inax_flux = inax2d
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

                for metric, info_dict in average_variable_dict.items():
                    if metric == "actual_rejection":
                        ax = ax_rej
                        # inax=inax_rej
                    elif metric == "flux":
                        ax = ax_flux
                        # inax=inax_flux
                    elif metric == "H_feed":
                        ax = ax_hfeed
                        if inset:
                            inax = inax_hfeed
                    elif metric == "H_perm":
                        ax = ax_hperm
                        if inset:
                            inax = inax_hperm

                    ax.plot(
                        x_value,
                        info_dict["avg"],
                        color=color,
                        marker=marker,
                        alpha=alpha,
                        markersize=markersize,
                    )
                    ax.errorbar(
                        x_value,
                        info_dict["avg"],
                        yerr=info_dict["spread"],
                        ecolor="grey",
                        capsize=4,
                    )

                    if inset and (metric == "H_feed" or metric == "H_perm"):
                        inax.plot(
                            x_value,
                            info_dict["avg"],
                            color=color,
                            marker=marker,
                            alpha=alpha,
                            markersize=markersize,
                        )
                        inax.errorbar(
                            x_value,
                            info_dict["avg"],
                            yerr=info_dict["spread"],
                            ecolor="grey",
                            capsize=4,
                        )

    if save_figure:
        plt.savefig(f"rejection_flux_partition_versus_{x_axis}.png", dpi=600)


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


def plot_flux_contributions(percentages=True, total=True):
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
        3, 2, figsize=(15, 10), dpi=75, constrained_layout=True
    )
    fig2, ((ax4a, ax4b), (ax5a, ax5b), (ax6a, ax6b)) = plt.subplots(
        3, 2, figsize=(15, 10), dpi=75, constrained_layout=True
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
    # flux_color = tol_bright_hex[6]
    flux_color = "black"

    model_folder_1 = Path("multi_component_case_studies/single_salt")
    # 41 characters (0-40) make up folder name before model name
    # multi_component_case_studies/single_salt/

    lithium_flux_averages = []
    cobalt_flux_averages = []
    aluminum_flux_averages = []
    cl_lithium_flux_averages = []
    cl_cobalt_flux_averages = []
    cl_aluminum_flux_averages = []

    lithium_bl_convection_averages = []
    lithium_bl_diffusion_averages = []
    lithium_bl_electromigration_averages = []

    cobalt_bl_convection_averages = []
    cobalt_bl_diffusion_averages = []
    cobalt_bl_electromigration_averages = []

    aluminum_bl_convection_averages = []
    aluminum_bl_diffusion_averages = []
    aluminum_bl_electromigration_averages = []

    cl_lithium_bl_convection_averages = []
    cl_lithium_bl_diffusion_averages = []
    cl_lithium_bl_electromigration_averages = []

    cl_cobalt_bl_convection_averages = []
    cl_cobalt_bl_diffusion_averages = []
    cl_cobalt_bl_electromigration_averages = []

    cl_aluminum_bl_convection_averages = []
    cl_aluminum_bl_diffusion_averages = []
    cl_aluminum_bl_electromigration_averages = []

    lithium_mem_convection_averages = []
    lithium_mem_diffusion_averages = []
    lithium_mem_electromigration_averages = []

    cobalt_mem_convection_averages = []
    cobalt_mem_diffusion_averages = []
    cobalt_mem_electromigration_averages = []

    aluminum_mem_convection_averages = []
    aluminum_mem_diffusion_averages = []
    aluminum_mem_electromigration_averages = []

    cl_lithium_mem_convection_averages = []
    cl_lithium_mem_diffusion_averages = []
    cl_lithium_mem_electromigration_averages = []

    cl_cobalt_mem_convection_averages = []
    cl_cobalt_mem_diffusion_averages = []
    cl_cobalt_mem_electromigration_averages = []

    cl_aluminum_mem_convection_averages = []
    cl_aluminum_mem_diffusion_averages = []
    cl_aluminum_mem_electromigration_averages = []

    case_study_list = []
    for case_study_file in model_folder_1.iterdir():
        case_study_list.append(case_study_file)
    case_study_list.sort()

    for case_study in case_study_list:
        cation = str(case_study)[47:49]
        chloride_multiplier = float(str(case_study)[51])
        concentration = float(50)  # mM
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
            H_feed_guess=1,
            H_permeate_guess=1,
            NFE_args=NFE_args,
            initialize=False,
        )
        from_json(model, fname=case_study)

        for solute in model.fs.membrane.solutes:
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
                        flux.append(
                            value(model.fs.membrane.molar_ion_flux[t, x, solute])
                        )
                        for (
                            z_bl
                        ) in model.fs.membrane.dimensionless_boundary_layer_thickness:
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
                        electromigration_mem_dict_by_x[f"{x}"] = (
                            electromigration_mem_by_x
                        )

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
                sum(electromigration_bl_dict_by_x[k])
                / len(electromigration_bl_dict_by_x[k])
                for k in electromigration_bl_dict_by_x.keys()
            ]
            avg_bl_electromigration = np.average(bl_electromigration_averaged_over_z)
            spread_bl_electromigration = calculate_spread(
                bl_electromigration_averaged_over_z
            )
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
                sum(electromigration_mem_dict_by_x[k])
                / len(electromigration_mem_dict_by_x[k])
                for k in electromigration_mem_dict_by_x.keys()
            ]
            avg_mem_electromigration = np.average(mem_electromigration_averaged_over_z)
            spread_mem_electromigration = calculate_spread(
                mem_electromigration_averaged_over_z
            )
            avg_mem_electromigration_percent = (
                np.average(mem_electromigration_averaged_over_z)
                / np.average(flux)
                * 100
            )

            if percentages:
                plot_bl_convection = avg_bl_convection_percent
                plot_bl_diffusion = avg_bl_diffusion_percent
                plot_bl_electromigration = avg_bl_electromigration_percent
                plot_mem_convection = avg_mem_convection_percent
                plot_mem_diffusion = avg_mem_diffusion_percent
                plot_mem_electromigration = avg_mem_electromigration_percent
            else:
                plot_bl_convection = avg_bl_convection
                plot_bl_diffusion = avg_bl_diffusion
                plot_bl_electromigration = avg_bl_electromigration
                plot_mem_convection = avg_mem_convection
                plot_mem_diffusion = avg_mem_diffusion
                plot_mem_electromigration = avg_mem_electromigration

            if solute == "Li":
                lithium_flux_averages.append(avg_flux)
                lithium_bl_convection_averages.append(plot_bl_convection)
                lithium_bl_diffusion_averages.append(plot_bl_diffusion)
                lithium_bl_electromigration_averages.append(plot_bl_electromigration)

                lithium_mem_convection_averages.append(plot_mem_convection)
                lithium_mem_diffusion_averages.append(plot_mem_diffusion)
                lithium_mem_electromigration_averages.append(plot_mem_electromigration)
            elif solute == "Co":
                cobalt_flux_averages.append(avg_flux)
                cobalt_bl_convection_averages.append(plot_bl_convection)
                cobalt_bl_diffusion_averages.append(plot_bl_diffusion)
                cobalt_bl_electromigration_averages.append(plot_bl_electromigration)

                cobalt_mem_convection_averages.append(plot_mem_convection)
                cobalt_mem_diffusion_averages.append(plot_mem_diffusion)
                cobalt_mem_electromigration_averages.append(plot_mem_electromigration)
            elif solute == "Al":
                aluminum_flux_averages.append(avg_flux)
                aluminum_bl_convection_averages.append(plot_bl_convection)
                aluminum_bl_diffusion_averages.append(plot_bl_diffusion)
                aluminum_bl_electromigration_averages.append(plot_bl_electromigration)

                aluminum_mem_convection_averages.append(plot_mem_convection)
                aluminum_mem_diffusion_averages.append(plot_mem_diffusion)
                aluminum_mem_electromigration_averages.append(plot_mem_electromigration)
            elif solute == "Cl":
                if model.fs.membrane.cations.at(1) == "Li":
                    cl_lithium_flux_averages.append(avg_flux)
                    cl_lithium_bl_convection_averages.append(plot_bl_convection)
                    cl_lithium_bl_diffusion_averages.append(plot_bl_diffusion)
                    cl_lithium_bl_electromigration_averages.append(
                        plot_bl_electromigration
                    )

                    cl_lithium_mem_convection_averages.append(plot_mem_convection)
                    cl_lithium_mem_diffusion_averages.append(plot_mem_diffusion)
                    cl_lithium_mem_electromigration_averages.append(
                        plot_mem_electromigration
                    )
                elif model.fs.membrane.cations.at(1) == "Co":
                    cl_cobalt_flux_averages.append(avg_flux)
                    cl_cobalt_bl_convection_averages.append(plot_bl_convection)
                    cl_cobalt_bl_diffusion_averages.append(plot_bl_diffusion)
                    cl_cobalt_bl_electromigration_averages.append(
                        plot_bl_electromigration
                    )

                    cl_cobalt_mem_convection_averages.append(plot_mem_convection)
                    cl_cobalt_mem_diffusion_averages.append(plot_mem_diffusion)
                    cl_cobalt_mem_electromigration_averages.append(
                        plot_mem_electromigration
                    )
                elif model.fs.membrane.cations.at(1) == "Al":
                    cl_aluminum_flux_averages.append(avg_flux)
                    cl_aluminum_bl_convection_averages.append(plot_bl_convection)
                    cl_aluminum_bl_diffusion_averages.append(plot_bl_diffusion)
                    cl_aluminum_bl_electromigration_averages.append(
                        plot_bl_electromigration
                    )

                    cl_aluminum_mem_convection_averages.append(plot_mem_convection)
                    cl_aluminum_mem_diffusion_averages.append(plot_mem_diffusion)
                    cl_aluminum_mem_electromigration_averages.append(
                        plot_mem_electromigration
                    )

    ionic_strengths = ["50", "75", "100", "150", "200", "400", "600", "800"]
    x = np.arange(len(ionic_strengths))
    for ax in fig1.axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ionic_strengths)
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.axhline(0, color="black", linewidth=1.5)
    for ax in fig2.axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ionic_strengths)
        ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
        ax.axhline(0, color="black", linewidth=1.5)

    fig1.suptitle(
        "Average Cation Flux Contributions (Excluding x,z=0)",
        fontsize=fontsize + 2,
        fontweight="bold",
    )
    fig2.suptitle(
        "Average Anion Flux Contributions (Excluding x,z=0)",
        fontsize=fontsize + 2,
        fontweight="bold",
    )
    ax1a.set_ylabel(
        "Contribution to\nLithium Flux", fontsize=fontsize, fontweight="bold"
    )
    ax2a.set_ylabel(
        "Contribution to\nCobalt Flux", fontsize=fontsize, fontweight="bold"
    )
    ax3a.set_ylabel(
        "Contribution to\nAluminum Flux", fontsize=fontsize, fontweight="bold"
    )
    for ax in [ax4a, ax5a, ax6a]:
        ax.set_ylabel(
            "Contribution to\nChloride Flux",
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
            "Inlet Feed Ionic Strength (mol/m$^3$)",
            fontsize=fontsize,
            fontweight="bold",
        )

    conv = mpatches.Patch(color=conv_color, label="Convection")
    diff = mpatches.Patch(color=diff_color, label="Diffusion")
    elec = mpatches.Patch(color=elec_color, label="Electromigration")
    for ax in [ax1a, ax4a]:
        ax.legend(handles=[conv, diff, elec], loc="upper center")

    bar_width = 0.2

    ax1a.bar(
        x - bar_width,
        abs_list(lithium_bl_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(lithium_bl_convection_averages),
    )
    ax1a.bar(
        x,
        abs_list(lithium_bl_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(lithium_bl_diffusion_averages),
    )
    ax1a.bar(
        x + bar_width,
        abs_list(lithium_bl_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(lithium_bl_electromigration_averages),
    )

    ax1b.bar(
        x - bar_width,
        abs_list(lithium_mem_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(lithium_mem_convection_averages),
    )
    ax1b.bar(
        x,
        abs_list(lithium_mem_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(lithium_mem_diffusion_averages),
    )
    ax1b.bar(
        x + bar_width,
        abs_list(lithium_mem_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(lithium_mem_electromigration_averages),
    )

    ax2a.bar(
        x - bar_width,
        abs_list(cobalt_bl_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(cobalt_bl_convection_averages),
    )
    ax2a.bar(
        x,
        abs_list(cobalt_bl_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(cobalt_bl_diffusion_averages),
    )
    ax2a.bar(
        x + bar_width,
        abs_list(cobalt_bl_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(cobalt_bl_electromigration_averages),
    )

    ax2b.bar(
        x - bar_width,
        abs_list(cobalt_mem_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(cobalt_mem_convection_averages),
    )
    ax2b.bar(
        x,
        abs_list(cobalt_mem_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(cobalt_mem_diffusion_averages),
    )
    ax2b.bar(
        x + bar_width,
        abs_list(cobalt_mem_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(cobalt_mem_electromigration_averages),
    )

    ax3a.bar(
        x - bar_width,
        abs_list(aluminum_bl_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(aluminum_bl_convection_averages),
    )
    ax3a.bar(
        x,
        abs_list(aluminum_bl_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(aluminum_bl_diffusion_averages),
    )
    ax3a.bar(
        x + bar_width,
        abs_list(aluminum_bl_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(aluminum_bl_electromigration_averages),
    )

    ax3b.bar(
        x - bar_width,
        abs_list(aluminum_mem_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(aluminum_mem_convection_averages),
    )
    ax3b.bar(
        x,
        abs_list(aluminum_mem_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(aluminum_mem_diffusion_averages),
    )
    ax3b.bar(
        x + bar_width,
        abs_list(aluminum_mem_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(aluminum_mem_electromigration_averages),
    )

    ax4a.bar(
        x - bar_width,
        abs_list(cl_lithium_bl_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(cl_lithium_bl_convection_averages),
    )
    ax4a.bar(
        x,
        abs_list(cl_lithium_bl_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(cl_lithium_bl_diffusion_averages),
    )
    ax4a.bar(
        x + bar_width,
        abs_list(cl_lithium_bl_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(cl_lithium_bl_electromigration_averages),
    )

    ax4b.bar(
        x - bar_width,
        abs_list(cl_lithium_mem_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(cl_lithium_mem_convection_averages),
    )
    ax4b.bar(
        x,
        abs_list(cl_lithium_mem_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(cl_lithium_mem_diffusion_averages),
    )
    ax4b.bar(
        x + bar_width,
        abs_list(cl_lithium_mem_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(cl_lithium_mem_electromigration_averages),
    )

    ax5a.bar(
        x - bar_width,
        abs_list(cl_cobalt_bl_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(cl_cobalt_bl_convection_averages),
    )
    ax5a.bar(
        x,
        abs_list(cl_cobalt_bl_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(cl_cobalt_bl_diffusion_averages),
    )
    ax5a.bar(
        x + bar_width,
        abs_list(cl_cobalt_bl_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(cl_cobalt_bl_electromigration_averages),
    )

    ax5b.bar(
        x - bar_width,
        abs_list(cl_cobalt_mem_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(cl_cobalt_mem_convection_averages),
    )
    ax5b.bar(
        x,
        abs_list(cl_cobalt_mem_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(cl_cobalt_mem_diffusion_averages),
    )
    ax5b.bar(
        x + bar_width,
        abs_list(cl_cobalt_mem_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(cl_cobalt_mem_electromigration_averages),
    )

    ax6a.bar(
        x - bar_width,
        abs_list(cl_aluminum_bl_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(cl_aluminum_bl_convection_averages),
    )
    ax6a.bar(
        x,
        abs_list(cl_aluminum_bl_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(cl_aluminum_bl_diffusion_averages),
    )
    ax6a.bar(
        x + bar_width,
        abs_list(cl_aluminum_bl_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(cl_aluminum_bl_electromigration_averages),
    )

    ax6b.bar(
        x - bar_width,
        abs_list(cl_aluminum_mem_convection_averages),
        bar_width,
        # label="Convection",
        color=conv_color,
        hatch=hatch_func(cl_aluminum_mem_convection_averages),
    )
    ax6b.bar(
        x,
        abs_list(cl_aluminum_mem_diffusion_averages),
        bar_width,
        # label="Diffusion",
        color=diff_color,
        hatch=hatch_func(cl_aluminum_mem_diffusion_averages),
    )
    ax6b.bar(
        x + bar_width,
        abs_list(cl_aluminum_mem_electromigration_averages),
        bar_width,
        # label="Electromigration",
        color=elec_color,
        hatch=hatch_func(cl_aluminum_mem_electromigration_averages),
    )

    if total:
        ax1a_flux = ax1a.twinx()
        ax1b_flux = ax1b.twinx()
        ax2a_flux = ax2a.twinx()
        ax2b_flux = ax2b.twinx()
        ax3a_flux = ax3a.twinx()
        ax3b_flux = ax3b.twinx()
        ax4a_flux = ax4a.twinx()
        ax4b_flux = ax4b.twinx()
        ax5a_flux = ax5a.twinx()
        ax5b_flux = ax5b.twinx()
        ax6a_flux = ax6a.twinx()
        ax6b_flux = ax6b.twinx()

        ax1a_flux.plot(
            ionic_strengths,
            lithium_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax1b_flux.plot(
            ionic_strengths,
            lithium_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax2a_flux.plot(
            ionic_strengths,
            cobalt_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax2b_flux.plot(
            ionic_strengths,
            cobalt_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax3a_flux.plot(
            ionic_strengths,
            aluminum_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax3b_flux.plot(
            ionic_strengths,
            aluminum_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax4a_flux.plot(
            ionic_strengths,
            cl_lithium_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax4b_flux.plot(
            ionic_strengths,
            cl_lithium_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax5a_flux.plot(
            ionic_strengths,
            cl_cobalt_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax5b_flux.plot(
            ionic_strengths,
            cl_cobalt_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax6a_flux.plot(
            ionic_strengths,
            cl_aluminum_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )
        ax6b_flux.plot(
            ionic_strengths,
            cl_aluminum_flux_averages,
            color=flux_color,
            marker="s",
            markersize=markersize,
        )

        for ax in [
            ax1a_flux,
            ax1b_flux,
            ax2a_flux,
            ax2b_flux,
            ax3a_flux,
            ax3b_flux,
            ax4a_flux,
            ax4b_flux,
            ax5a_flux,
            ax5b_flux,
            ax6a_flux,
            ax6b_flux,
        ]:
            ax.set_ylabel(
                "Ion Flux mol m$^{-2}$ h$^{-1}$)",
                color=flux_color,
                fontsize=fontsize,
                fontweight="bold",
            )
            ax.tick_params(axis="y", labelcolor=flux_color, labelsize=fontsize)


def abs_list(values):
    return [abs(x) for x in values]


def hatch_func(values):
    return ["XX" if val < 0 else "" for val in values]


if __name__ == "__main__":
    main()
