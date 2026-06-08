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
    solve_and_save_models(
        water_flux=0.02, run_single_salt=run_single_salt, run_two_salt=run_two_salt
    )

    single_salt_plots()
    two_salt_plots()

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


def solve_and_save_models(water_flux=0.02, run_single_salt=True, run_two_salt=True):
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

    if run_single_salt:
        feed = {
            "Li": [75, 150, 300, 450, 600, 900],
            "Co": [25, 50, 100, 150, 200, 300],
            "Al": [12.5, 25, 50, 75, 100, 150],
        }

        H_feed_guesses = np.arange(0.5, 2.1, 0.1)
        H_permeate_guesses = np.arange(0.5, 2.1, 0.1)
        H_guesses = np.column_stack((H_feed_guesses, H_permeate_guesses))

        for cation in feed.keys():
            if cation == "Li":
                chloride_multiplier = 1
            elif cation == "Co":
                chloride_multiplier = 2
            elif cation == "Al":
                chloride_multiplier = 3

            for concentration in feed[cation]:
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
                        to_json(
                            model,
                            fname=f"multi_component_case_studies/single_salt/{cation}Cl{chloride_multiplier}_{concentration}mM",
                        )
                        break
                    except:
                        continue
    if run_two_salt:
        cation_pairs = ["Li_Co", "Li_Al", "Co_Al"]
        feed_concentrations = [12.5, 25, 50, 75, 100, 150, 200]

        H_feed_guesses = np.arange(0.5, 2.1, 0.1)
        H_permeate_guesses = np.arange(0.5, 2.1, 0.1)
        H_guesses = np.column_stack((H_feed_guesses, H_permeate_guesses))

        for salt in cation_pairs:
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

            for concentration in feed_concentrations:
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
        cation = str(case_study_file)[41:43]
        chloride_multiplier = float(str(case_study_file)[45])
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
                color = "blue"
                ax_bl_flux = ax3c
                ax_mem_flux = ax3d
            elif solute == "Al":
                color = "green"
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
                    color = "blue"
                    ax_bl_flux = ax4c
                    ax_mem_flux = ax4d
                elif model.fs.membrane.cations.at(1) == "Al":
                    color = "green"
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
                color = "blue"
            elif solute == "Al":
                color = "green"
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


if __name__ == "__main__":
    main()
