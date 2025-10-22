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
    # Objective,
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    # maximize,
    units,
    value,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.util.constants import Constants
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

import matplotlib.pyplot as plt
import numpy as np

from prommis.nanofiltration.diafiltration_stream_properties import (
    DiafiltrationStreamParameter as DiafiltrationTwoSaltStreamParameter,
)
from prommis.nanofiltration.diafiltration_solute_properties import (
    SoluteParameter as SoluteTwoSaltParameter,
)
from prommis.nanofiltration.diafiltration_two_salt import TwoSaltDiafiltration

from prommis.nanofiltration.diafiltration_three_salt_stream_properties import (
    DiafiltrationStreamParameter as DiafiltrationThreeSaltStreamParameter,
)
from prommis.nanofiltration.diafiltration_three_salt_solute_properties import (
    SoluteParameter as SoluteThreeSaltParameter,
)
from prommis.nanofiltration.diafiltration_three_salt import ThreeSaltDiafiltration
from prommis.nanofiltration.flowsheets.diafiltration_flowsheet_three_salt import (
    plot_results,
    plot_membrane_results,
)


def main():
    m_two_salt = build_two_salt_model()
    m_two_salt.fs.membrane.diafiltrate_flow_volume.fix(1e-10)
    solve_model(m_two_salt)
    two_salt_model_checks(m_two_salt)

    # # initialize three salt model
    m_three_salt = build_three_salt_model()
    solve_model(m_three_salt)
    three_salt_model_checks(m_three_salt)

    m_three_salt.fs.membrane.diafiltrate_flow_volume.fix(1e-10)
    m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Al"].fix(25)
    solve_model(m_three_salt)
    three_salt_model_checks(m_three_salt)
    # plot_results(m_three_salt)
    # plot_membrane_results(m_three_salt)

    plot_relative_rejections(m_two_salt, m_three_salt)
    # plot_concentrations(m_two_salt, m_three_salt)

    # plot_relative_flux()

    # print("------------------------")
    # conc_list_2 = [
    #     [10, 20],
    #     [15, 30],
    #     [20, 40],
    #     [25, 50],
    #     [30, 60],
    #     [35, 70],
    #     [40, 80],
    #     [45, 90],
    #     [50, 100],
    #     [55, 110],
    #     [60, 120],
    #     [65, 130],
    #     [70, 140],
    #     [75, 150],
    #     [80, 160],
    #     [85, 170],
    #     [90, 180],
    #     [95, 190],
    #     [100, 200],
    # ]
    # for conc_pair in conc_list_2:
    #     pi_feed = calculate_osmotic_pressure_two_salt(conc_pair[0], conc_pair[1])
    #     print(value(pi_feed))

    # print("------------------------")
    # conc_list_3 = [
    #     [10, 20, 1],
    #     [15, 30, 1.5],
    #     [20, 40, 2],
    #     [25, 50, 2.5],
    #     [30, 60, 3],
    #     [35, 70, 3.5],
    #     [40, 80, 4],
    #     [45, 90, 4.5],
    #     [50, 100, 5],
    #     [55, 110, 5.5],
    #     [60, 120, 6],
    #     [65, 130, 6.5],
    #     [70, 140, 7],
    #     [75, 150, 7.5],
    #     [80, 160, 8],
    #     [85, 170, 8.5],
    #     [90, 180, 9],
    #     [95, 190, 9.5],
    #     [100, 200, 10],
    # ]
    # for conc_pair in conc_list_3:
    #     pi_feed = calculate_osmotic_pressure_three_salt(conc_pair[0], conc_pair[1], conc_pair[2])
    #     print(value(pi_feed))

    # print("------------------------")

    # source = ["Atacama Salar\nBrine, Chile", "Uyuni Salar\nBrine, Bolivia", "East Taijinar,\nChina", "West Taijinar,\nChina"]#, "Chott Djerid Salt\nLake, Tunisia", "Longmucuo, China", "North Arm Salt\nLake, USA"]
    # ion_conc = {
    #     'Li+': [3.02, 0.84, 0.14, 0.26],#, 0.06, 1.21, 0.04],
    #     'Mg2+': [17.6, 16.7, 5.64, 15.36],#, 3.4, 89.5, 9.38],
    #     'Na+': [61.9, 105.4, 117.03, 102.4],#, 80, 0, 100.8],
    #     'Ca2+': [0.41, 3.33, 0.43, 0.19],#, 1.6, 0, 0.35],
    #     'K+': [28.2, 15.7, 3.79, 8.44],#, 5.6, 0, 5.5],
    #     'B': [1.72, 0.7, 0, 0],#, 0, 0, 0.3],
    #     'SO42+': [37.9, 21.3, 0, 0],#, 6.7, 0, 19.7],
    # }

    # x = np.arange(len(source))  # the label locations
    # width = 0.1  # the width of the bars
    # multiplier = 0

    # fig, ax = plt.subplots(1, 1, dpi=125, figsize=(6, 4))

    # for attribute, measurement in ion_conc.items():
    #     offset = width * multiplier
    #     ax.bar(x + offset, measurement, width, label=attribute)
    #     multiplier += 1

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Concentration (g/L)')
    # ax.set_xticks(x + width, source)
    # # plt.xticks(rotation=60)
    # ax.legend(loc='upper left')
    # # ax.set_ylim(0, 250)

    # plt.show()


def plot_relative_rejections(m2, m3):
    """
    Plots relative solute rejection across the length of the membrane module.
    Rejections normalized to initial rejection (x=0).
    Compares two and three salt models.

    Args:
        m: Pyomo model
    """
    # store values for x-coordinate
    x_axis_values = []

    # store values for lithium rejection
    lithium_rejection_two_salt = []
    lithium_rejection_two_salt_norm = []
    lithium_rejection_three_salt = []
    lithium_rejection_three_salt_norm = []
    # store values for cobalt rejection
    cobalt_rejection_two_salt = []
    cobalt_rejection_two_salt_norm = []
    cobalt_rejection_three_salt = []
    cobalt_rejection_three_salt_norm = []
    # aluminum values for cobalt rejection
    aluminum_rejection_three_salt = []
    aluminum_rejection_three_salt_norm = []

    for x_val in m2.fs.membrane.dimensionless_module_length:
        if x_val != 0:
            x_axis_values.append(
                x_val
                * value(m2.fs.membrane.total_module_length)
                * value(m2.fs.membrane.total_membrane_length)
            )

            lith_rej_two_salt = (
                1
                - (
                    value(m2.fs.membrane.permeate_conc_mol_comp[0, x_val, "Li"])
                    / value(m2.fs.membrane.retentate_conc_mol_comp[0, x_val, "Li"])
                )
            ) * 100

            cob_rej_two_salt = (
                1
                - (
                    value(m2.fs.membrane.permeate_conc_mol_comp[0, x_val, "Co"])
                    / value(m2.fs.membrane.retentate_conc_mol_comp[0, x_val, "Co"])
                )
            ) * 100

            lith_rej_three_salt = (
                1
                - (
                    value(m3.fs.membrane.permeate_conc_mol_comp[0, x_val, "Li"])
                    / value(m3.fs.membrane.retentate_conc_mol_comp[0, x_val, "Li"])
                )
            ) * 100

            cob_rej_three_salt = (
                1
                - (
                    value(m3.fs.membrane.permeate_conc_mol_comp[0, x_val, "Co"])
                    / value(m3.fs.membrane.retentate_conc_mol_comp[0, x_val, "Co"])
                )
            ) * 100
            al_rej_three_salt = (
                1
                - (
                    value(m3.fs.membrane.permeate_conc_mol_comp[0, x_val, "Al"])
                    / value(m3.fs.membrane.retentate_conc_mol_comp[0, x_val, "Al"])
                )
            ) * 100

            lithium_rejection_two_salt.append(lith_rej_two_salt)
            lithium_rejection_three_salt.append(lith_rej_three_salt)
            cobalt_rejection_two_salt.append(cob_rej_two_salt)
            cobalt_rejection_three_salt.append(cob_rej_three_salt)
            aluminum_rejection_three_salt.append(al_rej_three_salt)

    lithium_rejection_two_salt_norm = [
        (i - lithium_rejection_two_salt[0]) / abs(lithium_rejection_two_salt[0]) * 100
        for i in lithium_rejection_two_salt
    ]
    lithium_rejection_three_salt_norm = [
        (i - lithium_rejection_three_salt[0])
        / abs(lithium_rejection_three_salt[0])
        * 100
        for i in lithium_rejection_three_salt
    ]
    cobalt_rejection_two_salt_norm = [
        (i - cobalt_rejection_two_salt[0]) / abs(cobalt_rejection_two_salt[0]) * 100
        for i in cobalt_rejection_two_salt
    ]
    cobalt_rejection_three_salt_norm = [
        (i - cobalt_rejection_three_salt[0]) / abs(cobalt_rejection_three_salt[0]) * 100
        for i in cobalt_rejection_three_salt
    ]
    aluminum_rejection_three_salt_norm = [
        (i - aluminum_rejection_three_salt[0])
        / abs(aluminum_rejection_three_salt[0])
        * 100
        for i in aluminum_rejection_three_salt
    ]

    # fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(9, 5))

    fig1, ax2 = plt.subplots(1, 1, dpi=125, figsize=(5, 4))

    # ax1.plot(
    #     x_axis_values, lithium_rejection_two_salt, "m-", linewidth=2
    # )  # , label="Lithium (Li-Co)")
    # ax1.plot(
    #     x_axis_values, cobalt_rejection_two_salt, "c-", linewidth=2
    # )  # , label="Cobalt (Li-Co)")
    # ax1.plot(
    #     x_axis_values, lithium_rejection_three_salt, "m--", linewidth=2
    # )  # , label="Lithium (Li-Co-Al)")
    # ax1.plot(
    #     x_axis_values, cobalt_rejection_three_salt, "c--", linewidth=2
    # )  # , label="Cobalt (Li-Co-Al)")
    # ax1.plot(
    #     x_axis_values, aluminum_rejection_three_salt, "g--", linewidth=2
    # )  # , label="Aluminum (Li-Co-Al)")
    # ax1.set_xlabel("Membrane Area (m$^2$)", fontsize=12, fontweight="bold")
    # ax1.set_ylabel("Solute Rejection (%)", fontsize=12, fontweight="bold")
    # ax1.tick_params(direction="in", labelsize=10)
    # # ax1.legend()

    ax2.plot(
        x_axis_values, lithium_rejection_two_salt_norm, "m-", linewidth=2
    )  # , label="Lithium (Li-Co)")
    ax2.plot(
        x_axis_values, cobalt_rejection_two_salt_norm, "c-", linewidth=2
    )  # , label="Cobalt (Li-Co)")
    ax2.plot(
        x_axis_values, lithium_rejection_three_salt_norm, "m--", linewidth=2
    )  # , label="Lithium (Li-Co-Al)")
    ax2.plot(
        x_axis_values, cobalt_rejection_three_salt_norm, "c--", linewidth=2
    )  # , label="Cobalt (Li-Co-Al)")
    ax2.plot(
        x_axis_values, aluminum_rejection_three_salt_norm, "g--", linewidth=2
    )  # , label="Aluminum (Li-Co-Al)")
    ax2.set_xlabel("Membrane Area (m$^2$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel(
        "Percent Change in Solute Rejection (%)", fontsize=10, fontweight="bold"
    )
    ax2.tick_params(direction="in", top=True, right=True, labelsize=10)

    ax2.plot([0, 164], [0, 0], "k-", linewidth=0.5)

    ax2.set_xlim(0, 164)
    ax2.set_ylim(-12, 2)

    # legend points
    # ax2.plot([],[], marker='None', linestyle='None', label="Solution (linestyle)")
    ax2.plot([], [], "k-", linewidth=2, label="Li-Co")
    ax2.plot([], [], "k--", linewidth=2, label="Li-Co-Al")
    ax2.plot([], [], marker="None", linestyle="None", label="Solute (color)")
    ax2.plot([], [], "ms", markersize=8, label="Lithium")
    ax2.plot([], [], "cs", markersize=8, label="Cobalt")
    ax2.plot([], [], "gs", markersize=8, label="Aluminum")

    ax2.legend(loc="best", title="Solution (linestyle)", bbox_to_anchor=(0.43, 0.54))

    plt.tight_layout()

    plt.show()


def calculate_ionic_strength_two_salt(m):
    return 0.5 * (
        (
            value(m.fs.membrane.retentate_conc_mol_comp[0, 0, "Li"])
            * value(m.fs.membrane.config.property_package.charge["Li"]) ** 2
        )
        + (
            value(m.fs.membrane.retentate_conc_mol_comp[0, 0, "Co"])
            * value(m.fs.membrane.config.property_package.charge["Co"]) ** 2
        )
        + (
            value(m.fs.membrane.retentate_conc_mol_comp[0, 0, "Cl"])
            * value(m.fs.membrane.config.property_package.charge["Cl"]) ** 2
        )
    )


def calculate_osmotic_pressure_two_salt(c1, c2):
    z1 = 1
    z2 = 2
    z3 = -1
    c3 = -(z1 / z3) * c1 - (z2 / z3) * c2

    pi_feed = units.convert(
        (
            Constants.gas_constant  # J / mol / K
            * 298
            * units.K
            * (
                (1 * 1 * c1 * units.mol / units.m**3)  # n * sigma * c
                + (1 * 1 * c2 * units.mol / units.m**3)  # n * sigma * c
                + (3 * 1 * c3 * units.mol / units.m**3)  # n * sigma * c
            )
        ),
        to_units=units.bar,
    )

    return pi_feed


def calculate_osmotic_pressure_three_salt(c1, c2, c3):
    z1 = 1
    z2 = 2
    z3 = 3
    z4 = -1
    c4 = -(z1 / z4) * c1 - (z2 / z4) * c2 - (z3 / z4) * c3
    return units.convert(
        (
            Constants.gas_constant  # J / mol / K
            * 298
            * units.K
            * (
                (1 * 1 * c1 * units.mol / units.m**3)  # n * sigma * c
                + (1 * 1 * c2 * units.mol / units.m**3)  # n * sigma * c
                + (1 * 1 * c3 * units.mol / units.m**3)  # n * sigma * c
                + (6 * 1 * c4 * units.mol / units.m**3)  # n * sigma * c
            )
        ),
        to_units=units.bar,
    )


def calculate_ionic_strength_three_salt(m):
    return 0.5 * (
        (
            value(m.fs.membrane.retentate_conc_mol_comp[0, 0, "Li"])
            * value(m.fs.membrane.config.property_package.charge["Li"]) ** 2
        )
        + (
            value(m.fs.membrane.retentate_conc_mol_comp[0, 0, "Co"])
            * value(m.fs.membrane.config.property_package.charge["Co"]) ** 2
        )
        + (
            value(m.fs.membrane.retentate_conc_mol_comp[0, 0, "Al"])
            * value(m.fs.membrane.config.property_package.charge["Al"]) ** 2
        )
        + (
            value(m.fs.membrane.retentate_conc_mol_comp[0, 0, "Cl"])
            * value(m.fs.membrane.config.property_package.charge["Cl"]) ** 2
        )
    )


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
    solve_model(m_two_salt)
    two_salt_model_checks(m_two_salt)

    for conc in conc_list_2:
        m_two_salt.fs.membrane.feed_conc_mol_comp[0, "Li"].fix(conc[0])
        m_two_salt.fs.membrane.feed_conc_mol_comp[0, "Co"].fix(conc[1])

        solve_model(m_two_salt)
        two_salt_model_checks(m_two_salt)

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
    solve_model(m_three_salt)
    three_salt_model_checks(m_three_salt)
    # m_three_salt.fs.membrane.diafiltrate_flow_volume.fix(1e-10)

    for conc in conc_list_3:
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Li"].fix(conc[0])
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Co"].fix(conc[1])
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Al"].fix(conc[2])

        solve_model(m_three_salt)
        three_salt_model_checks(m_three_salt)

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
    solve_model(m_three_salt)
    three_salt_model_checks(m_three_salt)
    # m_three_salt.fs.membrane.diafiltrate_flow_volume.fix(1e-10)

    for conc in conc_list_3:
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Li"].fix(conc[0])
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Co"].fix(conc[1])
        m_three_salt.fs.membrane.feed_conc_mol_comp[0, "Al"].fix(conc[2])

        solve_model(m_three_salt)
        three_salt_model_checks(m_three_salt)

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


def build_two_salt_model():
    # build flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.stream_properties = DiafiltrationTwoSaltStreamParameter()
    m.fs.properties = SoluteTwoSaltParameter()

    # add feed blocks for feed and diafiltrate
    m.fs.feed_block = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block = Feed(property_package=m.fs.stream_properties)

    surrogate_model_file_dict = {
        "D_11": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d11_scaled",
        "D_12": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d12_scaled",
        "D_21": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d21_scaled",
        "D_22": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d22_scaled",
        "alpha_1": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_alpha_1",
        "alpha_2": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_alpha_2",
    }

    # add the membrane unit model
    m.fs.membrane = TwoSaltDiafiltration(
        property_package=m.fs.properties,
        NFE_module_length=20,
        NFE_membrane_thickness=10,
        charged_membrane=True,
        surrogate_model_files=surrogate_model_file_dict,
        diffusion_surrogate_scaling_factor=1e-07,
    )

    # add product blocks for retentate and permeate
    m.fs.retentate_block = Product(property_package=m.fs.stream_properties)
    m.fs.permeate_block = Product(property_package=m.fs.stream_properties)

    # fix the degrees of freedom to their default values
    # fix degrees of freedom in the membrane
    m.fs.membrane.total_module_length.fix()
    m.fs.membrane.total_membrane_length.fix()
    m.fs.membrane.applied_pressure.fix()

    # fix degrees of freedom in the flowsheet
    m.fs.membrane.feed_flow_volume.fix()
    m.fs.membrane.feed_conc_mol_comp[0, "Li"].fix()
    m.fs.membrane.feed_conc_mol_comp[0, "Co"].fix()
    m.fs.membrane.feed_conc_mol_comp[0, "Cl"].fix()

    m.fs.membrane.diafiltrate_flow_volume.fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "Li"].fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "Co"].fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "Cl"].fix()

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


def two_salt_model_checks(m):
    for x in m.fs.membrane.dimensionless_module_length:
        for z in m.fs.membrane.dimensionless_membrane_thickness:
            # skip check at x=0 as the concentration is expected to be 0 and the
            # diffusion coefficient calculation is not needed
            if x == 0:
                pass
            elif not (40 < value(m.fs.membrane.membrane_conc_mol_lithium[x, z]) < 190):
                raise ValueError(
                    "WARNING: Membrane concentration for lithium ("
                    f"{value(m.fs.membrane.membrane_conc_mol_lithium[x, z])} mM at "
                    f"x={x * value(m.fs.membrane.total_module_length)} m and "
                    f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                    "of the valid range for the diffusion coefficient surrogate model "
                    "(40-190 mM). Consider re-training the surrogate model."
                )
            elif not (40 < value(m.fs.membrane.membrane_conc_mol_cobalt[x, z]) < 190):
                raise ValueError(
                    "WARNING: Membrane concentration for cobalt ("
                    f"{value(m.fs.membrane.membrane_conc_mol_cobalt[x, z])} mM at "
                    f"x={x * value(m.fs.membrane.total_module_length)} m and "
                    f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                    "of the valid range for the diffusion coefficient surrogate model "
                    "(40-190 mM). Consider re-training the surrogate model."
                )


def build_three_salt_model():
    # build flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.stream_properties = DiafiltrationThreeSaltStreamParameter()
    m.fs.properties = SoluteThreeSaltParameter()

    # add feed blocks for feed and diafiltrate
    m.fs.feed_block = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block = Feed(property_package=m.fs.stream_properties)

    surrogate_model_file_dict = {
        "D_11": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d11_scaled",
        "D_12": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d12_scaled",
        "D_13": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d13_scaled",
        "D_21": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d21_scaled",
        "D_22": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d22_scaled",
        "D_23": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d23_scaled",
        "D_31": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d31_scaled",
        "D_32": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d32_scaled",
        "D_33": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d33_scaled",
        "alpha_1": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_alpha1",
        "alpha_2": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_alpha2",
        "alpha_3": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_alpha3",
    }

    # add the membrane unit model
    m.fs.membrane = ThreeSaltDiafiltration(
        property_package=m.fs.properties,
        NFE_module_length=20,
        NFE_membrane_thickness=10,
        charged_membrane=True,
        surrogate_model_files=surrogate_model_file_dict,
        diffusion_surrogate_scaling_factor=1e-07,
    )

    # add product blocks for retentate and permeate
    m.fs.retentate_block = Product(property_package=m.fs.stream_properties)
    m.fs.permeate_block = Product(property_package=m.fs.stream_properties)

    # fix the degrees of freedom to their default values
    # fix degrees of freedom in the membrane
    m.fs.membrane.total_module_length.fix()
    m.fs.membrane.total_membrane_length.fix()
    m.fs.membrane.applied_pressure.fix()

    # fix degrees of freedom in the flowsheet
    m.fs.membrane.feed_flow_volume.fix()
    m.fs.membrane.feed_conc_mol_comp[0, "Li"].fix()
    m.fs.membrane.feed_conc_mol_comp[0, "Co"].fix()
    m.fs.membrane.feed_conc_mol_comp[0, "Al"].fix()
    m.fs.membrane.feed_conc_mol_comp[0, "Cl"].fix()

    m.fs.membrane.diafiltrate_flow_volume.fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "Li"].fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "Co"].fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "Al"].fix()
    m.fs.membrane.diafiltrate_conc_mol_comp[0, "Cl"].fix()

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


def three_salt_model_checks(m):
    for x in m.fs.membrane.dimensionless_module_length:
        for z in m.fs.membrane.dimensionless_membrane_thickness:
            # skip check at x=0 as the concentration is expected to be 0 and the
            # diffusion coefficient calculation is not needed
            if x == 0:
                pass
            elif not (40 < value(m.fs.membrane.membrane_conc_mol_lithium[x, z]) < 190):
                raise ValueError(
                    "WARNING: Membrane concentration for lithium ("
                    f"{value(m.fs.membrane.membrane_conc_mol_lithium[x, z])} mM at "
                    f"x={x * value(m.fs.membrane.total_module_length)} m and "
                    f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                    "of the valid range for the diffusion coefficient surrogate model "
                    "(40-190 mM). Consider re-training the surrogate model."
                )
            elif not (40 < value(m.fs.membrane.membrane_conc_mol_cobalt[x, z]) < 190):
                raise ValueError(
                    "WARNING: Membrane concentration for cobalt ("
                    f"{value(m.fs.membrane.membrane_conc_mol_cobalt[x, z])} mM at "
                    f"x={x * value(m.fs.membrane.total_module_length)} m and "
                    f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                    "of the valid range for the diffusion coefficient surrogate model "
                    "(50-190 mM). Consider re-training the surrogate model."
                )
            elif not (1 < value(m.fs.membrane.membrane_conc_mol_aluminum[x, z]) < 151):
                raise ValueError(
                    "WARNING: Membrane concentration for aluminum ("
                    f"{value(m.fs.membrane.membrane_conc_mol_aluminum[x, z])} mM at "
                    f"x={x * value(m.fs.membrane.total_module_length)} m and "
                    f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                    "of the valid range for the diffusion coefficient surrogate model "
                    "(1-151 mM). Consider re-training the surrogate model."
                )


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

    dt = DiagnosticsToolbox(m)
    # check numerical warnings
    dt.assert_no_numerical_warnings()
    # dt.report_numerical_issues()
    # dt.display_variables_at_or_outside_bounds()


if __name__ == "__main__":
    main()
