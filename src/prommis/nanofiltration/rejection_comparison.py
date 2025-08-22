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
    Objective,
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    maximize,
    value,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

import matplotlib.pyplot as plt

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


def main():
    m_two_salt = build_two_salt_model()
    solve_model(m_two_salt)
    two_salt_model_checks(m_two_salt)
    # m_two_salt.fs.membrane.total_module_length.unfix()
    # m_two_salt.fs.membrane.total_membrane_length.unfix()
    # m_two_salt.fs.membrane.applied_pressure.unfix()
    # m_two_salt.fs.membrane.diafiltrate_flow_volume.unfix()

    # m_two_salt.obj = Objective(
    #     expr=(
    #         ((
    #             m_two_salt.fs.membrane.permeate_conc_mol_comp[0, 1, "Li"]
    #             / m_two_salt.fs.membrane.retentate_conc_mol_comp[0, 0, "Li"]
    #         )
    #         + (
    #             m_two_salt.fs.membrane.permeate_conc_mol_comp[0, 1, "Co"]
    #             / m_two_salt.fs.membrane.retentate_conc_mol_comp[0, 0, "Co"]
    #         ))/(
    #             m_two_salt.fs.membrane.permeate_conc_mol_comp[0, 1, "Li"]
    #             / m_two_salt.fs.membrane.retentate_conc_mol_comp[0, 0, "Li"]
    #         )
    #     ),
    #     sense=maximize,
    # )
    # solve_model(m_two_salt, simulation=False)
    # two_salt_model_checks(m_two_salt)

    m_three_salt = build_three_salt_model()
    solve_model(m_three_salt)
    three_salt_model_checks(m_three_salt)

    plot_relative_rejections(m_two_salt, m_three_salt)


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
            x_axis_values.append(x_val * value(m2.fs.membrane.total_module_length) * value(m2.fs.membrane.total_membrane_length))

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
            
    lithium_rejection_two_salt_norm = [(i-lithium_rejection_two_salt[0]) / lithium_rejection_two_salt[0] * 100 for i in lithium_rejection_two_salt]
    lithium_rejection_three_salt_norm = [(i-lithium_rejection_three_salt[0]) / lithium_rejection_three_salt[0] * 100 for i in lithium_rejection_three_salt]
    cobalt_rejection_two_salt_norm = [(i-cobalt_rejection_two_salt[0]) / cobalt_rejection_two_salt[0] * 100 for i in cobalt_rejection_two_salt]
    cobalt_rejection_three_salt_norm = [(i-cobalt_rejection_three_salt[0]) / cobalt_rejection_three_salt[0] * 100 for i in cobalt_rejection_three_salt]
    aluminum_rejection_three_salt_norm = [(i-aluminum_rejection_three_salt[0]) / aluminum_rejection_three_salt[0] * 100 for i in aluminum_rejection_three_salt]

    # fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(9, 5))

    fig1, ax2 = plt.subplots(
        1, 1, dpi=100, figsize=(5, 4)
    )

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
    # ax1.set_xlabel("Module Length (m)", fontsize=10, fontweight="bold")
    # ax1.set_ylabel("Solute Rejection (%)", fontsize=10, fontweight="bold")
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
    ax2.set_ylabel("Percent Change in Solute Rejection (%)", fontsize=10, fontweight="bold")
    ax2.tick_params(direction="in", labelsize=10)

    ax2.plot([0,160],[0,0],"k-", linewidth=0.5)

    ax2.set_xlim(0,160)

    # legend points
    # ax2.plot([],[], marker='None', linestyle='None', label="Solution (linestyle)")
    ax2.plot([], [], "k-", linewidth=2, label="Li-Co")
    ax2.plot([], [], "k--", linewidth=2, label="Li-Co-Al")
    ax2.plot([],[], marker='None', linestyle='None', label="Solute (color)")
    ax2.plot([], [], "ms", markersize=8, label="Lithium")
    ax2.plot([], [], "cs", markersize=8, label="Cobalt")
    ax2.plot([], [], "gs", markersize=8, label="Aluminum")
    ax2.legend(loc="best", title="Solution (linestyle)")#, bbox_to_anchor=(1, 0.39))

    plt.tight_layout()

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

    # add the membrane unit model
    m.fs.membrane = TwoSaltDiafiltration(
        property_package=m.fs.properties,
        NFE_module_length=20,
        NFE_membrane_thickness=5,
        charged_membrane=True,
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
            elif not (50 < value(m.fs.membrane.membrane_conc_mol_lithium[x, z]) < 80):
                raise ValueError(
                    "Membrane concentration for lithium ("
                    f"{value(m.fs.membrane.membrane_conc_mol_lithium[x, z])} mM at "
                    f"x={x * value(m.fs.membrane.total_module_length)} m and "
                    f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                    "of the valid range for the diffusion coefficient approximations "
                    "(50-80 mM). The linearized approximation should be re-calculated."
                )
    if m.fs.membrane.config.charged_membrane:
        for x in m.fs.membrane.dimensionless_module_length:
            for z in m.fs.membrane.dimensionless_membrane_thickness:
                # skip check at x=0 as the concentration is expected to be 0 and the
                # diffusion coefficient calculation is not needed
                if x == 0:
                    pass
                elif not (
                    80 < value(m.fs.membrane.membrane_conc_mol_cobalt[x, z]) < 110
                ):
                    raise ValueError(
                        "Membrane concentration for cobalt ("
                        f"{value(m.fs.membrane.membrane_conc_mol_cobalt[x, z])} mM at "
                        f"x={x * value(m.fs.membrane.total_module_length)} m and "
                        f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                        "of the valid range for the diffusion coefficient approximations "
                        "(80-110 mM). The linearized approximation should be re-calculated."
                    )
    else:
        for x in m.fs.membrane.dimensionless_module_length:
            for z in m.fs.membrane.dimensionless_membrane_thickness:
                # skip check at x=0 as the concentration is expected to be 0 and the
                # diffusion coefficient calculation is not needed
                if x == 0:
                    pass
                elif not (
                    50 < value(m.fs.membrane.membrane_conc_mol_cobalt[x, z]) < 80
                ):
                    raise ValueError(
                        "Membrane concentration for cobalt ("
                        f"{value(m.fs.membrane.membrane_conc_mol_cobalt[x, z])} mM at "
                        f"x={x * value(m.fs.membrane.total_module_length)} m and "
                        f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                        "of the valid range for the diffusion coefficient approximations "
                        "(50-80 mM). The linearized approximation should be re-calculated."
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

    # add the membrane unit model
    m.fs.membrane = ThreeSaltDiafiltration(
        property_package=m.fs.properties,
        NFE_module_length=20,
        NFE_membrane_thickness=5,
        charged_membrane=True,
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
            elif not (50 <= value(m.fs.membrane.membrane_conc_mol_lithium[x, z]) <= 80):
                raise ValueError(
                    "Membrane concentration for lithium ("
                    f"{value(m.fs.membrane.membrane_conc_mol_lithium[x, z])} mM at "
                    f"x={x * value(m.fs.membrane.total_module_length)} m and "
                    f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                    "of the valid range for the diffusion coefficient approximations "
                    "(50-80 mM). The linearized approximation should be re-calculated."
                )
            elif not (2 <= value(m.fs.membrane.membrane_conc_mol_aluminum[x, z]) <= 32):
                raise ValueError(
                    "Membrane concentration for aluminum ("
                    f"{value(m.fs.membrane.membrane_conc_mol_aluminum[x, z])} mM at "
                    f"x={x * value(m.fs.membrane.total_module_length)} m and "
                    f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                    "of the valid range for the diffusion coefficient approximations "
                    "(2-32 mM). The linearized approximation should be re-calculated."
                )
    if m.fs.membrane.config.charged_membrane:
        for x in m.fs.membrane.dimensionless_module_length:
            for z in m.fs.membrane.dimensionless_membrane_thickness:
                # skip check at x=0 as the concentration is expected to be 0 and the
                # diffusion coefficient calculation is not needed
                if x == 0:
                    pass
                elif not (
                    80 <= value(m.fs.membrane.membrane_conc_mol_cobalt[x, z]) <= 110
                ):
                    raise ValueError(
                        "Membrane concentration for cobalt ("
                        f"{value(m.fs.membrane.membrane_conc_mol_cobalt[x, z])} mM at "
                        f"x={x * value(m.fs.membrane.total_module_length)} m and "
                        f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                        "of the valid range for the diffusion coefficient approximations "
                        "(80-110 mM). The linearized approximation should be re-calculated."
                    )
    else:
        for x in m.fs.membrane.dimensionless_module_length:
            for z in m.fs.membrane.dimensionless_membrane_thickness:
                # skip check at x=0 as the concentration is expected to be 0 and the
                # diffusion coefficient calculation is not needed
                if x == 0:
                    pass
                elif not (
                    50 <= value(m.fs.membrane.membrane_conc_mol_cobalt[x, z]) <= 80
                ):
                    raise ValueError(
                        "Membrane concentration for cobalt ("
                        f"{value(m.fs.membrane.membrane_conc_mol_cobalt[x, z])} mM at "
                        f"x={x * value(m.fs.membrane.total_module_length)} m and "
                        f"z={z * value(m.fs.membrane.total_membrane_thickness)} m) is outside "
                        "of the valid range for the diffusion coefficient approximations "
                        "(50-80 mM). The linearized approximation should be re-calculated."
                    )


def solve_model(m, simulation=True):
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
    if simulation:
        dt.assert_no_numerical_warnings()
    else:
        dt.report_numerical_issues()
        dt.display_variables_at_or_outside_bounds()


if __name__ == "__main__":
    main()
