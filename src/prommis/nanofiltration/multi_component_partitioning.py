import idaes

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Param,
    Set,
    SolverFactory,
    Var,
    assert_optimal_termination,
    value,
)


def main():
    lithium_H = 1.5
    cobalt_H = 0.5
    chlorine_H = 1

    lithium_dict = {"num": 1, "z": 1, "H": lithium_H, "conc_sol": 20}
    cobalt_dict = {"num": 2, "z": 2, "H": cobalt_H, "conc_sol": 20}
    chlorine_dict = {"num": 3, "z": -1, "H": chlorine_H, "conc_sol": 20}

    ion_dict_LiCl = {
        "lithium": lithium_dict,
        "chlorine": chlorine_dict,
    }
    m_general_lithium = single_salt_partitioning_model(ion_dict_LiCl)
    solve_model(m_general_lithium)

    ion_dict_CoCl2 = {
        "cobalt": cobalt_dict,
        "chlorine": chlorine_dict,
    }
    m_general_cobalt = single_salt_partitioning_model(ion_dict_CoCl2)
    solve_model(m_general_cobalt)

    ion_dict_LiCoCl = {
        "lithium": lithium_dict,
        "cobalt": cobalt_dict,
        "chlorine": chlorine_dict,
    }
    m_double_salt_independent = double_salt_independent_partitioning_model(
        ion_dict_LiCoCl
    )
    m_double_salt_independent.chi = -150
    solve_model(m_double_salt_independent)

    # m_double_salt = double_salt_partitioning_model(ion_dict_LiCoCl)
    # # m_double_salt.chi = -20
    # solve_model(m_double_salt)

    print("Lithium Chloride")
    print_info(m_general_lithium)
    print("\n")

    print("Cobalt Chloride")
    print_info(m_general_cobalt)
    print("\n")

    print("Double Salt, No Interaction")
    print_info(m_double_salt_independent)
    print("\n")

    # print("Double Salt")
    # print_info(m_double_salt)
    # print("\n")

    # plot_partitioning_behavior(single=True, double=False)
    # plot_partitioning_behavior(single=False, double=True, independent=True)
    # plot_partitioning_behavior(single=False, double=True, independent=False)

    plot_chi_sensitivity(independent=True)
    # plot_chi_sensitivity(independent=False)

    # visualize_3d_trends(independent=True)
    # visualize_3d_trends(independent=False)


def add_model_sets(m, ion_info):
    """
    Args:
        ion_info: {
            ion: {
                "num": int value
                "z": int val,
                "H": float val,
                "conc_sol": float val,
            },
            ...
        }
    """
    m.solutes = Set(initialize=[dict["num"] for dict in ion_info.values()])


def add_model_params(m, ion_info):
    """
    Args:
        ion_info: {
            ion: {
                "num": int value
                "z": int val,
                "H": float val,
                "conc_sol": float val,
            },
            ...
        }
    """

    m.chi = Param(
        initialize=0,
        mutable=True,
    )

    def initialize_z_params(m, j):
        vals = {dict["num"]: dict["z"] for dict in ion_info.values()}
        return vals[j]

    m.z = Param(
        m.solutes,
        initialize=initialize_z_params,
    )

    def initialize_H_params(m, j):
        vals = {dict["num"]: dict["H"] for dict in ion_info.values()}
        return vals[j]

    m.H = Param(
        m.solutes,
        initialize=initialize_H_params,
    )


def add_model_vars(m, ion_info):
    """
    Args:
        ion_info: {
            ion: {
                "num": int value
                "z": int val,
                "H": float val,
                "conc_sol": float val,
            },
            ...
        }
    """

    def initialize_conc_sol(m, j):
        vals = {dict["num"]: dict["conc_sol"] for dict in ion_info.values()}
        return vals[j]

    m.conc_sol = Var(
        m.solutes,
        initialize=initialize_conc_sol,
        bounds=[0, None],
    )
    # fix solution concentrations for cations assuming final ion is anion
    # concentration of anion gets verified later with electoneutrality constraint
    for i in m.solutes:
        if i != m.solutes.at(-1):
            m.conc_sol[i].fix()

    def initialize_conc_mem(m, j):
        vals = {dict["num"]: 10 for dict in ion_info.values()}
        return vals[j]

    m.conc_mem = Var(
        m.solutes,
        initialize=initialize_conc_mem,
        bounds=[0, None],
    )


# TODO: add generalized function for adding constraints; combiine into build_model


def single_salt_partitioning_model(ion_info):
    """
    Generalized model equations for the partitioning of ions in a single
    salt aqueous solution.

    Assumes 1 is cation, 2 is anion

    Args:
        ion_info: {
            ion: {
                "num": int value
                "z": int val,
                "H": float val,
                "conc_sol": float val,
            },
            ...
        }

    Returns:
        m: the Pyomo model
    """
    m = ConcreteModel()

    add_model_sets(m, ion_info)
    add_model_params(m, ion_info)
    add_model_vars(m, ion_info)

    # electroneutrality in the solution phase: z1c1+z2c2=0 --> c2=-(z1/z2)c1
    def _electoneutrality_solution(m):
        return (
            m.conc_sol[m.solutes.at(2)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(2)])
            * m.conc_sol[m.solutes.at(1)]
        )

    m.electroneutrality_solution = Constraint(rule=_electoneutrality_solution)

    # electroneutrality in the membrane phase: z1c1+z2c2+chi=0 --> c2=-(z1/z2)c1-(1/z2)chi
    def _electoneutrality_membrane(m):
        return (
            m.conc_mem[m.solutes.at(2)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(2)])
            * m.conc_mem[m.solutes.at(1)]
            - m.chi / m.z[m.solutes.at(2)]
        )

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # generalized single salt partitioning
    def _single_salt_partitioning(m):
        return m.H[m.solutes.at(1)] ** (1 / m.z[m.solutes.at(1)]) * m.H[
            m.solutes.at(2)
        ] ** (-1 / m.z[m.solutes.at(2)]) == (
            m.conc_mem[m.solutes.at(2)] / m.conc_sol[m.solutes.at(2)]
        ) ** (
            -1 / m.z[m.solutes.at(2)]
        ) * (
            m.conc_mem[m.solutes.at(1)] / m.conc_sol[m.solutes.at(1)]
        ) ** (
            1 / m.z[m.solutes.at(1)]
        )

    m.single_salt_partitioning = Constraint(rule=_single_salt_partitioning)

    return m


def double_salt_independent_partitioning_model(ion_info):
    """
    Generalized model equations for the partitioning of ions in a two salt
    (common anion) aqueous solution, assuming no cation interactions.

    Assumes 1 and 2 are cations, 3 is anion

    Args:
        ion_info: {
            ion: {
                "num": int value
                "z": int val,
                "H": float val,
                "conc_sol": float val,
            },
            ...
        }

    Returns:
        m: the Pyomo model
    """
    m = ConcreteModel()

    add_model_sets(m, ion_info)
    add_model_params(m, ion_info)
    add_model_vars(m, ion_info)

    # electroneutrality in the solution phase: z1c1+z2c2+z3c3=0 --> c3=-(z1/z3)c1-(z2/z3)c2
    def _electoneutrality_solution(m):
        return (
            m.conc_sol[m.solutes.at(3)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(3)])
            * m.conc_sol[m.solutes.at(1)]
            - (m.z[m.solutes.at(2)] / m.z[m.solutes.at(3)])
            * m.conc_sol[m.solutes.at(2)]
        )

    m.electroneutrality_solution = Constraint(rule=_electoneutrality_solution)

    # electroneutrality in the membrane phase: z1c1+z2c2+z3c3+chi=0 --> c3=-(z1/z3)c1-(c2/c3)c2-(1/z3)chi
    def _electoneutrality_membrane(m):
        return (
            m.conc_mem[m.solutes.at(3)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(3)])
            * m.conc_mem[m.solutes.at(1)]
            - (m.z[m.solutes.at(2)] / m.z[m.solutes.at(3)])
            * m.conc_mem[m.solutes.at(2)]
            - m.chi / m.z[m.solutes.at(3)]
        )

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # generalized single salt partitioning
    def _salt_1_partitioning(m):
        return m.H[m.solutes.at(1)] ** (1 / m.z[m.solutes.at(1)]) * m.H[
            m.solutes.at(3)
        ] ** (-1 / m.z[m.solutes.at(3)]) == (
            m.conc_mem[m.solutes.at(3)] / m.conc_sol[m.solutes.at(3)]
        ) ** (
            -1 / m.z[m.solutes.at(3)]
        ) * (
            m.conc_mem[m.solutes.at(1)] / m.conc_sol[m.solutes.at(1)]
        ) ** (
            1 / m.z[m.solutes.at(1)]
        )

    m.salt_1_partitioning = Constraint(rule=_salt_1_partitioning)

    def _salt_2_partitioning(m):
        return m.H[m.solutes.at(2)] ** (1 / m.z[m.solutes.at(2)]) * m.H[
            m.solutes.at(3)
        ] ** (-1 / m.z[m.solutes.at(3)]) == (
            m.conc_mem[m.solutes.at(3)] / m.conc_sol[m.solutes.at(3)]
        ) ** (
            -1 / m.z[m.solutes.at(3)]
        ) * (
            m.conc_mem[m.solutes.at(2)] / m.conc_sol[m.solutes.at(2)]
        ) ** (
            1 / m.z[m.solutes.at(2)]
        )

    m.salt_2_partitioning = Constraint(rule=_salt_2_partitioning)

    return m


def double_salt_partitioning_model(ion_info):
    """
    Generalized model equations for the partitioning of ions in a two salt
    (common anion) aqueous solution, assuming there are ion interactions.

    Assumes 1 and 2 are cations, 3 is anion

    Args:
        ion_info: {
            ion: {
                "num": int value
                "z": int val,
                "H": float val,
                "conc_sol": float val,
            },
            ...
        }

    Returns:
        m: the Pyomo model
    """
    m = ConcreteModel()

    add_model_sets(m, ion_info)
    add_model_params(m, ion_info)
    add_model_vars(m, ion_info)

    # electroneutrality in the solution phase: z1c1+z2c2+z3c3=0
    def _electoneutrality_solution(m):
        return (
            m.conc_sol[m.solutes.at(3)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(3)])
            * m.conc_sol[m.solutes.at(1)]
            - (m.z[m.solutes.at(2)] / m.z[m.solutes.at(3)])
            * m.conc_sol[m.solutes.at(2)]
        )

    m.electroneutrality_solution = Constraint(rule=_electoneutrality_solution)

    # electroneutrality in the membrane phase: z1c1+z2c2+z3c3+chi=0
    def _electoneutrality_membrane(m):
        return (
            m.conc_mem[m.solutes.at(3)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(3)])
            * m.conc_mem[m.solutes.at(1)]
            - (m.z[m.solutes.at(2)] / m.z[m.solutes.at(3)])
            * m.conc_mem[m.solutes.at(2)]
            - m.chi / m.z[m.solutes.at(3)]
        )

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # partitioning rules
    # cation 1 concentration in the membrane
    def _cation_1_partitioning(m):
        return (m.H[m.solutes.at(1)] * m.conc_sol[m.solutes.at(1)]) ** (
            1 / m.z[m.solutes.at(1)]
        ) * (
            (
                -m.z[m.solutes.at(1)]
                / m.z[m.solutes.at(3)]
                * m.conc_sol[m.solutes.at(1)]
                - m.z[m.solutes.at(2)]
                / m.z[m.solutes.at(3)]
                * m.conc_sol[m.solutes.at(2)]
            )
            ** (-1 / m.z[m.solutes.at(3)])
        ) == (
            (m.H[m.solutes.at(3)]) ** (1 / m.z[m.solutes.at(3)])
            * (m.conc_mem[m.solutes.at(1)]) ** (1 / m.z[m.solutes.at(1)])
            * (
                -m.z[m.solutes.at(1)]
                / m.z[m.solutes.at(3)]
                * m.conc_mem[m.solutes.at(1)]
                - (
                    (
                        (
                            m.z[m.solutes.at(2)]
                            * m.H[m.solutes.at(2)]
                            * m.conc_sol[m.solutes.at(2)]
                        )
                        / (m.z[m.solutes.at(3)])
                    )
                    * (
                        (m.conc_mem[m.solutes.at(1)])
                        / (m.H[m.solutes.at(1)] * m.conc_sol[m.solutes.at(1)])
                    )
                    ** (m.z[m.solutes.at(2)] / m.z[m.solutes.at(1)])
                )
                - (m.chi / m.z[m.solutes.at(3)])
            )
            ** (-1 / m.z[m.solutes.at(3)])
        )

    m.cation_1_partitioning = Constraint(rule=_cation_1_partitioning)

    # cation 2 concentration in the membrane
    def _cation_2_partitioning(m):
        return (m.H[m.solutes.at(2)] * m.conc_sol[m.solutes.at(2)]) ** (
            1 / m.z[m.solutes.at(2)]
        ) * (
            -m.z[m.solutes.at(1)] / m.z[m.solutes.at(3)] * m.conc_sol[m.solutes.at(1)]
            - m.z[m.solutes.at(2)] / m.z[m.solutes.at(3)] * m.conc_sol[m.solutes.at(2)]
        ) ** (
            -1 / m.z[m.solutes.at(3)]
        ) == (
            (m.H[m.solutes.at(3)]) ** (1 / m.z[m.solutes.at(3)])
            * (m.conc_mem[m.solutes.at(2)]) ** (1 / m.z[m.solutes.at(2)])
            * (
                -m.z[m.solutes.at(2)]
                / m.z[m.solutes.at(3)]
                * m.conc_mem[m.solutes.at(2)]
                - (
                    (
                        (
                            m.z[m.solutes.at(1)]
                            * m.H[m.solutes.at(1)]
                            * m.conc_sol[m.solutes.at(1)]
                        )
                        / (m.z[m.solutes.at(3)])
                    )
                    * (
                        (m.conc_mem[m.solutes.at(2)])
                        / (m.H[m.solutes.at(2)] * m.conc_sol[m.solutes.at(2)])
                    )
                    ** (m.z[m.solutes.at(1)] / m.z[m.solutes.at(2)])
                )
                - (m.chi / m.z[m.solutes.at(3)])
            )
            ** (-1 / m.z[m.solutes.at(3)])
        )

    m.cation_2_partitioning = Constraint(rule=_cation_2_partitioning)

    return m


def solve_model(m, tee=False):
    solver = SolverFactory("ipopt")
    result = solver.solve(m, tee=tee)
    assert_optimal_termination(result)


def print_info(m):
    print(
        " \t valence \t partition coefficient \t fixed membrane charge (mM) \t solution concentration (mM) \t membrane concentration (mM)"
    )
    for i in m.solutes:
        print(
            f"ion {i} \t {value(m.z[i])} \t\t {value(m.H[i])} \t\t\t {value(m.chi)} \t\t\t\t {value(m.conc_sol[i])} \t\t\t\t {value(m.conc_mem[i])}"
        )


def single_salt_sensitivity(cation_num, cation_z, c_1_sol_vals, h_1_vals, h_2_vals, ax):
    c_1_mem_vals = []

    for h1 in h_1_vals:
        for h2 in h_2_vals:
            cation_dict = {"num": cation_num, "z": cation_z, "H": h1, "conc_sol": 20}
            chlorine_dict = {"num": 3, "z": -1, "H": h2, "conc_sol": 20}
            ion_dict = {
                "cation": cation_dict,
                "chlorine": chlorine_dict,
            }
            m = single_salt_partitioning_model(ion_dict)
            for c1 in c_1_sol_vals:
                m.conc_sol[cation_dict["num"]].fix(c1)
                solve_model(m)
                c_1_mem_vals.append(value(m.conc_mem[cation_dict["num"]]))
            if (
                (h1 == 1 and h2 == 0.5)
                or (h1 == 1.5 and h2 == 0.5)
                or (h1 == 1.5 and h2 == 1)
            ):
                ax.plot(
                    c_1_sol_vals,
                    c_1_mem_vals,
                    linewidth=2,
                    linestyle="dashed",
                    label=f"{h1},{h2}",
                )
            else:
                ax.plot(c_1_sol_vals, c_1_mem_vals, linewidth=2, label=f"{h1},{h2}")
            c_1_mem_vals = []

    ax.plot([0, 300], [0, 300], "k--")

    ax.set_xlim(left=0, right=110)
    ax.set_ylim(bottom=0, top=180)
    ax.tick_params(direction="in", right=True, labelsize=10)
    ax.set_title(label="", fontweight="bold")
    ax.set_xlabel(xlabel="", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel="", fontsize=10, fontweight="bold")
    ax.legend(title="", loc="best", ncol=2)


def double_salt_sensitivity(
    c_primary_sol_vals,
    primary_cation,
    secondary_cation,
    h_1_vals,
    h_2_vals,
    ax,
    independent=False,
):
    c_primary_mem_vals = []
    h3 = 1

    for h1 in h_1_vals:
        for h2 in h_2_vals:
            lithium_dict = {"num": 1, "z": 1, "H": h1, "conc_sol": 20}
            cobalt_dict = {"num": 2, "z": 2, "H": h2, "conc_sol": 20}
            chlorine_dict = {"num": 3, "z": -1, "H": h3, "conc_sol": 20}
            ion_dict = {
                "lithium": lithium_dict,
                "cobalt": cobalt_dict,
                "chlorine": chlorine_dict,
            }
            if independent:
                m = double_salt_independent_partitioning_model(ion_dict)
            else:
                m = double_salt_partitioning_model(ion_dict)
            for c1 in c_primary_sol_vals:
                m.conc_sol[primary_cation].fix(c1)
                m.conc_sol[secondary_cation].fix(50)
                solve_model(m)
                c_primary_mem_vals.append(value(m.conc_mem[primary_cation]))
            ax.plot(
                c_primary_sol_vals, c_primary_mem_vals, linewidth=2, label=f"{h1},{h2}"
            )
            c_primary_mem_vals = []

    ax.plot([0, 300], [0, 300], "k--")

    ax.set_xlim(left=0, right=110)
    ax.set_ylim(bottom=0, top=180)
    ax.tick_params(direction="in", right=True, labelsize=10)
    ax.set_title(label="", fontweight="bold")
    ax.set_xlabel(xlabel="", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel="", fontsize=10, fontweight="bold")
    ax.legend(title="", loc="best", ncol=2)


def plot_partitioning_behavior(single=True, double=True, independent=False):
    # plot membrane concentration versus solution for different H's

    if single:
        # first look at the single salt models
        c_1_sol_vals = np.arange(5, 105, 5)  # mM
        h_1_vals = np.arange(0.5, 2, 0.5)
        h_2_vals = np.arange(0.5, 2, 0.5)

        fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=125, figsize=(10, 5))

        single_salt_sensitivity(1, 1, c_1_sol_vals, h_1_vals, h_2_vals, ax1)
        ax1.set_title(label="Lithium Chloride Only")
        ax1.set_xlabel(xlabel="Lithium Concentration, Solution (mM)")
        ax1.set_ylabel(ylabel="Lithium Concentration, \nMembrane (mM)")
        ax1.legend(title="$H_{Li},H_{Cl}$")

        single_salt_sensitivity(2, 2, c_1_sol_vals, h_1_vals, h_2_vals, ax2)
        ax2.set_title(label="Cobalt Chloride Only")
        ax2.set_xlabel(xlabel="Cobalt Concentration, Solution (mM)")
        ax2.set_ylabel(ylabel="Cobalt Concentration, \nMembrane (mM)")
        ax2.legend(title="$H_{Co},H_{Cl}$")

        plt.show()

    if double:
        # now look at the two salts together, assuming no interaction
        c_1_sol_vals = np.arange(5, 105, 5)  # mM
        c_2_sol_vals = np.arange(5, 105, 5)  # mM
        h_1_vals = np.arange(0.5, 2, 0.5)
        h_2_vals = np.arange(0.5, 2, 0.5)

        fig2, (ax3, ax4) = plt.subplots(1, 2, dpi=125, figsize=(10, 5))
        fig2.suptitle("Lithium Chloride, Cobalt Chloride System", fontweight="bold")

        double_salt_sensitivity(
            c_1_sol_vals, 1, 2, h_1_vals, h_2_vals, ax3, independent=independent
        )

        ax3.set_title(
            "Cobalt Chloride Concentration = 50 mM \n $H_{Cl}$ = 1, $\chi$=0 mM"
        )
        ax3.set_xlabel("Lithium Concentration, Solution (mM)")
        ax3.set_ylabel("Lithium Concentration, \nMembrane (mM)")
        ax3.legend(title="$H_{Li},H_{Co}$")

        double_salt_sensitivity(
            c_2_sol_vals, 2, 1, h_1_vals, h_2_vals, ax4, independent=independent
        )

        ax4.set_title(
            "Lithium Chloride Concentration = 50 mM \n $H_{Cl}$ = 1, $\chi$=0 mM"
        )
        ax4.set_xlabel("Cobalt Concentration, Solution (mM)")
        ax4.set_ylabel("Cobalt Concentration, \nMembrane (mM)")
        ax4.legend(title="$H_{Li},H_{Co}$")

        plt.show()


def plot_chi_sensitivity(independent=False):
    # TODO: add option for single salt models
    # TODO: generalize code

    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, dpi=125, figsize=(12.5, 10)
    )

    c_1_sol_vals = np.arange(5, 105, 5)  # mM
    c_2_sol_vals = np.arange(5, 105, 5)  # mM
    # chi_vals = np.arange(-10, 20, 10)  # mM
    chi_vals = [-200, -20, 0, 20, 200]  # mM

    c_1_mem_vals = []
    c_2_mem_vals = []

    lithium_dict = {"num": 1, "z": 1, "H": 1.5, "conc_sol": 20}
    cobalt_dict = {"num": 2, "z": 2, "H": 0.5, "conc_sol": 20}
    chlorine_dict = {"num": 3, "z": -1, "H": 1, "conc_sol": 20}
    ion_dict = {
        "lithium": lithium_dict,
        "cobalt": cobalt_dict,
        "chlorine": chlorine_dict,
    }

    for chi in chi_vals:
        if independent:
            m = double_salt_independent_partitioning_model(ion_dict)
        else:
            m = double_salt_partitioning_model(ion_dict)
        m.chi = chi
        for c1 in c_1_sol_vals:
            m.conc_sol[1].fix(c1)
            m.conc_sol[2].fix(50)
            solve_model(m)
            c_1_mem_vals.append(value(m.conc_mem[1]))

        if chi == -200:
            ax1.plot(c_1_sol_vals, c_1_mem_vals, "c-.", linewidth=2)
        if chi == -20:
            ax1.plot(c_1_sol_vals, c_1_mem_vals, "g-.", linewidth=2)
        if chi == 0:
            ax1.plot(c_1_sol_vals, c_1_mem_vals, "m-.", linewidth=2)
        if chi == 20:
            ax1.plot(c_1_sol_vals, c_1_mem_vals, "b-.", linewidth=2)
        if chi == 200:
            ax1.plot(c_1_sol_vals, c_1_mem_vals, "r-.", linewidth=2)

        c_1_mem_vals = []

    for chi in chi_vals:
        if independent:
            m = double_salt_independent_partitioning_model(ion_dict)
        else:
            m = double_salt_partitioning_model(ion_dict)
        m.chi = chi
        for c2 in c_2_sol_vals:
            m.conc_sol[2].fix(c2)
            m.conc_sol[1].fix(50)
            solve_model(m)
            c_2_mem_vals.append(value(m.conc_mem[2]))

        if chi == -200:
            ax2.plot(c_2_sol_vals, c_2_mem_vals, "c-.", linewidth=2)
        if chi == -20:
            ax2.plot(c_2_sol_vals, c_2_mem_vals, "g-.", linewidth=2)
        if chi == 0:
            ax2.plot(c_2_sol_vals, c_2_mem_vals, "m-.", linewidth=2)
        if chi == 20:
            ax2.plot(c_2_sol_vals, c_2_mem_vals, "b-.", linewidth=2)
        if chi == 200:
            ax2.plot(c_2_sol_vals, c_2_mem_vals, "r-.", linewidth=2)

        c_2_mem_vals = []

    lithium_dict = {"num": 1, "z": 1, "H": 1, "conc_sol": 20}
    cobalt_dict = {"num": 2, "z": 2, "H": 1, "conc_sol": 20}
    chlorine_dict = {"num": 3, "z": -1, "H": 1, "conc_sol": 20}
    ion_dict = {
        "lithium": lithium_dict,
        "cobalt": cobalt_dict,
        "chlorine": chlorine_dict,
    }

    for chi in chi_vals:
        if independent:
            m = double_salt_independent_partitioning_model(ion_dict)
        else:
            m = double_salt_partitioning_model(ion_dict)
        m.chi = chi
        for c1 in c_1_sol_vals:
            m.conc_sol[1].fix(c1)
            m.conc_sol[2].fix(50)
            solve_model(m)
            c_1_mem_vals.append(value(m.conc_mem[1]))

        if chi == -200:
            ax3.plot(c_1_sol_vals, c_1_mem_vals, "c-", linewidth=2)
        if chi == -20:
            ax3.plot(c_1_sol_vals, c_1_mem_vals, "g-", linewidth=2)
        if chi == 0:
            ax3.plot(c_1_sol_vals, c_1_mem_vals, "m-", linewidth=2)
        if chi == 20:
            ax3.plot(c_1_sol_vals, c_1_mem_vals, "b-", linewidth=2)
        if chi == 200:
            ax3.plot(c_1_sol_vals, c_1_mem_vals, "r-", linewidth=2)

        c_1_mem_vals = []

    for chi in chi_vals:
        if independent:
            m = double_salt_independent_partitioning_model(ion_dict)
        else:
            m = double_salt_partitioning_model(ion_dict)
        m.chi = chi
        for c2 in c_2_sol_vals:
            m.conc_sol[2].fix(c2)
            m.conc_sol[1].fix(50)
            solve_model(m)
            c_2_mem_vals.append(value(m.conc_mem[2]))

        if chi == -200:
            ax4.plot(c_2_sol_vals, c_2_mem_vals, "c-", linewidth=2)
        if chi == -20:
            ax4.plot(c_2_sol_vals, c_2_mem_vals, "g-", linewidth=2)
        if chi == 0:
            ax4.plot(c_2_sol_vals, c_2_mem_vals, "m-", linewidth=2)
        if chi == 20:
            ax4.plot(c_2_sol_vals, c_2_mem_vals, "b-", linewidth=2)
        if chi == 200:
            ax4.plot(c_2_sol_vals, c_2_mem_vals, "r-", linewidth=2)

        c_2_mem_vals = []

    lithium_dict = {"num": 1, "z": 1, "H": 0.5, "conc_sol": 20}
    cobalt_dict = {"num": 2, "z": 2, "H": 1.5, "conc_sol": 20}
    chlorine_dict = {"num": 3, "z": -1, "H": 1, "conc_sol": 20}
    ion_dict = {
        "lithium": lithium_dict,
        "cobalt": cobalt_dict,
        "chlorine": chlorine_dict,
    }

    for chi in chi_vals:
        if independent:
            m = double_salt_independent_partitioning_model(ion_dict)
        else:
            m = double_salt_partitioning_model(ion_dict)
        m.chi = chi
        for c1 in c_1_sol_vals:
            m.conc_sol[1].fix(c1)
            m.conc_sol[2].fix(50)
            solve_model(m)
            c_1_mem_vals.append(value(m.conc_mem[1]))

        if chi == -200:
            ax5.plot(c_1_sol_vals, c_1_mem_vals, "c:", linewidth=2)
        if chi == -20:
            ax5.plot(c_1_sol_vals, c_1_mem_vals, "g:", linewidth=2)
        if chi == 0:
            ax5.plot(c_1_sol_vals, c_1_mem_vals, "m:", linewidth=2)
        if chi == 20:
            ax5.plot(c_1_sol_vals, c_1_mem_vals, "b:", linewidth=2)
        if chi == 200:
            ax5.plot(c_1_sol_vals, c_1_mem_vals, "r:", linewidth=2)

        c_1_mem_vals = []

    for chi in chi_vals:
        if independent:
            m = double_salt_independent_partitioning_model(ion_dict)
        else:
            m = double_salt_partitioning_model(ion_dict)
        m.chi = chi
        for c2 in c_2_sol_vals:
            m.conc_sol[2].fix(c2)
            m.conc_sol[1].fix(50)
            solve_model(m)
            c_2_mem_vals.append(value(m.conc_mem[2]))

        if chi == -200:
            ax6.plot(c_2_sol_vals, c_2_mem_vals, "c:", linewidth=2)
        if chi == -20:
            ax6.plot(c_2_sol_vals, c_2_mem_vals, "g:", linewidth=2)
        if chi == 0:
            ax6.plot(c_2_sol_vals, c_2_mem_vals, "m:", linewidth=2)
        if chi == 20:
            ax6.plot(c_2_sol_vals, c_2_mem_vals, "b:", linewidth=2)
        if chi == 200:
            ax6.plot(c_2_sol_vals, c_2_mem_vals, "r:", linewidth=2)

        c_2_mem_vals = []

    # ghost points for legends
    ax3.plot(
        [], [], "k-.", linewidth=2, label="($H_{Li}$,$H_{Co}$,$H_{Cl}$)=(1.5,0.5,1)"
    )
    ax3.plot([], [], "k-", linewidth=2, label="($H_{Li}$,$H_{Co}$,$H_{Cl}$)=(1,1,1)")
    ax3.plot(
        [], [], "k:", linewidth=2, label="($H_{Li}$,$H_{Co}$,$H_{Cl}$)=(0.5,1.5,1)"
    )
    ax3.legend(loc="best")
    ax5.plot([], [], "c", linewidth=2, label="$\chi$=-200 mM")
    ax5.plot([], [], "g", linewidth=2, label="$\chi$=-20 mM")
    ax5.plot([], [], "m", linewidth=2, label="$\chi$=0 mM")
    ax5.plot([], [], "b", linewidth=2, label="$\chi$=20 mM")
    ax5.plot([], [], "r", linewidth=2, label="$\chi$=200 mM")
    ax5.legend(loc="best")

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.plot([0, 300], [0, 300], "k--")
        ax.set_xlim(left=0, right=110)
        ax.set_ylim(bottom=0, top=180)
        ax.tick_params(direction="in", right=True, labelsize=12)

    ax1.set_title(
        label="Cobalt Concentration = 50 mM",
        fontweight="bold",
    )
    ax5.set_xlabel(
        xlabel="Lithium Concentration, Solution (mM)", fontsize=12, fontweight="bold"
    )
    for ax in (ax1, ax3, ax5):
        ax.set_ylabel(
            ylabel="Lithium Concentration, \nMembrane (mM)",
            fontsize=12,
            fontweight="bold",
        )

    ax2.set_title(
        label="Lithium Concentration = 50 mM",
        fontweight="bold",
    )
    ax6.set_xlabel(
        xlabel="Cobalt Concentration, Solution (mM)", fontsize=12, fontweight="bold"
    )
    for ax in (ax2, ax4, ax6):
        ax.set_ylabel(
            ylabel="Cobalt Concentration, \nMembrane (mM)",
            fontsize=12,
            fontweight="bold",
        )

    plt.show()


def visualize_3d_trends(independent=False):
    c_1_sol_vals = np.arange(5, 105, 5)  # mM
    c_2_sol_vals = np.arange(5, 105, 5)  # mM
    c1_s, c2_s = np.meshgrid(c_1_sol_vals, c_2_sol_vals)
    c1_m = []
    c1m_list = []
    c2_m = []
    c2m_list = []

    ax = plt.figure().add_subplot(projection="3d")

    lithium_dict = {"num": 1, "z": 1, "H": 1.5, "conc_sol": 20}
    cobalt_dict = {"num": 2, "z": 2, "H": 0.5, "conc_sol": 20}
    chlorine_dict = {"num": 3, "z": -1, "H": 1, "conc_sol": 20}
    ion_dict = {
        "lithium": lithium_dict,
        "cobalt": cobalt_dict,
        "chlorine": chlorine_dict,
    }
    if independent:
        m = double_salt_independent_partitioning_model(ion_dict)
    else:
        m = double_salt_partitioning_model(ion_dict)
    m.chi = -5

    for c2 in c_2_sol_vals:
        for c1 in c_1_sol_vals:
            m.conc_sol[1].fix(c1)
            m.conc_sol[2].fix(c2)
            solve_model(m)
            c1m_list.append(value(m.conc_mem[1]))
            c2m_list.append(value(m.conc_mem[2]))
        c1_m.append(c1m_list)
        c1m_list = []
        c2_m.append(c2m_list)
        c2m_list = []

    ax.plot_surface(c1_s, c2_s, np.array(c1_m), label="lithium")
    ax.plot_surface(c1_s, c2_s, np.array(c2_m), label="cobalt")

    ax.set_xlabel(
        xlabel="Lithium Concentration, Solution (mM)", fontsize=10, fontweight="bold"
    )
    ax.set_ylabel(
        ylabel="Cobalt Concentration, Solution (mM)", fontsize=10, fontweight="bold"
    )
    ax.set_title("Membrane Concentration (mM)", fontsize=10, fontweight="bold")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
