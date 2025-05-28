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
    lithium_H = 1
    cobalt_H = 1
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
    solve_model(m_double_salt_independent)

    m_double_salt = double_salt_partitioning_model(ion_dict_LiCoCl)
    solve_model(m_double_salt)

    print("Lithium Chloride")
    print_info(m_general_lithium)
    print("\n")

    print("Cobalt Chloride")
    print_info(m_general_cobalt)
    print("\n")

    print("Double Salt, No Interaction")
    print_info(m_double_salt_independent)
    print("\n")

    print("Double Salt")
    print_info(m_double_salt)
    print("\n")

    # plot_partitioning_behavior(single=True, double=True)


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

    # electroneutrality in the solution phase: z1c1+z2c2=0
    def _electoneutrality_solution(m):
        return (
            m.conc_sol[m.solutes.at(2)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(2)])
            * m.conc_sol[m.solutes.at(1)]
        )

    m.electroneutrality_solution = Constraint(rule=_electoneutrality_solution)

    # electroneutrality in the membrane phase: z1c1+z2c2=0
    def _electoneutrality_membrane(m):
        return (
            m.conc_mem[m.solutes.at(2)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(2)])
            * m.conc_mem[m.solutes.at(1)]
        )

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # generalized single salt partitioning
    def _single_salt_partitioning(m):
        return m.H[m.solutes.at(1)] * m.H[m.solutes.at(2)] == (
            m.conc_mem[m.solutes.at(2)] / m.conc_sol[m.solutes.at(2)]
        ) * (m.conc_mem[m.solutes.at(1)] / m.conc_sol[m.solutes.at(1)]) ** (
            -m.z[m.solutes.at(2)] / m.z[m.solutes.at(1)]
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

    # electroneutrality in the membrane phase: z1c1+z2c2+z3c3=0
    def _electoneutrality_membrane(m):
        return (
            m.conc_mem[m.solutes.at(3)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(3)])
            * m.conc_mem[m.solutes.at(1)]
            - (m.z[m.solutes.at(2)] / m.z[m.solutes.at(3)])
            * m.conc_mem[m.solutes.at(2)]
        )

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # generalized single salt partitioning
    def _salt_1_partitioning(m):
        return m.H[m.solutes.at(1)] * m.H[m.solutes.at(3)] == (
            m.conc_mem[m.solutes.at(3)] / m.conc_sol[m.solutes.at(3)]
        ) * (m.conc_mem[m.solutes.at(1)] / m.conc_sol[m.solutes.at(1)]) ** (
            -m.z[m.solutes.at(3)] / m.z[m.solutes.at(1)]
        )

    m.salt_1_partitioning = Constraint(rule=_salt_1_partitioning)

    def _salt_2_partitioning(m):
        return m.H[m.solutes.at(2)] * m.H[m.solutes.at(3)] == (
            m.conc_mem[m.solutes.at(3)] / m.conc_sol[m.solutes.at(3)]
        ) * (m.conc_mem[m.solutes.at(2)] / m.conc_sol[m.solutes.at(2)]) ** (
            -m.z[m.solutes.at(3)] / m.z[m.solutes.at(2)]
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

    # electroneutrality in the membrane phase: z1c1+z2c2+z3c3=0
    def _electoneutrality_membrane(m):
        return (
            m.conc_mem[m.solutes.at(3)]
            == -(m.z[m.solutes.at(1)] / m.z[m.solutes.at(3)])
            * m.conc_mem[m.solutes.at(1)]
            - (m.z[m.solutes.at(2)] / m.z[m.solutes.at(3)])
            * m.conc_mem[m.solutes.at(2)]
        )

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # partitioning rules
    # TODO generalize by adding z parameters
    # lithium concentration in the membrane
    def _lithium_chloride_partitioning(m):
        return m.H[m.solutes.at(1)] * m.conc_sol[m.solutes.at(1)] * (
            m.conc_sol[m.solutes.at(1)] + 2 * m.conc_sol[m.solutes.at(2)]
        ) == (
            m.H[m.solutes.at(3)]
            * m.conc_mem[m.solutes.at(1)]
            * (
                m.conc_mem[m.solutes.at(1)]
                + (
                    (
                        2
                        * m.H[m.solutes.at(2)]
                        * m.conc_sol[m.solutes.at(2)]
                        * (m.conc_mem[m.solutes.at(1)]) ** 2
                    )
                    / ((m.H[m.solutes.at(1)]) ** 2 * (m.conc_sol[m.solutes.at(1)]) ** 2)
                )
            )
        )

    m.lithium_chloride_partitioning = Constraint(rule=_lithium_chloride_partitioning)

    # cobalt concentration in the membrane
    def _cobalt_chloride_partitioning(m):
        return (m.H[m.solutes.at(2)]) ** (1 / 2) * (m.conc_sol[m.solutes.at(2)]) ** (
            1 / 2
        ) * (m.conc_sol[m.solutes.at(1)] + 2 * m.conc_sol[m.solutes.at(2)]) == (
            m.H[m.solutes.at(3)]
            * (m.conc_mem[m.solutes.at(2)]) ** (1 / 2)
            * (
                2 * m.conc_mem[m.solutes.at(2)]
                + (
                    (
                        m.H[m.solutes.at(1)]
                        * m.conc_sol[m.solutes.at(1)]
                        * (m.conc_mem[m.solutes.at(2)]) ** (1 / 2)
                    )
                    / (
                        (m.H[m.solutes.at(2)]) ** (1 / 2)
                        * (m.conc_sol[m.solutes.at(2)]) ** (1 / 2)
                    )
                )
            )
        )

    m.cobalt_chloride_partitioning = Constraint(rule=_cobalt_chloride_partitioning)

    return m


def solve_model(m):
    solver = SolverFactory("ipopt")
    result = solver.solve(m, tee=True)
    assert_optimal_termination(result)


def print_info(m):
    print(
        " \t valence \t partition coefficient \t solution concentration (mM) \t membrane concentration (mM)"
    )
    for i in m.solutes:
        print(
            f"ion {i} \t {value(m.z[i])} \t\t {value(m.H[i])} \t\t\t {value(m.conc_sol[i])} \t\t\t\t {value(m.conc_mem[i])}"
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
    c_primary_sol_vals, primary_cation, secondary_cation, h_1_vals, h_2_vals, ax
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
            m = double_salt_independent_partitioning_model(ion_dict)
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


def plot_partitioning_behavior(single=True, double=True):
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

        double_salt_sensitivity(c_1_sol_vals, 1, 2, h_1_vals, h_2_vals, ax3)

        ax3.set_title("Cobalt Chloride Concentration = 50 mM \n $H_{Cl}$ = 1")
        ax3.set_xlabel("Lithium Concentration, Solution (mM)")
        ax3.set_ylabel("Lithium Concentration, \nMembrane (mM)")
        ax3.legend(title="$H_{Li},H_{Co}$")

        double_salt_sensitivity(c_2_sol_vals, 2, 1, h_1_vals, h_2_vals, ax4)

        ax4.set_title("Lithium Chloride Concentration = 50 mM \n $H_{Cl}$ = 1")
        ax4.set_xlabel("Cobalt Concentration, Solution (mM)")
        ax4.set_ylabel("Cobalt Concentration, \nMembrane (mM)")
        ax4.legend(title="$H_{Li},H_{Co}$")

        plt.show()


if __name__ == "__main__":
    main()
