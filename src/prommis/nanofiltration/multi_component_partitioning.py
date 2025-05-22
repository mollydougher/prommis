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
    m_single_symmetric = single_symmetric_salt_partitioning_model(h1=1.5, h2=1)
    solve_model(m_single_symmetric)
    m_single_asymmetric = single_asymmetric_salt_partitioning_model(h1=0.5, h2=1)
    solve_model(m_single_asymmetric)
    m_double_salt_independent = double_salt_independent_partitioning_model(
        h1=1.5, h2=0.5, h3=1
    )
    solve_model(m_double_salt_independent)

    print("Single, Symmetric Salt")
    print_info(m_single_symmetric)
    print("\n")

    print("Single, Asymmetric Salt")
    print_info(m_single_asymmetric)
    print("\n")

    print("Double Salt, Independent")
    print_info(m_double_salt_independent)
    print("\n")

    plot_partitioning_behavior(single=True, double=True)


def single_symmetric_salt_partitioning_model(h1=1, h2=1):
    m = ConcreteModel()

    m.solutes = Set(initialize=["Li", "Cl"])

    m.H_1 = Param(initialize=h1)
    m.H_2 = Param(initialize=h2)
    m.z_1 = Param(initialize=1)  # monovalent cation
    m.z_2 = Param(initialize=-1)  # monovalent anion

    m.conc_1_sol = Var(initialize=20)  # mM
    m.conc_1_sol.fix()
    m.conc_2_sol = Var(initialize=20)  # mM

    m.conc_1_mem = Var(initialize=10)
    m.conc_2_mem = Var(initialize=10)

    # electroneutrality in the solution phase: z1c1+z2c2=0
    def _electoneutrality_solution(m):
        return m.conc_2_sol == -(m.z_1 / m.z_2) * m.conc_1_sol

    m.electroneutrality_solution = Constraint(rule=_electoneutrality_solution)

    # electroneutrality in the membrane phase: z1c1+z2c2=0
    def _electoneutrality_membrane(m):
        return m.conc_2_mem == -(m.z_1 / m.z_2) * m.conc_1_mem

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # partitioning
    def _single_symmetric_partitioning(m):
        return m.H_1 * m.H_2 == (m.conc_1_mem * m.conc_2_mem) / (
            m.conc_1_sol * m.conc_2_sol
        )

    m.single_symmetric_partitioning = Constraint(rule=_single_symmetric_partitioning)

    return m


def single_asymmetric_salt_partitioning_model(h1=1, h2=1):
    m = ConcreteModel()

    m.solutes = Set(initialize=["Co", "Cl"])

    m.H_1 = Param(initialize=h1)
    m.H_2 = Param(initialize=h2)
    m.z_1 = Param(initialize=2)  # divalent cation
    m.z_2 = Param(initialize=-1)  # monovalent anion

    m.conc_1_sol = Var(initialize=20)  # mM
    m.conc_1_sol.fix()
    m.conc_2_sol = Var(initialize=20)  # mM

    m.conc_1_mem = Var(initialize=10)
    m.conc_2_mem = Var(initialize=10)

    # electroneutrality in the solution phase: z1c1+z2c2=0
    def _electoneutrality_solution(m):
        return m.conc_2_sol == -(m.z_1 / m.z_2) * m.conc_1_sol

    m.electroneutrality_solution = Constraint(rule=_electoneutrality_solution)

    # electroneutrality in the membrane phase: z1c1+z2c2=0
    def _electoneutrality_membrane(m):
        return m.conc_2_mem == -(m.z_1 / m.z_2) * m.conc_1_mem

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # partitioning
    def _single_asymmetric_partitioning(m):
        return m.H_1 * m.H_2 == ((m.conc_1_mem) ** (1 / 2) * m.conc_2_mem) / (
            (m.conc_1_sol) ** (1 / 2) * m.conc_2_sol
        )

    m.single_asymmetric_partitioning = Constraint(rule=_single_asymmetric_partitioning)

    return m


def double_salt_independent_partitioning_model(h1=1, h2=1, h3=1):
    m = ConcreteModel()

    m.solutes = Set(initialize=["Li", "Co", "Cl"])

    m.H_1 = Param(initialize=h1)
    m.H_2 = Param(initialize=h2)
    m.H_3 = Param(initialize=h3)
    m.z_1 = Param(initialize=1)  # monovalent cation
    m.z_2 = Param(initialize=2)  # divalent cation
    m.z_3 = Param(initialize=-1)  # monovalent anion

    m.conc_1_sol = Var(initialize=20)  # mM
    m.conc_1_sol.fix()
    m.conc_2_sol = Var(initialize=20)  # mM
    m.conc_2_sol.fix()
    m.conc_3_sol = Var(initialize=20)  # mM

    m.conc_1_mem = Var(initialize=10)
    m.conc_2_mem = Var(initialize=10)
    m.conc_3_mem = Var(initialize=10)

    # electroneutrality in the solution phase: z1c1+z2c2+z3c3=0
    def _electoneutrality_solution(m):
        return (
            m.conc_3_sol
            == -(m.z_1 / m.z_3) * m.conc_1_sol - (m.z_2 / m.z_3) * m.conc_2_sol
        )

    m.electroneutrality_solution = Constraint(rule=_electoneutrality_solution)

    # electroneutrality in the membrane phase: z1c1+z2c2+z3c3=0
    def _electoneutrality_membrane(m):
        return (
            m.conc_3_mem
            == -(m.z_1 / m.z_3) * m.conc_1_mem - (m.z_2 / m.z_3) * m.conc_2_mem
        )

    m.electoneutrality_membrane = Constraint(rule=_electoneutrality_membrane)

    # partitioning
    def _single_symmetric_partitioning(m):
        return m.H_1 * m.H_3 == (m.conc_1_mem * m.conc_3_mem) / (
            m.conc_1_sol * m.conc_3_sol
        )

    m.single_symmetric_partitioning = Constraint(rule=_single_symmetric_partitioning)

    def _single_asymmetric_partitioning(m):
        return m.H_2 * m.H_3 == ((m.conc_2_mem) ** (1 / 2) * m.conc_3_mem) / (
            (m.conc_2_sol) ** (1 / 2) * m.conc_3_sol
        )

    m.single_asymmetric_partitioning = Constraint(rule=_single_asymmetric_partitioning)

    return m


def solve_model(m):
    solver = SolverFactory("ipopt")
    result = solver.solve(m, tee=True)
    assert_optimal_termination(result)


def print_info(m):
    print(
        " \t valence \t partition coefficient \t solution concentration (mM) \t membrane concentration (mM)"
    )
    print(
        f"ion 1 \t {value(m.z_1)} \t\t {value(m.H_1)} \t\t\t {value(m.conc_1_sol)} \t\t\t\t {value(m.conc_1_mem)}"
    )
    print(
        f"ion 2 \t {value(m.z_2)} \t\t {value(m.H_2)} \t\t\t {value(m.conc_2_sol)} \t\t\t\t {value(m.conc_2_mem)}"
    )
    if len(m.solutes) == 3:
        print(
            f"ion 3 \t {value(m.z_3)} \t\t {value(m.H_3)} \t\t\t {value(m.conc_3_sol)} \t\t\t\t {value(m.conc_3_mem)}"
        )


def plot_partitioning_behavior(single=True, double=True):
    # plot membrane concentration versus solution for different H's

    if single:
        # first look at the single salt models
        c_1_sol_vals = np.arange(5, 105, 5)  # mM
        h_1_vals = np.arange(0.5, 2, 0.5)
        h_2_vals = np.arange(0.5, 2, 0.5)

        c_1_mem_vals = []

        fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=125, figsize=(10, 5))

        for h1 in h_1_vals:
            for h2 in h_2_vals:
                m = single_symmetric_salt_partitioning_model(h1=h1, h2=h2)
                for c1 in c_1_sol_vals:
                    m.conc_1_sol.fix(c1)
                    solve_model(m)
                    c_1_mem_vals.append(value(m.conc_1_mem))
                if (
                    (h1 == 1 and h2 == 0.5)
                    or (h1 == 1.5 and h2 == 0.5)
                    or (h1 == 1.5 and h2 == 1)
                ):
                    ax1.plot(
                        c_1_sol_vals,
                        c_1_mem_vals,
                        linewidth=2,
                        linestyle="dashed",
                        label=f"{h1},{h2}",
                    )
                else:
                    ax1.plot(
                        c_1_sol_vals, c_1_mem_vals, linewidth=2, label=f"{h1},{h2}"
                    )
                c_1_mem_vals = []
        ax1.plot([0, 300], [0, 300], "k--")

        ax1.set_title("Lithium Chloride Only", fontweight="bold")
        ax1.set_xlabel(
            "Lithium Concentration, Solution (mM)",
            fontsize=10,
            fontweight="bold",
        )
        ax1.set_ylabel(
            "Lithium Concentration, \nMembrane (mM)",
            fontsize=10,
            fontweight="bold",
        )
        ax1.set_xlim(left=0, right=110)
        ax1.set_ylim(bottom=0, top=180)
        ax1.tick_params(direction="in", right=True, labelsize=10)
        ax1.legend(loc="best", ncol=2, title="$H_{Li},H_{Cl}$")

        for h1 in h_1_vals:
            for h2 in h_2_vals:
                m = single_asymmetric_salt_partitioning_model(h1=h1, h2=h2)
                for c1 in c_1_sol_vals:
                    m.conc_1_sol.fix(c1)
                    solve_model(m)
                    c_1_mem_vals.append(value(m.conc_1_mem))
                if (
                    (h1 == 1 and h2 == 0.5)
                    or (h1 == 1.5 and h2 == 0.5)
                    or (h1 == 1.5 and h2 == 1)
                ):
                    ax2.plot(
                        c_1_sol_vals,
                        c_1_mem_vals,
                        linewidth=2,
                        linestyle="dashed",
                        label=f"{h1},{h2}",
                    )
                else:
                    ax2.plot(
                        c_1_sol_vals, c_1_mem_vals, linewidth=2, label=f"{h1},{h2}"
                    )
                c_1_mem_vals = []
        ax2.plot([0, 300], [0, 300], "k--")

        ax2.set_title("Cobalt Chloride Only", fontweight="bold")
        ax2.set_xlabel(
            "Cobalt Concentration, Solution (mM)",
            fontsize=10,
            fontweight="bold",
        )
        ax2.set_ylabel(
            "Cobalt Concentration, \nMembrane (mM)",
            fontsize=10,
            fontweight="bold",
        )
        ax2.set_xlim(left=0, right=110)
        ax2.set_ylim(bottom=0, top=180)
        ax2.tick_params(direction="in", right=True, labelsize=10)
        ax2.legend(loc="best", ncol=2, title="$H_{Co},H_{Cl}$")

        plt.show()

    if double:
        # now look at the two salts together, assuming no interaction
        c_1_sol_vals = np.arange(5, 105, 5)  # mM
        c_2_sol_vals = np.arange(5, 105, 5)  # mM
        h_1_vals = np.arange(0.5, 2, 0.5)
        h_2_vals = np.arange(0.5, 2, 0.5)
        # h_3_vals = np.arange(0.5,2,0.5)

        c_1_mem_vals = []
        c_2_mem_vals = []

        fig2, (ax3, ax4) = plt.subplots(1, 2, dpi=125, figsize=(10, 5))
        fig2.suptitle("Lithium Chloride, Cobalt Chloride System", fontweight="bold")

        for h1 in h_1_vals:
            for h2 in h_2_vals:
                m = double_salt_independent_partitioning_model(h1=h1, h2=h2, h3=1)
                for c1 in c_1_sol_vals:
                    m.conc_1_sol.fix(c1)
                    m.conc_2_sol.fix(50)
                    solve_model(m)
                    c_1_mem_vals.append(value(m.conc_1_mem))
                ax3.plot(c_1_sol_vals, c_1_mem_vals, linewidth=2, label=f"{h1},{h2}")
                c_1_mem_vals = []
        ax3.plot([0, 300], [0, 300], "k--")

        ax3.set_title(
            "Cobalt Chloride Concentration = 50 mM \n $H_{Cl}$ = 1", fontweight="bold"
        )
        ax3.set_xlabel(
            "Lithium Concentration, Solution (mM)",
            fontsize=10,
            fontweight="bold",
        )
        ax3.set_ylabel(
            "Lithium Concentration, \nMembrane (mM)",
            fontsize=10,
            fontweight="bold",
        )
        ax3.set_xlim(left=0, right=110)
        ax3.set_ylim(bottom=0, top=180)
        ax3.tick_params(direction="in", right=True, labelsize=10)
        ax3.legend(loc="best", ncol=2, title="$H_{Li},H_{Co}$")

        for h1 in h_1_vals:
            for h2 in h_2_vals:
                m = double_salt_independent_partitioning_model(h1=h1, h2=h2, h3=1)
                for c2 in c_2_sol_vals:
                    m.conc_1_sol.fix(50)
                    m.conc_2_sol.fix(c2)
                    solve_model(m)
                    c_2_mem_vals.append(value(m.conc_2_mem))
                ax4.plot(c_2_sol_vals, c_2_mem_vals, linewidth=2, label=f"{h1},{h2}")
                c_2_mem_vals = []
        ax4.plot([0, 300], [0, 300], "k--")

        ax4.set_title(
            "Lithium Chloride Concentration = 50 mM \n $H_{Cl}$ = 1", fontweight="bold"
        )
        ax4.set_xlabel(
            "Cobalt Concentration, Solution (mM)",
            fontsize=10,
            fontweight="bold",
        )
        ax4.set_ylabel(
            "Cobalt Concentration, \nMembrane (mM)",
            fontsize=10,
            fontweight="bold",
        )
        ax4.set_xlim(left=0, right=110)
        ax4.set_ylim(bottom=0, top=180)
        ax4.tick_params(direction="in", right=True, labelsize=10)
        ax4.legend(loc="best", ncol=2, title="$H_{Li},H_{Co}$")

        plt.show()


if __name__ == "__main__":
    main()
