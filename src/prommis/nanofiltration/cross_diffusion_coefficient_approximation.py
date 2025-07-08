import idaes

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pandas import DataFrame

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    Set,
    SolverFactory,
    Var,
)


def main():
    (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df) = (
        calculate_diffusion_coefficients(chi=-140)
    )
    calculate_linearized_diffusion_coefficients(
        D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df
    )
    plot_2D_diffusion_coefficients(
        D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df
    )
    plot_3D_diffusion_coefficients()


def calculate_D_denominator(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )
    return D_denom


def calculate_D_11(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_11 = (
        (z1 * z3 * D1 * D3 - (z1**2) * D1 * D3) * c1
        + (z2 * z3 * D1 * D3 - (z2**2) * D1 * D2) * c2
        + (z3 * D1 * D3 * chi)
    ) / D_denom
    return D_11


def calculate_D_12(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_12 = ((z1 * z2 * D1 * D2 - z1 * z2 * D1 * D3) * c1) / D_denom
    return D_12


def calculate_D_21(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_21 = ((z1 * z2 * D1 * D2 - z1 * z2 * D2 * D3) * c2) / D_denom
    return D_21


def calculate_D_22(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_22 = (
        (z1 * z3 * D2 * D3 - (z1**2) * D1 * D2) * c1
        + (z2 * z3 * D2 * D3 - (z2**2) * D2 * D3) * c2
        + (z3 * D2 * D3 * chi)
    ) / D_denom
    return D_22


def calculate_alpha_1(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    alpha_1 = 1 + (z1 * D1 * chi) / D_denom
    return alpha_1


def calculate_alpha_2(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    alpha_2 = 1 + (z2 * D2 * chi) / D_denom
    return alpha_2


def set_parameter_values(chi=-140):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.7e-6  # m2/h
    D2 = 2.64e-6  # m2/h
    D3 = 7.3e-6  # m2/h

    chi = chi

    return (z1, z2, z3, D1, D2, D3, chi)


def set_concentration_ranges():
    # c1_vals = np.arange(0.1, 5, 0.1) # kg/m3
    # c2_vals = np.arange(10, 15, 0.1) # kg/m3

    c1_vals = np.arange(50, 80, 1)  # mol/m3 = mM
    c2_vals = np.arange(50, 80, 1)  # mol/m3 = mM

    return (c1_vals, c2_vals)


def calculate_diffusion_coefficients(chi=-140):
    (z1, z2, z3, D1, D2, D3, chi) = set_parameter_values(chi=chi)
    (c1_vals, c2_vals) = set_concentration_ranges()

    c2_list = []
    D_11_vals = []
    D_12_vals = []
    D_21_vals = []
    D_22_vals = []
    alpha_1_vals = []
    alpha_2_vals = []
    d11 = {}
    d12 = {}
    d21 = {}
    d22 = {}
    alpha1 = {}
    alpha2 = {}

    for c2 in c2_vals:
        c2_list.append(c2.round(1))

    for c1 in c1_vals:
        for c2 in c2_vals:
            D_11_vals.append((calculate_D_11(z1, z2, z3, D1, D2, D3, c1, c2, chi)))
            D_12_vals.append((calculate_D_12(z1, z2, z3, D1, D2, D3, c1, c2, chi)))
            D_21_vals.append((calculate_D_21(z1, z2, z3, D1, D2, D3, c1, c2, chi)))
            D_22_vals.append((calculate_D_22(z1, z2, z3, D1, D2, D3, c1, c2, chi)))
            alpha_1_vals.append(
                (calculate_alpha_1(z1, z2, z3, D1, D2, D3, c1, c2, chi))
            )
            alpha_2_vals.append(
                (calculate_alpha_2(z1, z2, z3, D1, D2, D3, c1, c2, chi))
            )
        d11[f"{c1.round(1)}"] = D_11_vals
        d12[f"{c1.round(1)}"] = D_12_vals
        d21[f"{c1.round(1)}"] = D_21_vals
        d22[f"{c1.round(1)}"] = D_22_vals
        alpha1[f"{c1.round(1)}"] = alpha_1_vals
        alpha2[f"{c1.round(1)}"] = alpha_2_vals
        D_11_vals = []
        D_12_vals = []
        D_21_vals = []
        D_22_vals = []
        alpha_1_vals = []
        alpha_2_vals = []

    D_11_df = DataFrame(index=c2_list, data=d11)
    D_12_df = DataFrame(index=c2_list, data=d12)
    D_21_df = DataFrame(index=c2_list, data=d21)
    D_22_df = DataFrame(index=c2_list, data=d22)
    alpha_1_df = DataFrame(index=c2_list, data=alpha1)
    alpha_2_df = DataFrame(index=c2_list, data=alpha2)

    return (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df)


def plot_2D_diffusion_coefficients(
    D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df
):
    figs, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, dpi=125, figsize=(14, 10)
    )
    sns.heatmap(
        ax=ax1,
        data=D_11_df,
        cmap="mako",
    )
    ax1.tick_params(axis="x", labelrotation=45)
    ax1.set_ylabel("Cobalt \nConcentration (mM)", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_title("D_11 (m2/h)", fontsize=12, fontweight="bold")
    ax1.tick_params(direction="in", labelsize=10)

    sns.heatmap(
        ax=ax2,
        data=D_12_df,
        cmap="mako",
    )
    ax2.tick_params(axis="x", labelrotation=45)
    ax2.invert_yaxis()
    ax2.set_title("D_12 (m2/h)", fontsize=12, fontweight="bold")
    ax2.tick_params(direction="in", labelsize=10)

    sns.heatmap(
        ax=ax3,
        data=D_21_df,
        cmap="mako",
    )
    ax3.tick_params(axis="x", labelrotation=45)
    ax3.set_ylabel("Cobalt \nConcentration (mM)", fontsize=13, fontweight="bold")
    ax3.invert_yaxis()
    ax3.set_title("D_21 (m2/h)", fontsize=12, fontweight="bold")
    ax3.tick_params(direction="in", labelsize=10)

    sns.heatmap(
        ax=ax4,
        data=D_22_df,
        cmap="mako",
    )
    ax4.tick_params(axis="x", labelrotation=45)
    ax4.invert_yaxis()
    ax4.set_title("D_22 (m2/h)", fontsize=12, fontweight="bold")
    ax4.tick_params(direction="in", labelsize=10)

    sns.heatmap(
        ax=ax5,
        data=alpha_1_df,
        cmap="mako",
    )
    ax5.tick_params(axis="x", labelrotation=45)
    ax5.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
    ax5.set_ylabel("Cobalt \nConcentration (mM)", fontsize=13, fontweight="bold")
    ax5.invert_yaxis()
    ax5.set_title("alpha_1", fontsize=12, fontweight="bold")
    ax5.tick_params(direction="in", labelsize=10)

    sns.heatmap(
        ax=ax6,
        data=alpha_2_df,
        cmap="mako",
    )
    ax6.tick_params(axis="x", labelrotation=45)
    ax6.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
    ax6.invert_yaxis()
    ax6.set_title("alpha_2", fontsize=12, fontweight="bold")
    ax6.tick_params(direction="in", labelsize=10)

    plt.show()


def linear_regression(dataframe):
    """
    y = beta_0 + beta_1*c1 + beta_2*c2
    """
    m = ConcreteModel()

    m.beta_0 = Var(initialize=0)
    m.beta_1 = Var(initialize=1)
    m.beta_2 = Var(initialize=1)

    m.c1_data = Set(initialize=[float(c1) for c1 in dataframe.columns])
    m.c2_data = Set(initialize=[float(c2) for c2 in dataframe.index])

    m.diffusion_prediction = Var(m.c1_data, m.c2_data, initialize=1e-6)

    def diffusion_calculation(m, c1, c2):
        return m.diffusion_prediction[c1, c2] == (
            m.beta_0 + m.beta_1 * c1 + m.beta_2 * c2
        )

    m.model_eqn = Constraint(m.c1_data, m.c2_data, rule=diffusion_calculation)

    residual = 0
    for c1 in dataframe.columns:
        for c2 in dataframe.index:
            residual += (
                m.diffusion_prediction[float(c1), float(c2)] - dataframe.loc[c2][c1]
            ) ** 2

    m.objective = Objective(expr=residual)

    solver = SolverFactory("ipopt")
    solver.solve(m, tee=True)

    return (m.beta_0.value, m.beta_1.value, m.beta_2.value)


def calculate_linearized_diffusion_coefficients(
    D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df
):

    (c1_vals, c2_vals) = set_concentration_ranges()

    c2_list = []

    D_11_vals = []
    D_12_vals = []
    D_21_vals = []
    D_22_vals = []
    alpha_1_vals = []
    alpha_2_vals = []

    d11 = {}
    d12 = {}
    d21 = {}
    d22 = {}
    alpha1 = {}
    alpha2 = {}

    for c2 in c2_vals:
        c2_list.append(c2.round(1))

    (beta_0_D_11, beta_1_D_11, beta_2_D_11) = linear_regression(D_11_df)
    (beta_0_D_12, beta_1_D_12, beta_2_D_12) = linear_regression(D_12_df)
    (beta_0_D_21, beta_1_D_21, beta_2_D_21) = linear_regression(D_21_df)
    (beta_0_D_22, beta_1_D_22, beta_2_D_22) = linear_regression(D_22_df)
    (beta_0_alpha_1, beta_1_alpha_1, beta_2_alpha_1) = linear_regression(alpha_1_df)
    (beta_0_alpha_2, beta_1_alpha_2, beta_2_alpha_2) = linear_regression(alpha_2_df)

    print(" \t beta_0 (m2/h) \t beta_1 (m5/mol/h) \t beta_2 (m5/mol/h)")
    print(
        f"D_11 \t {round(beta_0_D_11,12)} \t {round(beta_1_D_11,12)} \t\t {round(beta_2_D_11,12)}"
    )
    print(
        f"D_12 \t {round(beta_0_D_12,12)} \t {round(beta_1_D_12,12)} \t\t {round(beta_2_D_12,12)}"
    )
    print(
        f"D_21 \t {round(beta_0_D_21,12)} \t {round(beta_1_D_21,12)} \t\t {round(beta_2_D_21,12)}"
    )
    print(
        f"D_22 \t {round(beta_0_D_22,12)} \t {round(beta_1_D_22,12)} \t\t {round(beta_2_D_22,12)}"
    )
    print(f"alpha_1 \t {beta_0_alpha_1} \t {beta_1_alpha_1} \t\t {beta_2_alpha_1}")
    print(f"alpha_2 \t {beta_0_alpha_2} \t {beta_1_alpha_2} \t\t {beta_2_alpha_2}")

    for c1 in c1_vals:
        for c2 in c2_vals:
            D_11_vals.append(beta_0_D_11 + beta_1_D_11 * c1 + beta_2_D_11 * c2)
            D_12_vals.append(beta_0_D_12 + beta_1_D_12 * c1 + beta_2_D_12 * c2)
            D_21_vals.append(beta_0_D_21 + beta_1_D_21 * c1 + beta_2_D_21 * c2)
            D_22_vals.append(beta_0_D_22 + beta_1_D_22 * c1 + beta_2_D_22 * c2)
            alpha_1_vals.append(
                beta_0_alpha_1 + beta_1_alpha_1 * c1 + beta_2_alpha_1 * c2
            )
            alpha_2_vals.append(
                beta_0_alpha_2 + beta_1_alpha_2 * c1 + beta_2_alpha_2 * c2
            )
        d11[f"{c1.round(1)}"] = D_11_vals
        d12[f"{c1.round(1)}"] = D_12_vals
        d21[f"{c1.round(1)}"] = D_21_vals
        d22[f"{c1.round(1)}"] = D_22_vals
        alpha1[f"{c1.round(1)}"] = alpha_1_vals
        alpha2[f"{c1.round(1)}"] = alpha_2_vals
        D_11_vals = []
        D_12_vals = []
        D_21_vals = []
        D_22_vals = []
        alpha_1_vals = []
        alpha_2_vals = []

    D_11_df_linearized = DataFrame(index=c2_list, data=d11)
    D_12_df_linearized = DataFrame(index=c2_list, data=d12)
    D_21_df_linearized = DataFrame(index=c2_list, data=d21)
    D_22_df_linearized = DataFrame(index=c2_list, data=d22)
    alpha_1_df_linearized = DataFrame(index=c2_list, data=alpha1)
    alpha_2_df_linearized = DataFrame(index=c2_list, data=alpha2)

    return (
        D_11_df_linearized,
        D_12_df_linearized,
        D_21_df_linearized,
        D_22_df_linearized,
        alpha_1_df_linearized,
        alpha_2_df_linearized,
    )


def plot_3D_diffusion_coefficients(chi=-140):
    (z1, z2, z3, D1, D2, D3, chi) = set_parameter_values()
    (c1_vals, c2_vals) = set_concentration_ranges()

    c1, c2 = np.meshgrid(c1_vals, c2_vals)
    D_11 = calculate_D_11(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_12 = calculate_D_12(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_21 = calculate_D_21(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_22 = calculate_D_22(z1, z2, z3, D1, D2, D3, c1, c2, chi)

    (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df) = (
        calculate_diffusion_coefficients(chi=chi)
    )
    (
        D_11_df_linearized,
        D_12_df_linearized,
        D_21_df_linearized,
        D_22_df_linearized,
        alpha_1_linearized,
        alpha_2_linearized,
    ) = calculate_linearized_diffusion_coefficients(
        D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df
    )

    ax1 = plt.figure().add_subplot(projection="3d")
    ax1.plot_surface(
        c1,
        c2,
        D_11,
    )
    ax1.plot_surface(
        c1,
        c2,
        D_11_df_linearized,
    )
    ax1.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
    ax1.set_title("D_11 (m2/h)", fontsize=14, fontweight="bold")
    ax1.tick_params(labelsize=12)

    ax2 = plt.figure().add_subplot(projection="3d")
    ax2.plot_surface(
        c1,
        c2,
        D_12,
    )
    ax2.plot_surface(
        c1,
        c2,
        D_12_df_linearized,
    )
    ax2.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
    ax2.set_title("D_12 (m2/h)", fontsize=14, fontweight="bold")
    ax2.tick_params(labelsize=12)

    ax3 = plt.figure().add_subplot(projection="3d")
    ax3.plot_surface(
        c1,
        c2,
        D_21,
    )
    ax3.plot_surface(
        c1,
        c2,
        D_21_df_linearized,
    )
    ax3.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
    ax3.set_title("D_21 (m2/h)", fontsize=14, fontweight="bold")
    ax3.tick_params(labelsize=12)

    ax4 = plt.figure().add_subplot(projection="3d")
    ax4.plot_surface(
        c1,
        c2,
        D_22,
    )
    ax4.plot_surface(
        c1,
        c2,
        D_22_df_linearized,
    )
    ax4.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
    ax4.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
    ax4.set_title("D_22 (m2/h)", fontsize=14, fontweight="bold")
    ax4.tick_params(labelsize=12)

    plt.show()


if __name__ == "__main__":
    main()
