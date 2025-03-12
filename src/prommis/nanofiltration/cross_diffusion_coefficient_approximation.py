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
    (D_11_df, D_12_df, D_21_df, D_22_df) = calculate_diffusion_coefficients()
    calculate_linearized_diffusion_coefficients(D_11_df, D_12_df, D_21_df, D_22_df)
    # plot_2D_diffusion_coefficients(D_11_df, D_12_df, D_21_df, D_22_df)
    # plot_3D_with_regression()


def calculate_D_11(z1, z2, z3, D1, D2, D3, c1, c2):
    D_11 = (
        (z1 * z3 * D1 * D3 - (z1**2) * D1 * D3) * c1
        + (z2 * z3 * D1 * D3 - (z2**2) * D1 * D2) * c2
    ) / (((z1**2) * D1 - z1 * z3 * D3) * c1 + ((z2**2) * D2 - z2 * z3 * D3) * c2)
    return D_11


def calculate_D_12(z1, z2, z3, D1, D2, D3, c1, c2):
    D_12 = ((z1 * z2 * D1 * D2 - z1 * z2 * D1 * D3) * c1) / (
        ((z1**2) * D1 - z1 * z3 * D3) * c1 + ((z2**2) * D2 - z2 * z3 * D3) * c2
    )
    return D_12


def calculate_D_21(z1, z2, z3, D1, D2, D3, c1, c2):
    D_21 = ((z1 * z2 * D1 * D2 - z1 * z2 * D2 * D3) * c2) / (
        ((z1**2) * D1 - z1 * z3 * D3) * c1 + ((z2**2) * D2 - z2 * z3 * D3) * c2
    )
    return D_21


def calculate_D_22(z1, z2, z3, D1, D2, D3, c1, c2):
    D_22 = (
        (z1 * z2 * D2 * D3 - (z1**2) * D1 * D2) * c1
        + (z2 * z3 * D2 * D3 - (z2**2) * D2 * D3) * c2
    ) / (((z1**2) * D1 - z1 * z3 * D3) * c1 + ((z2**2) * D2 - z2 * z3 * D3) * c2)
    return D_22


def set_parameter_values():
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.7e-6  # m2/h
    D2 = 2.64e-6  # m2/h
    D3 = 7.3e-6  # m2/h

    return (z1, z2, z3, D1, D2, D3)


def set_concentration_ranges():
    c1_vals = np.arange(0.1, 5, 0.1)
    c2_vals = np.arange(4, 8, 0.1)

    return (c1_vals, c2_vals)


def calculate_diffusion_coefficients():
    (z1, z2, z3, D1, D2, D3) = set_parameter_values()
    (c1_vals, c2_vals) = set_concentration_ranges()

    c2_list = []
    D_11_vals = []
    D_12_vals = []
    D_21_vals = []
    D_22_vals = []
    d11 = {}
    d12 = {}
    d21 = {}
    d22 = {}

    for c2 in c2_vals:
        c2_list.append(c2.round(1))

    for c1 in c1_vals:
        for c2 in c2_vals:
            D_11_vals.append((calculate_D_11(z1, z2, z3, D1, D2, D3, c1, c2)))
            D_12_vals.append((calculate_D_12(z1, z2, z3, D1, D2, D3, c1, c2)))
            D_21_vals.append((calculate_D_21(z1, z2, z3, D1, D2, D3, c1, c2)))
            D_22_vals.append((calculate_D_22(z1, z2, z3, D1, D2, D3, c1, c2)))
        d11[f"{c1.round(1)}"] = D_11_vals
        d12[f"{c1.round(1)}"] = D_12_vals
        d21[f"{c1.round(1)}"] = D_21_vals
        d22[f"{c1.round(1)}"] = D_22_vals
        D_11_vals = []
        D_12_vals = []
        D_21_vals = []
        D_22_vals = []

    D_11_df = DataFrame(index=c2_list, data=d11)
    D_12_df = DataFrame(index=c2_list, data=d12)
    D_21_df = DataFrame(index=c2_list, data=d21)
    D_22_df = DataFrame(index=c2_list, data=d22)

    return (D_11_df, D_12_df, D_21_df, D_22_df)


def plot_2D_diffusion_coefficients(D_11_df, D_12_df, D_21_df, D_22_df):
    figs, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    sns.heatmap(
        ax=ax1,
        data=D_11_df,
        cmap="mako",
    )
    ax1.tick_params(axis="x", labelrotation=45)
    ax1.set_xlabel("Lithium Concentration (kg/m3)")
    ax1.set_ylabel("Cobalt Concentration (kg/m3)")
    ax1.invert_yaxis()
    ax1.set_title("D_11 (m2/h)")

    sns.heatmap(
        ax=ax2,
        data=D_12_df,
        cmap="mako",
    )
    ax2.tick_params(axis="x", labelrotation=45)
    ax2.set_xlabel("Lithium Concentration (kg/m3)")
    ax2.set_ylabel("Cobalt Concentration (kg/m3)")
    ax2.invert_yaxis()
    ax2.set_title("D_12 (m2/h)")

    sns.heatmap(
        ax=ax3,
        data=D_21_df,
        cmap="mako",
    )
    ax3.tick_params(axis="x", labelrotation=45)
    ax3.set_xlabel("Lithium Concentration (kg/m3)")
    ax3.set_ylabel("Cobalt Concentration (kg/m3)")
    ax3.invert_yaxis()
    ax3.set_title("D_21 (m2/h)")

    sns.heatmap(
        ax=ax4,
        data=D_22_df,
        cmap="mako",
    )
    ax4.tick_params(axis="x", labelrotation=45)
    ax4.set_xlabel("Lithium Concentration (kg/m3)")
    ax4.set_ylabel("Cobalt Concentration (kg/m3)")
    ax4.invert_yaxis()
    ax4.set_title("D_22 (m2/h)")

    plt.show()


def plot_3D_diffusion_coefficients():
    ax = plt.figure().add_subplot(projection="3d")

    (z1, z2, z3, D1, D2, D3) = set_parameter_values()
    (c1_vals, c2_vals) = set_concentration_ranges()

    c1, c2 = np.meshgrid(c1_vals, c2_vals)
    D_11 = (
        (z1 * z3 * D1 * D3 - (z1**2) * D1 * D3) * c1
        + (z2 * z3 * D1 * D3 - (z2**2) * D1 * D2) * c2
    ) / (((z1**2) * D1 - z1 * z3 * D3) * c1 + ((z2**2) * D2 - z2 * z3 * D3) * c2)

    ax.plot_surface(
        c1,
        c2,
        D_11,
        cmap="mako",
    )

    ax.set_xlabel("Lithium Concentration (kg/m3)")
    ax.set_ylabel("Cobalt Concentration (kg/m3)")

    plt.show()


def linear_regression(cross_diffusion_dataframe):
    m = ConcreteModel()

    m.beta_0 = Var(initialize=0)
    m.beta_1 = Var(initialize=1)
    m.beta_2 = Var(initialize=1)

    m.c1_data = Set(initialize=[float(c1) for c1 in cross_diffusion_dataframe.columns])
    m.c2_data = Set(initialize=[float(c2) for c2 in cross_diffusion_dataframe.index])

    m.diffusion_prediction = Var(m.c1_data, m.c2_data, initialize=1e-6)

    def diffusion_calculation(m, c1, c2):
        return m.diffusion_prediction[c1, c2] == (
            m.beta_0 + m.beta_1 * c1 + m.beta_2 * c2
        )

    m.model_eqn = Constraint(m.c1_data, m.c2_data, rule=diffusion_calculation)

    residual = 0
    for c1 in cross_diffusion_dataframe.columns:
        for c2 in cross_diffusion_dataframe.index:
            residual += (
                m.diffusion_prediction[float(c1), float(c2)]
                - cross_diffusion_dataframe.loc[c2][c1]
            ) ** 2

    m.objective = Objective(expr=residual)

    solver = SolverFactory("ipopt")
    solver.solve(m, tee=True)

    return (m.beta_0.value, m.beta_1.value, m.beta_2.value)


def calculate_linearized_diffusion_coefficients(D_11_df, D_12_df, D_21_df, D_22_df):

    (c1_vals, c2_vals) = set_concentration_ranges()

    c2_list = []

    D_11_vals = []
    D_12_vals = []
    D_21_vals = []
    D_22_vals = []

    d11 = {}
    d12 = {}
    d21 = {}
    d22 = {}

    for c2 in c2_vals:
        c2_list.append(c2.round(1))

    (beta_0_D_11, beta_1_D_11, beta_2_D_11) = linear_regression(D_11_df)
    (beta_0_D_12, beta_1_D_12, beta_2_D_12) = linear_regression(D_12_df)
    (beta_0_D_21, beta_1_D_21, beta_2_D_21) = linear_regression(D_21_df)
    (beta_0_D_22, beta_1_D_22, beta_2_D_22) = linear_regression(D_22_df)

    print(" \t beta_0 (m2/h) \t beta_1 (m5/kg/h) \t beta_2 (m5/kg/h)")
    print(
        f"D_11 \t {round(beta_0_D_11,8)} \t {round(beta_1_D_11,10)} \t\t {round(beta_2_D_11,10)}"
    )
    print(
        f"D_12 \t {round(beta_0_D_12,9)} \t {round(beta_1_D_12,9)} \t\t {round(beta_2_D_12,10)}"
    )
    print(
        f"D_21 \t {round(beta_0_D_21,9)} \t {round(beta_1_D_21,10)} \t\t {round(beta_2_D_21,10)}"
    )
    print(
        f"D_22 \t {round(beta_0_D_22,8)} \t {round(beta_1_D_22,9)} \t\t {round(beta_2_D_22,9)}"
    )

    for c1 in c1_vals:
        for c2 in c2_vals:
            D_11_vals.append(beta_0_D_11 + beta_1_D_11 * c1 + beta_2_D_11 * c2)
            D_12_vals.append(beta_0_D_12 + beta_1_D_12 * c1 + beta_2_D_12 * c2)
            D_21_vals.append(beta_0_D_21 + beta_1_D_21 * c1 + beta_2_D_21 * c2)
            D_22_vals.append(beta_0_D_22 + beta_1_D_22 * c1 + beta_2_D_22 * c2)
        d11[f"{c1.round(1)}"] = D_11_vals
        d12[f"{c1.round(1)}"] = D_12_vals
        d21[f"{c1.round(1)}"] = D_21_vals
        d22[f"{c1.round(1)}"] = D_22_vals
        D_11_vals = []
        D_12_vals = []
        D_21_vals = []
        D_22_vals = []

    D_11_df_linearized = DataFrame(index=c2_list, data=d11)
    D_12_df_linearized = DataFrame(index=c2_list, data=d12)
    D_21_df_linearized = DataFrame(index=c2_list, data=d21)
    D_22_df_linearized = DataFrame(index=c2_list, data=d22)

    return (
        D_11_df_linearized,
        D_12_df_linearized,
        D_21_df_linearized,
        D_22_df_linearized,
    )


def plot_3D_with_regression():
    (z1, z2, z3, D1, D2, D3) = set_parameter_values()
    (c1_vals, c2_vals) = set_concentration_ranges()

    c1, c2 = np.meshgrid(c1_vals, c2_vals)
    D_11 = calculate_D_11(z1, z2, z3, D1, D2, D3, c1, c2)
    D_12 = calculate_D_12(z1, z2, z3, D1, D2, D3, c1, c2)
    D_21 = calculate_D_21(z1, z2, z3, D1, D2, D3, c1, c2)
    D_22 = calculate_D_22(z1, z2, z3, D1, D2, D3, c1, c2)

    (D_11_df, D_12_df, D_21_df, D_22_df) = calculate_diffusion_coefficients()
    (D_11_df_linearized, D_12_df_linearized, D_21_df_linearized, D_22_df_linearized) = (
        calculate_linearized_diffusion_coefficients(D_11_df, D_12_df, D_21_df, D_22_df)
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
    ax1.set_xlabel("Lithium Concentration (kg/m3)")
    ax1.set_ylabel("Cobalt Concentration (kg/m3)")
    ax1.set_title("D_11 (m2/h)")

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
    ax2.set_xlabel("Lithium Concentration (kg/m3)")
    ax2.set_ylabel("Cobalt Concentration (kg/m3)")
    ax2.set_title("D_12 (m2/h)")

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
    ax3.set_xlabel("Lithium Concentration (kg/m3)")
    ax3.set_ylabel("Cobalt Concentration (kg/m3)")
    ax3.set_title("D_21 (m2/h)")

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
    ax4.set_xlabel("Lithium Concentration (kg/m3)")
    ax4.set_ylabel("Cobalt Concentration (kg/m3)")
    ax4.set_title("D_22 (m2/h)")

    plt.show()


if __name__ == "__main__":
    main()
