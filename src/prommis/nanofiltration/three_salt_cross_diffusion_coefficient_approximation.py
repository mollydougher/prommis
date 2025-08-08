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
    assert_optimal_termination,
)


def main(plot=False):
    for chi_val in [0, -140]:
        report_linear_regression(chi_val)

        if plot:
            (
                D_11_df,
                D_12_df,
                D_13_df,
                D_21_df,
                D_22_df,
                D_23_df,
                D_31_df,
                D_32_df,
                D_33_df,
                alpha_1_df,
                alpha_2_df,
                alpha_3_df,
            ) = calculate_diffusion_coefficients(chi=chi_val)

            # calculate_linearized_diffusion_coefficients(
            #     D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df, chi_val
            # )

            plot_2D_diffusion_coefficients(
                D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df
            )
            # plot_3D_diffusion_coefficients(chi=chi_val)


def calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = (
        ((((z1**2) * D1) - (z1 * z4 * D4)) * c1)
        + ((((z2**2) * D2) - (z2 * z4 * D4)) * c2)
        + ((((z3**2) * D3) - (z3 * z4 * D4)) * c3)
        - (z4 * D4 * chi)
    )
    return D_denom


def calculate_D_11(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_11 = (
        (((z1 * z4 * D1 * D4) - ((z1**2) * D1 * D4)) * c1)
        + (((z2 * z4 * D1 * D4) - ((z2**2) * D1 * D2)) * c2)
        + (((z3 * z4 * D1 * D4) - ((z3**2) * D1 * D3)) * c3)
        + (z4 * D1 * D4 * chi)
    ) / D_denom
    return D_11


def calculate_D_12(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_12 = (((z1 * z2 * D1 * D2) - (z1 * z2 * D1 * D4)) * c1) / D_denom
    return D_12


def calculate_D_13(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_13 = (((z1 * z3 * D1 * D3) - (z1 * z3 * D1 * D4)) * c1) / D_denom
    return D_13


def calculate_D_21(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_21 = (((z1 * z2 * D1 * D2) - (z1 * z2 * D2 * D4)) * c2) / D_denom
    return D_21


def calculate_D_22(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_22 = (
        (((z1 * z4 * D2 * D4) - ((z1**2) * D1 * D2)) * c1)
        + (((z2 * z4 * D2 * D4) - ((z2**2) * D2 * D4)) * c2)
        + (((z3 * z4 * D2 * D4) - ((z3**2) * D2 * D3)) * c3)
        + (z4 * D2 * D4 * chi)
    ) / D_denom
    return D_22


def calculate_D_23(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_23 = (((z2 * z3 * D2 * D3) - (z2 * z3 * D2 * D4)) * c2) / D_denom
    return D_23


def calculate_D_31(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_31 = (((z1 * z3 * D1 * D3) - (z1 * z3 * D3 * D4)) * c3) / D_denom
    return D_31


def calculate_D_32(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_32 = (((z2 * z3 * D2 * D3) - (z2 * z3 * D3 * D4)) * c3) / D_denom
    return D_32


def calculate_D_33(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    D_33 = (
        (((z1 * z4 * D3 * D4) - ((z1**2) * D1 * D3)) * c1)
        + (((z2 * z4 * D3 * D4) - ((z2**2) * D2 * D3)) * c2)
        + (((z3 * z4 * D3 * D4) - ((z3**2) * D3 * D4)) * c3)
        + (z4 * D3 * D4 * chi)
    ) / D_denom
    return D_33


def calculate_alpha_1(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    alpha_1 = 1 + (z1 * D1 * chi) / D_denom
    return alpha_1


def calculate_alpha_2(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    alpha_2 = 1 + (z2 * D2 * chi) / D_denom
    return alpha_2


def calculate_alpha_3(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)
    alpha_3 = 1 + (z3 * D3 * chi) / D_denom
    return alpha_3


def set_parameter_values_and_concentration_ranges(chi):
    z1 = 1
    z2 = 2
    z3 = 3
    z4 = -1

    # TODO: pick consistent number of signiicant digits
    D1 = 3.7e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 2.01e-6  # m2/h (aluminum)
    D4 = 7.3e-6  # m2/h (chloride)

    chi = chi

    if chi == 0:
        c1_vals = np.arange(50, 81, 1)  # mol/m3 = mM
        c2_vals = np.arange(50, 81, 1)  # mol/m3 = mM
        c3_vals = np.arange(50, 81, 1)  # mol/m3 = mM
        # c3_vals = [50]  # mol/m3 = mM
    elif chi == -140:
        c1_vals = np.arange(50, 81, 1)  # mol/m3 = mM
        c2_vals = np.arange(80, 111, 1)  # mol/m3 = mM
        c3_vals = np.arange(80, 111, 1)  # mol/m3 = mM
        # c3_vals = [50]  # mol/m3 = mM

    return (z1, z2, z3, z4, D1, D2, D3, D4, chi, c1_vals, c2_vals, c3_vals)


def calculate_diffusion_coefficients(chi):
    (z1, z2, z3, z4, D1, D2, D3, D4, chi, c1_vals, c2_vals, c3_vals) = (
        set_parameter_values_and_concentration_ranges(chi=chi)
    )

    c2_list = []
    c3_list = []
    D_11_vals = []
    D_12_vals = []
    D_13_vals = []
    D_21_vals = []
    D_22_vals = []
    D_23_vals = []
    D_31_vals = []
    D_32_vals = []
    D_33_vals = []
    alpha_1_vals = []
    alpha_2_vals = []
    alpha_3_vals = []
    d11 = {}
    d12 = {}
    d13 = {}
    d21 = {}
    d22 = {}
    d23 = {}
    d31 = {}
    d32 = {}
    d33 = {}
    alpha1 = {}
    alpha2 = {}
    alpha3 = {}

    for c2 in c2_vals:
        c2_list.append(c2.round(1))

    c3 = 50  # look at a single value for now
    for c1 in c1_vals:
        for c2 in c2_vals:
            D_11_vals.append(
                (calculate_D_11(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            D_12_vals.append(
                (calculate_D_12(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            D_13_vals.append(
                (calculate_D_13(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            D_21_vals.append(
                (calculate_D_21(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            D_22_vals.append(
                (calculate_D_22(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            D_23_vals.append(
                (calculate_D_23(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            D_31_vals.append(
                (calculate_D_31(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            D_32_vals.append(
                (calculate_D_32(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            D_33_vals.append(
                (calculate_D_33(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            alpha_1_vals.append(
                (calculate_alpha_1(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            alpha_2_vals.append(
                (calculate_alpha_2(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
            alpha_3_vals.append(
                (calculate_alpha_3(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi))
            )
        d11[f"{c1.round(1)}"] = D_11_vals
        d12[f"{c1.round(1)}"] = D_12_vals
        d13[f"{c1.round(1)}"] = D_13_vals
        d21[f"{c1.round(1)}"] = D_21_vals
        d22[f"{c1.round(1)}"] = D_22_vals
        d23[f"{c1.round(1)}"] = D_23_vals
        d31[f"{c1.round(1)}"] = D_31_vals
        d32[f"{c1.round(1)}"] = D_32_vals
        d33[f"{c1.round(1)}"] = D_33_vals
        alpha1[f"{c1.round(1)}"] = alpha_1_vals
        alpha2[f"{c1.round(1)}"] = alpha_2_vals
        alpha3[f"{c1.round(1)}"] = alpha_3_vals
        D_11_vals = []
        D_12_vals = []
        D_13_vals = []
        D_21_vals = []
        D_22_vals = []
        D_23_vals = []
        D_31_vals = []
        D_32_vals = []
        D_33_vals = []
        alpha_1_vals = []
        alpha_2_vals = []
        alpha_3_vals = []

    D_11_df = DataFrame(index=c2_list, data=d11)
    D_12_df = DataFrame(index=c2_list, data=d12)
    D_13_df = DataFrame(index=c2_list, data=d13)
    D_21_df = DataFrame(index=c2_list, data=d21)
    D_22_df = DataFrame(index=c2_list, data=d22)
    D_23_df = DataFrame(index=c2_list, data=d23)
    D_31_df = DataFrame(index=c2_list, data=d31)
    D_32_df = DataFrame(index=c2_list, data=d32)
    D_33_df = DataFrame(index=c2_list, data=d33)
    alpha_1_df = DataFrame(index=c2_list, data=alpha1)
    alpha_2_df = DataFrame(index=c2_list, data=alpha2)
    alpha_3_df = DataFrame(index=c2_list, data=alpha3)

    return (
        D_11_df,
        D_12_df,
        D_13_df,
        D_21_df,
        D_22_df,
        D_23_df,
        D_31_df,
        D_32_df,
        D_33_df,
        alpha_1_df,
        alpha_2_df,
        alpha_3_df,
    )


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


def linear_regression(coeff_function_calc, chi, tee=False):
    """
    D_pred = beta_0 + beta_1*c1 + beta_2*c2 + beta_3*c3
    """
    (z1, z2, z3, z4, D1, D2, D3, D4, chi, c1_vals, c2_vals, c3_vals) = (
        set_parameter_values_and_concentration_ranges(chi)
    )

    m = ConcreteModel()

    m.beta_0 = Var(initialize=0)
    m.beta_1 = Var(initialize=1)
    m.beta_2 = Var(initialize=1)
    m.beta_3 = Var(initialize=1)

    m.c1_data = Set(initialize=[c1_val for c1_val in c1_vals])
    m.c2_data = Set(initialize=[c2_val for c2_val in c2_vals])
    m.c3_data = Set(initialize=[c3_val for c3_val in c3_vals])

    m.diffusion_prediction = Var(m.c1_data, m.c2_data, m.c3_data, initialize=1e-6)

    def diffusion_calculation(m, c1, c2, c3):
        return m.diffusion_prediction[c1, c2, c3] == (
            m.beta_0 + (m.beta_1 * c1) + (m.beta_2 * c2) + (m.beta_3 * c3)
        )

    m.model_eqn = Constraint(
        m.c1_data, m.c2_data, m.c3_data, rule=diffusion_calculation
    )

    m.diffusion_actual = Var(m.c1_data, m.c2_data, m.c3_data, initialize=1e-6)

    def diffusion_calculation_actual(m, c1, c2, c3):
        return m.diffusion_actual[c1, c2, c3] == coeff_function_calc(
            z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
        )

    m.model_eqn_actual = Constraint(
        m.c1_data, m.c2_data, m.c3_data, rule=diffusion_calculation_actual
    )

    residual = 0
    for c1 in m.c1_data:
        for c2 in m.c2_data:
            for c3 in m.c3_data:
                residual += (
                    m.diffusion_prediction[c1, c2, c3] - m.diffusion_actual[c1, c2, c3]
                ) ** 2

    m.objective = Objective(expr=residual)

    solver = SolverFactory("ipopt")
    results = solver.solve(m, tee=tee)
    assert_optimal_termination(results)

    return (m.beta_0.value, m.beta_1.value, m.beta_2.value, m.beta_3.value)


def report_linear_regression(chi):

    (z1, z2, z3, z4, D1, D2, D3, D4, chi, c1_vals, c2_vals, c3_vals) = (
        set_parameter_values_and_concentration_ranges(chi)
    )

    c1_list = []
    c2_list = []
    c3_list = []

    for c1 in c1_vals:
        c1_list.append(c1.round(1))
    for c2 in c2_vals:
        c2_list.append(c2.round(1))
    for c3 in c3_vals:
        c3_list.append(c3.round(1))

    (beta_0_D_11, beta_1_D_11, beta_2_D_11, beta_3_D_11) = linear_regression(
        calculate_D_11, chi
    )
    (beta_0_D_12, beta_1_D_12, beta_2_D_12, beta_3_D_12) = linear_regression(
        calculate_D_12, chi
    )
    (beta_0_D_13, beta_1_D_13, beta_2_D_13, beta_3_D_13) = linear_regression(
        calculate_D_13, chi
    )

    (beta_0_D_21, beta_1_D_21, beta_2_D_21, beta_3_D_21) = linear_regression(
        calculate_D_21, chi
    )
    (beta_0_D_22, beta_1_D_22, beta_2_D_22, beta_3_D_22) = linear_regression(
        calculate_D_22, chi
    )
    (beta_0_D_23, beta_1_D_23, beta_2_D_23, beta_3_D_23) = linear_regression(
        calculate_D_23, chi
    )

    (beta_0_D_31, beta_1_D_31, beta_2_D_31, beta_3_D_31) = linear_regression(
        calculate_D_31, chi
    )
    (beta_0_D_32, beta_1_D_32, beta_2_D_32, beta_3_D_32) = linear_regression(
        calculate_D_32, chi
    )
    (beta_0_D_33, beta_1_D_33, beta_2_D_33, beta_3_D_33) = linear_regression(
        calculate_D_33, chi
    )

    (beta_0_alpha_1, beta_1_alpha_1, beta_2_alpha_1, beta_3_alpha_1) = (
        linear_regression(calculate_alpha_1, chi)
    )
    (beta_0_alpha_2, beta_1_alpha_2, beta_2_alpha_2, beta_3_alpha_2) = (
        linear_regression(calculate_alpha_2, chi)
    )
    (beta_0_alpha_3, beta_1_alpha_3, beta_2_alpha_3, beta_3_alpha_3) = (
        linear_regression(calculate_alpha_3, chi)
    )

    print(
        "=========================================================================================="
    )
    print(f"chi = {chi} mM")
    print(f"lithium conc range = {c1_list[0]} to {c1_list[-1]} mM")
    print(f"cobalt conc range = {c2_list[0]} to {c2_list[-1]} mM")
    print(f"aluminum conc range = {c3_list[0]} to {c3_list[-1]} mM")
    print(
        "------------------------------------------------------------------------------------------"
    )
    print(
        " \t beta_0 (m2/h) \t beta_1 (m5/mol/h) \t beta_2 (m5/mol/h) \t beta_3 (m5/mol/h)"
    )
    print(
        "------------------------------------------------------------------------------------------"
    )
    print(
        f"D_11 \t {round(beta_0_D_11,12)} \t {round(beta_1_D_11,12)} \t\t {round(beta_2_D_11,12)} \t\t {round(beta_3_D_11,12)}"
    )
    print(
        f"D_12 \t {round(beta_0_D_12,12)} \t {round(beta_1_D_12,12)} \t\t {round(beta_2_D_12,12)} \t\t {round(beta_3_D_12,12)}"
    )
    print(
        f"D_13 \t {round(beta_0_D_13,12)} \t {round(beta_1_D_13,12)} \t\t {round(beta_2_D_13,12)} \t\t {round(beta_3_D_13,12)}"
    )
    print(
        f"D_21 \t {round(beta_0_D_21,12)} \t {round(beta_1_D_21,12)} \t\t {round(beta_2_D_21,12)} \t\t {round(beta_3_D_21,12)}"
    )
    print(
        f"D_22 \t {round(beta_0_D_22,12)} \t {round(beta_1_D_22,12)} \t\t {round(beta_2_D_22,12)} \t\t {round(beta_3_D_22,12)}"
    )
    print(
        f"D_23 \t {round(beta_0_D_23,12)} \t {round(beta_1_D_23,12)} \t\t {round(beta_2_D_23,12)} \t\t {round(beta_3_D_23,12)}"
    )
    print(
        f"D_31 \t {round(beta_0_D_31,12)} \t {round(beta_1_D_31,12)} \t\t {round(beta_2_D_31,12)} \t\t {round(beta_3_D_31,12)}"
    )
    print(
        f"D_32 \t {round(beta_0_D_32,12)} \t {round(beta_1_D_32,12)} \t\t {round(beta_2_D_32,12)} \t\t {round(beta_3_D_32,12)}"
    )
    print(
        f"D_33 \t {round(beta_0_D_33,12)} \t {round(beta_1_D_33,12)} \t\t {round(beta_2_D_33,12)} \t\t {round(beta_3_D_33,12)}"
    )
    print(
        "------------------------------------------------------------------------------------------"
    )
    print(" \t omega_0 \t omega_1 (m3/mol) \t omega_2 (m3/mol) \t omega_3 (m3/mol)")
    print(
        "------------------------------------------------------------------------------------------"
    )
    print(
        f"alpha_1 \t {round(beta_0_alpha_1, 6)} \t {round(beta_1_alpha_1, 6)} \t\t {round(beta_2_alpha_1, 6)} \t\t {round(beta_3_alpha_1, 6)}"
    )
    print(
        f"alpha_2 \t {round(beta_0_alpha_2, 6)} \t {round(beta_1_alpha_2, 6)} \t\t {round(beta_2_alpha_2, 6)} \t\t {round(beta_3_alpha_2, 6)}"
    )
    print(
        f"alpha_3 \t {round(beta_0_alpha_3, 6)} \t {round(beta_1_alpha_3, 6)} \t\t {round(beta_2_alpha_3, 6)} \t\t {round(beta_3_alpha_3, 6)}"
    )
    print(
        "=========================================================================================="
    )


# def calculate_linearized_diffusion_coefficients(
#     D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df, chi
# ):

#     (z1, z2, z3, D1, D2, D3, chi, c1_vals, c2_vals) = (
#         set_parameter_values_and_concentration_ranges(chi=chi)
#     )

#     c1_list = []
#     c2_list = []

#     D_11_vals = []
#     D_12_vals = []
#     D_21_vals = []
#     D_22_vals = []
#     alpha_1_vals = []
#     alpha_2_vals = []

#     d11 = {}
#     d12 = {}
#     d21 = {}
#     d22 = {}
#     alpha1 = {}
#     alpha2 = {}

#     for c1 in c1_vals:
#         c1_list.append(c1.round(1))
#     for c2 in c2_vals:
#         c2_list.append(c2.round(1))

#     (beta_0_D_11, beta_1_D_11, beta_2_D_11) = linear_regression(D_11_df)
#     (beta_0_D_12, beta_1_D_12, beta_2_D_12) = linear_regression(D_12_df)
#     (beta_0_D_21, beta_1_D_21, beta_2_D_21) = linear_regression(D_21_df)
#     (beta_0_D_22, beta_1_D_22, beta_2_D_22) = linear_regression(D_22_df)
#     (beta_0_alpha_1, beta_1_alpha_1, beta_2_alpha_1) = linear_regression(alpha_1_df)
#     (beta_0_alpha_2, beta_1_alpha_2, beta_2_alpha_2) = linear_regression(alpha_2_df)

#     print("==================================================================")
#     print(f"chi = {chi} mM")
#     print(f"lithium conc range = {c1_list[0]} to {c1_list[-1]} mM")
#     print(f"cobalt conc range = {c2_list[0]} to {c2_list[-1]} mM")
#     print("------------------------------------------------------------------")
#     print(" \t beta_0 (m2/h) \t beta_1 (m5/mol/h) \t beta_2 (m5/mol/h)")
#     print("------------------------------------------------------------------")
#     print(
#         f"D_11 \t {round(beta_0_D_11,12)} \t {round(beta_1_D_11,12)} \t\t {round(beta_2_D_11,12)}"
#     )
#     print(
#         f"D_12 \t {round(beta_0_D_12,12)} \t {round(beta_1_D_12,12)} \t\t {round(beta_2_D_12,12)}"
#     )
#     print(
#         f"D_21 \t {round(beta_0_D_21,12)} \t {round(beta_1_D_21,12)} \t\t {round(beta_2_D_21,12)}"
#     )
#     print(
#         f"D_22 \t {round(beta_0_D_22,12)} \t {round(beta_1_D_22,12)} \t\t {round(beta_2_D_22,12)}"
#     )
#     print("------------------------------------------------------------------")
#     print(" \t omega_0 \t omega_1 (m3/mol) \t omega_2 (m3/mol)")
#     print("------------------------------------------------------------------")
#     print(
#         f"alpha_1 \t {round(beta_0_alpha_1, 6)} \t {round(beta_1_alpha_1, 6)} \t\t {round(beta_2_alpha_1, 6)}"
#     )
#     print(
#         f"alpha_2 \t {round(beta_0_alpha_2, 6)} \t {round(beta_1_alpha_2, 6)} \t\t {round(beta_2_alpha_2, 6)}"
#     )
#     print("==================================================================")

#     for c1 in c1_vals:
#         for c2 in c2_vals:
#             D_11_vals.append(beta_0_D_11 + beta_1_D_11 * c1 + beta_2_D_11 * c2)
#             D_12_vals.append(beta_0_D_12 + beta_1_D_12 * c1 + beta_2_D_12 * c2)
#             D_21_vals.append(beta_0_D_21 + beta_1_D_21 * c1 + beta_2_D_21 * c2)
#             D_22_vals.append(beta_0_D_22 + beta_1_D_22 * c1 + beta_2_D_22 * c2)
#             alpha_1_vals.append(
#                 beta_0_alpha_1 + beta_1_alpha_1 * c1 + beta_2_alpha_1 * c2
#             )
#             alpha_2_vals.append(
#                 beta_0_alpha_2 + beta_1_alpha_2 * c1 + beta_2_alpha_2 * c2
#             )
#         d11[f"{c1.round(1)}"] = D_11_vals
#         d12[f"{c1.round(1)}"] = D_12_vals
#         d21[f"{c1.round(1)}"] = D_21_vals
#         d22[f"{c1.round(1)}"] = D_22_vals
#         alpha1[f"{c1.round(1)}"] = alpha_1_vals
#         alpha2[f"{c1.round(1)}"] = alpha_2_vals
#         D_11_vals = []
#         D_12_vals = []
#         D_21_vals = []
#         D_22_vals = []
#         alpha_1_vals = []
#         alpha_2_vals = []

#     D_11_df_linearized = DataFrame(index=c2_list, data=d11)
#     D_12_df_linearized = DataFrame(index=c2_list, data=d12)
#     D_21_df_linearized = DataFrame(index=c2_list, data=d21)
#     D_22_df_linearized = DataFrame(index=c2_list, data=d22)
#     alpha_1_df_linearized = DataFrame(index=c2_list, data=alpha1)
#     alpha_2_df_linearized = DataFrame(index=c2_list, data=alpha2)

#     return (
#         D_11_df_linearized,
#         D_12_df_linearized,
#         D_21_df_linearized,
#         D_22_df_linearized,
#         alpha_1_df_linearized,
#         alpha_2_df_linearized,
#     )


# def plot_3D_diffusion_coefficients(chi):
#     (z1, z2, z3, D1, D2, D3, chi, c1_vals, c2_vals) = (
#         set_parameter_values_and_concentration_ranges(chi=chi)
#     )

#     c1, c2 = np.meshgrid(c1_vals, c2_vals)
#     D_11 = calculate_D_11(z1, z2, z3, D1, D2, D3, c1, c2, chi)
#     D_12 = calculate_D_12(z1, z2, z3, D1, D2, D3, c1, c2, chi)
#     D_21 = calculate_D_21(z1, z2, z3, D1, D2, D3, c1, c2, chi)
#     D_22 = calculate_D_22(z1, z2, z3, D1, D2, D3, c1, c2, chi)
#     alpha_1 = calculate_alpha_1(z1, z2, z3, D1, D2, D3, c1, c2, chi)
#     alpha_2 = calculate_alpha_2(z1, z2, z3, D1, D2, D3, c1, c2, chi)

#     (
#         D_11_df,
#         D_12_df,
#         D_13_df,
#         D_21_df,
#         D_22_df,
#         D_23_df,
#         D_31_df,
#         D_32_df,
#         D_33_df,
#         alpha_1_df,
#         alpha_2_df,
#         alpha_3_df,
#     ) = calculate_diffusion_coefficients(chi=chi)
#     (
#         D_11_df_linearized,
#         D_12_df_linearized,
#         D_21_df_linearized,
#         D_22_df_linearized,
#         alpha_1_df_linearized,
#         alpha_2_df_linearized,
#     ) = calculate_linearized_diffusion_coefficients(
#         D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df, chi
#     )

#     ax1 = plt.figure().add_subplot(projection="3d")
#     ax1.plot_surface(
#         c1,
#         c2,
#         D_11,
#     )
#     ax1.plot_surface(
#         c1,
#         c2,
#         D_11_df_linearized,
#     )
#     ax1.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
#     ax1.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
#     ax1.set_title("D_11 (m2/h)", fontsize=14, fontweight="bold")
#     ax1.tick_params(labelsize=12)

#     ax2 = plt.figure().add_subplot(projection="3d")
#     ax2.plot_surface(
#         c1,
#         c2,
#         D_12,
#     )
#     ax2.plot_surface(
#         c1,
#         c2,
#         D_12_df_linearized,
#     )
#     ax2.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
#     ax2.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
#     ax2.set_title("D_12 (m2/h)", fontsize=14, fontweight="bold")
#     ax2.tick_params(labelsize=12)

#     ax3 = plt.figure().add_subplot(projection="3d")
#     ax3.plot_surface(
#         c1,
#         c2,
#         D_21,
#     )
#     ax3.plot_surface(
#         c1,
#         c2,
#         D_21_df_linearized,
#     )
#     ax3.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
#     ax3.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
#     ax3.set_title("D_21 (m2/h)", fontsize=14, fontweight="bold")
#     ax3.tick_params(labelsize=12)

#     ax4 = plt.figure().add_subplot(projection="3d")
#     ax4.plot_surface(
#         c1,
#         c2,
#         D_22,
#     )
#     ax4.plot_surface(
#         c1,
#         c2,
#         D_22_df_linearized,
#     )
#     ax4.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
#     ax4.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
#     ax4.set_title("D_22 (m2/h)", fontsize=14, fontweight="bold")
#     ax4.tick_params(labelsize=12)

#     ax5 = plt.figure().add_subplot(projection="3d")
#     ax5.plot_surface(
#         c1,
#         c2,
#         alpha_1,
#     )
#     ax5.plot_surface(
#         c1,
#         c2,
#         alpha_1_df_linearized,
#     )
#     ax5.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
#     ax5.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
#     ax5.set_title("alpha_1", fontsize=14, fontweight="bold")
#     ax5.tick_params(labelsize=12)

#     ax6 = plt.figure().add_subplot(projection="3d")
#     ax6.plot_surface(
#         c1,
#         c2,
#         alpha_2,
#     )
#     ax6.plot_surface(
#         c1,
#         c2,
#         alpha_2_df_linearized,
#     )
#     ax6.set_xlabel("Lithium Concentration (mM)", fontsize=14, fontweight="bold")
#     ax6.set_ylabel("Cobalt Concentration (mM)", fontsize=14, fontweight="bold")
#     ax6.set_title("alpha_2", fontsize=14, fontweight="bold")
#     ax6.tick_params(labelsize=12)

#     plt.show()


if __name__ == "__main__":
    main()
