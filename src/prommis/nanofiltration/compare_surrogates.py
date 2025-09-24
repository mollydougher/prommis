from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


def main():
    plot_residuals_two_salt()
    plot_residuals_three_salt()


def plot_residuals_two_salt():
    surrogate_model_file_dict_two_salt = {
        "D_11": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d11_scaled.json",
        "D_12": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d12_scaled.json",
        "D_21": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d21_scaled.json",
        "D_22": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d22_scaled.json",
        "alpha_1": "surrogate_models/rlithium_cobalt_chloride/bf_pysmo_surrogate_alpha1.json",
        "alpha_2": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_alpha2.json",
    }

    _surrogates_obj_D_11 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_two_salt["D_11"]
    )

    c1_vals = []
    c2_vals = []
    for c1 in np.arange(50, 201, 1):
        c1_vals.append(c1)
    for c2 in np.arange(50, 201, 1):
        c2_vals.append(c2)

    linear_surrogate_vals = []
    RBF_surrogate_vals = []
    actual_vals = []
    percent_error_vals_linear = []
    percent_error_vals_rbf = []

    linear_surrogate_vals_dict = {}
    RBF_surrogate_vals_dict = {}
    actual_vals_dict = {}
    percent_error_dict_linear = {}
    percent_error_dict_rbf = {}

    for c1 in np.arange(50, 201, 1):
        for c2 in np.arange(50, 201, 1):
            linear_val = evaluate_linear_surrogate_two_salt(c1, c2)
            linear_surrogate_vals.append(linear_val)

            rbf_val = evaluate_surrogate_two_salt(
                c1, c2, -140, _surrogates_obj_D_11, "D_11_scaled"
            )
            RBF_surrogate_vals.append(rbf_val)

            actual_val = calculate_true_value_two_salt(c1, c2)
            actual_vals.append(actual_val)

            percent_error_val_linear = (
                abs(linear_val - actual_val) / abs(actual_val) * 100
            )
            percent_error_vals_linear.append(percent_error_val_linear)

            percent_error_val_rbf = abs(rbf_val - actual_val) / abs(actual_val) * 100
            percent_error_vals_rbf.append(percent_error_val_rbf)

        linear_surrogate_vals_dict[f"{c1}"] = linear_surrogate_vals
        RBF_surrogate_vals_dict[f"{c1}"] = RBF_surrogate_vals
        actual_vals_dict[f"{c1}"] = actual_vals
        percent_error_dict_linear[f"{c1}"] = percent_error_vals_linear
        percent_error_dict_rbf[f"{c1}"] = percent_error_vals_rbf

        linear_surrogate_vals = []
        RBF_surrogate_vals = []
        actual_vals = []
        percent_error_vals_linear = []
        percent_error_vals_rbf = []

    linear_surrogate_vals_df = DataFrame(index=c2_vals, data=linear_surrogate_vals_dict)
    RBF_surrogate_vals_df = DataFrame(index=c2_vals, data=RBF_surrogate_vals_dict)
    actual_vals_df = DataFrame(index=c2_vals, data=actual_vals_dict)
    percent_error_df_linear = DataFrame(index=c2_vals, data=percent_error_dict_linear)
    percent_error_df_rbf = DataFrame(index=c2_vals, data=percent_error_dict_rbf)

    # fig1, (ax1,ax2, ax3) = plt.subplots(1,3)

    # linear_plot = ax1.pcolor(c1_vals, c2_vals, linear_surrogate_vals_df)
    # ax2.pcolor(c1_vals,c2_vals,RBF_surrogate_vals_df)
    # ax3.pcolor(c1_vals,c2_vals,actual_vals_df)

    # fig1.colorbar(linear_plot, ax=[ax1,ax2,ax3])

    # plt.show()

    fig2, (ax4, ax5) = plt.subplots(1, 2)
    percent_error_plot_linear = ax4.pcolor(c1_vals, c2_vals, percent_error_df_linear)
    ax4.set_title("Percent Error (%)\nLinear Surrogate", fontsize=12, fontweight="bold")
    ax4.set_ylabel(
        "Cobalt Concentration in Membrane (mM)",
        fontsize=12,
        fontweight="bold",
    )
    percent_error_plot_rbf = ax5.pcolor(c1_vals, c2_vals, percent_error_df_rbf)
    ax5.set_title("Percent Error (%)\nRBF Surrogate", fontsize=12, fontweight="bold")

    for ax in (ax4, ax5):
        ax.tick_params(direction="in", labelsize=10)
        ax.set_xlabel(
            "Lithium Concentration\n in Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )

    ax4.plot([50, 80], [80, 80], "m-")
    ax4.plot([50, 80], [110, 110], "m-")
    ax4.plot([50, 50], [80, 110], "m-")
    ax4.plot([80, 80], [80, 110], "m-")

    fig2.colorbar(percent_error_plot_linear, ax=ax4)
    fig2.colorbar(percent_error_plot_rbf, ax=ax5)

    plt.show()


def evaluate_linear_surrogate_two_salt(c1, c2):
    val = -4.33e-06 + -4.21e-09 * c1 + 5.10e-09 * c2

    return val


def evaluate_linear_surrogate_three_salt(c1, c2, c3):
    val = -4.15e-06 + -3.61e-09 * c1 + 3.03e-09 * c2 + 4.75e-09 * c3

    return val


def plot_residuals_three_salt():
    surrogate_model_file_dict_three_salt = {
        "D_11": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d11_scaled",
        # "D_12": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d12_scaled.json",
        # "D_21": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d21_scaled.json",
        # "D_22": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_d22_scaled.json",
        # "alpha_1": "surrogate_models/rlithium_cobalt_chloride/bf_pysmo_surrogate_alpha1.json",
        # "alpha_2": "surrogate_models/lithium_cobalt_chloride/rbf_pysmo_surrogate_alpha2.json",
    }

    _surrogates_obj_D_11 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_three_salt["D_11"]
    )

    c1_vals = []
    c2_vals = []
    c3_vals = []
    for c1 in np.arange(50, 210, 10):
        c1_vals.append(c1)
    for c2 in np.arange(50, 210, 10):
        c2_vals.append(c2)
    for c3 in [75, 100, 125]:
        c3_vals.append(c3)

    linear_surrogate_vals = []
    RBF_surrogate_vals = []
    actual_vals = []
    percent_error_vals_linear = []
    percent_error_vals_rbf = []

    for c3 in [75, 100, 125]:
        for c1 in np.arange(50, 210, 10):
            for c2 in np.arange(50, 210, 10):
                linear_val = evaluate_linear_surrogate_three_salt(c1, c2, c3)
                linear_surrogate_vals.append(linear_val)

                rbf_val = evaluate_surrogate_three_salt(
                    c1, c2, c3, -140, _surrogates_obj_D_11, "D_11_scaled"
                )
                RBF_surrogate_vals.append(rbf_val)

                actual_val = calculate_true_value_three_salt(c1, c2, c3)
                actual_vals.append(actual_val)

                percent_error_val_linear = (
                    abs(linear_val - actual_val) / abs(actual_val) * 100
                )
                percent_error_vals_linear.append(percent_error_val_linear)

                percent_error_val_rbf = (
                    abs(rbf_val - actual_val) / abs(actual_val) * 100
                )
                percent_error_vals_rbf.append(percent_error_val_rbf)

            if c3 == 75:
                if c1 == 50:
                    linear_surrogate_vals_dict_l = {f"{c1})": linear_surrogate_vals}
                    RBF_surrogate_vals_dict_l = {f"{c1}": RBF_surrogate_vals}
                    actual_vals_dict_l = {f"{c1}": actual_vals}
                    percent_error_dict_linear_l = {f"{c1}": percent_error_vals_linear}
                    percent_error_dict_rbf_l = {f"{c1}": percent_error_vals_rbf}
                else:
                    linear_surrogate_vals_dict_l[f"{c1})"] = linear_surrogate_vals
                    RBF_surrogate_vals_dict_l[f"{c1}"] = RBF_surrogate_vals
                    actual_vals_dict_l[f"{c1}"] = actual_vals
                    percent_error_dict_linear_l[f"{c1}"] = percent_error_vals_linear
                    percent_error_dict_rbf_l[f"{c1}"] = percent_error_vals_rbf
            elif c3 == 100:
                if c1 == 50:
                    linear_surrogate_vals_dict_m = {f"{c1})": linear_surrogate_vals}
                    RBF_surrogate_vals_dict_m = {f"{c1}": RBF_surrogate_vals}
                    actual_vals_dict_m = {f"{c1}": actual_vals}
                    percent_error_dict_linear_m = {f"{c1}": percent_error_vals_linear}
                    percent_error_dict_rbf_m = {f"{c1}": percent_error_vals_rbf}
                else:
                    linear_surrogate_vals_dict_m[f"{c1})"] = linear_surrogate_vals
                    RBF_surrogate_vals_dict_m[f"{c1}"] = RBF_surrogate_vals
                    actual_vals_dict_m[f"{c1}"] = actual_vals
                    percent_error_dict_linear_m[f"{c1}"] = percent_error_vals_linear
                    percent_error_dict_rbf_m[f"{c1}"] = percent_error_vals_rbf
            elif c3 == 125:
                if c1 == 50:
                    linear_surrogate_vals_dict_h = {f"{c1})": linear_surrogate_vals}
                    RBF_surrogate_vals_dict_h = {f"{c1}": RBF_surrogate_vals}
                    actual_vals_dict_h = {f"{c1}": actual_vals}
                    percent_error_dict_linear_h = {f"{c1}": percent_error_vals_linear}
                    percent_error_dict_rbf_h = {f"{c1}": percent_error_vals_rbf}
                else:
                    linear_surrogate_vals_dict_h[f"{c1})"] = linear_surrogate_vals
                    RBF_surrogate_vals_dict_h[f"{c1}"] = RBF_surrogate_vals
                    actual_vals_dict_h[f"{c1}"] = actual_vals
                    percent_error_dict_linear_h[f"{c1}"] = percent_error_vals_linear
                    percent_error_dict_rbf_h[f"{c1}"] = percent_error_vals_rbf

            linear_surrogate_vals = []
            RBF_surrogate_vals = []
            actual_vals = []
            percent_error_vals_linear = []
            percent_error_vals_rbf = []

    linear_surrogate_vals_df_l = DataFrame(
        index=c2_vals, data=linear_surrogate_vals_dict_l
    )
    RBF_surrogate_vals_df_l = DataFrame(index=c2_vals, data=RBF_surrogate_vals_dict_l)
    actual_vals_df_l = DataFrame(index=c2_vals, data=actual_vals_dict_l)
    percent_error_df_linear_l = DataFrame(
        index=c2_vals, data=percent_error_dict_linear_l
    )
    percent_error_df_rbf_l = DataFrame(index=c2_vals, data=percent_error_dict_rbf_l)

    linear_surrogate_vals_df_m = DataFrame(
        index=c2_vals, data=linear_surrogate_vals_dict_m
    )
    RBF_surrogate_vals_df_m = DataFrame(index=c2_vals, data=RBF_surrogate_vals_dict_m)
    actual_vals_df_m = DataFrame(index=c2_vals, data=actual_vals_dict_m)
    percent_error_df_linear_m = DataFrame(
        index=c2_vals, data=percent_error_dict_linear_m
    )
    percent_error_df_rbf_m = DataFrame(index=c2_vals, data=percent_error_dict_rbf_m)

    linear_surrogate_vals_df_h = DataFrame(
        index=c2_vals, data=linear_surrogate_vals_dict_h
    )
    RBF_surrogate_vals_df_h = DataFrame(index=c2_vals, data=RBF_surrogate_vals_dict_h)
    actual_vals_df_h = DataFrame(index=c2_vals, data=actual_vals_dict_h)
    percent_error_df_linear_h = DataFrame(
        index=c2_vals, data=percent_error_dict_linear_h
    )
    percent_error_df_rbf_h = DataFrame(index=c2_vals, data=percent_error_dict_rbf_h)

    # fig1, (ax1,ax2, ax3) = plt.subplots(1,3)

    # linear_plot = ax1.pcolor(c1_vals, c2_vals, linear_surrogate_vals_df)
    # ax2.pcolor(c1_vals,c2_vals,RBF_surrogate_vals_df)
    # ax3.pcolor(c1_vals,c2_vals,actual_vals_df)

    # fig1.colorbar(linear_plot, ax=[ax1,ax2,ax3])

    # plt.show()

    fig2, ((ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(2, 3)
    percent_error_plot_linear_l = ax4.pcolor(
        c1_vals, c2_vals, percent_error_df_linear_l
    )
    percent_error_plot_linear_m = ax5.pcolor(
        c1_vals, c2_vals, percent_error_df_linear_m
    )
    percent_error_plot_linear_h = ax6.pcolor(
        c1_vals, c2_vals, percent_error_df_linear_h
    )

    percent_error_plot_rbf_l = ax7.pcolor(c1_vals, c2_vals, percent_error_df_rbf_l)
    percent_error_plot_rbf_m = ax8.pcolor(c1_vals, c2_vals, percent_error_df_rbf_m)
    percent_error_plot_rbf_h = ax9.pcolor(c1_vals, c2_vals, percent_error_df_rbf_h)

    for ax in (ax4, ax5, ax6):
        ax.plot([50, 80], [80, 80], "m-")
        ax.plot([50, 80], [110, 110], "m-")
        ax.plot([50, 50], [80, 110], "m-")
        ax.plot([80, 80], [80, 110], "m-")

    # ax4.plot([73], [108], "ms", markersize=7)
    # ax5.plot([73], [108], "ms", markersize=7)

    for ax in (ax4, ax5, ax6, ax7, ax8, ax9):
        ax.tick_params(direction="in", labelsize=10)
    for ax in (ax4, ax5, ax6):
        ax.set_title(
            "Percent Error (%)\nLinear Surrogate", fontsize=12, fontweight="bold"
        )
    for ax in (ax7, ax8, ax9):
        ax.set_title("Percent Error (%)\nRBF Surrogate", fontsize=12, fontweight="bold")
        ax.set_xlabel(
            "Lithium Concentration\n in Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )
    for ax in (ax4, ax7):
        ax.set_ylabel(
            "Cobalt Concentration\nin Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )

    fig2.colorbar(percent_error_plot_linear_l, ax=ax4)
    fig2.colorbar(percent_error_plot_linear_m, ax=ax5)
    fig2.colorbar(percent_error_plot_linear_h, ax=ax6)
    fig2.colorbar(percent_error_plot_rbf_l, ax=ax7)
    fig2.colorbar(percent_error_plot_rbf_m, ax=ax8)
    fig2.colorbar(percent_error_plot_rbf_h, ax=ax9)

    plt.show()


def evaluate_surrogate_two_salt(conc1, conc2, chi, surrogate_obj, var):
    input_dict = {
        "conc_1": conc1,
        "conc_2": conc2,
        "chi": chi,
    }
    input_df = DataFrame(data=input_dict, index=[0])
    surrogate_value = surrogate_obj.evaluate_surrogate(input_df)[var][0] * 1e-7

    return surrogate_value


def evaluate_surrogate_three_salt(conc1, conc2, conc3, chi, surrogate_obj, var):
    input_dict = {
        "conc_1": conc1,
        "conc_2": conc2,
        "conc_3": conc3,
        "chi": chi,
    }
    input_df = DataFrame(data=input_dict, index=[0])
    surrogate_value = surrogate_obj.evaluate_surrogate(input_df)[var][0] * 1e-7

    return surrogate_value


def calculate_true_value_two_salt(c1, c2):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 7.31e-6  # m2/h (chloride)

    chi = -140

    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )

    D_11 = (
        (z1 * z3 * D1 * D3 - (z1**2) * D1 * D3) * c1
        + (z2 * z3 * D1 * D3 - (z2**2) * D1 * D2) * c2
        + (z3 * D1 * D3 * chi)
    ) / D_denom

    return D_11


def calculate_true_value_three_salt(c1, c2, c3):
    z1 = 1  # m2/h (lithium)
    z2 = 2  # m2/h (cobalt)
    z3 = 3  # m2/h (aluminum)
    z4 = -1  # m2/h (chloride)

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 2.01e-6  # m2/h (aluminum)
    D4 = 7.31e-6  # m2/h (chloride)

    chi = -140

    D_denom = (
        ((((z1**2) * D1) - (z1 * z4 * D4)) * c1)
        + ((((z2**2) * D2) - (z2 * z4 * D4)) * c2)
        + ((((z3**2) * D3) - (z3 * z4 * D4)) * c3)
        - (z4 * D4 * chi)
    )

    D_11 = (
        (((z1 * z4 * D1 * D4) - ((z1**2) * D1 * D4)) * c1)
        + (((z2 * z4 * D1 * D4) - ((z2**2) * D1 * D2)) * c2)
        + (((z3 * z4 * D1 * D4) - ((z3**2) * D1 * D3)) * c3)
        + (z4 * D1 * D4 * chi)
    ) / D_denom

    return D_11


if __name__ == "__main__":
    main()
