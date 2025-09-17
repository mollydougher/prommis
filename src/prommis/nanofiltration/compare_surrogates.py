from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

def main():

    surrogate_model_file_dict = {
        "D_11": "surrogate_models/rbf_pysmo_surrogate_d11_scaled.json",
        "D_12": "surrogate_models/rbf_pysmo_surrogate_d12_scaled.json",
        "D_21": "surrogate_models/rbf_pysmo_surrogate_d21_scaled.json",
        "D_22": "surrogate_models/rbf_pysmo_surrogate_d22_scaled.json",
        "alpha_1": "surrogate_models/rbf_pysmo_surrogate_alpha1.json",
        "alpha_2": "surrogate_models/rbf_pysmo_surrogate_alpha2.json",
    }

    _surrogates_obj_D_11 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict["D_11"]
    )

    c1_vals = []
    c2_vals = []
    for c1 in np.arange(50,201,1):
        c1_vals.append(c1)
    for c2 in np.arange(50,201,1):
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

    for c1 in np.arange(50,201,1):
        for c2 in np.arange(50,201,1):
            linear_val = evaluate_linear_surrogate(c1,c2)
            linear_surrogate_vals.append(linear_val)

            rbf_val = evaluate_surrogate(c1, c2, -140, _surrogates_obj_D_11, "D_11_scaled")
            RBF_surrogate_vals.append(rbf_val)

            actual_val = calculate_true_value(c1, c2)
            actual_vals.append(actual_val)

            percent_error_val_linear = abs(linear_val-actual_val)/abs(actual_val)*100
            percent_error_vals_linear.append(percent_error_val_linear)

            percent_error_val_rbf = abs(rbf_val-actual_val)/abs(actual_val)*100
            percent_error_vals_rbf.append(percent_error_val_rbf)

        linear_surrogate_vals_dict[f"{c1}"] = linear_surrogate_vals
        RBF_surrogate_vals_dict[f"{c1}"] = RBF_surrogate_vals
        actual_vals_dict[f"{c1}"] = actual_vals
        percent_error_dict_linear[f"{c1}"] = percent_error_vals_linear
        percent_error_dict_rbf[f"{c1}"] = percent_error_vals_rbf

        linear_surrogate_vals = []
        RBF_surrogate_vals = []
        actual_vals= []
        percent_error_vals_linear = []
        percent_error_vals_rbf = []
    
    linear_surrogate_vals_df = DataFrame(index=c2_vals,data=linear_surrogate_vals_dict)
    RBF_surrogate_vals_df = DataFrame(index=c2_vals,data=RBF_surrogate_vals_dict)
    actual_vals_df = DataFrame(index=c2_vals,data=actual_vals_dict)
    percent_error_df_linear = DataFrame(index=c2_vals,data=percent_error_dict_linear)
    percent_error_df_rbf = DataFrame(index=c2_vals,data=percent_error_dict_rbf)

    # fig1, (ax1,ax2, ax3) = plt.subplots(1,3)

    # linear_plot = ax1.pcolor(c1_vals, c2_vals, linear_surrogate_vals_df)
    # ax2.pcolor(c1_vals,c2_vals,RBF_surrogate_vals_df)
    # ax3.pcolor(c1_vals,c2_vals,actual_vals_df)

    # fig1.colorbar(linear_plot, ax=[ax1,ax2,ax3])

    # plt.show()

    fig2, (ax4,ax5) = plt.subplots(1,2)
    percent_error_plot_linear = ax4.pcolor(c1_vals, c2_vals, percent_error_df_linear)
    percent_error_plot_rbf = ax5.pcolor(c1_vals, c2_vals, percent_error_df_rbf)

    ax4.plot([73],[108],'ms',markersize=7)
    ax5.plot([73],[108],'ms',markersize=7)

    fig2.colorbar(percent_error_plot_linear,ax=[ax4,ax5])
    
    plt.show()

def evaluate_linear_surrogate(c1, c2):
    val = -4.33e-06 + -4.21e-09 * c1 + 5.10e-09 * c2

    return val


def evaluate_surrogate(conc1, conc2, chi, surrogate_obj, var):
    input_dict = {
        "conc_1": conc1,
        "conc_2": conc2,
        "chi": chi,
    }
    input_df = DataFrame(data=input_dict, index=[0])
    surrogate_value = surrogate_obj.evaluate_surrogate(input_df)[var][0] * 1e-7

    return surrogate_value


def calculate_true_value(c1, c2):
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



if __name__ == "__main__":
    main()
