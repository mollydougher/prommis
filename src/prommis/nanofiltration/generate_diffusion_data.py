import numpy as np

from pandas import DataFrame


def main():
    (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df) = generate_data(
        scaled=False
    )
    D_11_df.to_csv("surrogate_data/D_11.csv", index=False)
    D_12_df.to_csv("surrogate_data/D_12.csv", index=False)
    D_21_df.to_csv("surrogate_data/D_21.csv", index=False)
    D_22_df.to_csv("surrogate_data/D_22.csv", index=False)
    alpha_1_df.to_csv("surrogate_data/alpha_1.csv", index=False)
    alpha_2_df.to_csv("surrogate_data/alpha_2.csv", index=False)

    (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df) = generate_data(
        scaled=True
    )
    D_11_df.to_csv("surrogate_data/D_11_scaled.csv", index=False)
    D_12_df.to_csv("surrogate_data/D_12_scaled.csv", index=False)
    D_21_df.to_csv("surrogate_data/D_21_scaled.csv", index=False)
    D_22_df.to_csv("surrogate_data/D_22_scaled.csv", index=False)
    alpha_1_df.to_csv("surrogate_data/alpha_1_scaled.csv", index=False)
    alpha_2_df.to_csv("surrogate_data/alpha_2_scaled.csv", index=False)


def set_parameter_values_and_concentration_ranges():
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.7e-6  # m2/h
    D2 = 2.64e-6  # m2/h
    D3 = 7.3e-6  # m2/h

    # chi = chi

    # nominal concentrations
    # chi=0
    # c,li,m ~57
    # c,co,m ~66

    # chi=-140
    # c,li,m ~73
    # c,co,m ~108

    # if chi == 0:
    #     c1_vals = np.arange(40, 91, 1)  # mol/m3 = mM
    #     c2_vals = np.arange(50, 101, 1)  # mol/m3 = mM
    # if chi == -140:
    #     c1_vals = np.arange(60, 111, 1)  # mol/m3 = mM
    #     c2_vals = np.arange(90, 141, 1)  # mol/m3 = mM

    c1_vals = np.arange(50, 111, 1)  # mol/m3 = mM
    c2_vals = np.arange(50, 141, 1)  # mol/m3 = mM
    chi_vals = np.arange(-140, 10, 10)  # mol/m3 = mM

    return (z1, z2, z3, D1, D2, D3, c1_vals, c2_vals, chi_vals)


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


def generate_data(scaled):
    (z1, z2, z3, D1, D2, D3, c1_vals, c2_vals, chi_vals) = (
        set_parameter_values_and_concentration_ranges()
    )

    c1_list = []
    c2_list = []
    chi_list = []
    D_11_vals = []
    D_12_vals = []
    D_21_vals = []
    D_22_vals = []
    alpha_1_vals = []
    alpha_2_vals = []

    for c1 in c1_vals:
        for c2 in c2_vals:
            for chi in chi_vals:
                c1_list.append(c1)
                c2_list.append(c2)
                chi_list.append(chi)

    if scaled:
        scale_factor = 1e7
    else:
        scale_factor = 1

    for c1 in c1_vals:
        for c2 in c2_vals:
            for chi in chi_vals:
                D_11_vals.append(
                    scale_factor * (calculate_D_11(z1, z2, z3, D1, D2, D3, c1, c2, chi))
                )
                D_12_vals.append(
                    scale_factor * (calculate_D_12(z1, z2, z3, D1, D2, D3, c1, c2, chi))
                )
                D_21_vals.append(
                    scale_factor * (calculate_D_21(z1, z2, z3, D1, D2, D3, c1, c2, chi))
                )
                D_22_vals.append(
                    scale_factor * (calculate_D_22(z1, z2, z3, D1, D2, D3, c1, c2, chi))
                )
                alpha_1_vals.append(
                    scale_factor
                    * (calculate_alpha_1(z1, z2, z3, D1, D2, D3, c1, c2, chi))
                )
                alpha_2_vals.append(
                    scale_factor
                    * (calculate_alpha_2(z1, z2, z3, D1, D2, D3, c1, c2, chi))
                )

    if scaled:
        d_11_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "D_11_scaled": D_11_vals,
        }
        d_12_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "D_12_scaled": D_12_vals,
        }
        d_21_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "D_21_scaled": D_21_vals,
        }
        d_22_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "D_22_scaled": D_22_vals,
        }
        alpha_1_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "alpha_1_scaled": alpha_1_vals,
        }
        alpha_2_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "alpha_2_scaled": alpha_2_vals,
        }
    else:
        d_11_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "D_11": D_11_vals,
        }
        d_12_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "D_12": D_12_vals,
        }
        d_21_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "D_21": D_21_vals,
        }
        d_22_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "D_22": D_22_vals,
        }
        alpha_1_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "alpha_1": alpha_1_vals,
        }
        alpha_2_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "alpha_2": alpha_2_vals,
        }

    D_11_df = DataFrame(data=d_11_dict)
    D_12_df = DataFrame(data=d_12_dict)
    D_21_df = DataFrame(data=d_21_dict)
    D_22_df = DataFrame(data=d_22_dict)
    alpha_1_df = DataFrame(data=alpha_1_dict)
    alpha_2_df = DataFrame(data=alpha_2_dict)

    return (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df)


if __name__ == "__main__":
    main()
