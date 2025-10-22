import numpy as np

from pandas import DataFrame

import itertools


def main():
    generate_data_li_co_cl()
    # generate_two_salt_data()
    # set_three_level_doe()
    # generate_data_li_co_al_cl(kriging_or_rbf=True)

def generate_three_level_doe_design():
    levels = [-1, 0, 1]
    full_cube_list = list(itertools.product(levels, repeat=3))
    full_cube_array = np.asarray(full_cube_list)
    full_cube_bool = np.asarray([full_cube_array[i,:] == 0 for i in range(len(full_cube_array))])
    index_to_keep = []
    index=0
    for point in full_cube_bool:
        if point.sum() == 1:
            pass
        else: index_to_keep.append(index)
        index+=1
    fractional_doe = full_cube_array[index_to_keep,:]
    return fractional_doe


def set_three_level_doe():
    design_structure = generate_three_level_doe_design()
    print(design_structure)
    c_values = [1, 101, 201]
    chi_values = [-150, -50, 50]
    final_design = []
    for design_point in design_structure:
        final_point = []
        var = 0
        for conc_level in design_point:
            if conc_level == -1:
                if var == 0 or var == 1:
                    val = c_values[0]
                if var == 2:
                    val = chi_values[0]
                final_point.append(val)
            elif conc_level == 0:
                if var == 0 or var == 1:
                    val = c_values[1]
                if var == 2:
                    val = chi_values[1]
                final_point.append(val)
            elif conc_level == 1:
                if var == 0 or var == 1:
                    val = c_values[2]
                if var == 2:
                    val = chi_values[2]
                final_point.append(val)
            var +=1
        final_design.append(final_point)
    return final_design
    # print(final_design)


def generate_data_li_co_cl():
    kriging_or_rbf = True
    if kriging_or_rbf:
        (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df) = (
            generate_two_salt_data(
                system="li_co_cl",
                # kriging_or_rbf=kriging_or_rbf,
            )
        )
        D_11_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_chloride/D_11_scaled.csv",
            index=False,
        )
        D_12_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_chloride/D_12_scaled.csv",
            index=False,
        )
        D_21_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_chloride/D_21_scaled.csv",
            index=False,
        )
        D_22_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_chloride/D_22_scaled.csv",
            index=False,
        )
        alpha_1_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_chloride/alpha_1.csv",
            index=False,
        )
        alpha_2_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_chloride/alpha_2.csv",
            index=False,
        )
    else:
        (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df) = (
            generate_two_salt_data(
                system="li_co_cl",
                # kriging_or_rbf=kriging_or_rbf,
            )
        )
        D_11_df.to_csv("surrogate_data/D_11_scaled.csv", index=False)
        D_12_df.to_csv("surrogate_data/D_12_scaled.csv", index=False)
        D_21_df.to_csv("surrogate_data/D_21_scaled.csv", index=False)
        D_22_df.to_csv("surrogate_data/D_22_scaled.csv", index=False)
        alpha_1_df.to_csv("surrogate_data/alpha_1.csv", index=False)
        alpha_2_df.to_csv("surrogate_data/alpha_2.csv", index=False)


def generate_data_li_co_al_cl(kriging_or_rbf):
    if kriging_or_rbf:
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
        ) = generate_three_salt_data(
            system="li_co_al_cl",
            kriging_or_rbf=kriging_or_rbf,
        )
        D_11_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_11_scaled.csv",
            index=False,
        )
        D_12_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_12_scaled.csv",
            index=False,
        )
        D_13_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_13_scaled.csv",
            index=False,
        )
        D_21_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_21_scaled.csv",
            index=False,
        )
        D_22_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_22_scaled.csv",
            index=False,
        )
        D_23_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_23_scaled.csv",
            index=False,
        )
        D_31_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_31_scaled.csv",
            index=False,
        )
        D_32_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_32_scaled.csv",
            index=False,
        )
        D_33_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/D_33_scaled.csv",
            index=False,
        )
        alpha_1_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/alpha_1.csv",
            index=False,
        )
        alpha_2_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/alpha_2.csv",
            index=False,
        )
        alpha_3_df.to_csv(
            "surrogate_data/kriging_or_rbf/lithium_cobalt_aluminum_chloride/alpha_3.csv",
            index=False,
        )
    else:
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
        ) = generate_three_salt_data(
            system="li_co_al_cl",
            kriging_or_rbf=kriging_or_rbf,
        )
        D_11_df.to_csv("surrogate_data/D_11_scaled.csv", index=False)
        D_12_df.to_csv("surrogate_data/D_12_scaled.csv", index=False)
        D_13_df.to_csv("surrogate_data/D_13_scaled.csv", index=False)
        D_21_df.to_csv("surrogate_data/D_21_scaled.csv", index=False)
        D_22_df.to_csv("surrogate_data/D_22_scaled.csv", index=False)
        D_23_df.to_csv("surrogate_data/D_23_scaled.csv", index=False)
        D_31_df.to_csv("surrogate_data/D_31_scaled.csv", index=False)
        D_32_df.to_csv("surrogate_data/D_32_scaled.csv", index=False)
        D_33_df.to_csv("surrogate_data/D_33_scaled.csv", index=False)
        alpha_1_df.to_csv("surrogate_data/alpha_1.csv", index=False)
        alpha_2_df.to_csv("surrogate_data/alpha_2.csv", index=False)
        alpha_3_df.to_csv("surrogate_data/alpha_3.csv", index=False)


def set_two_salt_concentration_ranges(system):#, kriging_or_rbf):
    # nominal membrane concentrations (min and max)
    # chi=0
    # c,li,m ~57
    # c,co,m ~66

    # chi=-140
    # c,li,m ~73
    # c,co,m ~108

    if system == "li_co_cl":
        z1 = 1  # m2/h (lithium)
        z2 = 2  # m2/h (cobalt)
        z3 = -1  # m2/h (chloride)

        D1 = 3.71e-6  # m2/h (lithium)
        D2 = 2.64e-6  # m2/h (cobalt)
        D3 = 7.31e-6  # m2/h (chloride)

        # if kriging_or_rbf:
        #     c1_vals = [40, 115, 190]  # mol/m3 = mM
        #     c2_vals = [40, 115, 190]  # mol/m3 = mM
        #     # c1_vals = np.arange(1, 300, 100)  # mol/m3 = mM
        #     # c2_vals = np.arange(50, 275, 75)  # mol/m3 = mM
        #     chi_vals = np.arange(-150, 75, 75)  # mol/m3 = mM
        # else:
        #     c1_vals = np.arange(50, 237.5, 37.5)  # mol/m3 = mM
        #     c2_vals = np.arange(50, 237.5, 37.5)  # mol/m3 = mM
        #     chi_vals = np.arange(-150, 37.5, 37.5)  # mol/m3 = mM

        # print(c1_vals)
        # print(c2_vals)
        # print(chi_vals)

    # return (z1, z2, z3, D1, D2, D3, c1_vals, c2_vals, chi_vals)
    return (z1, z2, z3, D1, D2, D3)


def set_three_salt_concentration_ranges(system, kriging_or_rbf):
    # battery leachte aluminum conc: 22-89 mM

    if system == "li_co_al_cl":
        z1 = 1  # m2/h (lithium)
        z2 = 2  # m2/h (cobalt)
        z3 = 3  # m2/h (aluminum)
        z4 = -1  # m2/h (chloride)

        D1 = 3.71e-6  # m2/h (lithium)
        D2 = 2.64e-6  # m2/h (cobalt)
        D3 = 2.01e-6  # m2/h (aluminum)
        D4 = 7.31e-6  # m2/h (chloride)

        if kriging_or_rbf:
            c1_vals = [40, 115, 190]  # mol/m3 = mM
            c2_vals = [40, 115, 190]  # mol/m3 = mM
            c3_vals = [1, 76, 151]  # mol/m3 = mM
            # c1_vals = np.arange(75, 300, 75)  # mol/m3 = mM
            # c2_vals = np.arange(50, 275, 75)  # mol/m3 = mM
            # c3_vals = np.arange(1, 226, 75)  # mol/m3 = mM
            chi_vals = np.arange(-150, 75, 75)  # mol/m3 = mM
        else:
            c1_vals = np.arange(50, 237.5, 37.5)  # mol/m3 = mM
            c2_vals = np.arange(50, 237.5, 37.5)  # mol/m3 = mM
            c3_vals = np.arange(50, 237.5, 37.5)  # mol/m3 = mM
            chi_vals = np.arange(-150, 37.5, 37.5)  # mol/m3 = mM

        print(c1_vals)
        print(c2_vals)
        print(c3_vals)
        print(chi_vals)

    return (z1, z2, z3, z4, D1, D2, D3, D4, c1_vals, c2_vals, c3_vals, chi_vals)


def calculate_D_denominator_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )
    return D_denom


def calculate_D_11_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_11 = (
        (z1 * z3 * D1 * D3 - (z1**2) * D1 * D3) * c1
        + (z2 * z3 * D1 * D3 - (z2**2) * D1 * D2) * c2
        + (z3 * D1 * D3 * chi)
    ) / D_denom
    return D_11


def calculate_D_12_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_12 = ((z1 * z2 * D1 * D2 - z1 * z2 * D1 * D3) * c1) / D_denom
    return D_12


def calculate_D_21_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_21 = ((z1 * z2 * D1 * D2 - z1 * z2 * D2 * D3) * c2) / D_denom
    return D_21


def calculate_D_22_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    D_22 = (
        (z1 * z3 * D2 * D3 - (z1**2) * D1 * D2) * c1
        + (z2 * z3 * D2 * D3 - (z2**2) * D2 * D3) * c2
        + (z3 * D2 * D3 * chi)
    ) / D_denom
    return D_22


def calculate_alpha_1_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    alpha_1 = 1 + (z1 * D1 * chi) / D_denom
    return alpha_1


def calculate_alpha_2_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = calculate_D_denominator_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi)
    alpha_2 = 1 + (z2 * D2 * chi) / D_denom
    return alpha_2


def calculate_D_denominator_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = (
        ((((z1**2) * D1) - (z1 * z4 * D4)) * c1)
        + ((((z2**2) * D2) - (z2 * z4 * D4)) * c2)
        + ((((z3**2) * D3) - (z3 * z4 * D4)) * c3)
        - (z4 * D4 * chi)
    )
    return D_denom


def calculate_D_11_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_11 = (
        (((z1 * z4 * D1 * D4) - ((z1**2) * D1 * D4)) * c1)
        + (((z2 * z4 * D1 * D4) - ((z2**2) * D1 * D2)) * c2)
        + (((z3 * z4 * D1 * D4) - ((z3**2) * D1 * D3)) * c3)
        + (z4 * D1 * D4 * chi)
    ) / D_denom
    return D_11


def calculate_D_12_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_12 = (((z1 * z2 * D1 * D2) - (z1 * z2 * D1 * D4)) * c1) / D_denom
    return D_12


def calculate_D_13_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_13 = (((z1 * z3 * D1 * D3) - (z1 * z3 * D1 * D4)) * c1) / D_denom
    return D_13


def calculate_D_21_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_21 = (((z1 * z2 * D1 * D2) - (z1 * z2 * D2 * D4)) * c2) / D_denom
    return D_21


def calculate_D_22_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_22 = (
        (((z1 * z4 * D2 * D4) - ((z1**2) * D1 * D2)) * c1)
        + (((z2 * z4 * D2 * D4) - ((z2**2) * D2 * D4)) * c2)
        + (((z3 * z4 * D2 * D4) - ((z3**2) * D2 * D3)) * c3)
        + (z4 * D2 * D4 * chi)
    ) / D_denom
    return D_22


def calculate_D_23_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_23 = (((z2 * z3 * D2 * D3) - (z2 * z3 * D2 * D4)) * c2) / D_denom
    return D_23


def calculate_D_31_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_31 = (((z1 * z3 * D1 * D3) - (z1 * z3 * D3 * D4)) * c3) / D_denom
    return D_31


def calculate_D_32_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_32 = (((z2 * z3 * D2 * D3) - (z2 * z3 * D3 * D4)) * c3) / D_denom
    return D_32


def calculate_D_33_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    D_33 = (
        (((z1 * z4 * D3 * D4) - ((z1**2) * D1 * D3)) * c1)
        + (((z2 * z4 * D3 * D4) - ((z2**2) * D2 * D3)) * c2)
        + (((z3 * z4 * D3 * D4) - ((z3**2) * D3 * D4)) * c3)
        + (z4 * D3 * D4 * chi)
    ) / D_denom
    return D_33


def calculate_alpha_1_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    alpha_1 = 1 + (z1 * D1 * chi) / D_denom
    return alpha_1


def calculate_alpha_2_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    alpha_2 = 1 + (z2 * D2 * chi) / D_denom
    return alpha_2


def calculate_alpha_3_three_salt(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = calculate_D_denominator_three_salt(
        z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
    )
    alpha_3 = 1 + (z3 * D3 * chi) / D_denom
    return alpha_3


def generate_two_salt_data(system="li_co_cl", scaled_diff=True):
    training_ponts = set_three_level_doe()

    # (z1, z2, z3, D1, D2, D3, c1_vals, c2_vals, chi_vals) = (
    (z1, z2, z3, D1, D2, D3) = (
        set_two_salt_concentration_ranges(system)#, kriging_or_rbf)
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

    if scaled_diff:
        scale_factor = 1e7
    else:
        scale_factor = 1

    for point in training_ponts:
        c1 = point[0]
        c2 = point[1]
        chi = point[2]

        c1_list.append(c1)
        c2_list.append(c2)
        chi_list.append(chi)

        D_11_vals.append(
            scale_factor
            * (calculate_D_11_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi))
        )
        D_12_vals.append(
            scale_factor
            * (calculate_D_12_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi))
        )
        D_21_vals.append(
            scale_factor
            * (calculate_D_21_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi))
        )
        D_22_vals.append(
            scale_factor
            * (calculate_D_22_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi))
        )
        alpha_1_vals.append(
            (calculate_alpha_1_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi))
        )
        alpha_2_vals.append(
            (calculate_alpha_2_two_salt(z1, z2, z3, D1, D2, D3, c1, c2, chi))
        )

    if scaled_diff:
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
    # print(D_11_df)


def generate_three_salt_data(system, kriging_or_rbf, scaled_diff=True):
    (z1, z2, z3, z4, D1, D2, D3, D4, c1_vals, c2_vals, c3_vals, chi_vals) = (
        set_three_salt_concentration_ranges(system, kriging_or_rbf)
    )

    c1_list = []
    c2_list = []
    c3_list = []
    chi_list = []
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

    for c1 in c1_vals:
        for c2 in c2_vals:
            for c3 in c3_vals:
                for chi in chi_vals:
                    c1_list.append(c1)
                    c2_list.append(c2)
                    c3_list.append(c3)
                    chi_list.append(chi)

    if scaled_diff:
        scale_factor = 1e7
    else:
        scale_factor = 1

    for c1 in c1_vals:
        for c2 in c2_vals:
            for c3 in c3_vals:
                for chi in chi_vals:
                    D_11_vals.append(
                        scale_factor
                        * (
                            calculate_D_11_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    D_12_vals.append(
                        scale_factor
                        * (
                            calculate_D_12_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    D_13_vals.append(
                        scale_factor
                        * (
                            calculate_D_13_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    D_21_vals.append(
                        scale_factor
                        * (
                            calculate_D_21_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    D_22_vals.append(
                        scale_factor
                        * (
                            calculate_D_22_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    D_23_vals.append(
                        scale_factor
                        * (
                            calculate_D_23_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    D_31_vals.append(
                        scale_factor
                        * (
                            calculate_D_31_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    D_32_vals.append(
                        scale_factor
                        * (
                            calculate_D_32_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    D_33_vals.append(
                        scale_factor
                        * (
                            calculate_D_33_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    alpha_1_vals.append(
                        (
                            calculate_alpha_1_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    alpha_2_vals.append(
                        (
                            calculate_alpha_2_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )
                    alpha_3_vals.append(
                        (
                            calculate_alpha_3_three_salt(
                                z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi
                            )
                        )
                    )

    if scaled_diff:
        d_11_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_11_scaled": D_11_vals,
        }
        d_12_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_12_scaled": D_12_vals,
        }
        d_13_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_13_scaled": D_13_vals,
        }
        d_21_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_21_scaled": D_21_vals,
        }
        d_22_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_22_scaled": D_22_vals,
        }
        d_23_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_23_scaled": D_23_vals,
        }
        d_31_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_31_scaled": D_31_vals,
        }
        d_32_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_32_scaled": D_32_vals,
        }
        d_33_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_33_scaled": D_33_vals,
        }
    else:
        d_11_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_11": D_11_vals,
        }
        d_12_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_12": D_12_vals,
        }
        d_13_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_13": D_13_vals,
        }
        d_21_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_21": D_21_vals,
        }
        d_22_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_22": D_22_vals,
        }
        d_23_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_23": D_23_vals,
        }
        d_31_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_31": D_31_vals,
        }
        d_32_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_32": D_32_vals,
        }
        d_33_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "chi": chi_list,
            "D_33": D_33_vals,
        }
    alpha_1_dict = {
        "conc_1": c1_list,
        "conc_2": c2_list,
        "conc_3": c3_list,
        "chi": chi_list,
        "alpha_1": alpha_1_vals,
    }
    alpha_2_dict = {
        "conc_1": c1_list,
        "conc_2": c2_list,
        "conc_3": c3_list,
        "chi": chi_list,
        "alpha_2": alpha_2_vals,
    }
    alpha_3_dict = {
        "conc_1": c1_list,
        "conc_2": c2_list,
        "conc_3": c3_list,
        "chi": chi_list,
        "alpha_3": alpha_3_vals,
    }

    D_11_df = DataFrame(data=d_11_dict)
    D_12_df = DataFrame(data=d_12_dict)
    D_13_df = DataFrame(data=d_13_dict)
    D_21_df = DataFrame(data=d_21_dict)
    D_22_df = DataFrame(data=d_22_dict)
    D_23_df = DataFrame(data=d_23_dict)
    D_31_df = DataFrame(data=d_31_dict)
    D_32_df = DataFrame(data=d_32_dict)
    D_33_df = DataFrame(data=d_33_dict)
    alpha_1_df = DataFrame(data=alpha_1_dict)
    alpha_2_df = DataFrame(data=alpha_2_dict)
    alpha_3_df = DataFrame(data=alpha_3_dict)

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


if __name__ == "__main__":
    main()
