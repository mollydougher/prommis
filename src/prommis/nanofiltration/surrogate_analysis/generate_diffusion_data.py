import numpy as np

from pandas import DataFrame

import itertools


def main():
    test = False
    if test:
        # test relevent scenarios
        for fractional_Bool in [True, False]:
            for extra_center_Bool in [True, False]:
                print(
                    f"# inputs: 4, fractional: {fractional_Bool}, extra center point(s): {extra_center_Bool}"
                )
                generate_three_level_doe_design(
                    num_inputs=4,
                    input_bounds={
                        "c1": [1, 101, 201],
                        "c2": [1, 101, 201],
                        "c3": [1, 101, 201],
                        "chi": [-150, -50, 50],
                    },
                    fractional=fractional_Bool,
                    remove_sum=1,
                    extra_center=extra_center_Bool,
                )
                print(
                    f"# inputs: 3, fractional: {fractional_Bool}, extra center point(s): {extra_center_Bool}"
                )
                generate_three_level_doe_design(
                    num_inputs=3,
                    input_bounds={
                        "c1": [1, 101, 201],
                        "c2": [1, 101, 201],
                        "chi": [-150, -50, 50],
                    },
                    fractional=fractional_Bool,
                    remove_sum=1,
                    extra_center=extra_center_Bool,
                )
                print(
                    f"# inputs: 2, fractional: {fractional_Bool}, extra center point(s): {extra_center_Bool}"
                )
                generate_three_level_doe_design(
                    num_inputs=2,
                    input_bounds={"c1": [1, 101, 201], "c2": [1, 101, 201]},
                    fractional=fractional_Bool,
                    remove_sum=1,
                    extra_center=extra_center_Bool,
                )

    for fractional_Bool in [True, False]:
        for sum_option in [1, 2]:
            for extra_center_Bool in [True, False]:
                generate_two_salt_data(
                    system="li_co_cl",
                    vary_chi=True,
                    input_bounds={
                        "c1": [1, 101, 201],
                        "c2": [1, 101, 201],
                        "chi": [-150, -50, 50],
                    },
                    fractional=fractional_Bool,
                    remove_sum=sum_option,
                    extra_center=extra_center_Bool,
                    save=True,
                    folder_name="lithium_cobalt_chloride",
                )
                generate_two_salt_data(
                    system="li_co_cl",
                    vary_chi=False,
                    input_bounds={"c1": [1, 101, 201], "c2": [1, 101, 201]},
                    fractional=fractional_Bool,
                    remove_sum=sum_option,
                    extra_center=extra_center_Bool,
                    save=True,
                    folder_name="lithium_cobalt_chloride",
                )

                generate_three_salt_data(
                    system="li_co_al_cl",
                    vary_chi=True,
                    input_bounds={
                        "c1": [1, 101, 201],
                        "c2": [1, 101, 201],
                        "c3": [1, 101, 201],
                        "chi": [-150, -50, 50],
                    },
                    fractional=fractional_Bool,
                    remove_sum=sum_option,
                    extra_center=extra_center_Bool,
                    save=True,
                    folder_name="lithium_cobalt_aluminum_chloride",
                )
                generate_three_salt_data(
                    system="li_co_al_cl",
                    vary_chi=False,
                    input_bounds={
                        "c1": [1, 101, 201],
                        "c2": [1, 101, 201],
                        "c3": [1, 101, 201],
                    },
                    fractional=fractional_Bool,
                    remove_sum=sum_option,
                    extra_center=extra_center_Bool,
                    save=True,
                    folder_name="lithium_cobalt_aluminum_chloride",
                )


def generate_three_level_doe_design(
    num_inputs,
    input_bounds,
    fractional=True,
    remove_sum=1,
    extra_center=True,
):
    # set the number of levels in the doe (high, medium, low)
    levels = [-1, 0, 1]

    # create the full factorial
    full_cube_list = list(itertools.product(levels, repeat=num_inputs))
    full_cube_array = np.asarray(full_cube_list)

    # return the full factorial if desired
    if not fractional:
        design_structure = full_cube_array

    # otherwise create the fractional factorial
    else:
        # express the full factorial as Boolean (True when 0)
        full_cube_bool = np.asarray(
            [full_cube_array[i, :] == 0 for i in range(len(full_cube_array))]
        )

        # create a list to store index values to keep
        index_to_keep = []
        index = 0

        # for each experiement in the full factorial doe (Bool)
        for point in full_cube_bool:
            # check the sum of the Bool exp.
            if point.sum() == remove_sum:
                pass
            # keep if not equal to the specified sum
            # default: remove_sum = 1, meaning removal of the edge midpoints
            # if remove_sum = 2, it will remove the face midponts
            else:
                index_to_keep.append(index)
            index += 1

        # create the fractional doe based on indices to keep
        fractional_doe = full_cube_array[index_to_keep, :]

        design_structure = fractional_doe

    if not extra_center:
        print("---------------")
        print("DOE Skeleton")
        print(design_structure)

    else:
        # add multiple instances of the center point
        middle_point = np.array([0] * num_inputs)

        for j in range(num_inputs - 1):
            design_structure = np.vstack((design_structure, middle_point))

        # re-sort the design
        sorted_indicies = np.lexsort(
            [design_structure[:, k] for k in range(num_inputs - 1, -1, -1)]
        )
        design_structure = design_structure[sorted_indicies]

        print("---------------")
        print("DOE Skeleton")
        print(design_structure)

    # set the bounds for the input variables
    # input_bounds = {var: [min, mid, max], ...}
    bounds_list = []
    for bounds in input_bounds.values():
        bounds_list.append(bounds)

    # for each experiment
    for design_point in design_structure:
        # replace each condition with the appropriate value
        for i in range(num_inputs):
            if design_point[i] == -1:
                np.put(design_point, i, bounds_list[i][0])
            elif design_point[i] == 0:
                np.put(design_point, i, bounds_list[i][1])
            elif design_point[i] == 1:
                np.put(design_point, i, bounds_list[i][2])

    print("Final DOE Design")
    print(design_structure)
    print("---------------")

    return design_structure


def set_parameters(system, vary_chi):
    z_lithium = 1
    z_cobalt = 2
    z_aluminum = 3
    z_chloride = -1

    D_lithium = 3.71e-6  # m2/h
    D_cobalt = 2.64e-6  # m2/h
    D_aluminum = 2.01e-6  # m2/h
    D_chloride = 7.31e-6  # m2/h

    nominal_chi = -140  # mM

    if system == "li_co_cl":
        z1 = z_lithium
        z2 = z_cobalt
        z3 = z_chloride

        D1 = D_lithium
        D2 = D_cobalt
        D3 = D_chloride

        if vary_chi:
            return (z1, z2, z3, D1, D2, D3)
        else:
            return (z1, z2, z3, D1, D2, D3, nominal_chi)

    elif system == "li_co_al_cl":
        z1 = z_lithium
        z2 = z_cobalt
        z3 = z_aluminum
        z4 = z_chloride

        D1 = D_lithium
        D2 = D_cobalt
        D3 = D_aluminum
        D4 = D_chloride

        if vary_chi:
            return (z1, z2, z3, z4, D1, D2, D3, D4)
        else:
            return (z1, z2, z3, z4, D1, D2, D3, D4, nominal_chi)


def two_salt_calculations(z1, z2, z3, D1, D2, D3, c1, c2, chi):
    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )

    D_11_calc = (
        (z1 * z3 * D1 * D3 - (z1**2) * D1 * D3) * c1
        + (z2 * z3 * D1 * D3 - (z2**2) * D1 * D2) * c2
        + (z3 * D1 * D3 * chi)
    ) / D_denom
    D_12_calc = ((z1 * z2 * D1 * D2 - z1 * z2 * D1 * D3) * c1) / D_denom

    D_21_calc = ((z1 * z2 * D1 * D2 - z1 * z2 * D2 * D3) * c2) / D_denom
    D_22_calc = (
        (z1 * z3 * D2 * D3 - (z1**2) * D1 * D2) * c1
        + (z2 * z3 * D2 * D3 - (z2**2) * D2 * D3) * c2
        + (z3 * D2 * D3 * chi)
    ) / D_denom

    alpha_1_calc = 1 + (z1 * D1 * chi) / D_denom
    alpha_2_calc = 1 + (z2 * D2 * chi) / D_denom

    return (D_11_calc, D_12_calc, D_21_calc, D_22_calc, alpha_1_calc, alpha_2_calc)


def three_salt_calculations(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi):
    D_denom = (
        ((((z1**2) * D1) - (z1 * z4 * D4)) * c1)
        + ((((z2**2) * D2) - (z2 * z4 * D4)) * c2)
        + ((((z3**2) * D3) - (z3 * z4 * D4)) * c3)
        - (z4 * D4 * chi)
    )

    D_11_calc = (
        (((z1 * z4 * D1 * D4) - ((z1**2) * D1 * D4)) * c1)
        + (((z2 * z4 * D1 * D4) - ((z2**2) * D1 * D2)) * c2)
        + (((z3 * z4 * D1 * D4) - ((z3**2) * D1 * D3)) * c3)
        + (z4 * D1 * D4 * chi)
    ) / D_denom
    D_12_calc = (((z1 * z2 * D1 * D2) - (z1 * z2 * D1 * D4)) * c1) / D_denom
    D_13_calc = (((z1 * z3 * D1 * D3) - (z1 * z3 * D1 * D4)) * c1) / D_denom

    D_21_calc = (((z1 * z2 * D1 * D2) - (z1 * z2 * D2 * D4)) * c2) / D_denom
    D_22_calc = (
        (((z1 * z4 * D2 * D4) - ((z1**2) * D1 * D2)) * c1)
        + (((z2 * z4 * D2 * D4) - ((z2**2) * D2 * D4)) * c2)
        + (((z3 * z4 * D2 * D4) - ((z3**2) * D2 * D3)) * c3)
        + (z4 * D2 * D4 * chi)
    ) / D_denom
    D_23_calc = (((z2 * z3 * D2 * D3) - (z2 * z3 * D2 * D4)) * c2) / D_denom

    D_31_calc = (((z1 * z3 * D1 * D3) - (z1 * z3 * D3 * D4)) * c3) / D_denom
    D_32_calc = (((z2 * z3 * D2 * D3) - (z2 * z3 * D3 * D4)) * c3) / D_denom
    D_33_calc = (
        (((z1 * z4 * D3 * D4) - ((z1**2) * D1 * D3)) * c1)
        + (((z2 * z4 * D3 * D4) - ((z2**2) * D2 * D3)) * c2)
        + (((z3 * z4 * D3 * D4) - ((z3**2) * D3 * D4)) * c3)
        + (z4 * D3 * D4 * chi)
    ) / D_denom

    alpha_1_calc = 1 + (z1 * D1 * chi) / D_denom
    alpha_2_calc = 1 + (z2 * D2 * chi) / D_denom
    alpha_3_calc = 1 + (z3 * D3 * chi) / D_denom

    return (
        D_11_calc,
        D_12_calc,
        D_13_calc,
        D_21_calc,
        D_22_calc,
        D_23_calc,
        D_31_calc,
        D_32_calc,
        D_33_calc,
        alpha_1_calc,
        alpha_2_calc,
        alpha_3_calc,
    )


def generate_two_salt_data(
    system,
    vary_chi,
    input_bounds,
    fractional=True,
    remove_sum=1,
    extra_center=True,
    save=True,
    folder_name=None,
):
    if vary_chi:
        num_inputs = 3
    else:
        num_inputs = 2

    training_ponts = generate_three_level_doe_design(
        num_inputs,
        input_bounds,
        fractional,
        remove_sum,
        extra_center,
    )

    if vary_chi:
        (z1, z2, z3, D1, D2, D3) = set_parameters(system, vary_chi)
    else:
        (z1, z2, z3, D1, D2, D3, nominal_chi) = set_parameters(system, vary_chi)

    c1_list = []
    c2_list = []
    if vary_chi:
        chi_list = []
    D_11_vals = []
    D_12_vals = []
    D_21_vals = []
    D_22_vals = []
    alpha_1_vals = []
    alpha_2_vals = []

    scale_factor = 1e7

    for point in training_ponts:
        c1 = point[0]
        c2 = point[1]

        if vary_chi:
            chi = point[2]
        else:
            chi = nominal_chi

        c1_list.append(c1)
        c2_list.append(c2)
        if vary_chi:
            chi_list.append(chi)

        (D_11_calc, D_12_calc, D_21_calc, D_22_calc, alpha_1_calc, alpha_2_calc) = (
            two_salt_calculations(z1, z2, z3, D1, D2, D3, c1, c2, chi)
        )

        D_11_vals.append(scale_factor * D_11_calc)
        D_12_vals.append(scale_factor * D_12_calc)
        D_21_vals.append(scale_factor * D_21_calc)
        D_22_vals.append(scale_factor * D_22_calc)
        alpha_1_vals.append(alpha_1_calc)
        alpha_2_vals.append(alpha_2_calc)

    if vary_chi:
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
            "alpha_1": alpha_1_vals,
        }
        alpha_2_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "chi": chi_list,
            "alpha_2": alpha_2_vals,
        }
    else:
        d_11_dict = {"conc_1": c1_list, "conc_2": c2_list, "D_11_scaled": D_11_vals}
        d_12_dict = {"conc_1": c1_list, "conc_2": c2_list, "D_12_scaled": D_12_vals}
        d_21_dict = {"conc_1": c1_list, "conc_2": c2_list, "D_21_scaled": D_21_vals}
        d_22_dict = {"conc_1": c1_list, "conc_2": c2_list, "D_22_scaled": D_22_vals}
        alpha_1_dict = {"conc_1": c1_list, "conc_2": c2_list, "alpha_1": alpha_1_vals}
        alpha_2_dict = {"conc_1": c1_list, "conc_2": c2_list, "alpha_2": alpha_2_vals}

    D_11_df = DataFrame(data=d_11_dict)
    D_12_df = DataFrame(data=d_12_dict)
    D_21_df = DataFrame(data=d_21_dict)
    D_22_df = DataFrame(data=d_22_dict)
    alpha_1_df = DataFrame(data=alpha_1_dict)
    alpha_2_df = DataFrame(data=alpha_2_dict)

    if save:
        dataframe_list = [D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df]

        if vary_chi:
            chi_folder_name = "with_chi_input"
        else:
            chi_folder_name = "without_chi_input"

        if fractional:
            if remove_sum == 1:
                factorial_folder_name = "fractional_factorial_1"
            elif remove_sum == 2:
                factorial_folder_name = "fractional_factorial_2"
        else:
            factorial_folder_name = "full_factorial"

        if extra_center:
            center_folder_name = "with_extra_center"
        else:
            center_folder_name = "without_extra_center"

        for dataframe in dataframe_list:
            dataframe.to_csv(
                f"surrogate_data/{folder_name}/{chi_folder_name}/{factorial_folder_name}/{center_folder_name}/{dataframe.columns[-1]}.csv",
                index=False,
            )
    else:
        return (D_11_df, D_12_df, D_21_df, D_22_df, alpha_1_df, alpha_2_df)


def generate_three_salt_data(
    system,
    vary_chi,
    input_bounds,
    fractional=True,
    remove_sum=1,
    extra_center=True,
    save=True,
    folder_name=None,
):
    if vary_chi:
        num_inputs = 4
    else:
        num_inputs = 3

    training_ponts = generate_three_level_doe_design(
        num_inputs,
        input_bounds,
        fractional,
        remove_sum,
        extra_center,
    )

    if vary_chi:
        (z1, z2, z3, z4, D1, D2, D3, D4) = set_parameters(system, vary_chi)
    else:
        (z1, z2, z3, z4, D1, D2, D3, D4, nominal_chi) = set_parameters(system, vary_chi)

    c1_list = []
    c2_list = []
    c3_list = []
    if vary_chi:
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

    scale_factor = 1e7

    for point in training_ponts:
        c1 = point[0]
        c2 = point[1]
        c3 = point[2]

        if vary_chi:
            chi = point[3]
        else:
            chi = nominal_chi

        c1_list.append(c1)
        c2_list.append(c2)
        c3_list.append(c3)
        if vary_chi:
            chi_list.append(chi)

        (
            D_11_calc,
            D_12_calc,
            D_13_calc,
            D_21_calc,
            D_22_calc,
            D_23_calc,
            D_31_calc,
            D_32_calc,
            D_33_calc,
            alpha_1_calc,
            alpha_2_calc,
            alpha_3_calc,
        ) = three_salt_calculations(z1, z2, z3, z4, D1, D2, D3, D4, c1, c2, c3, chi)

        D_11_vals.append(scale_factor * D_11_calc)
        D_12_vals.append(scale_factor * D_12_calc)
        D_13_vals.append(scale_factor * D_13_calc)
        D_21_vals.append(scale_factor * D_21_calc)
        D_22_vals.append(scale_factor * D_22_calc)
        D_23_vals.append(scale_factor * D_23_calc)
        D_31_vals.append(scale_factor * D_31_calc)
        D_32_vals.append(scale_factor * D_32_calc)
        D_33_vals.append(scale_factor * D_33_calc)
        alpha_1_vals.append(alpha_1_calc)
        alpha_2_vals.append(alpha_2_calc)
        alpha_3_vals.append(alpha_3_calc)

    if vary_chi:
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
    else:
        d_11_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_11_scaled": D_11_vals,
        }
        d_12_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_12_scaled": D_12_vals,
        }
        d_13_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_13_scaled": D_13_vals,
        }
        d_21_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_21_scaled": D_21_vals,
        }
        d_22_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_22_scaled": D_22_vals,
        }
        d_23_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_23_scaled": D_23_vals,
        }
        d_31_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_31_scaled": D_31_vals,
        }
        d_32_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_32_scaled": D_32_vals,
        }
        d_33_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "D_33_scaled": D_33_vals,
        }
        alpha_1_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "alpha_1": alpha_1_vals,
        }
        alpha_2_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
            "alpha_2": alpha_2_vals,
        }
        alpha_3_dict = {
            "conc_1": c1_list,
            "conc_2": c2_list,
            "conc_3": c3_list,
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

    if save:
        dataframe_list = [
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
        ]

        if vary_chi:
            chi_folder_name = "with_chi_input"
        else:
            chi_folder_name = "without_chi_input"

        if fractional:
            if remove_sum == 1:
                factorial_folder_name = "fractional_factorial_1"
            elif remove_sum == 2:
                factorial_folder_name = "fractional_factorial_2"
        else:
            factorial_folder_name = "full_factorial"

        if extra_center:
            center_folder_name = "with_extra_center"
        else:
            center_folder_name = "without_extra_center"

        for dataframe in dataframe_list:
            dataframe.to_csv(
                f"surrogate_data/{folder_name}/{chi_folder_name}/{factorial_folder_name}/{center_folder_name}/{dataframe.columns[-1]}.csv",
                index=False,
            )
    else:
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
