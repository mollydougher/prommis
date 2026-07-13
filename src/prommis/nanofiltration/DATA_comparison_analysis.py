#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Model comparison for multi-salt diafiltration.

Author: Molly Dougher
"""

from pathlib import Path

from pyomo.environ import value

from idaes.core.util import from_json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

from prommis.nanofiltration.multi_component_analysis import (
    build_model,
    get_model_averages,
)


def main():
    # save_data_comparison_csv(Cl_phi_key = "005")
    generate_data_comparison_heat_maps()

    plt.show()


def generate_data_comparison_heat_maps():
    fontsize = 14

    # load data from csv files
    Na_MAE_Cl_phi_005_df = load_data("005", "Na_MAE")
    Ca_MAE_Cl_phi_005_df = load_data("005", "Ca_MAE")
    La_MAE_Cl_phi_005_df = load_data("005", "La_MAE")
    Na_MSE_Cl_phi_005_df = load_data("005", "Na_MSE")
    Ca_MSE_Cl_phi_005_df = load_data("005", "Ca_MSE")
    La_MSE_Cl_phi_005_df = load_data("005", "La_MSE")

    Na_MAE_Cl_phi_010_df = load_data("010", "Na_MAE")
    Ca_MAE_Cl_phi_010_df = load_data("010", "Ca_MAE")
    La_MAE_Cl_phi_010_df = load_data("010", "La_MAE")
    Na_MSE_Cl_phi_010_df = load_data("010", "Na_MSE")
    Ca_MSE_Cl_phi_010_df = load_data("010", "Ca_MSE")
    La_MSE_Cl_phi_010_df = load_data("010", "La_MSE")

    Na_MAE_Cl_phi_020_df = load_data("020", "Na_MAE")
    Ca_MAE_Cl_phi_020_df = load_data("020", "Ca_MAE")
    La_MAE_Cl_phi_020_df = load_data("020", "La_MAE")
    Na_MSE_Cl_phi_020_df = load_data("020", "Na_MSE")
    Ca_MSE_Cl_phi_020_df = load_data("020", "Ca_MSE")
    La_MSE_Cl_phi_020_df = load_data("020", "La_MSE")

    Na_MAE_Cl_phi_030_df = load_data("030", "Na_MAE")
    Ca_MAE_Cl_phi_030_df = load_data("030", "Ca_MAE")
    La_MAE_Cl_phi_030_df = load_data("030", "La_MAE")
    Na_MSE_Cl_phi_030_df = load_data("030", "Na_MSE")
    Ca_MSE_Cl_phi_030_df = load_data("030", "Ca_MSE")
    La_MSE_Cl_phi_030_df = load_data("030", "La_MSE")

    Na_MAE_Cl_phi_040_df = load_data("040", "Na_MAE")
    Ca_MAE_Cl_phi_040_df = load_data("040", "Ca_MAE")
    La_MAE_Cl_phi_040_df = load_data("040", "La_MAE")
    Na_MSE_Cl_phi_040_df = load_data("040", "Na_MSE")
    Ca_MSE_Cl_phi_040_df = load_data("040", "Ca_MSE")
    La_MSE_Cl_phi_040_df = load_data("040", "La_MSE")

    df_mae_list = [
        Na_MAE_Cl_phi_005_df,
        Ca_MAE_Cl_phi_005_df,
        La_MAE_Cl_phi_005_df,
        Na_MAE_Cl_phi_010_df,
        Ca_MAE_Cl_phi_010_df,
        La_MAE_Cl_phi_010_df,
        Na_MAE_Cl_phi_020_df,
        Ca_MAE_Cl_phi_020_df,
        La_MAE_Cl_phi_020_df,
        Na_MAE_Cl_phi_030_df,
        Ca_MAE_Cl_phi_030_df,
        La_MAE_Cl_phi_030_df,
        Na_MAE_Cl_phi_040_df,
        Ca_MAE_Cl_phi_040_df,
        La_MAE_Cl_phi_040_df,
    ]
    df_mse_list = [
        Na_MSE_Cl_phi_005_df,
        Ca_MSE_Cl_phi_005_df,
        La_MSE_Cl_phi_005_df,
        Na_MSE_Cl_phi_010_df,
        Ca_MSE_Cl_phi_010_df,
        La_MSE_Cl_phi_010_df,
        Na_MSE_Cl_phi_020_df,
        Ca_MSE_Cl_phi_020_df,
        La_MSE_Cl_phi_020_df,
        Na_MSE_Cl_phi_030_df,
        Ca_MSE_Cl_phi_030_df,
        La_MSE_Cl_phi_030_df,
        Na_MSE_Cl_phi_040_df,
        Ca_MSE_Cl_phi_040_df,
        La_MSE_Cl_phi_040_df,
    ]

    vmin_mae = min(df.values.min() for df in df_mae_list)
    vmax_mae = max(df.values.max() for df in df_mae_list)

    vmin_mse = min(df.values.min() for df in df_mse_list)
    vmax_mse = max(df.values.max() for df in df_mse_list)

    # add stars at the minumum of each subplot
    Na_MAE_Cl_phi_005_df_min = find_minimums(Na_MAE_Cl_phi_005_df)
    Ca_MAE_Cl_phi_005_df_min = find_minimums(Ca_MAE_Cl_phi_005_df)
    La_MAE_Cl_phi_005_df_min = find_minimums(La_MAE_Cl_phi_005_df)
    Na_MSE_Cl_phi_005_df_min = find_minimums(Na_MSE_Cl_phi_005_df)
    Ca_MSE_Cl_phi_005_df_min = find_minimums(Ca_MSE_Cl_phi_005_df)
    La_MSE_Cl_phi_005_df_min = find_minimums(La_MSE_Cl_phi_005_df)

    Na_MAE_Cl_phi_010_df_min = find_minimums(Na_MAE_Cl_phi_010_df)
    Ca_MAE_Cl_phi_010_df_min = find_minimums(Ca_MAE_Cl_phi_010_df)
    La_MAE_Cl_phi_010_df_min = find_minimums(La_MAE_Cl_phi_010_df)
    Na_MSE_Cl_phi_010_df_min = find_minimums(Na_MSE_Cl_phi_010_df)
    Ca_MSE_Cl_phi_010_df_min = find_minimums(Ca_MSE_Cl_phi_010_df)
    La_MSE_Cl_phi_010_df_min = find_minimums(La_MSE_Cl_phi_010_df)

    Na_MAE_Cl_phi_020_df_min = find_minimums(Na_MAE_Cl_phi_020_df)
    Ca_MAE_Cl_phi_020_df_min = find_minimums(Ca_MAE_Cl_phi_020_df)
    La_MAE_Cl_phi_020_df_min = find_minimums(La_MAE_Cl_phi_020_df)
    Na_MSE_Cl_phi_020_df_min = find_minimums(Na_MSE_Cl_phi_020_df)
    Ca_MSE_Cl_phi_020_df_min = find_minimums(Ca_MSE_Cl_phi_020_df)
    La_MSE_Cl_phi_020_df_min = find_minimums(La_MSE_Cl_phi_020_df)

    Na_MAE_Cl_phi_030_df_min = find_minimums(Na_MAE_Cl_phi_030_df)
    Ca_MAE_Cl_phi_030_df_min = find_minimums(Ca_MAE_Cl_phi_030_df)
    La_MAE_Cl_phi_030_df_min = find_minimums(La_MAE_Cl_phi_030_df)
    Na_MSE_Cl_phi_030_df_min = find_minimums(Na_MSE_Cl_phi_030_df)
    Ca_MSE_Cl_phi_030_df_min = find_minimums(Ca_MSE_Cl_phi_030_df)
    La_MSE_Cl_phi_030_df_min = find_minimums(La_MSE_Cl_phi_030_df)

    Na_MAE_Cl_phi_040_df_min = find_minimums(Na_MAE_Cl_phi_040_df)
    Ca_MAE_Cl_phi_040_df_min = find_minimums(Ca_MAE_Cl_phi_040_df)
    La_MAE_Cl_phi_040_df_min = find_minimums(La_MAE_Cl_phi_040_df)
    Na_MSE_Cl_phi_040_df_min = find_minimums(Na_MSE_Cl_phi_040_df)
    Ca_MSE_Cl_phi_040_df_min = find_minimums(Ca_MSE_Cl_phi_040_df)
    La_MSE_Cl_phi_040_df_min = find_minimums(La_MSE_Cl_phi_040_df)

    fig1, (
        (ax1a, ax1b, ax1c, ax1d, ax1e),
        (ax2a, ax2b, ax2c, ax2d, ax2e),
        (ax3a, ax3b, ax3c, ax3d, ax3e),
    ) = plt.subplots(
        3,
        5,
        dpi=75,
        figsize=(25, 14.1),
        sharex=True,
        sharey=True,
    )
    fig1.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
    fig2, (
        (ax4a, ax4b, ax4c, ax4d, ax4e),
        (ax5a, ax5b, ax5c, ax5d, ax5e),
        (ax6a, ax6b, ax6c, ax6d, ax6e),
    ) = plt.subplots(
        3,
        5,
        dpi=75,
        figsize=(25, 14.1),
        sharex=True,
        sharey=True,
    )
    fig2.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])

    for ax in fig1.axes:
        ax.tick_params(labelsize=fontsize)
    for ax in fig2.axes:
        ax.tick_params(labelsize=fontsize)

    fig1.suptitle("Mean Absolute Error", fontsize=fontsize, fontweight="bold")
    fig2.suptitle("Mean Squared Error", fontsize=fontsize, fontweight="bold")
    for ax in [ax1a, ax4a]:
        ax.set_title(
            "$\mathbf{\phi_{Cl}}$ = 0.05", fontsize=fontsize, fontweight="bold"
        )
    for ax in [ax1b, ax4b]:
        ax.set_title("$\mathbf{\phi_{Cl}}$ = 0.1", fontsize=fontsize, fontweight="bold")
    for ax in [ax1c, ax4c]:
        ax.set_title("$\mathbf{\phi_{Cl}}$ = 0.2", fontsize=fontsize, fontweight="bold")
    for ax in [ax1d, ax4d]:
        ax.set_title("$\mathbf{\phi_{Cl}}$ = 0.3", fontsize=fontsize, fontweight="bold")
    for ax in [ax1e, ax4e]:
        ax.set_title("$\mathbf{\phi_{Cl}}$ = 0.4", fontsize=fontsize, fontweight="bold")

    cbar_ax_mae = fig1.add_axes([0.91, 0.2, 0.02, 0.6])
    cbar_ax_mae.tick_params(labelsize=fontsize)

    cbar_ax_mse = fig2.add_axes([0.91, 0.2, 0.02, 0.6])
    cbar_ax_mse.tick_params(labelsize=fontsize)

    plot_data(
        Na_MAE_Cl_phi_005_df,
        Na_MAE_Cl_phi_005_df_min,
        ax1a,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        Ca_MAE_Cl_phi_005_df,
        Ca_MAE_Cl_phi_005_df_min,
        ax2a,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        La_MAE_Cl_phi_005_df,
        La_MAE_Cl_phi_005_df_min,
        ax3a,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )

    plot_data(
        Na_MAE_Cl_phi_010_df,
        Na_MAE_Cl_phi_010_df_min,
        ax1b,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        Ca_MAE_Cl_phi_010_df,
        Ca_MAE_Cl_phi_010_df_min,
        ax2b,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        La_MAE_Cl_phi_010_df,
        La_MAE_Cl_phi_010_df_min,
        ax3b,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )

    plot_data(
        Na_MAE_Cl_phi_020_df,
        Na_MAE_Cl_phi_020_df_min,
        ax1c,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        Ca_MAE_Cl_phi_020_df,
        Ca_MAE_Cl_phi_020_df_min,
        ax2c,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        La_MAE_Cl_phi_020_df,
        La_MAE_Cl_phi_020_df_min,
        ax3c,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )

    plot_data(
        Na_MAE_Cl_phi_030_df,
        Na_MAE_Cl_phi_030_df_min,
        ax1d,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        Ca_MAE_Cl_phi_030_df,
        Ca_MAE_Cl_phi_030_df_min,
        ax2d,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        La_MAE_Cl_phi_030_df,
        La_MAE_Cl_phi_030_df_min,
        ax3d,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )

    plot_data(
        Na_MAE_Cl_phi_040_df,
        Na_MAE_Cl_phi_040_df_min,
        ax1e,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        Ca_MAE_Cl_phi_040_df,
        Ca_MAE_Cl_phi_040_df_min,
        ax2e,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )
    plot_data(
        La_MAE_Cl_phi_040_df,
        La_MAE_Cl_phi_040_df_min,
        ax3e,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
    )

    plot_data(
        Na_MSE_Cl_phi_005_df,
        Na_MSE_Cl_phi_005_df_min,
        ax4a,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        Ca_MSE_Cl_phi_005_df,
        Ca_MSE_Cl_phi_005_df_min,
        ax5a,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        La_MSE_Cl_phi_005_df,
        La_MSE_Cl_phi_005_df_min,
        ax6a,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )

    plot_data(
        Na_MSE_Cl_phi_010_df,
        Na_MSE_Cl_phi_010_df_min,
        ax4b,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        Ca_MSE_Cl_phi_010_df,
        Ca_MSE_Cl_phi_010_df_min,
        ax5b,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        La_MSE_Cl_phi_010_df,
        La_MSE_Cl_phi_010_df_min,
        ax6b,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )

    plot_data(
        Na_MSE_Cl_phi_020_df,
        Na_MSE_Cl_phi_020_df_min,
        ax4c,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        Ca_MSE_Cl_phi_020_df,
        Ca_MSE_Cl_phi_020_df_min,
        ax5c,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        La_MSE_Cl_phi_020_df,
        La_MSE_Cl_phi_020_df_min,
        ax6c,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )

    plot_data(
        Na_MSE_Cl_phi_030_df,
        Na_MSE_Cl_phi_030_df_min,
        ax4d,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        Ca_MSE_Cl_phi_030_df,
        Ca_MSE_Cl_phi_030_df_min,
        ax5d,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        La_MSE_Cl_phi_030_df,
        La_MSE_Cl_phi_030_df_min,
        ax6d,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )

    plot_data(
        Na_MSE_Cl_phi_040_df,
        Na_MSE_Cl_phi_040_df_min,
        ax4e,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        Ca_MSE_Cl_phi_040_df,
        Ca_MSE_Cl_phi_040_df_min,
        ax5e,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )
    plot_data(
        La_MSE_Cl_phi_040_df,
        La_MSE_Cl_phi_040_df_min,
        ax6e,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
    )

    for ax in fig1.axes:
        ax.tick_params(axis="y", rotation=0)
    for ax in fig2.axes:
        ax.tick_params(axis="y", rotation=0)

    for ax in [ax3a, ax3b, ax3c, ax3d, ax3e, ax6a, ax6b, ax6c, ax6d, ax6e]:
        ax.set_xlabel(
            "$\mathbf{D_m}$/$\mathbf{l_m}$ (um/s)",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax1a, ax4a]:
        ax.set_ylabel(
            "$\mathbf{\phi_{Na}}$",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax2a, ax5a]:
        ax.set_ylabel(
            "$\mathbf{\phi_{Ca}}$",
            fontsize=fontsize,
            fontweight="bold",
        )
    for ax in [ax3a, ax6a]:
        ax.set_ylabel(
            "$\mathbf{\phi_{La}}$",
            fontsize=fontsize,
            fontweight="bold",
        )

    fig1.savefig("sieving_data_mae.png", dpi=600)
    fig2.savefig("sieving_data_mse.png", dpi=600)


def load_data(Cl_phi_key, filename):
    indices = [0.05, 0.1, 0.2, 0.3, 0.4]

    df = pd.read_csv(f"heat_map_data/Cl_phi_{Cl_phi_key}/{filename}.csv", index_col=0)

    # fixes indexing issues of the minimum annotation if NaNs present
    df = df.reindex(indices)

    return df


def find_minimums(df):
    indices = [0.05, 0.1, 0.2, 0.3, 0.4]
    columns = ["5", "10", "20", "40", "80"]

    df_min = pd.DataFrame(columns=columns, index=indices)
    min_val = df.min().min()
    df_min.at[df.stack().idxmin()] = f"*\n{min_val:.2e}"
    df_min.fillna("", inplace=True)

    return df_min


def plot_data(df, df_min, ax, cbar_ax, vmin, vmax):
    fontsize = 12

    tol_bright_hex = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "monotone_magenta", ["#ffffff", tol_bright_hex[5]], N=256
    )

    sns.heatmap(
        df,
        annot=df_min,
        fmt="",
        annot_kws={"size": fontsize, "fontweight": "bold"},
        ax=ax,
        cbar=True,
        cbar_ax=cbar_ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    ).invert_yaxis()


def save_data_comparison_csv(Cl_phi_key):
    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 15
    NFE_boundary_layer_thickness = 5
    NFE_membrane_thickness = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    Cl_Dm = 2.03  # um2/s

    # NF270_MC5_07_23_24_NaCl = {conc: sieving, ...}
    NF270_MC5_07_23_24_NaCl = {
        # 9.6199: 0.1875,
        30.5114: 0.2296,
        43.1343: 0.3566,
        54.5286: 0.4416,
        64.7312: 0.5049,
        74.1947: 0.5511,
        83.0378: 0.5910,
        91.3303: 0.6251,
        98.4631: 0.6356,
        105.1806: 0.6680,
    }
    NF270_MC3_07_11_24_SCaCl2 = {
        # 3.0588: 0.1893,
        10.4631: 0.2077,
        15.2044: 0.3179,
        19.6055: 0.3572,
        23.6953: 0.3761,
        27.5362: 0.3927,
        31.0213: 0.4157,
        34.3589: 0.4313,
        37.4959: 0.4388,
        40.3890: 0.4500,
    }
    NF270_MC2_05_21_24_LaCl3 = {
        # 1.4129: 0.0429,
        4.8072: 0.0240,
        7.0910: 0.0286,
        9.2925: 0.0312,
        11.3234: 0.0335,
        13.5457: 0.0343,
        15.5382: 0.0358,
        17.4718: 0.0377,
        19.3323: 0.0391,
        21.1177: 0.0419,
    }

    sample_folder = Path(
        f"multi_component_case_studies/DATA_comparison/Cl_phi_{Cl_phi_key}/"
    )
    # 80 characters (0-79) make up folder name before model name
    # multi_component_case_studies/DATA_comparison/Cl_phi_XXX/80umpers/cation_phi_XXX/

    case_study_list = [file for file in sample_folder.rglob("*") if file.is_file()]

    # sieving_dict = {Dm/l: {cation_phi: {conc: sieving, ...}, ...}, ...}
    Na_sieving_dict = {
        5: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        10: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        20: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        40: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        80: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
    }
    Ca_sieving_dict = {
        5: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        10: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        20: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        40: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        80: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
    }
    La_sieving_dict = {
        5: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        10: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        20: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        40: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
        80: {0.05: {}, 0.10: {}, 0.20: {}, 0.30: {}, 0.40: {}},
    }

    Dm_over_l_sensitivity = [80, 40, 20, 10, 5]  # um/s
    Dm_over_l_sensitivity_keys = ["80", "40", "20", "10", "05"]  # um/s

    phi_star_sensitivity = [0.4, 0.3, 0.2, 0.1, 0.05]
    phi_star_sensitivity_keys = ["040", "030", "020", "010", "005"]

    for case_study in case_study_list:
        concentration = float(50)  # mM

        Dm_over_l_key = str(case_study)[56:58]
        Dm_over_l = Dm_over_l_sensitivity[
            Dm_over_l_sensitivity_keys.index(Dm_over_l_key)
        ]

        cation_phi_key = str(case_study)[76:79]
        cation_phi = phi_star_sensitivity[
            phi_star_sensitivity_keys.index(cation_phi_key)
        ]

        cation = str(case_study)[80:82]
        cation_list = [cation]

        if cation == "Na":
            chloride_multiplier = 1
            sieving_dict = Na_sieving_dict
        elif cation == "Ca":
            chloride_multiplier = 2
            sieving_dict = Ca_sieving_dict
        elif cation == "La":
            chloride_multiplier = 3
            sieving_dict = La_sieving_dict

        inlet_concentration = {
            "feed": {
                cation: concentration,
                "Cl": chloride_multiplier * concentration,
            },
            "diafiltrate": {
                cation: 1e-10,
                "Cl": chloride_multiplier * 1e-10,
            },
        }

        non_Donnan_partition_dict = {
            cation: cation_phi,
            "Cl": 0.4,
        }

        l_um = Cl_Dm / Dm_over_l  # um
        l_m = l_um / 1e6  # m

        model = build_model(
            cation_list=cation_list,
            inlet_concentration=inlet_concentration,
            default_args=default_args,
            H_feed_guess=1,
            H_permeate_guess=1,
            non_Donnan_partition_dict=non_Donnan_partition_dict,
            data_membrane_thickness=l_m,
            NFE_args=NFE_args,
            initialize=False,
        )
        from_json(model, fname=case_study)

        average_variable_dict = get_model_averages(model, cation)

        feed_conc_predicted = value(
            model.fs.membrane.retentate_conc_mol_comp[0, 0, cation]
        )
        obs_sieving_predicted = average_variable_dict["observed_sieving"]["avg"]

        sieving_dict[Dm_over_l][cation_phi].update(
            {round(feed_conc_predicted, 4): obs_sieving_predicted}
        )

    # residual = actual - predicted
    Na_sieving_residual_dict = {
        Dm_over_l: {
            cation_phi: {
                conc: NF270_MC5_07_23_24_NaCl[conc]
                - Na_sieving_dict[Dm_over_l][cation_phi][conc]
                for conc in Na_sieving_dict[Dm_over_l][cation_phi].keys()
            }
            for cation_phi in Na_sieving_dict[Dm_over_l].keys()
        }
        for Dm_over_l in Na_sieving_dict.keys()
    }
    Ca_sieving_residual_dict = {
        Dm_over_l: {
            cation_phi: {
                conc: NF270_MC3_07_11_24_SCaCl2[conc]
                - Ca_sieving_dict[Dm_over_l][cation_phi][conc]
                for conc in Ca_sieving_dict[Dm_over_l][cation_phi].keys()
            }
            for cation_phi in Ca_sieving_dict[Dm_over_l].keys()
        }
        for Dm_over_l in Ca_sieving_dict.keys()
    }
    La_sieving_residual_dict = {
        Dm_over_l: {
            cation_phi: {
                conc: NF270_MC2_05_21_24_LaCl3[conc]
                - La_sieving_dict[Dm_over_l][cation_phi][conc]
                for conc in La_sieving_dict[Dm_over_l][cation_phi].keys()
            }
            for cation_phi in La_sieving_dict[Dm_over_l].keys()
        }
        for Dm_over_l in La_sieving_dict.keys()
    }

    Na_MAE_dict = {
        Dm_over_l: {
            cation_phi: (
                1 / len(Na_sieving_residual_dict[Dm_over_l][cation_phi].keys())
            )
            * sum(
                abs(residual)
                for residual in Na_sieving_residual_dict[Dm_over_l][cation_phi].values()
            )
            for cation_phi in Na_sieving_residual_dict[Dm_over_l].keys()
            if len(Na_sieving_residual_dict[Dm_over_l][cation_phi].keys()) != 0
        }
        for Dm_over_l in Na_sieving_residual_dict.keys()
    }
    Ca_MAE_dict = {
        Dm_over_l: {
            cation_phi: (
                1 / len(Ca_sieving_residual_dict[Dm_over_l][cation_phi].keys())
            )
            * sum(
                abs(residual)
                for residual in Ca_sieving_residual_dict[Dm_over_l][cation_phi].values()
            )
            for cation_phi in Ca_sieving_residual_dict[Dm_over_l].keys()
            if len(Ca_sieving_residual_dict[Dm_over_l][cation_phi].keys()) != 0
        }
        for Dm_over_l in Ca_sieving_residual_dict.keys()
    }
    La_MAE_dict = {
        Dm_over_l: {
            cation_phi: (
                1 / len(La_sieving_residual_dict[Dm_over_l][cation_phi].keys())
            )
            * sum(
                abs(residual)
                for residual in La_sieving_residual_dict[Dm_over_l][cation_phi].values()
            )
            for cation_phi in La_sieving_residual_dict[Dm_over_l].keys()
            if len(La_sieving_residual_dict[Dm_over_l][cation_phi].keys()) != 0
        }
        for Dm_over_l in La_sieving_residual_dict.keys()
    }

    Na_MSE_dict = {
        Dm_over_l: {
            cation_phi: (
                1 / len(Na_sieving_residual_dict[Dm_over_l][cation_phi].keys())
            )
            * sum(
                residual**2
                for residual in Na_sieving_residual_dict[Dm_over_l][cation_phi].values()
            )
            for cation_phi in Na_sieving_residual_dict[Dm_over_l].keys()
            if len(Na_sieving_residual_dict[Dm_over_l][cation_phi].keys()) != 0
        }
        for Dm_over_l in Na_sieving_residual_dict.keys()
    }
    Ca_MSE_dict = {
        Dm_over_l: {
            cation_phi: (
                1 / len(Ca_sieving_residual_dict[Dm_over_l][cation_phi].keys())
            )
            * sum(
                residual**2
                for residual in Ca_sieving_residual_dict[Dm_over_l][cation_phi].values()
            )
            for cation_phi in Ca_sieving_residual_dict[Dm_over_l].keys()
            if len(Ca_sieving_residual_dict[Dm_over_l][cation_phi].keys()) != 0
        }
        for Dm_over_l in Ca_sieving_residual_dict.keys()
    }
    La_MSE_dict = {
        Dm_over_l: {
            cation_phi: (
                1 / len(La_sieving_residual_dict[Dm_over_l][cation_phi].keys())
            )
            * sum(
                residual**2
                for residual in La_sieving_residual_dict[Dm_over_l][cation_phi].values()
            )
            for cation_phi in La_sieving_residual_dict[Dm_over_l].keys()
            if len(La_sieving_residual_dict[Dm_over_l][cation_phi].keys()) != 0
        }
        for Dm_over_l in La_sieving_residual_dict.keys()
    }

    Na_MAE_df = pd.DataFrame(Na_MAE_dict)
    Ca_MAE_df = pd.DataFrame(Ca_MAE_dict)
    La_MAE_df = pd.DataFrame(La_MAE_dict)

    Na_MSE_df = pd.DataFrame(Na_MSE_dict)
    Ca_MSE_df = pd.DataFrame(Ca_MSE_dict)
    La_MSE_df = pd.DataFrame(La_MSE_dict)

    print(f"Na (MAE)\n{Na_MAE_df}")
    print(f"Na (MSE)\n{Na_MSE_df}")

    print(f"Ca (MAE)\n{Ca_MAE_df}")
    print(f"Ca (MSE)\n{Ca_MSE_df}")

    print(f"La (MAE)\n{La_MAE_df}")
    print(f"La (MSE)\n{La_MSE_df}")

    Na_MAE_df.to_csv(f"heat_map_data/Cl_phi_{Cl_phi_key}/Na_MAE.csv")
    Ca_MAE_df.to_csv(f"heat_map_data/Cl_phi_{Cl_phi_key}/Ca_MAE.csv")
    La_MAE_df.to_csv(f"heat_map_data/Cl_phi_{Cl_phi_key}/La_MAE.csv")

    Na_MSE_df.to_csv(f"heat_map_data/Cl_phi_{Cl_phi_key}/Na_MSE.csv")
    Ca_MSE_df.to_csv(f"heat_map_data/Cl_phi_{Cl_phi_key}/Ca_MSE.csv")
    La_MSE_df.to_csv(f"heat_map_data/Cl_phi_{Cl_phi_key}/La_MSE.csv")


if __name__ == "__main__":
    main()
