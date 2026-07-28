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
import numpy as np

from prommis.nanofiltration.multi_component_analysis import (
    build_model,
    get_model_averages,
)


def main():
    # save_data_comparison_csv(Cl_phi_key="040")
    # save_data_comparison_csv(Cl_phi_key="030")
    # save_data_comparison_csv(Cl_phi_key="020")
    # save_data_comparison_csv(Cl_phi_key="010")
    # save_data_comparison_csv(Cl_phi_key="005")

    generate_data_comparison_heat_maps()
    generate_N_data_heat_maps()

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

    # Na_MAE_Cl_phi_020_df = load_data("020", "Na_MAE")
    # Ca_MAE_Cl_phi_020_df = load_data("020", "Ca_MAE")
    # La_MAE_Cl_phi_020_df = load_data("020", "La_MAE")
    # Na_MSE_Cl_phi_020_df = load_data("020", "Na_MSE")
    # Ca_MSE_Cl_phi_020_df = load_data("020", "Ca_MSE")
    # La_MSE_Cl_phi_020_df = load_data("020", "La_MSE")

    # Na_MAE_Cl_phi_030_df = load_data("030", "Na_MAE")
    # Ca_MAE_Cl_phi_030_df = load_data("030", "Ca_MAE")
    # La_MAE_Cl_phi_030_df = load_data("030", "La_MAE")
    # Na_MSE_Cl_phi_030_df = load_data("030", "Na_MSE")
    # Ca_MSE_Cl_phi_030_df = load_data("030", "Ca_MSE")
    # La_MSE_Cl_phi_030_df = load_data("030", "La_MSE")

    # Na_MAE_Cl_phi_040_df = load_data("040", "Na_MAE")
    # Ca_MAE_Cl_phi_040_df = load_data("040", "Ca_MAE")
    # La_MAE_Cl_phi_040_df = load_data("040", "La_MAE")
    # Na_MSE_Cl_phi_040_df = load_data("040", "Na_MSE")
    # Ca_MSE_Cl_phi_040_df = load_data("040", "Ca_MSE")
    # La_MSE_Cl_phi_040_df = load_data("040", "La_MSE")

    vmin_mae = 0
    vmax_mae = 0.51

    vmin_mse = 0
    vmax_mse = 0.51

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

    # Na_MAE_Cl_phi_020_df_min = find_minimums(Na_MAE_Cl_phi_020_df)
    # Ca_MAE_Cl_phi_020_df_min = find_minimums(Ca_MAE_Cl_phi_020_df)
    # La_MAE_Cl_phi_020_df_min = find_minimums(La_MAE_Cl_phi_020_df)
    # Na_MSE_Cl_phi_020_df_min = find_minimums(Na_MSE_Cl_phi_020_df)
    # Ca_MSE_Cl_phi_020_df_min = find_minimums(Ca_MSE_Cl_phi_020_df)
    # La_MSE_Cl_phi_020_df_min = find_minimums(La_MSE_Cl_phi_020_df)

    # Na_MAE_Cl_phi_030_df_min = find_minimums(Na_MAE_Cl_phi_030_df)
    # Ca_MAE_Cl_phi_030_df_min = find_minimums(Ca_MAE_Cl_phi_030_df)
    # La_MAE_Cl_phi_030_df_min = find_minimums(La_MAE_Cl_phi_030_df)
    # Na_MSE_Cl_phi_030_df_min = find_minimums(Na_MSE_Cl_phi_030_df)
    # Ca_MSE_Cl_phi_030_df_min = find_minimums(Ca_MSE_Cl_phi_030_df)
    # La_MSE_Cl_phi_030_df_min = find_minimums(La_MSE_Cl_phi_030_df)

    # Na_MAE_Cl_phi_040_df_min = find_minimums(Na_MAE_Cl_phi_040_df)
    # Ca_MAE_Cl_phi_040_df_min = find_minimums(Ca_MAE_Cl_phi_040_df)
    # La_MAE_Cl_phi_040_df_min = find_minimums(La_MAE_Cl_phi_040_df)
    # Na_MSE_Cl_phi_040_df_min = find_minimums(Na_MSE_Cl_phi_040_df)
    # Ca_MSE_Cl_phi_040_df_min = find_minimums(Ca_MSE_Cl_phi_040_df)
    # La_MSE_Cl_phi_040_df_min = find_minimums(La_MSE_Cl_phi_040_df)

    fig1, (
        (ax1a, ax1b),  # , ax1c, ax1d, ax1e),
        (ax2a, ax2b),  # , ax2c, ax2d, ax2e),
        (ax3a, ax3b),  # , ax3c, ax3d, ax3e),
    ) = plt.subplots(
        3,
        2,
        dpi=75,
        figsize=(11, 14.1),
        sharex=True,
    )
    fig1.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
    fig2, (
        (ax4a, ax4b),  # , ax4c, ax4d, ax4e),
        (ax5a, ax5b),  # , ax5c, ax5d, ax5e),
        (ax6a, ax6b),  # , ax6c, ax6d, ax6e),
    ) = plt.subplots(
        3,
        2,
        dpi=75,
        figsize=(11, 14.1),
        sharex=True,
    )
    fig2.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])

    for ax in fig1.axes:
        ax.tick_params(bottom=False, labelsize=fontsize)
    for ax in fig2.axes:
        ax.tick_params(bottom=False, labelsize=fontsize)

    fig1.suptitle("Mean Absolute Error", fontsize=fontsize, fontweight="bold")
    fig2.suptitle("Mean Squared Error", fontsize=fontsize, fontweight="bold")
    for ax in [ax1a, ax4a]:
        ax.set_title(
            "$\mathbf{\phi_{Cl}}$ = 0.05", fontsize=fontsize, fontweight="bold"
        )
    for ax in [ax1b, ax4b]:
        ax.set_title("$\mathbf{\phi_{Cl}}$ = 0.1", fontsize=fontsize, fontweight="bold")
    # for ax in [ax1c, ax4c]:
    #     ax.set_title("$\mathbf{\phi_{Cl}}$ = 0.2", fontsize=fontsize, fontweight="bold")
    # for ax in [ax1d, ax4d]:
    #     ax.set_title("$\mathbf{\phi_{Cl}}$ = 0.3", fontsize=fontsize, fontweight="bold")
    # for ax in [ax1e, ax4e]:
    #     ax.set_title("$\mathbf{\phi_{Cl}}$ = 0.4", fontsize=fontsize, fontweight="bold")

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
        yticklabels=True,
    )
    plot_data(
        Ca_MAE_Cl_phi_005_df,
        Ca_MAE_Cl_phi_005_df_min,
        ax2a,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
        yticklabels=True,
    )
    plot_data(
        La_MAE_Cl_phi_005_df,
        La_MAE_Cl_phi_005_df_min,
        ax3a,
        cbar_ax_mae,
        vmin_mae,
        vmax_mae,
        yticklabels=True,
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

    # plot_data(
    #     Na_MAE_Cl_phi_020_df,
    #     Na_MAE_Cl_phi_020_df_min,
    #     ax1c,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )
    # plot_data(
    #     Ca_MAE_Cl_phi_020_df,
    #     Ca_MAE_Cl_phi_020_df_min,
    #     ax2c,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )
    # plot_data(
    #     La_MAE_Cl_phi_020_df,
    #     La_MAE_Cl_phi_020_df_min,
    #     ax3c,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )

    # plot_data(
    #     Na_MAE_Cl_phi_030_df,
    #     Na_MAE_Cl_phi_030_df_min,
    #     ax1d,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )
    # plot_data(
    #     Ca_MAE_Cl_phi_030_df,
    #     Ca_MAE_Cl_phi_030_df_min,
    #     ax2d,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )
    # plot_data(
    #     La_MAE_Cl_phi_030_df,
    #     La_MAE_Cl_phi_030_df_min,
    #     ax3d,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )

    # plot_data(
    #     Na_MAE_Cl_phi_040_df,
    #     Na_MAE_Cl_phi_040_df_min,
    #     ax1e,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )
    # plot_data(
    #     Ca_MAE_Cl_phi_040_df,
    #     Ca_MAE_Cl_phi_040_df_min,
    #     ax2e,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )
    # plot_data(
    #     La_MAE_Cl_phi_040_df,
    #     La_MAE_Cl_phi_040_df_min,
    #     ax3e,
    #     cbar_ax_mae,
    #     vmin_mae,
    #     vmax_mae,
    # )

    plot_data(
        Na_MSE_Cl_phi_005_df,
        Na_MSE_Cl_phi_005_df_min,
        ax4a,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
        yticklabels=True,
    )
    plot_data(
        Ca_MSE_Cl_phi_005_df,
        Ca_MSE_Cl_phi_005_df_min,
        ax5a,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
        yticklabels=True,
    )
    plot_data(
        La_MSE_Cl_phi_005_df,
        La_MSE_Cl_phi_005_df_min,
        ax6a,
        cbar_ax_mse,
        vmin_mse,
        vmax_mse,
        yticklabels=True,
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

    # plot_data(
    #     Na_MSE_Cl_phi_020_df,
    #     Na_MSE_Cl_phi_020_df_min,
    #     ax4c,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )
    # plot_data(
    #     Ca_MSE_Cl_phi_020_df,
    #     Ca_MSE_Cl_phi_020_df_min,
    #     ax5c,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )
    # plot_data(
    #     La_MSE_Cl_phi_020_df,
    #     La_MSE_Cl_phi_020_df_min,
    #     ax6c,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )

    # plot_data(
    #     Na_MSE_Cl_phi_030_df,
    #     Na_MSE_Cl_phi_030_df_min,
    #     ax4d,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )
    # plot_data(
    #     Ca_MSE_Cl_phi_030_df,
    #     Ca_MSE_Cl_phi_030_df_min,
    #     ax5d,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )
    # plot_data(
    #     La_MSE_Cl_phi_030_df,
    #     La_MSE_Cl_phi_030_df_min,
    #     ax6d,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )

    # plot_data(
    #     Na_MSE_Cl_phi_040_df,
    #     Na_MSE_Cl_phi_040_df_min,
    #     ax4e,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )
    # plot_data(
    #     Ca_MSE_Cl_phi_040_df,
    #     Ca_MSE_Cl_phi_040_df_min,
    #     ax5e,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )
    # plot_data(
    #     La_MSE_Cl_phi_040_df,
    #     La_MSE_Cl_phi_040_df_min,
    #     ax6e,
    #     cbar_ax_mse,
    #     vmin_mse,
    #     vmax_mse,
    # )

    for ax in fig1.axes:
        ax.tick_params(axis="y", rotation=0)
    for ax in fig2.axes:
        ax.tick_params(axis="y", rotation=0)

    for ax in [ax3a, ax3b, ax6a, ax6b]:
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

    fig1.savefig("sieving_data_mae_fine.png", dpi=600)
    fig2.savefig("sieving_data_mse_fine.png", dpi=600)


def generate_N_data_heat_maps():
    fontsize = 14

    # load data from csv files
    Na_N_Cl_phi_005_df = load_data("005", "Na_N", N=True)
    Ca_N_Cl_phi_005_df = load_data("005", "Ca_N", N=True)
    La_N_Cl_phi_005_df = load_data("005", "La_N", N=True)

    Na_N_Cl_phi_010_df = load_data("010", "Na_N", N=True)
    Ca_N_Cl_phi_010_df = load_data("010", "Ca_N", N=True)
    La_N_Cl_phi_010_df = load_data("010", "La_N", N=True)

    # Na_N_Cl_phi_020_df = load_data("020", "Na_N", N=True)
    # Ca_N_Cl_phi_020_df = load_data("020", "Ca_N", N=True)
    # La_N_Cl_phi_020_df = load_data("020", "La_N", N=True)

    # Na_N_Cl_phi_030_df = load_data("030", "Na_N", N=True)
    # Ca_N_Cl_phi_030_df = load_data("030", "Ca_N", N=True)
    # La_N_Cl_phi_030_df = load_data("030", "La_N", N=True)

    # Na_N_Cl_phi_040_df = load_data("040", "Na_N", N=True)
    # Ca_N_Cl_phi_040_df = load_data("040", "Ca_N", N=True)
    # La_N_Cl_phi_040_df = load_data("040", "La_N", N=True)

    # add stars at the minumum of each subplot
    fig1, (
        (ax1a, ax1b),  # , ax1c, ax1d, ax1e),
        (ax2a, ax2b),  # , ax2c, ax2d, ax2e),
        (ax3a, ax3b),  # , ax3c, ax3d, ax3e),
    ) = plt.subplots(
        3,
        2,
        dpi=75,
        figsize=(11, 14.1),
        sharex=True,
    )
    fig1.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])

    for ax in fig1.axes:
        ax.tick_params(bottom=False, labelsize=fontsize)

    fig1.suptitle("Number of Points", fontsize=fontsize, fontweight="bold")
    ax1a.set_title("$\mathbf{\phi_{Cl}}$ = 0.05", fontsize=fontsize, fontweight="bold")
    ax1b.set_title("$\mathbf{\phi_{Cl}}$ = 0.1", fontsize=fontsize, fontweight="bold")
    # ax1c.set_title("$\mathbf{\phi_{Cl}}$ = 0.2", fontsize=fontsize, fontweight="bold")
    # ax1d.set_title("$\mathbf{\phi_{Cl}}$ = 0.3", fontsize=fontsize, fontweight="bold")
    # ax1e.set_title("$\mathbf{\phi_{Cl}}$ = 0.4", fontsize=fontsize, fontweight="bold")

    cbar_ax = fig1.add_axes([0.91, 0.2, 0.02, 0.6])
    cbar_ax.tick_params(labelsize=fontsize)

    plot_N_data(Na_N_Cl_phi_005_df, ax1a, cbar_ax, yticklabels=True)
    plot_N_data(Ca_N_Cl_phi_005_df, ax2a, cbar_ax, yticklabels=True)
    plot_N_data(La_N_Cl_phi_005_df, ax3a, cbar_ax, yticklabels=True)

    plot_N_data(Na_N_Cl_phi_010_df, ax1b, cbar_ax)
    plot_N_data(Ca_N_Cl_phi_010_df, ax2b, cbar_ax)
    plot_N_data(La_N_Cl_phi_010_df, ax3b, cbar_ax)

    # plot_N_data(Na_N_Cl_phi_020_df, ax1c, cbar_ax)
    # plot_N_data(Ca_N_Cl_phi_020_df, ax2c, cbar_ax)
    # plot_N_data(La_N_Cl_phi_020_df, ax3c, cbar_ax)

    # plot_N_data(Na_N_Cl_phi_030_df, ax1d, cbar_ax)
    # plot_N_data(Ca_N_Cl_phi_030_df, ax2d, cbar_ax)
    # plot_N_data(La_N_Cl_phi_030_df, ax3d, cbar_ax)

    # plot_N_data(Na_N_Cl_phi_040_df, ax1e, cbar_ax)
    # plot_N_data(Ca_N_Cl_phi_040_df, ax2e, cbar_ax)
    # plot_N_data(La_N_Cl_phi_040_df, ax3e, cbar_ax)

    for ax in fig1.axes:
        ax.tick_params(axis="y", rotation=0)

    for ax in [ax3a, ax3b]:  # , ax3c, ax3d, ax3e]:
        ax.set_xlabel(
            "$\mathbf{D_m}$/$\mathbf{l_m}$ (um/s)",
            fontsize=fontsize,
            fontweight="bold",
        )
    ax1a.set_ylabel(
        "$\mathbf{\phi_{Na}}$",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax2a.set_ylabel(
        "$\mathbf{\phi_{Ca}}$",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax3a.set_ylabel(
        "$\mathbf{\phi_{La}}$",
        fontsize=fontsize,
        fontweight="bold",
    )

    fig1.savefig("N_data_fine.png", dpi=600)


def load_data(Cl_phi_key, filename, N=False):
    if filename[0:2] == "Na":
        # indices = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        indices = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    elif filename[0:2] == "La":
        # indices = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
        indices = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    else:
        # indices = [0.05, 0.1, 0.2, 0.3, 0.4]
        indices = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    if N:
        folder = "N_data"
    else:
        folder = "heat_map_data"

    df = pd.read_csv(f"{folder}/Cl_phi_{Cl_phi_key}/{filename}.csv", index_col=0)

    # fixes indexing issues of the minimum annotation if NaNs present
    df = df.reindex(indices)

    return df


def find_minimums(df):
    Na_indices = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    Ca_indices = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    La_indices = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

    if df.index.tolist() == Na_indices:
        indices = Na_indices
    elif df.index.tolist() == La_indices:
        indices = La_indices
    else:
        indices = Ca_indices

    # columns = ["5", "10", "20", "40", "80"]
    columns = ["40", "50", "60", "70", "80"]

    df_min = pd.DataFrame(columns=columns, index=indices)
    min_val = df.min().min()
    df_min.at[df.stack().idxmin()] = f"*\n{min_val:.2e}"
    # fixes any indexing issues
    df_min = df_min.reindex(indices)
    df_min.fillna("", inplace=True)

    return df_min


def plot_data(df, df_min, ax, cbar_ax, vmin, vmax, yticklabels=False):
    fontsize = 12

    tol_iridescent_sequential_hex = [
        # "#FEFBE9",
        # "#FCF7D5",
        # "#F5F3C1",
        "#EAF0B5",
        "#DDECBF",
        "#D0E7CA",
        "#C2E3D2",
        "#B5DDD8",
        "#A8D8DC",
        "#9BD2E1",
        "#8DCBE4",
        "#81C4E7",
        "#7BBCE7",
        "#7EB2E4",
        "#88A5DD",
        "#9398D2",
        "#9B8AC4",
        "#9D7DB2",
        "#9A709E",
        "#906388",
        "#805770",
        "#684957",
        "#46353A",
    ]
    # tol_discrete_rainbow_hex = [
    #     "#D9CCE3",
    #     "#CAACCB",
    #     "#BA8DB4",
    #     "#AA6F9E",
    #     "#994F88",
    #     "#882E72",
    #     "#1965B0",
    #     "#437DBF",
    #     "#6195CF",
    #     "#7BAFDE",
    #     "#4EB265",
    #     "#90C987",
    #     "#CAE0AB",
    #     "#F7F056",
    #     "#F6C141",
    #     "#F1932D",
    #     "#E8601C",
    #     "#DC050C",
    #     "#A5170E",
    #     "#72190E",
    # ]
    cmap = mcolors.ListedColormap(tol_iridescent_sequential_hex)
    bins = np.linspace(0, 0.51, 21).tolist()
    norm = mcolors.BoundaryNorm(bins, cmap.N)

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
        norm=norm,
        yticklabels=yticklabels,
    ).invert_yaxis()


def plot_N_data(df, ax, cbar_ax, yticklabels=False):
    fontsize = 12

    vmin = 0
    vmax = 9

    tol_iridescent_ten_sequential_hex = [
        # "#FEFBE9",
        # "#FCF7D5",
        "#F5F3C1",
        # "#EAF0B5",
        "#DDECBF",
        # "#D0E7CA",
        "#C2E3D2",
        # "#B5DDD8",
        "#A8D8DC",
        # "#9BD2E1",
        "#8DCBE4",
        # "#81C4E7",
        "#7BBCE7",
        # "#7EB2E4",
        "#88A5DD",
        # "#9398D2",
        "#9B8AC4",
        # "#9D7DB2",
        "#9A709E",
        # "#906388",
        "#805770",
        # "#684957",
        "#46353A",
    ]
    cmap = mcolors.ListedColormap(tol_iridescent_ten_sequential_hex)
    bins = np.linspace(-0.5, 9.5, 11).tolist()

    norm = mcolors.BoundaryNorm(bins, cmap.N)

    sns.heatmap(
        df,
        annot=True,
        ax=ax,
        cbar=True,
        cbar_ax=cbar_ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        norm=norm,
        yticklabels=yticklabels,
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
    # 81 characters (0-80) make up folder name before model name
    # multi_component_case_studies/DATA_comparison/Cl_phi_XXX/80umpers/cation_phi_XXXX/

    case_study_list = [file for file in sample_folder.rglob("*") if file.is_file()]

    # sieving_dict = {Dm/l: {cation_phi: {conc: sieving, ...}, ...}, ...}
    Na_sieving_dict = {
        40: {
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
            0.55: {},
            0.60: {},
            0.65: {},
            0.70: {},
        },
        50: {
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
            0.55: {},
            0.60: {},
            0.65: {},
            0.70: {},
        },
        60: {
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
            0.55: {},
            0.60: {},
            0.65: {},
            0.70: {},
        },
        70: {
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
            0.55: {},
            0.60: {},
            0.65: {},
            0.70: {},
        },
        80: {
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
            0.55: {},
            0.60: {},
            0.65: {},
            0.70: {},
        },
    }
    Ca_sieving_dict = {
        40: {
            0.15: {},
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
        },
        50: {
            0.15: {},
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
        },
        60: {
            0.15: {},
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
        },
        70: {
            0.15: {},
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
        },
        80: {
            0.15: {},
            0.20: {},
            0.25: {},
            0.30: {},
            0.35: {},
            0.40: {},
            0.45: {},
            0.50: {},
        },
    }
    La_sieving_dict = {
        40: {
            0.0005: {},
            0.001: {},
            0.005: {},
            0.01: {},
            0.05: {},
            0.10: {},
            0.15: {},
            0.20: {},
            0.25: {},
        },
        50: {
            0.0005: {},
            0.001: {},
            0.005: {},
            0.01: {},
            0.05: {},
            0.10: {},
            0.15: {},
            0.20: {},
            0.25: {},
        },
        60: {
            0.0005: {},
            0.001: {},
            0.005: {},
            0.01: {},
            0.05: {},
            0.10: {},
            0.15: {},
            0.20: {},
            0.25: {},
        },
        70: {
            0.0005: {},
            0.001: {},
            0.005: {},
            0.01: {},
            0.05: {},
            0.10: {},
            0.15: {},
            0.20: {},
            0.25: {},
        },
        80: {
            0.0005: {},
            0.001: {},
            0.005: {},
            0.01: {},
            0.05: {},
            0.10: {},
            0.15: {},
            0.20: {},
            0.25: {},
        },
    }

    Dm_over_l_sensitivity = [80, 70, 60, 50, 40]  # um/s
    Dm_over_l_sensitivity_keys = ["80", "70", "60", "50", "40"]  # um/s

    cation_phi_star_sensitivity = [
        0.7,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.05,
        0.01,
        0.005,
        0.001,
        0.0005,
    ]
    cation_phi_star_sensitivity_keys = [
        "0700",
        "0650",
        "0600",
        "0550",
        "0500",
        "0450",
        "0400",
        "0350",
        "0300",
        "0250",
        "0200",
        "0150",
        "0100",
        "0050",
        "0010",
        "0005",
        "0001",
        "0000",
    ]

    chloride_phi_star_sensitivity = [0.1, 0.05]
    chloride_phi_star_sensitivity_keys = ["010", "005"]

    for case_study in case_study_list:
        concentration = float(50)  # mM

        Dm_over_l_key = str(case_study)[56:58]
        Dm_over_l = Dm_over_l_sensitivity[
            Dm_over_l_sensitivity_keys.index(Dm_over_l_key)
        ]

        cation_phi_key = str(case_study)[76:80]
        cation_phi = cation_phi_star_sensitivity[
            cation_phi_star_sensitivity_keys.index(cation_phi_key)
        ]

        cation = str(case_study)[81:83]
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
            "Cl": chloride_phi_star_sensitivity[
                chloride_phi_star_sensitivity_keys.index(Cl_phi_key)
            ],
        }

        l_um = Cl_Dm / Dm_over_l  # um
        l_m = l_um / 1e6  # m

        model = build_model(
            cation_list=cation_list,
            inlet_concentration=inlet_concentration,
            default_args=default_args,
            # H_feed_guess=1,
            # H_permeate_guess=1,
            non_Donnan_partition_dict=non_Donnan_partition_dict,
            data_membrane_thickness=l_m,
            NFE_args=NFE_args,
            initialize_and_solve=False,
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

    Na_N_dict = {
        Dm_over_l: {
            cation_phi: len(Na_sieving_dict[Dm_over_l][cation_phi])
            for cation_phi in Na_sieving_dict[Dm_over_l].keys()
        }
        for Dm_over_l in Na_sieving_dict.keys()
    }
    Ca_N_dict = {
        Dm_over_l: {
            cation_phi: len(Ca_sieving_dict[Dm_over_l][cation_phi])
            for cation_phi in Ca_sieving_dict[Dm_over_l].keys()
        }
        for Dm_over_l in Ca_sieving_dict.keys()
    }
    La_N_dict = {
        Dm_over_l: {
            cation_phi: len(La_sieving_dict[Dm_over_l][cation_phi])
            for cation_phi in La_sieving_dict[Dm_over_l].keys()
        }
        for Dm_over_l in La_sieving_dict.keys()
    }

    Na_N_df = pd.DataFrame(Na_N_dict)
    Ca_N_df = pd.DataFrame(Ca_N_dict)
    La_N_df = pd.DataFrame(La_N_dict)

    print(f"Na (N)\n{Na_N_df}")
    print(f"Ca (N)\n{Ca_N_df}")
    print(f"La (N)\n{La_N_df}")

    Na_N_df.to_csv(f"N_data/Cl_phi_{Cl_phi_key}/Na_N.csv")
    Ca_N_df.to_csv(f"N_data/Cl_phi_{Cl_phi_key}/Ca_N.csv")
    La_N_df.to_csv(f"N_data/Cl_phi_{Cl_phi_key}/La_N.csv")


if __name__ == "__main__":
    main()
