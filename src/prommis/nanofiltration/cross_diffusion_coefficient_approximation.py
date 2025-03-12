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
    # (D_11_df, D_12_df, D_21_df, D_22_df, c1, c2) = calculate_diffusion_coefficients()
    # plot_2D_diffusion_coefficients(D_11_df, D_12_df, D_21_df, D_22_df)
    # plot_3D_diffusion_coefficients()
    # linear_regression(D_11_df)
    # calculate_linearized_diffusion_coefficients(D_11_df)
    plot_3D_with_regression()


def calculate_D_11(z1,z2,z3,D1,D2,D3,c1,c2):
    D_11 = ((z1*z3*D1*D3 - (z1**2)*D1*D3)*c1 + (z2*z3*D1*D3 - (z2**2)*D1*D2)*c2) / (
        ((z1**2)*D1 - z1*z3*D3)*c1 + ((z2**2)*D2 - z2*z3*D3)*c2
    )
    return D_11

def calculate_D_12(z1,z2,z3,D1,D2,D3,c1,c2):
    D_12 = ((z1*z2*D1*D2 - z1*z2*D1*D3)*c1) / (
        ((z1**2)*D1 - z1*z3*D3)*c1 + ((z2**2)*D2 - z2*z3*D3)*c2
    )
    return D_12

def calculate_D_21(z1,z2,z3,D1,D2,D3,c1,c2):
    D_21 = ((z1*z2*D1*D2 - z1*z2*D2*D3)*c2) / (
        ((z1**2)*D1 - z1*z3*D3)*c1 + ((z2**2)*D2 - z2*z3*D3)*c2
    )
    return D_21

def calculate_D_22(z1,z2,z3,D1,D2,D3,c1,c2):
    D_22 = ((z1*z2*D2*D3 - (z1**2)*D1*D2)*c1 + (z2*z3*D2*D3 - (z2**2)*D2*D3)*c2) / (
        ((z1**2)*D1 - z1*z3*D3)*c1 + ((z2**2)*D2 - z2*z3*D3)*c2
    )
    return D_22

def set_concentration_ranges():
    c1_vals = np.arange(0.1,10,0.1)
    c2_vals = np.arange(0.1,10,0.1)

    return (c1_vals, c2_vals)

def calculate_diffusion_coefficients():
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.7e-6 # m2/h
    D2 = 2.64e-6 # m2/h
    D3 = 7.3e-6 # m2/h

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
            D_11_vals.append((calculate_D_11(z1,z2,z3,D1,D2,D3,c1,c2)))
            D_12_vals.append((calculate_D_12(z1,z2,z3,D1,D2,D3,c1,c2)))
            D_21_vals.append((calculate_D_21(z1,z2,z3,D1,D2,D3,c1,c2)))
            D_22_vals.append((calculate_D_22(z1,z2,z3,D1,D2,D3,c1,c2)))
        d11[f"{c1.round(1)}"] = D_11_vals
        d12[f"{c1.round(1)}"] = D_12_vals
        d21[f"{c1.round(1)}"] = D_21_vals
        d22[f"{c1.round(1)}"] = D_22_vals
        D_11_vals = []
        D_12_vals = []
        D_21_vals = []
        D_22_vals = []

    D_11_df = DataFrame(index=c2_list,data=d11)
    D_12_df = DataFrame(index=c2_list,data=d12)
    D_21_df = DataFrame(index=c2_list,data=d21)
    D_22_df = DataFrame(index=c2_list,data=d22)
    
    return (D_11_df, D_12_df, D_21_df, D_22_df, c1_vals, c2_vals)

def plot_2D_diffusion_coefficients(D_11_df, D_12_df, D_21_df, D_22_df):
    figs, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)
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
    ax = plt.figure().add_subplot(projection='3d')

    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.7e-6 # m2/h
    D2 = 2.64e-6 # m2/h
    D3 = 7.3e-6 # m2/h

    (c1_vals, c2_vals) = set_concentration_ranges()

    c1, c2 = np.meshgrid(c1_vals, c2_vals)
    D_11 = ((z1*z3*D1*D3 - (z1**2)*D1*D3)*c1 + (z2*z3*D1*D3 - (z2**2)*D1*D2)*c2) / (
        ((z1**2)*D1 - z1*z3*D3)*c1 + ((z2**2)*D2 - z2*z3*D3)*c2
    )

    ax.plot_surface(
        c1,
        c2,
        D_11,
        cmap='mako',
    )

    ax.set_xlabel("Lithium Concentration (kg/m3)")
    ax.set_ylabel("Cobalt Concentration (kg/m3)")

    plt.show()

def linear_regression(D_11_df):
    m = ConcreteModel()

    m.beta_0 = Var(initialize=0)
    m.beta_1 = Var(initialize=1)
    m.beta_2 = Var(initialize=1)

    m.c1_data = Set(initialize=[float(c1) for c1 in D_11_df.columns])
    m.c2_data = Set(initialize=[float(c2) for c2 in D_11_df.index])

    m.D_11_prediction = Var(m.c1_data, m.c2_data,initialize=1e-6)

    def D_11_calculation(m,c1,c2):
        return m.D_11_prediction[c1,c2] == (m.beta_0 + m.beta_1*c1 + m.beta_2*c2)
    m.model_eqn = Constraint(m.c1_data, m.c2_data, rule=D_11_calculation)

    residual = 0
    for c1 in D_11_df.columns:
        for c2 in D_11_df.index:
            residual += (m.D_11_prediction[float(c1),float(c2)] - D_11_df.loc[c2][c1])**2

    m.objective = Objective(expr=residual)

    solver = SolverFactory('ipopt')
    solver.solve(m, tee=True)

    m.beta_0.display()
    m.beta_1.display()
    m.beta_2.display()

    return (m.beta_0.value, m.beta_1.value, m.beta_2.value)

def calculate_linearized_diffusion_coefficients(D_11_df):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.7e-6 # m2/h
    D2 = 2.64e-6 # m2/h
    D3 = 7.3e-6 # m2/h

    (c1_vals, c2_vals) = set_concentration_ranges()

    c2_list = []

    D_11_vals = []
    # D_12_vals = []
    # D_21_vals = []
    # D_22_vals = []

    d11 = {}
    # d12 = {}
    # d21 = {}
    # d22 = {}

    for c2 in c2_vals:
        c2_list.append(c2.round(1))

    (beta_0, beta_1, beta_2) = linear_regression(D_11_df)

    for c1 in c1_vals:
        for c2 in c2_vals:
            D_11_vals.append(beta_0 + beta_1*c1 + beta_2*c2)
            # D_12_vals.append((calculate_D_12(z1,z2,z3,D1,D2,D3,c1,c2)))
            # D_21_vals.append((calculate_D_21(z1,z2,z3,D1,D2,D3,c1,c2)))
            # D_22_vals.append((calculate_D_22(z1,z2,z3,D1,D2,D3,c1,c2)))
        d11[f"{c1.round(1)}"] = D_11_vals
        # d12[f"{c1.round(1)}"] = D_12_vals
        # d21[f"{c1.round(1)}"] = D_21_vals
        # d22[f"{c1.round(1)}"] = D_22_vals
        D_11_vals = []
        # D_12_vals = []
        # D_21_vals = []
        # D_22_vals = []

    D_11_df_linearized = DataFrame(index=c2_list,data=d11)
    # D_12_df = DataFrame(index=c2_list,data=d12)
    # D_21_df = DataFrame(index=c2_list,data=d21)
    # D_22_df = DataFrame(index=c2_list,data=d22)
    
    # return (D_11_df, D_12_df, D_21_df, D_22_df, c1_vals, c2_vals)
    print(D_11_df_linearized)
    return D_11_df_linearized

def plot_3D_with_regression():
    ax = plt.figure().add_subplot(projection='3d')

    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.7e-6 # m2/h
    D2 = 2.64e-6 # m2/h
    D3 = 7.3e-6 # m2/h

    (c1_vals, c2_vals) = set_concentration_ranges()

    c1, c2 = np.meshgrid(c1_vals, c2_vals)
    D_11 = ((z1*z3*D1*D3 - (z1**2)*D1*D3)*c1 + (z2*z3*D1*D3 - (z2**2)*D1*D2)*c2) / (
        ((z1**2)*D1 - z1*z3*D3)*c1 + ((z2**2)*D2 - z2*z3*D3)*c2
    )

    ax.plot_surface(
        c1,
        c2,
        D_11,
        cmap='mako',
    )
    (D_11_df, D_12_df, D_21_df, D_22_df, c1_list, c2_list) = calculate_diffusion_coefficients()
    linearized_D_11 = calculate_linearized_diffusion_coefficients(D_11_df)
    ax.plot_surface(
        c1,
        c2,
        linearized_D_11,
    )

    ax.set_xlabel("Lithium Concentration (kg/m3)")
    ax.set_ylabel("Cobalt Concentration (kg/m3)")

    plt.show()

if __name__ == "__main__":
    main()
