"""
Synthetic data generation for multi-component diafiltration model.

Author: Molly Dougher, edits from Claude
"""

from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    value,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.util.diagnostics_tools.diagnostics_toolbox import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

import numpy as np

from pandas import DataFrame

from prommis.nanofiltration.multi_component_diafiltration_stream_properties import (
    MultiComponentDiafiltrationStreamParameter,
)
from prommis.nanofiltration.multi_component_diafiltration_solute_properties import (
    MultiComponentDiafiltrationSoluteParameter,
)
from prommis.nanofiltration.multi_component_diafiltration import (
    MultiComponentDiafiltration,
)
from prommis.nanofiltration.multi_component_diafiltration_stage import (
    MultiComponentDiafiltrationStageFlowsheet,
)


def main():
    # generate_tube_data_per_salt(cation="Na")
    # generate_tube_data_per_salt(cation="Ca")
    # generate_tube_data_per_salt(cation="La")

    generate_single_salt_tube_data()  # adds water_flux output

    # generate_stage_data_per_salt(cation="Na")
    # generate_stage_data_per_salt(cation="Ca")
    # generate_stage_data_per_salt(cation="La")

    generate_single_salt_stage_data()


def build_tube_model(
    cation_list,
    inlet_concentration,
    default_args,
    NFE_args,
    applied_pressure,
    data_membrane_thickness,
    multiplier_H_feed=1.2,
    multiplier_H_perm=1,
    non_Donnan_partition_dict={},
    initialize_and_solve=True,
):
    anion_list, inlet_flow_volume, include_boundary_layer = default_args
    NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness = NFE_args
    # build flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.stream_properties = MultiComponentDiafiltrationStreamParameter(
        cation_list=cation_list,
        anion_list=anion_list,
    )
    m.fs.properties = MultiComponentDiafiltrationSoluteParameter(
        cation_list=cation_list,
        anion_list=anion_list,
        non_Donnan_partition_dict=non_Donnan_partition_dict,
    )

    # add feed blocks for feed and diafiltrate
    m.fs.feed_block = Feed(property_package=m.fs.stream_properties)
    m.fs.diafiltrate_block = Feed(property_package=m.fs.stream_properties)

    # add the membrane unit model
    m.fs.membrane = MultiComponentDiafiltration(
        property_package=m.fs.properties,
        cation_list=cation_list,
        anion_list=anion_list,
        include_boundary_layer=include_boundary_layer,
        total_membrane_thickness=data_membrane_thickness,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    # add product blocks for retentate and permeate
    m.fs.retentate_block = Product(property_package=m.fs.stream_properties)
    m.fs.permeate_block = Product(property_package=m.fs.stream_properties)

    # fix the degrees of freedom to their default values
    m.fs.membrane.total_module_length.fix()
    m.fs.membrane.total_membrane_length.fix()
    m.fs.membrane.feed_flow_volume.fix(inlet_flow_volume["feed"])
    m.fs.membrane.diafiltrate_flow_volume.fix(inlet_flow_volume["diafiltrate"])
    for t in m.fs.membrane.time:
        for j in m.fs.membrane.solutes:
            m.fs.membrane.feed_conc_mol_comp[t, j].fix(inlet_concentration["feed"][j])
            m.fs.membrane.diafiltrate_conc_mol_comp[t, j].fix(
                inlet_concentration["diafiltrate"][j]
            )
    if cation_list[0] == "Na":
        m.fs.membrane.applied_pressure.fix(3)
    elif cation_list[0] == "Ca":
        m.fs.membrane.applied_pressure.fix(8)
    elif cation_list[0] == "La":
        m.fs.membrane.applied_pressure.fix(10)

    # add and connect flowsheet streams
    m.fs.feed_stream = Arc(
        source=m.fs.feed_block.outlet,
        destination=m.fs.membrane.feed_inlet,
    )
    m.fs.diafiltrate_stream = Arc(
        source=m.fs.diafiltrate_block.outlet,
        destination=m.fs.membrane.diafiltrate_inlet,
    )
    m.fs.retentate_stream = Arc(
        source=m.fs.membrane.retentate_outlet,
        destination=m.fs.retentate_block.inlet,
    )
    m.fs.permeate_stream = Arc(
        source=m.fs.membrane.permeate_outlet,
        destination=m.fs.permeate_block.inlet,
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    # check structural warnings
    dt = DiagnosticsToolbox(m)
    dt.assert_no_structural_warnings()

    # initialize membrane model
    if initialize_and_solve:
        initialized_membrane_model = m.fs.membrane.default_initializer(
            multiplier_H_feed=multiplier_H_feed,
            multiplier_H_perm=multiplier_H_perm,
        )
        initialized_membrane_model.initialize(m.fs.membrane)

        if cation_list[0] == "Na":
            if (inlet_concentration["feed"][cation_list[0]] == 50) and (
                inlet_flow_volume["feed"] == 17
            ):
                m.fs.membrane.scaling_factor[
                    m.fs.membrane.membrane_D_tilde_calculation
                ] = 1e1
            elif (inlet_concentration["feed"][cation_list[0]] == 50) and (
                inlet_flow_volume["feed"] == 8
            ):
                m.fs.membrane.scaling_factor[m.fs.membrane.overall_mol_balance] = 1e1
                m.fs.membrane.scaling_factor[
                    m.fs.membrane.d_retentate_flow_volume_dx_disc_eq
                ] = 1e1

        if cation_list[0] == "Ca":
            if (
                (inlet_concentration["feed"][cation_list[0]] == 50)
                and (inlet_flow_volume["feed"] == 10)
                and (applied_pressure == 13)
            ):
                m.fs.membrane.scaling_factor[
                    m.fs.membrane.cation_bulk_flux_equation
                ] = 1e2
                m.fs.membrane.scaling_factor[
                    m.fs.membrane.membrane_D_tilde_calculation
                ] = 1e1

        if cation_list[0] == "La":
            if (inlet_concentration["feed"][cation_list[0]] == 50) and (
                inlet_flow_volume["feed"] == 12.5
            ):
                m.fs.membrane.scaling_factor[m.fs.membrane.molar_ion_flux] = 1e3

        solve_model(m)
        m.fs.membrane.applied_pressure.fix(applied_pressure)
        solve_model(m)

        dt = DiagnosticsToolbox(m)
        dt.assert_no_numerical_warnings()

    return m


def build_stage_model(
    cation_list,
    inlet_concentration,
    default_args,
    NFE_args,
    applied_pressure,
    pressure_drop,
    data_membrane_thickness,
    num_tubes,
    non_Donnan_partition_dict={},
    initialize_and_solve=True,
):
    anion_list, inlet_flow_volume, include_boundary_layer = default_args
    NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness = NFE_args

    membrane_stage = MultiComponentDiafiltrationStageFlowsheet(
        cation_list=cation_list,
        anion_list=anion_list,
        num_tubes=num_tubes,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration=inlet_concentration,
        non_Donnan_partition_dict=non_Donnan_partition_dict,
        include_boundary_layer=include_boundary_layer,
        total_membrane_thickness=data_membrane_thickness,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    model = membrane_stage.build_flowsheet()

    if initialize_and_solve:
        membrane_stage.initialize_flowsheet(model, applied_pressure, pressure_drop)
        membrane_stage.add_and_connect_streams(model)

        dt = DiagnosticsToolbox(model)
        dt.assert_no_structural_warnings()

        solve_model(model)

        dt.assert_no_numerical_warnings()

    return model


def solve_model(m):
    scaling = TransformationFactory("core.scale_model")
    scaled_model = scaling.create_using(m, rename=False)

    solver = SolverFactory("ipopt")
    results = solver.solve(scaled_model, tee=True)
    assert_optimal_termination(results)

    scaling.propagate_solution(scaled_model, m)

    return results


def generate_tube_data_per_salt(cation):
    anion_list = ["Cl"]
    diafiltrate = {"Na": 1e-10, "Ca": 1e-10, "La": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10

    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    concentrations = [50, 125, 200]
    feed_flows = [8, 12.5, 17]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_um = Dm_Cl / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m
    chloride_phi_star_value = 0.05

    if cation == "Na":
        chloride_multiplier = 1
        cation_phi_star_value = 0.75
        pressures = [3, 8, 13]
    elif cation == "Ca":
        chloride_multiplier = 2
        cation_phi_star_value = 0.4
        pressures = [5, 10, 15]
    elif cation == "La":
        chloride_multiplier = 3
        cation_phi_star_value = 0.0001
        pressures = [7, 12, 17]

    conc_list = []
    pressure_list = []
    feed_flow_list = []
    ion_flux_list = []

    if cation == "La":
        for feed_flow in feed_flows:
            for pressure in pressures:
                for concentration in concentrations:
                    print("=====================")
                    print("Solving for")
                    print(f"feed flow = {feed_flow}")
                    print(f"applied pressure = {pressure}")
                    print(f"concentration = {concentration}")
                    print("=====================")

                    inlet_flow_volume = {"feed": feed_flow, "diafiltrate": 1e-10}
                    default_args = (
                        anion_list,
                        inlet_flow_volume,
                        include_boundary_layer,
                    )

                    if concentration == concentrations[0]:
                        model = build_tube_model(
                            cation_list=[cation],
                            inlet_concentration={
                                "feed": {
                                    cation: concentration,
                                    "Cl": chloride_multiplier * concentration,
                                },
                                "diafiltrate": {
                                    cation: diafiltrate[cation],
                                    "Cl": 1e-10,
                                },
                            },
                            default_args=default_args,
                            NFE_args=NFE_args,
                            applied_pressure=pressure,
                            data_membrane_thickness=l_m,
                            multiplier_H_feed=1.2,
                            multiplier_H_perm=1,
                            non_Donnan_partition_dict={
                                cation: cation_phi_star_value,
                                "Cl": chloride_phi_star_value,
                            },
                            initialize_and_solve=True,
                        )

                        conc_list.append(concentration)
                        pressure_list.append(pressure)
                        feed_flow_list.append(feed_flow)
                        ion_flux_list.append(
                            get_average_flux(model, cation, stage_bool=False)
                        )

                    else:
                        for t in model.fs.membrane.time:
                            for j in model.fs.membrane.solutes:
                                model.fs.membrane.feed_conc_mol_comp[t, j].set_value(
                                    concentration
                                )

                        if concentration == 200:
                            model.fs.membrane.scaling_factor[
                                model.fs.membrane.molar_ion_flux
                            ] = 1e3

                        solve_model(model)
                        model.fs.membrane.applied_pressure.fix(pressure)
                        solve_model(model)

                        dt = DiagnosticsToolbox(model)
                        dt.assert_no_numerical_warnings(model)

                        conc_list.append(concentration)
                        pressure_list.append(pressure)
                        feed_flow_list.append(feed_flow)
                        ion_flux_list.append(
                            get_average_flux(model, cation, stage_bool=False)
                        )
    else:
        for feed_flow in feed_flows:
            for pressure in pressures:
                for concentration in concentrations:
                    print("=====================")
                    print("Solving for")
                    print(f"feed flow = {feed_flow}")
                    print(f"applied pressure = {pressure}")
                    print(f"concentration = {concentration}")
                    print("=====================")
                    inlet_flow_volume = {"feed": feed_flow, "diafiltrate": 1e-10}
                    default_args = (
                        anion_list,
                        inlet_flow_volume,
                        include_boundary_layer,
                    )

                    model = build_tube_model(
                        cation_list=[cation],
                        inlet_concentration={
                            "feed": {
                                cation: concentration,
                                "Cl": chloride_multiplier * concentration,
                            },
                            "diafiltrate": {
                                cation: diafiltrate[cation],
                                "Cl": 1e-10,
                            },
                        },
                        default_args=default_args,
                        NFE_args=NFE_args,
                        applied_pressure=pressure,
                        data_membrane_thickness=l_m,
                        multiplier_H_feed=1.2,
                        multiplier_H_perm=1,
                        non_Donnan_partition_dict={
                            cation: cation_phi_star_value,
                            "Cl": chloride_phi_star_value,
                        },
                        initialize_and_solve=True,
                    )

                    conc_list.append(concentration)
                    pressure_list.append(pressure)
                    feed_flow_list.append(feed_flow)
                    ion_flux_list.append(
                        get_average_flux(model, cation, stage_bool=False)
                    )

    dict = {
        "feed_conc": conc_list,
        "pressure": pressure_list,
        "feed_flow": feed_flow_list,
        "ion_flux": ion_flux_list,
    }
    df = DataFrame(data=dict)
    df.to_csv(f"training_data/tube/single_salt/{cation}_chloride", index=False)


def generate_single_salt_tube_data():
    anion_list = ["Cl"]
    diafiltrate = {"Na": 1e-10, "Ca": 1e-10, "La": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10

    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    cations = ["Na", "Ca", "La"]
    concentrations = [50, 125, 200]
    feed_flows = [8, 12.5, 17]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_um = Dm_Cl / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m
    chloride_phi_star_value = 0.05

    valence_list = []
    conc_list = []
    pressure_list = []
    feed_flow_list = []
    ion_flux_list = []
    water_flux_list = []  # tube-average water flux, m3/m2/h

    for cation in cations:
        if cation == "Na":
            chloride_multiplier = 1
            cation_phi_star_value = 0.75
            pressures = [3, 8, 13]
        elif cation == "Ca":
            chloride_multiplier = 2
            cation_phi_star_value = 0.4
            pressures = [5, 10, 15]
        elif cation == "La":
            chloride_multiplier = 3
            cation_phi_star_value = 0.0001
            pressures = [7, 12, 17]

        if cation == "La":
            for feed_flow in feed_flows:
                for pressure in pressures:
                    for concentration in concentrations:
                        inlet_flow_volume = {"feed": feed_flow, "diafiltrate": 1e-10}
                        default_args = (
                            anion_list,
                            inlet_flow_volume,
                            include_boundary_layer,
                        )

                        if concentration == concentrations[0]:
                            model = build_tube_model(
                                cation_list=[cation],
                                inlet_concentration={
                                    "feed": {
                                        cation: concentration,
                                        "Cl": chloride_multiplier * concentration,
                                    },
                                    "diafiltrate": {
                                        cation: diafiltrate[cation],
                                        "Cl": 1e-10,
                                    },
                                },
                                default_args=default_args,
                                NFE_args=NFE_args,
                                applied_pressure=pressure,
                                data_membrane_thickness=l_m,
                                multiplier_H_feed=1.2,
                                multiplier_H_perm=1,
                                non_Donnan_partition_dict={
                                    cation: cation_phi_star_value,
                                    "Cl": chloride_phi_star_value,
                                },
                                initialize_and_solve=True,
                            )

                            valence_list.append(chloride_multiplier)
                            conc_list.append(concentration)
                            pressure_list.append(pressure)
                            feed_flow_list.append(feed_flow)
                            ion_flux_list.append(
                                get_average_flux(model, cation, stage_bool=False)
                            )
                            water_flux_list.append(
                                get_average_water_flux(model, stage_bool=False)
                            )

                        else:
                            for t in model.fs.membrane.time:
                                for j in model.fs.membrane.solutes:
                                    model.fs.membrane.feed_conc_mol_comp[
                                        t, j
                                    ].set_value(concentration)

                            if concentration == 200:
                                model.fs.membrane.scaling_factor[
                                    model.fs.membrane.molar_ion_flux
                                ] = 1e3

                            solve_model(model)
                            model.fs.membrane.applied_pressure.fix(pressure)
                            solve_model(model)

                            dt = DiagnosticsToolbox(model)
                            dt.assert_no_numerical_warnings(model)

                            valence_list.append(chloride_multiplier)
                            conc_list.append(concentration)
                            pressure_list.append(pressure)
                            feed_flow_list.append(feed_flow)
                            ion_flux_list.append(
                                get_average_flux(model, cation, stage_bool=False)
                            )
                            water_flux_list.append(
                                get_average_water_flux(model, stage_bool=False)
                            )
        else:
            for feed_flow in feed_flows:
                for pressure in pressures:
                    for concentration in concentrations:
                        inlet_flow_volume = {"feed": feed_flow, "diafiltrate": 1e-10}
                        default_args = (
                            anion_list,
                            inlet_flow_volume,
                            include_boundary_layer,
                        )

                        model = build_tube_model(
                            cation_list=[cation],
                            inlet_concentration={
                                "feed": {
                                    cation: concentration,
                                    "Cl": chloride_multiplier * concentration,
                                },
                                "diafiltrate": {
                                    cation: diafiltrate[cation],
                                    "Cl": 1e-10,
                                },
                            },
                            default_args=default_args,
                            NFE_args=NFE_args,
                            applied_pressure=pressure,
                            data_membrane_thickness=l_m,
                            multiplier_H_feed=1.2,
                            multiplier_H_perm=1,
                            non_Donnan_partition_dict={
                                cation: cation_phi_star_value,
                                "Cl": chloride_phi_star_value,
                            },
                            initialize_and_solve=True,
                        )

                        valence_list.append(chloride_multiplier)
                        conc_list.append(concentration)
                        pressure_list.append(pressure)
                        feed_flow_list.append(feed_flow)
                        ion_flux_list.append(
                            get_average_flux(model, cation, stage_bool=False)
                        )
                        water_flux_list.append(
                            get_average_water_flux(model, stage_bool=False)
                        )

    dict = {
        "valency": valence_list,
        "feed_conc": conc_list,
        "pressure": pressure_list,
        "feed_flow": feed_flow_list,
        "ion_flux": ion_flux_list,
        "water_flux": water_flux_list,
    }
    df = DataFrame(data=dict)
    df.to_csv(f"training_data/tube/single_salt/mixed_chlorides", index=False)


def generate_stage_data_per_salt(cation):
    anion_list = ["Cl"]
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10
    num_tubes = 5

    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_um = Dm_Cl / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m
    chloride_phi_star_value = 0.05

    if cation == "Na":
        chloride_multiplier = 1
        cation_phi_star_value = 0.75

        concentrations = [150, 175, 200]
        pressures = [5, 7, 9]

    elif cation == "Ca":
        chloride_multiplier = 2
        cation_phi_star_value = 0.4

        concentrations = [150, 175, 200]
        pressures = [8, 10, 12]

    elif cation == "La":
        chloride_multiplier = 3
        cation_phi_star_value = 0.0001

        concentrations = [50, 75, 100]
        pressures = [12, 14, 16]

    conc_list = []
    pressure_list = []
    ion_flux_list = []

    # maximum tube feed flow rate is 17 m3/h
    # diafiltrate needs to be added later in the stage in order to look at higher pressures / lower concentrations
    # otherwise, the stage is drying out
    if cation == "La":
        inlet_flow_volume = {
            "feed": 17,
            "diafiltrate": [1e-10, 1e-10, 1e-10, 2.5, 1e-10],
        }
    else:
        inlet_flow_volume = {"feed": 17, "diafiltrate": [1e-10, 1e-10, 1e-10, 5, 1e-10]}
    default_args = (
        anion_list,
        inlet_flow_volume,
        include_boundary_layer,
    )

    if cation == "La":
        for concentration in concentrations:
            for pressure in pressures:
                print("=====================")
                print("Solving for")
                print(f"tube 1 applied pressure = {pressure}")
                print(f"tube 1 feed concentration = {concentration}")
                print("=====================")

                inlet_concentration = {
                    "feed": {
                        cation: concentration,
                        "Cl": chloride_multiplier * concentration,
                    },
                    "diafiltrate": {
                        cation: [1e-10, 1e-10, 1e-10, 10, 1e-10],
                        "Cl": [1e-10, 1e-10, 1e-10, chloride_multiplier * 10, 1e-10],
                    },
                }

                if pressure == pressures[0]:
                    model = build_stage_model(
                        cation_list=[cation],
                        inlet_concentration=inlet_concentration,
                        default_args=default_args,
                        NFE_args=NFE_args,
                        applied_pressure=pressure,
                        pressure_drop=0.5,  # assume a pressure drop per tube
                        data_membrane_thickness=l_m,
                        non_Donnan_partition_dict={
                            cation: cation_phi_star_value,
                            "Cl": chloride_phi_star_value,
                        },
                        num_tubes=num_tubes,
                        initialize_and_solve=True,
                    )

                    conc_list.append(concentration)
                    pressure_list.append(pressure)
                    ion_flux_list.append(
                        get_average_flux(
                            model, cation, stage_bool=True, num_tubes=num_tubes
                        )
                    )
                else:
                    for i in RangeSet(1, num_tubes):
                        if i == 1:
                            model.fs.membrane[i].applied_pressure.fix(pressure)
                        else:
                            model.fs.membrane[i].applied_pressure.fix(
                                pressure - (i - 1) * 0.5
                            )

                    solve_model(model)
                    for i in RangeSet(1, num_tubes):
                        for t in model.fs.membrane[i].time:
                            for j in model.fs.membrane[i].solutes:
                                model.fs.membrane[i].diafiltrate_conc_mol_comp[
                                    t, j
                                ].fix(inlet_concentration["diafiltrate"][j][i - 1])
                    solve_model(model)

                    dt = DiagnosticsToolbox(model)
                    dt.assert_no_numerical_warnings(model)

                    conc_list.append(concentration)
                    pressure_list.append(pressure)
                    ion_flux_list.append(
                        get_average_flux(
                            model, cation, stage_bool=True, num_tubes=num_tubes
                        )
                    )

    else:
        for pressure in pressures:
            for concentration in concentrations:
                print("=====================")
                print("Solving for")
                print(f"tube 1 applied pressure = {pressure}")
                print(f"tube 1 feed concentration = {concentration}")
                print("=====================")
                model = build_stage_model(
                    cation_list=[cation],
                    inlet_concentration={
                        "feed": {
                            cation: concentration,
                            "Cl": chloride_multiplier * concentration,
                        },
                        "diafiltrate": {
                            cation: [1e-10, 1e-10, 1e-10, 10, 1e-10],
                            "Cl": [
                                1e-10,
                                1e-10,
                                1e-10,
                                chloride_multiplier * 10,
                                1e-10,
                            ],
                        },
                    },
                    default_args=default_args,
                    NFE_args=NFE_args,
                    applied_pressure=pressure,
                    pressure_drop=0.5,  # assume a pressure drop per tube
                    data_membrane_thickness=l_m,
                    non_Donnan_partition_dict={
                        cation: cation_phi_star_value,
                        "Cl": chloride_phi_star_value,
                    },
                    num_tubes=num_tubes,
                    initialize_and_solve=True,
                )

                conc_list.append(concentration)
                pressure_list.append(pressure)
                ion_flux_list.append(
                    get_average_flux(
                        model, cation, stage_bool=True, num_tubes=num_tubes
                    )
                )

    dict = {
        "feed_conc": conc_list,
        "pressure": pressure_list,
        "ion_flux": ion_flux_list,
    }
    df = DataFrame(data=dict)
    df.to_csv(f"training_data/stage/single_salt/{cation}_chloride", index=False)


def generate_single_salt_stage_data():
    anion_list = ["Cl"]
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10
    num_tubes = 5

    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_um = Dm_Cl / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m
    chloride_phi_star_value = 0.05

    cations = ["Na", "Ca", "La"]

    valence_list = []
    conc_list = []
    pressure_list = []
    ion_flux_list = []
    tube_conditions = []  # per-tube operating conditions within the stage
    failed_cases = []  # cases that did not solve (La continuation)

    for cation in cations:
        if cation == "Na":
            chloride_multiplier = 1
            cation_phi_star_value = 0.75

            concentrations = [150, 175, 200]
            pressures = [5, 6, 7]

        elif cation == "Ca":
            chloride_multiplier = 2
            cation_phi_star_value = 0.4

            concentrations = [150, 175, 200]
            pressures = [8, 10, 12]

        elif cation == "La":
            chloride_multiplier = 3
            cation_phi_star_value = 0.0001

            concentrations = [50, 75, 100]
            pressures = [12, 14, 16]

        if cation == "La":
            inlet_flow_volume = {
                "feed": 17,
                "diafiltrate": [1e-10, 1e-10, 1e-10, 2.5, 1e-10],
            }
        else:
            inlet_flow_volume = {
                "feed": 17,
                "diafiltrate": [1e-10, 1e-10, 1e-10, 5, 1e-10],
            }
        default_args = (
            anion_list,
            inlet_flow_volume,
            include_boundary_layer,
        )

        if cation == "La":
            # Building the La stage from scratch is erratic (the sequential tube
            # initialization sometimes fails, even between conditions that
            # solve). The stage is therefore built once and every other
            # condition is reached by continuation: pressure is stepped at fixed
            # concentration, and concentration is stepped in small increments at
            # the end of a pressure sweep. Sweeps alternate direction so a new
            # concentration starts at the pressure the previous sweep ended on.
            # A case that fails is recorded and skipped; the next case then
            # rebuilds the stage from scratch.
            diafiltrate_conc = {
                cation: [1e-10, 1e-10, 1e-10, 10, 1e-10],
                "Cl": [1e-10, 1e-10, 1e-10, chloride_multiplier * 10, 1e-10],
            }
            model = None
            current_pressure = None
            current_concentration = None

            for c_index, concentration in enumerate(concentrations):
                sweep = pressures if c_index % 2 == 0 else pressures[::-1]
                for pressure in sweep:
                    print("=====================")
                    print("Solving for")
                    print(f"tube 1 applied pressure = {pressure}")
                    print(f"tube 1 feed concentration = {concentration}")
                    print("=====================")

                    try:
                        if model is None:
                            model = build_stage_model(
                                cation_list=[cation],
                                inlet_concentration={
                                    "feed": {
                                        cation: concentration,
                                        "Cl": chloride_multiplier * concentration,
                                    },
                                    "diafiltrate": diafiltrate_conc,
                                },
                                default_args=default_args,
                                NFE_args=NFE_args,
                                applied_pressure=pressure,
                                pressure_drop=0.5,  # assume a pressure drop per tube
                                data_membrane_thickness=l_m,
                                non_Donnan_partition_dict={
                                    cation: cation_phi_star_value,
                                    "Cl": chloride_phi_star_value,
                                },
                                num_tubes=num_tubes,
                                initialize_and_solve=True,
                            )
                        else:
                            if concentration != current_concentration:
                                step_stage_feed_conc(
                                    model,
                                    cation,
                                    chloride_multiplier,
                                    current_concentration,
                                    concentration,
                                )
                            if pressure != current_pressure:
                                set_stage_pressure(model, pressure, 0.5, num_tubes)
                                solve_model(model)
                                fix_stage_diafiltrate_conc(
                                    model, diafiltrate_conc, num_tubes
                                )
                                solve_model(model)

                            dt = DiagnosticsToolbox(model)
                            dt.assert_no_numerical_warnings(model)

                        flux = get_average_flux(
                            model, cation, stage_bool=True, num_tubes=num_tubes
                        )
                        tubes = get_stage_tube_conditions(
                            model,
                            cation,
                            chloride_multiplier,
                            concentration,
                            pressure,
                            num_tubes,
                        )
                    except Exception as e:
                        print(f"FAILED: {cation} P = {pressure}, c = {concentration}")
                        failed_cases.append(
                            {
                                "valency": chloride_multiplier,
                                "feed_conc": concentration,
                                "pressure": pressure,
                                "message": str(e).splitlines()[0][:120],
                            }
                        )
                        model = None
                        current_pressure = None
                        current_concentration = None
                        continue

                    valence_list.append(chloride_multiplier)
                    conc_list.append(concentration)
                    pressure_list.append(pressure)
                    ion_flux_list.append(flux)
                    tube_conditions.extend(tubes)
                    current_pressure = pressure
                    current_concentration = concentration

        else:
            for pressure in pressures:
                for concentration in concentrations:
                    print("=====================")
                    print("Solving for")
                    print(f"tube 1 applied pressure = {pressure}")
                    print(f"tube 1 feed concentration = {concentration}")
                    print("=====================")
                    model = build_stage_model(
                        cation_list=[cation],
                        inlet_concentration={
                            "feed": {
                                cation: concentration,
                                "Cl": chloride_multiplier * concentration,
                            },
                            "diafiltrate": {
                                cation: [1e-10, 1e-10, 1e-10, 10, 1e-10],
                                "Cl": [
                                    1e-10,
                                    1e-10,
                                    1e-10,
                                    chloride_multiplier * 10,
                                    1e-10,
                                ],
                            },
                        },
                        default_args=default_args,
                        NFE_args=NFE_args,
                        applied_pressure=pressure,
                        pressure_drop=0.5,  # assume a pressure drop per tube
                        data_membrane_thickness=l_m,
                        non_Donnan_partition_dict={
                            cation: cation_phi_star_value,
                            "Cl": chloride_phi_star_value,
                        },
                        num_tubes=num_tubes,
                        initialize_and_solve=True,
                    )

                    valence_list.append(chloride_multiplier)
                    conc_list.append(concentration)
                    pressure_list.append(pressure)
                    ion_flux_list.append(
                        get_average_flux(
                            model, cation, stage_bool=True, num_tubes=num_tubes
                        )
                    )
                    tube_conditions.extend(
                        get_stage_tube_conditions(
                            model,
                            cation,
                            chloride_multiplier,
                            concentration,
                            pressure,
                            num_tubes,
                        )
                    )

    dict = {
        "valency": valence_list,
        "feed_conc": conc_list,
        "pressure": pressure_list,
        "ion_flux": ion_flux_list,
    }
    df = DataFrame(data=dict)
    df.to_csv(f"training_data/stage/single_salt/mixed_chlorides", index=False)

    # conditions seen by each tube within the stage (feed is the mixture of the
    # upstream retentate and any diafiltrate); used to check that the tube
    # surrogate training bounds cover tubes 2-5 of the stage
    df_tubes = DataFrame(data=tube_conditions)
    df_tubes.to_csv(
        "training_data/stage/single_salt/mixed_chlorides_tube_conditions",
        index=False,
    )

    if failed_cases:
        print("The following stage cases did not solve and are not in the data:")
        for case in failed_cases:
            print(case)
        DataFrame(data=failed_cases).to_csv(
            "training_data/stage/single_salt/mixed_chlorides_failed_cases",
            index=False,
        )


def set_stage_pressure(model, pressure, pressure_drop, num_tubes):
    """Fix the applied pressure of each tube given the pressure of tube 1."""
    for i in RangeSet(1, num_tubes):
        model.fs.membrane[i].applied_pressure.fix(pressure - (i - 1) * pressure_drop)


def set_stage_feed_conc(model, cation, valence, concentration):
    """Fix the tube 1 feed concentration of the cation and chloride."""
    blk = model.fs.membrane[1]
    for t in blk.time:
        blk.feed_conc_mol_comp[t, cation].fix(concentration)
        blk.feed_conc_mol_comp[t, "Cl"].fix(valence * concentration)


def step_stage_feed_conc(model, cation, valence, from_conc, to_conc, max_step=5):
    """Move the tube 1 feed concentration in small steps, solving at each."""
    n_steps = max(1, int(np.ceil(abs(to_conc - from_conc) / max_step)))
    for k in range(1, n_steps + 1):
        set_stage_feed_conc(
            model, cation, valence, from_conc + (to_conc - from_conc) * k / n_steps
        )
        solve_model(model)


def fix_stage_diafiltrate_conc(model, diafiltrate_conc, num_tubes):
    """Fix the diafiltrate concentrations of each tube; keys are solute names."""
    for i in RangeSet(1, num_tubes):
        for t in model.fs.membrane[i].time:
            for j in model.fs.membrane[i].solutes:
                model.fs.membrane[i].diafiltrate_conc_mol_comp[t, j].fix(
                    diafiltrate_conc[j][i - 1]
                )


def _local_flux_values(blk, flux_var, solute=None):
    """Local values of a flux variable along the module length of one tube."""
    values = []
    for t in blk.time:
        for x in blk.dimensionless_module_length:
            if x != 0:
                if solute is None:
                    values.append(value(flux_var[t, x]))
                else:
                    values.append(value(flux_var[t, x, solute]))
    return values


def _average_flux(model, flux_name, solute, stage_bool, num_tubes):
    """
    Average of a local flux variable over the module length. For a stage, the
    local values of all tubes are pooled (tubes are the same length).
    """
    if stage_bool:
        blocks = [model.fs.membrane[i] for i in RangeSet(1, num_tubes)]
    else:
        blocks = [model.fs.membrane]

    local_flux = []
    for blk in blocks:
        local_flux += _local_flux_values(blk, getattr(blk, flux_name), solute)

    return np.average(local_flux)


def get_average_flux(model, solute, stage_bool, num_tubes=None):
    """Average of the local molar ion flux of a solute."""
    return _average_flux(model, "molar_ion_flux", solute, stage_bool, num_tubes)


def get_average_water_flux(model, stage_bool, num_tubes=None):
    """Average of the local water flux (m3/m2/h)."""
    return _average_flux(model, "volume_flux_water", None, stage_bool, num_tubes)


def get_stage_tube_conditions(
    model, cation, valence, tube_1_feed_conc, tube_1_pressure, num_tubes
):
    """
    Records the conditions of each tube in a solved stage. The tube inlet flow
    and concentration are those after mixing with the diafiltrate stream
    (i.e., the retentate state at x = 0), which is what a tube surrogate sees.
    """
    rows = []
    for i in RangeSet(1, num_tubes):
        blk = model.fs.membrane[i]
        t = blk.time.first()
        x_in = blk.dimensionless_module_length.first()
        x_out = blk.dimensionless_module_length.last()
        rows.append(
            {
                "valency": valence,
                "tube_1_feed_conc": tube_1_feed_conc,
                "tube_1_pressure": tube_1_pressure,
                "tube": i,
                "tube_pressure": value(blk.applied_pressure[t]),
                "tube_inlet_flow": value(blk.retentate_flow_volume[t, x_in]),
                "tube_inlet_conc": value(blk.retentate_conc_mol_comp[t, x_in, cation]),
                "tube_ion_flux": np.average(
                    _local_flux_values(blk, blk.molar_ion_flux, cation)
                ),
                "tube_water_flux": np.average(
                    _local_flux_values(blk, blk.volume_flux_water)
                ),
                "tube_retentate_flow": value(blk.retentate_flow_volume[t, x_out]),
                "tube_retentate_conc": value(
                    blk.retentate_conc_mol_comp[t, x_out, cation]
                ),
                "tube_permeate_flow": value(blk.permeate_outlet_flow[t]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
