"""
Synthetic data generation for multi-component diafiltration model.

Author: Molly Dougher
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
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
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

    generate_single_salt_tube_data()

    # generate_stage_data_per_salt(cation="Na")
    # generate_stage_data_per_salt(cation="Ca")
    # generate_stage_data_per_salt(cation="La")


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
        m.fs.membrane.applied_pressure.fix(5)
    else:
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
            if (
                (inlet_concentration["feed"]["Na"] == 125)
                and (inlet_flow_volume["feed"] == 10)
                and (applied_pressure == 15)
            ):
                m.fs.membrane.scaling_factor[m.fs.membrane.molar_ion_flux] = 1e3
                m.fs.membrane.scaling_factor[
                    m.fs.membrane.boundary_layer_conc_mol_comp
                ] = 1e3
                m.fs.membrane.scaling_factor[m.fs.membrane.membrane_conc_mol_comp] = 1e3

            elif (inlet_concentration["feed"][cation_list[0]] == 50) and (
                inlet_flow_volume["feed"] == 17
            ):
                m.fs.membrane.scaling_factor[
                    m.fs.membrane.membrane_D_tilde_calculation
                ] = 1e1

        if cation_list[0] == "Ca":
            if (inlet_concentration["feed"][cation_list[0]] == 50) and (
                inlet_flow_volume["feed"] == 10
            ):
                m.fs.membrane.scaling_factor[
                    m.fs.membrane.cation_bulk_flux_equation
                ] = 1e2
                m.fs.membrane.scaling_factor[
                    m.fs.membrane.membrane_D_tilde_calculation
                ] = 1e1

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
    water_flux,
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
        if inlet_concentration["feed"][cation_list[0]] < 30:
            pressure = 3
        elif (inlet_concentration["feed"][cation_list[0]] >= 30) and (
            inlet_concentration["feed"][cation_list[0]] < 99
        ):
            pressure = 5
        elif inlet_concentration["feed"][cation_list[0]] >= 99:
            if (cation_list[0] == "La") and (water_flux == 0.03):
                # higher valency & ionic strength at higher flux
                pressure = 10
            else:
                pressure = 5

        membrane_stage.initialize_flowsheet(model, pressure)
        membrane_stage.add_and_connect_streams(model)

        dt = DiagnosticsToolbox(model)
        dt.assert_no_structural_warnings()

        solve_model(model)
        unfix_pressure(
            model, water_flux_target=water_flux, stage_bool=True, num_tubes=num_tubes
        )
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
    pressures = [5, 10, 15]
    feed_flows = [10, 13.5, 17]
    # concentrations = [125]
    # pressures = [5]
    # feed_flows = [10]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_um = Dm_Cl / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m
    chloride_phi_star_value = 0.05

    if cation == "Na":
        chloride_multiplier = 1
        cation_phi_star_value = 0.75
    elif cation == "Ca":
        chloride_multiplier = 2
        cation_phi_star_value = 0.4
    elif cation == "La":
        chloride_multiplier = 3
        cation_phi_star_value = 0.0001

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
    pressures = [5, 10, 15]
    feed_flows = [10, 13.5, 17]

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

    for cation in cations:
        if cation == "Na":
            chloride_multiplier = 1
            cation_phi_star_value = 0.75
        elif cation == "Ca":
            chloride_multiplier = 2
            cation_phi_star_value = 0.4
        elif cation == "La":
            chloride_multiplier = 3
            cation_phi_star_value = 0.0001

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

                        else:
                            for t in model.fs.membrane.time:
                                for j in model.fs.membrane.solutes:
                                    model.fs.membrane.feed_conc_mol_comp[
                                        t, j
                                    ].set_value(concentration)

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

    dict = {
        "valency": valence_list,
        "feed_conc": conc_list,
        "pressure": pressure_list,
        "feed_flow": feed_flow_list,
        "ion_flux": ion_flux_list,
    }
    df = DataFrame(data=dict)
    df.to_csv(f"training_data/tube/single_salt/mixed_chlorides", index=False)


def generate_stage_data_per_salt(cation):
    anion_list = ["Cl"]
    # cannot operate at 0.02 m/h water flux or higher at current q_f (16.25 m3/h) and num_tubes (5)
    # for now, doubling the flow rate in the stage
    # TODO: investigate flow rate as an input in the tube surrogates so we can compare a stage surrogate
    # to a stage constructed of tube surrogates
    inlet_flow_volume = {"feed": 2 * (12.5 + 3.75), "diafiltrate": 1e-10}
    diafiltrate = {"Na": 1e-10, "Ca": 1e-10, "La": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10
    num_tubes = 5

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    concentrations = [25, 50, 100, 150, 200]
    water_fluxes = [0.01, 0.02, 0.03]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_um = Dm_Cl / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m
    chloride_phi_star_value = 0.05

    if cation == "Na":
        chloride_multiplier = 1
        cation_phi_star_value = 0.75
    elif cation == "Ca":
        chloride_multiplier = 2
        cation_phi_star_value = 0.4
    elif cation == "La":
        chloride_multiplier = 3
        cation_phi_star_value = 0.0001

    conc_list = []
    water_flux_list = []
    ion_flux_list = []

    for water_flux in water_fluxes:
        for concentration in concentrations:
            if water_flux == 0.03:
                # ionic strength in final tube gets too high
                # TODO: determine if changes needed to the "experiments"
                if (concentration == 150) or (concentration == 200):
                    continue
            print("=====================")
            print("Solving for")
            print(f"water flux = {water_flux}")
            print(f"concentration = {concentration}")
            print("=====================")
            model = build_stage_model(
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
                water_flux=water_flux,
                data_membrane_thickness=l_m,
                non_Donnan_partition_dict={
                    cation: cation_phi_star_value,
                    "Cl": chloride_phi_star_value,
                },
                num_tubes=num_tubes,
                initialize_and_solve=True,
            )

            conc_list.append(concentration)
            water_flux_list.append(water_flux)
            ion_flux_list.append(
                get_average_flux(model, cation, stage_bool=True, num_tubes=num_tubes)
            )

        dict = {
            "feed_conc": conc_list,
            "water_flux": water_flux_list,
            "ion_flux": ion_flux_list,
        }
        df = DataFrame(data=dict)
        df.to_csv(f"training_data/stage/single_salt/{cation}_chloride", index=False)


def get_average_flux(model, solute, stage_bool, num_tubes=None):
    local_flux = []

    if stage_bool:
        for i in RangeSet(1, num_tubes):
            for t in model.fs.membrane[i].time:
                for x in model.fs.membrane[i].dimensionless_module_length:
                    if x != 0:
                        local_flux.append(
                            value(model.fs.membrane[i].molar_ion_flux[t, x, solute])
                        )
    else:
        for t in model.fs.membrane.time:
            for x in model.fs.membrane.dimensionless_module_length:
                if x != 0:
                    local_flux.append(
                        value(model.fs.membrane.molar_ion_flux[t, x, solute])
                    )

    avg_local_flux = np.average(local_flux)

    return avg_local_flux


if __name__ == "__main__":
    main()
