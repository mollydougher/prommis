"""
Synthetic data generation for multi-component diafiltration model.

Author: Molly Dougher
"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
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

    # generate_single_salt_tube_data()

    generate_stage_data()


def build_model(
    cation_list,
    inlet_concentration,
    default_args,
    NFE_args,
    water_flux,
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
    # if len(cation_list) == 1:
    #     m.fs.membrane.applied_pressure.fix(5)
    # else:
    #     m.fs.membrane.applied_pressure.fix(10)
    m.fs.membrane.feed_flow_volume.fix(inlet_flow_volume["feed"])
    m.fs.membrane.diafiltrate_flow_volume.fix(inlet_flow_volume["diafiltrate"])
    for t in m.fs.membrane.time:
        for j in m.fs.membrane.solutes:
            m.fs.membrane.feed_conc_mol_comp[t, j].fix(inlet_concentration["feed"][j])
            m.fs.membrane.diafiltrate_conc_mol_comp[t, j].fix(
                inlet_concentration["diafiltrate"][j]
            )
    # initial pressure values may need to be tweaked to successfully initialize
    feed_ionic_strength = value(m.fs.membrane.feed_ionic_strength[0])
    if feed_ionic_strength < 201:
        m.fs.membrane.applied_pressure.fix(5)
    elif feed_ionic_strength >= 201:
        m.fs.membrane.applied_pressure.fix(15)

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

        solve_model(m)
        unfix_pressure(m, water_flux_target=water_flux)
        solve_model(m)

        dt = DiagnosticsToolbox(m)
        dt.assert_no_numerical_warnings()

    return m


def solve_model(m):
    scaling = TransformationFactory("core.scale_model")
    scaled_model = scaling.create_using(m, rename=False)

    solver = SolverFactory("ipopt")
    results = solver.solve(scaled_model, tee=True)
    assert_optimal_termination(results)

    scaling.propagate_solution(scaled_model, m)

    return results


def unfix_pressure(m, water_flux_target):
    m.fs.membrane.applied_pressure.unfix()

    def _water_flux_constraint(m):
        return (
            sum(
                m.fs.membrane.volume_flux_water[0, x]
                for x in m.fs.membrane.dimensionless_module_length
                if x != 0
            )
            / (len(m.fs.membrane.dimensionless_module_length) - 1)
            == water_flux_target
        )

    m.water_flux_constraint = Constraint(rule=_water_flux_constraint)


def generate_tube_data_per_salt(cation):
    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    diafiltrate = {"Na": 1e-10, "Ca": 1e-10, "La": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10

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

    if cation == "La":
        for water_flux in water_fluxes:
            for concentration in concentrations:
                if concentration == concentrations[0]:
                    model = build_model(
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
                        multiplier_H_feed=1.2,
                        multiplier_H_perm=1,
                        non_Donnan_partition_dict={
                            cation: cation_phi_star_value,
                            "Cl": chloride_phi_star_value,
                        },
                        initialize_and_solve=True,
                    )

                    conc_list.append(concentration)
                    water_flux_list.append(water_flux)
                    ion_flux_list.append(get_average_flux(model, cation))

                else:
                    model.del_component(model.water_flux_constraint)

                    for t in model.fs.membrane.time:
                        for j in model.fs.membrane.solutes:
                            model.fs.membrane.feed_conc_mol_comp[t, j].set_value(
                                concentration
                            )

                    solve_model(model)
                    unfix_pressure(model, water_flux_target=water_flux)
                    solve_model(model)

                    dt = DiagnosticsToolbox(model)
                    dt.assert_no_numerical_warnings(model)

                    conc_list.append(concentration)
                    water_flux_list.append(water_flux)
                    ion_flux_list.append(get_average_flux(model, cation))
    else:
        for water_flux in water_fluxes:
            for concentration in concentrations:
                model = build_model(
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
                    multiplier_H_feed=1.2,
                    multiplier_H_perm=1,
                    non_Donnan_partition_dict={
                        cation: cation_phi_star_value,
                        "Cl": chloride_phi_star_value,
                    },
                    initialize_and_solve=True,
                )

                conc_list.append(concentration)
                water_flux_list.append(water_flux)
                ion_flux_list.append(get_average_flux(model, cation))

    dict = {
        "feed_conc": conc_list,
        "water_flux": water_flux_list,
        "ion_flux": ion_flux_list,
    }
    df = DataFrame(data=dict)
    df.to_csv(f"training_data/single_salt/{cation}_chloride", index=False)


def generate_single_salt_tube_data():
    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    diafiltrate = {"Na": 1e-10, "Ca": 1e-10, "La": 1e-10}
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10

    default_args = (anion_list, inlet_flow_volume, include_boundary_layer)
    NFE_args = [NFE_module_length, NFE_boundary_layer_thickness, NFE_membrane_thickness]

    cations = ["Na", "Ca", "La"]
    concentrations = [25, 50, 100, 150, 200]
    water_fluxes = [0.01, 0.02, 0.03]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_um = Dm_Cl / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m
    chloride_phi_star_value = 0.05

    valence_list = []
    conc_list = []
    water_flux_list = []
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
            for water_flux in water_fluxes:
                for concentration in concentrations:
                    if concentration == concentrations[0]:
                        model = build_model(
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
                        water_flux_list.append(water_flux)
                        ion_flux_list.append(get_average_flux(model, cation))

                    else:
                        model.del_component(model.water_flux_constraint)

                        for t in model.fs.membrane.time:
                            for j in model.fs.membrane.solutes:
                                model.fs.membrane.feed_conc_mol_comp[t, j].set_value(
                                    concentration
                                )

                        solve_model(model)
                        unfix_pressure(model, water_flux_target=water_flux)
                        solve_model(model)

                        dt = DiagnosticsToolbox(model)
                        dt.assert_no_numerical_warnings(model)

                        valence_list.append(chloride_multiplier)
                        conc_list.append(concentration)
                        water_flux_list.append(water_flux)
                        ion_flux_list.append(get_average_flux(model, cation))
        else:
            for water_flux in water_fluxes:
                for concentration in concentrations:
                    model = build_model(
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
                    water_flux_list.append(water_flux)
                    ion_flux_list.append(get_average_flux(model, cation))

    dict = {
        "valency": valence_list,
        "feed_conc": conc_list,
        "water_flux": water_flux_list,
        "ion_flux": ion_flux_list,
    }
    df = DataFrame(data=dict)
    df.to_csv(f"training_data/single_salt/mixed_chlorides", index=False)


def generate_stage_data():
    cation_list = ["Na"]
    anion_list = ["Cl"]
    inlet_flow_volume = {"feed": 12.5 + 3.75, "diafiltrate": 1e-10}
    inlet_concentration = {
        "feed": {"Na": 200, "Cl": 200},
        "diafiltrate": {"Na": 1e-10, "Cl": 1e-10},
    }
    include_boundary_layer = True
    NFE_module_length = 10
    NFE_boundary_layer_thickness = 10
    NFE_membrane_thickness = 10

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_um = Dm_Cl / Dm_over_l_value  # um
    l_m = l_um / 1e6  # m

    membrane_stage = MultiComponentDiafiltrationStageFlowsheet(
        cation_list=cation_list,
        anion_list=anion_list,
        num_tubes=5,
        inlet_flow_volume=inlet_flow_volume,
        inlet_concentration=inlet_concentration,
        non_Donnan_partition_dict={
            "Na": 0.75,
            "Cl": 0.05,
        },
        include_boundary_layer=include_boundary_layer,
        total_membrane_thickness=l_m,
        NFE_module_length=NFE_module_length,
        NFE_boundary_layer_thickness=NFE_boundary_layer_thickness,
        NFE_membrane_thickness=NFE_membrane_thickness,
    )

    model = membrane_stage.build_flowsheet()


def get_average_flux(model, solute):
    local_flux = []
    for t in model.fs.membrane.time:
        for x in model.fs.membrane.dimensionless_module_length:
            if x != 0:
                local_flux.append(value(model.fs.membrane.molar_ion_flux[t, x, solute]))

    avg_local_flux = np.average(local_flux)

    return avg_local_flux


if __name__ == "__main__":
    main()
