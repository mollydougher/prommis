#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Multi-Component Diafiltration Stage Flowsheet Model

Author: Molly Dougher
"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    RangeSet,
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    maximize,
    units,
    value,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.models.unit_models import Feed, Product

from prommis.nanofiltration.multi_component_diafiltration_stream_properties import (
    MultiComponentDiafiltrationStreamParameter,
)
from prommis.nanofiltration.multi_component_diafiltration_solute_properties import (
    MultiComponentDiafiltrationSoluteParameter,
)
from prommis.nanofiltration.multi_component_diafiltration import (
    MultiComponentDiafiltration,
)

class MultiComponentDiafiltrationStageFlowsheet():
    """
    Multi-Component Diafiltration Stage Class.
    """

    def __init__(
        self,
        cation_list=["Na", "Ca"],
        anion_list=["Cl"],
        num_tubes=3,
        inlet_flow_volume = {"feed": 12.5, "diafiltrate": 3.75},
        inlet_concentration = {
            "feed": {"Na": 200, "Ca": 200, "Cl": 600},
            "diafiltrate": {"Na": 14, "Ca": 3, "Cl": 20},
        },
        non_Donnan_partition_dict={
            "Na": 0.75,
            "Ca": 0.4,
            "Cl": 0.05,
        },
        include_boundary_layer=True,
        total_membrane_thickness=1e-7,
        NFE_module_length=10,
        NFE_boundary_layer_thickness=10,
        NFE_membrane_thickness=10,
    ):
        self.cation_list = cation_list
        self.anion_list = anion_list
        self.num_tubes = num_tubes
        self.inlet_flow_volume = inlet_flow_volume
        self.inlet_concentration = inlet_concentration
        self.non_Donnan_partition_dict = non_Donnan_partition_dict
        self.include_boundary_layer = include_boundary_layer
        self.total_membrane_thickness = total_membrane_thickness
        self.NFE_module_length = NFE_module_length
        self.NFE_boundary_layer_thickness = NFE_boundary_layer_thickness
        self.NFE_membrane_thickness = NFE_membrane_thickness

    def build_flowsheet(self):
        """
        Build method for the multi-component diafiltration stage.
        """
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.stream_properties = MultiComponentDiafiltrationStreamParameter(
            cation_list=self.cation_list,
            anion_list=self.anion_list,
        )
        m.fs.properties = MultiComponentDiafiltrationSoluteParameter(
            cation_list=self.cation_list,
            anion_list=self.anion_list,
            non_Donnan_partition_dict=self.non_Donnan_partition_dict,
        )

        m.fs.membrane = MultiComponentDiafiltration(
            RangeSet(1, self.num_tubes),
            property_package=m.fs.properties,
            cation_list=self.cation_list,
            anion_list=self.anion_list,
            include_boundary_layer=self.include_boundary_layer,
            total_membrane_thickness=self.total_membrane_thickness,
            NFE_module_length=self.NFE_module_length,
            NFE_boundary_layer_thickness=self.NFE_boundary_layer_thickness,
            NFE_membrane_thickness=self.NFE_membrane_thickness,
        )

        m.fs.feed_block = Feed(RangeSet(1, self.num_tubes), property_package=m.fs.stream_properties)
        m.fs.diafiltrate_block = Feed(RangeSet(1, self.num_tubes), property_package=m.fs.stream_properties)
        # only collect final retentate
        m.fs.retentate_block = Product(property_package=m.fs.stream_properties)
        m.fs.permeate_block = Product(RangeSet(1, self.num_tubes), property_package=m.fs.stream_properties)

        return m

    def solve_sub_model(self, m):
        """
        Solves scaled model.

        Args:
            m: Pyomo model
        """
        scaling = TransformationFactory("core.scale_model")
        scaled_model = scaling.create_using(m.parent_block(), rename=False)

        solver = SolverFactory("ipopt")
        results = solver.solve(scaled_model, tee=True)
        assert_optimal_termination(results)

        scaling.propagate_solution(scaled_model, m.parent_block())

    def initialize_flowsheet(self, m, pressure, pressure_drop):
        for i in RangeSet(1, self.num_tubes):
            m.fs.membrane[i].total_module_length.fix()
            m.fs.membrane[i].total_membrane_length.fix()
            if i == 1:
                m.fs.membrane[i].applied_pressure.fix(pressure)
            else:
                m.fs.membrane[i].applied_pressure.fix(pressure - (i-1*pressure_drop))

        m.fs.membrane[1].feed_flow_volume.fix(self.inlet_flow_volume["feed"])
        m.fs.membrane[1].diafiltrate_flow_volume.fix(self.inlet_flow_volume["diafiltrate"])
        for t in m.fs.membrane[1].time:
            for j in m.fs.membrane[1].solutes:
                m.fs.membrane[1].feed_conc_mol_comp[t, j].fix(
                    self.inlet_concentration["feed"][j]
                )
                m.fs.membrane[1].diafiltrate_conc_mol_comp[t, j].fix(
                    self.inlet_concentration["diafiltrate"][j]
                )

        for i in RangeSet(2,self.num_tubes):
            m.fs.membrane[i].diafiltrate_flow_volume.fix(
                value(m.fs.membrane[i].numerical_zero_tolerance)
            )
            for t in m.fs.membrane[i].time:
                for j in m.fs.membrane[i].solutes:
                    m.fs.membrane[i].diafiltrate_conc_mol_comp[t, j].fix(
                        value(m.fs.membrane[i].numerical_zero_tolerance)
                    )

        initialized_membrane_1 = m.fs.membrane[1].default_initializer(
            multiplier_H_feed=1.2,  # increase can help solver performance
            multiplier_H_perm=1,
        )
        initialized_membrane_1.initialize(m.fs.membrane[1])

        dt_membrane_1 = DiagnosticsToolbox(m.fs.membrane[1])
        dt_membrane_1.assert_no_structural_warnings()

        for i in RangeSet(2, self.num_tubes):
            m.fs.membrane[i].deactivate()
        self.solve_sub_model(m.fs.membrane)

        for i in RangeSet(2, self.num_tubes):
            m.fs.membrane[i-1].deactivate()
            m.fs.membrane[i].activate()
            for t in m.fs.membrane[i].time:
                m.fs.membrane[i].feed_flow_volume[t].fix(
                    value(m.fs.membrane[i-1].retentate_flow_volume[t, 1])
                )
                for j in m.fs.membrane[i].solutes:
                    m.fs.membrane[i].feed_conc_mol_comp[t, j].fix(
                        value(m.fs.membrane[i-1].retentate_conc_mol_comp[t, 1, j])
                    )

            initialized_membrane = m.fs.membrane[i].default_initializer(
                multiplier_H_feed=1.2,  # increase can help solver performance
                multiplier_H_perm=1,
            )
            initialized_membrane.initialize(m.fs.membrane[i])

            dt_membrane_temp = DiagnosticsToolbox(m.fs.membrane[i])
            dt_membrane_temp.assert_no_structural_warnings()
            self.solve_sub_model(m.fs.membrane[i])

            for t in m.fs.membrane[i].time:
                m.fs.membrane[i].feed_flow_volume[t].unfix()
                for j in m.fs.membrane[i].solutes:
                    m.fs.membrane[i].feed_conc_mol_comp[t, j].unfix()

        for i in RangeSet(1, self.num_tubes):
            m.fs.membrane[i].activate()


    def add_and_connect_streams(self, m):
        m.fs.feed_stream = Arc(
            source=m.fs.feed_block[1].outlet,
            destination=m.fs.membrane[1].feed_inlet,
        )
        for i in RangeSet(1, self.num_tubes):
            m.fs.add_component(
                f"diafiltrate_stream_{i}",
                Arc(
                    source=m.fs.diafiltrate_block[i].outlet,
                    destination=m.fs.membrane[i].diafiltrate_inlet,
                ),
            )
            if i == self.num_tubes:
                m.fs.add_component(
                    f"retentate_stream_{i}",
                    Arc(
                        source=m.fs.membrane[i].retentate_outlet,
                        destination=m.fs.retentate_block.inlet,
                    ),
                )
            else:
                m.fs.add_component(
                    f"retentate_stream_{i}",
                    Arc(
                        source=m.fs.membrane[i].retentate_outlet,
                        destination=m.fs.membrane[i+1].feed_inlet,
                    ),
                )
            m.fs.add_component(
                f"permeate_stream_{i}",
                Arc(
                    source=m.fs.membrane[i].permeate_outlet,
                    destination=m.fs.permeate_block[i].inlet,
                ),
            )

        TransformationFactory("network.expand_arcs").apply_to(m)
