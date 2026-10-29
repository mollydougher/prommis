"""
Comparing flowsheets constructed with surrogate models for multi-component
diafiltration model at different length scales.

Author: Molly Dougher
"""

from pyomo.contrib.parmest.utils.ipopt_solver_wrapper import ipopt_solve_with_stats
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    RangeSet,
    SolverFactory,
    TransformationFactory,
    Var,
    assert_optimal_termination,
    maximize,
    units,
    value,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.util.diagnostics_tools.diagnostics_toolbox import DiagnosticsToolbox
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.models.unit_models import Feed, Product

import matplotlib.pyplot as plt


def main():
    m_tubes = build_tube_surrogate_flowsheet()
    m_stage = build_stage_surrogate_flowsheet()


def build_tube_surrogate_flowsheet(num_tubes=5):

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.tubes = RangeSet(1, num_tubes)

    # surrogate inputs
    m.fs.valence_cation = Var(initialize=1, doc="Charge of cation in chloride salt")
    m.fs.feed_flow = Var(m.fs.tubes, doc="Feed flow rate per tube")
    m.fs.feed_conc_cation = Var(
        m.fs.tubes, doc="Feed concentration of cation or chloride salt per tube"
    )
    m.fs.applied_pressure = Var(m.fs.tubes, doc="Applied pressure per tube")

    # surrogate output
    m.fs.flux_cation = Var(
        m.fs.tubes, initialize=1, doc="Solute flux of cation in chloride salt"
    )

    def _inputs(i):
        return [
            m.fs.valence_cation,
            m.fs.feed_conc_cation[i],
            m.fs.applied_pressure[i],
            m.fs.feed_flow[i],
        ]

    def _output(i):
        return [m.fs.flux_cation[i]]

    tube_surrogate = PysmoSurrogate.load_from_file(
        "surrogate_models/tube/single_salt/single_salt_kriging_surrogate"
    )

    m.fs.tube_surrogate = SurrogateBlock(m.fs.tubes)
    for i in m.fs.tubes:
        m.fs.tube_surrogate[i].build_model(
            tube_surrogate, input_vars=_inputs(i), output_vars=_output(i)
        )

    feed_flow_list = [17, 12, 10, 15, 13]
    feed_conc_cation_list = [70, 70, 70, 70, 70]
    applied_pressure_list = [5, 4.5, 4, 3.5, 3]
    m.fs.valence_cation.fix()
    for i in m.fs.tubes:
        m.fs.feed_flow[i].fix(feed_flow_list[i - 1])
        m.fs.feed_conc_cation[i].fix(feed_conc_cation_list[i - 1])
        m.fs.applied_pressure[i].fix(applied_pressure_list[i - 1])

    solver = SolverFactory("ipopt")
    results = solver.solve(m, tee=True)
    assert_optimal_termination(results)

    return m


def build_stage_surrogate_flowsheet():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # surrogate inputs
    m.fs.valence_cation = Var(initialize=1, doc="Charge of cation in chloride salt")
    m.fs.feed_conc_cation = Var(
        initialize=50, doc="Feed concentration of cation or chloride salt"
    )
    m.fs.applied_pressure_tube_1 = Var(
        initialize=5, doc="Applied pressure of the first tube in stage"
    )

    # surrogate output
    m.fs.flux_cation = Var(initialize=1, doc="Solute flux of cation in chloride salt")

    inputs = [m.fs.valence_cation, m.fs.feed_conc_cation, m.fs.applied_pressure_tube_1]
    output = [m.fs.flux_cation]

    stage_surrogate = PysmoSurrogate.load_from_file(
        "surrogate_models/stage/single_salt/single_salt_kriging_surrogate"
    )

    m.fs.stage_surrogate = SurrogateBlock()
    m.fs.stage_surrogate.build_model(
        stage_surrogate, input_vars=inputs, output_vars=output
    )

    m.fs.valence_cation.fix()
    m.fs.feed_conc_cation.fix()
    m.fs.applied_pressure_tube_1.fix()

    solver = SolverFactory("ipopt")
    results = solver.solve(m, tee=True)
    assert_optimal_termination(results)

    return m


if __name__ == "__main__":
    main()
