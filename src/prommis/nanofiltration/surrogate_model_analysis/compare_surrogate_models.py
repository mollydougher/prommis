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

    build_stage_surrogate_flowsheet()


def build_stage_surrogate_flowsheet():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # surrogate inputs
    m.fs.valence_cation = Var(initialize=1, doc="Charge of cation in chloride salt")
    m.fs.feed_conc_cation = Var(initialize=50, doc="Feed concentration of cation or chloride salt")
    m.fs.applied_pressure_tube_1 = Var(initialize=5, doc="Applied pressure of the first tube in stage")

    # surrogate output
    m.fs.flux_cation = Var(initialize=1, doc="Solute flux of cation in chloride salt")

    inputs = [m.fs.valence_cation, m.fs.feed_conc_cation, m.fs.applied_pressure_tube_1]
    output = [m.fs.flux_cation]

    stage_surrogate = PysmoSurrogate.load_from_file("surrogate_models/stage/single_salt/single_salt_kriging_surrogate")

    m.fs.stage_surrogate = SurrogateBlock()
    m.fs.stage_surrogate.build_model(stage_surrogate, input_vars=inputs, output_vars=output)

    m.fs.valence_cation.fix()
    m.fs.feed_conc_cation.fix()
    m.fs.applied_pressure_tube_1.fix()

    solver = SolverFactory('ipopt')
    results = solver.solve(m, tee=True)
    assert_optimal_termination(results)



if __name__ == "__main__":
    main()
