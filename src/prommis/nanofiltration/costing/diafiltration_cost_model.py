"""
Flowsheet costing block for diafiltration flowsheet model

Reference: Reference: watertap > watertap > costing > watertap_costing_package.py
"""

from idaes.core import declare_process_block_class, register_idaes_currency_units
from idaes.core.util.constants import Constants
from pyomo.environ import Constraint, Expression, Param, Var, units
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from watertap.costing.util import make_capital_cost_var, make_fixed_operating_cost_var

from prommis.nanofiltration.costing.diafiltration_cost_block import (
    DiafiltrationCostingBlockData,
)


@declare_process_block_class("DiafiltrationCosting")
class DiafiltrationCostingData(DiafiltrationCostingBlockData):
    """
    Simplified costing block for the diafiltration flowsheet
    """

    def build_global_params(self):
        # Register currency and conversion rates based on CE Index
        register_idaes_currency_units()

        # initialize the common global parameters
        self._build_common_global_params()

        # Set the base year for all costs
        self.base_currency = units.USD_2021

        # Set a base period for all operating costs
        self.base_period = units.year

        # the following global parameters are from the reference file
        self.factor_total_investment = Var(
            initialize=2,  # TODO: verify
            doc="Total investment factor [investment cost/equipment cost]",
            units=units.dimensionless,
        )
        self.factor_maintenance_labor_chemical = Var(
            initialize=0.03,  # TODO: verify
            doc="Maintenance-labor-chemical factor [fraction of investment cost/year]",
            units=units.year**-1,
        )
        self.factor_capital_annualization = Var(
            initialize=0.1,  # TODO: verify
            doc="Capital annualization factor [fraction of investment cost/year]",
            units=units.year**-1,
        )
        self.capital_recovery_factor.expr = self.factor_capital_annualization

        self.density = Param(
            initialize=1000,
            doc="Operating fluid density",
            units=units.kg / units.m**3,
        )
        self.specific_gravity = Param(
            initialize=1,
            doc="Operating fluid specific gravity",
            units=units.dimensionless,
        )
        self.operating_time = Var(
            initialize=8760,
            doc="Operational hours in a year",
            units=units.hr,
        )
        self.electricity_cost = Var(
            initialize=0.141,
            doc="Unit cost of electricity",
            units=units.USD_2021 / units.kWh,
        )

        # fix the parameters
        self.fix_all_vars()

    def build_process_costs(
        self,
    ):
        """
        Builds the process-wide cositng
        Using the same method as the reference file

        Arguments:

        """

        # add total_capital_cost and total_operating_cost
        self._build_common_process_costs()

        self.maintenance_labor_chemical_operating_cost = Var(
            initialize=1e3,
            doc="Maintenance-labor-chemical operating cost",
            units=self.base_currency / self.base_period,
        )
        self.total_capital_cost_constraint = Constraint(
            expr=self.total_capital_cost
            == self.factor_total_investment * self.aggregate_capital_cost
        )
        self.maintenance_labor_chemical_operating_cost_constraint = Constraint(
            expr=self.maintenance_labor_chemical_operating_cost
            == self.factor_maintenance_labor_chemical * self.total_capital_cost
        )

        # if (
        #     units.get_units(sum(self.aggregate_flow_costs.values()))
        # ) == units.dimensionless:
        #     self.total_operating_cost_constraint = Constraint(
        #         expr=self.total_operating_cost
        #         == self.maintenance_labor_chemical_operating_cost
        #         + self.aggregate_fixed_operating_cost
        #         + self.aggregate_variable_operating_cost
        #         + sum(self.aggregate_flow_costs.values())
        #         * self.base_currency
        #         / self.base_period
        #         * self.utilization_factor
        #     )
        # else:
        #     self.total_operating_cost_constraint = Constraint(
        #         expr=self.total_operating_cost
        #         == self.maintenance_labor_chemical_operating_cost
        #         + self.aggregate_fixed_operating_cost
        #         + self.aggregate_variable_operating_cost
        #         + sum(self.aggregate_flow_costs.values()) * self.utilization_factor
        #     )

        ##### from WaterTAPCostingBlockData
        self.total_fixed_operating_cost = Expression(
            expr=self.aggregate_fixed_operating_cost
            + self.maintenance_labor_chemical_operating_cost,
            doc="Total fixed operating costs",
        )

        self.total_variable_operating_cost = Expression(
            expr=(
                (
                    self.aggregate_variable_operating_cost
                    + sum(self.aggregate_flow_costs[f] for f in self.used_flows)
                    * self.utilization_factor
                )
                if self.used_flows
                else self.aggregate_variable_operating_cost
            ),
            doc="Total variable operating cost of process per operating period",
        )

        self.total_operating_cost_constraint = Constraint(
            expr=self.total_operating_cost
            == (self.total_fixed_operating_cost + self.total_variable_operating_cost),
            doc="Total operating cost of process per operating period",
        )
        #####

        self.total_annualized_cost = Expression(
            expr=(
                self.total_capital_cost * self.capital_recovery_factor
                + self.total_operating_cost
            ),
            doc="Total annualized cost of operation",
        )

    @staticmethod
    def initialize_build(self):
        """
        Same method as the reference file
        """
        calculate_variable_from_constraint(
            self.total_capital_cost, self.total_capital_cost_constraint
        )
        calculate_variable_from_constraint(
            self.maintenance_labor_chemical_operating_cost,
            self.maintenance_labor_chemical_operating_cost_constraint,
        )
        calculate_variable_from_constraint(
            self.total_operating_cost, self.total_operating_cost_constraint
        )

    def cost_membranes(
        blk,
        membrane_length,
        membrane_width,
        water_flux,
        vol_flow_feed,
        vol_flow_perm,
    ):
        """
        membrane:
        capital cost assumes a constant cost per (total) area of $50/m2.
            Reference: https://doi.org/10.1016/j.ijggc.2019.03.018
            TODO: Update this price value for typical NF. This value is for CO2 (RO?) membranes
        opereating costs assumes all membranes get replaced every 5 years (20% replaced every year)
        """

        blk.factor_membrane_replacement = Param(
            initialize=0.2,
            doc="Membrane replacement factor [fraction of membrane replaced/year]",
            units=units.year**-1,
        )
        blk.membrane_cost = Param(
            initialize=50,
            doc="Membrane cost",
            units=units.USD_2021 / (units.meter**2),  # TODO: validate reference year
        )
        blk.hydraulic_permeability = Param(
            initialize=3,
            doc="Hydraulic permeability (Lp) of the membrane",
            units=units.L / units.m**2 / units.hr / units.bar,
        )

        # create the capital and operating cost variables
        make_capital_cost_var(blk)
        make_fixed_operating_cost_var(blk)

        # calculate membrane area
        blk.membrane_area = Var(
            initialize=2700,
            doc="Membrane area in square meters",
            units=units.m**2,
        )

        @blk.Constraint()
        def membrane_area_equation(blk):
            return blk.membrane_area == units.convert(
                (membrane_length * membrane_width), to_units=units.m**2
            )

        # calculate pressure drop
        blk.pressure_drop = Var(
            initialize=483,
            doc="Pressure drop over the membrane",
            units=units.psi,
        )

        Lp = units.convert(
            blk.hydraulic_permeability,
            to_units=units.m**3 / units.m**2 / units.hr / units.bar,
        )

        @blk.Constraint()
        def pressure_drop_equation(blk):
            return blk.pressure_drop == units.convert(
                (water_flux / Lp), to_units=units.psi
            )

        # calculate specific energy consumption
        blk.SEC = Var(
            initialize=3,
            doc="Specific energy consumption of feed pump",
            units=units.kWh / units.m**3,
        )

        dP = units.convert(blk.pressure_drop, to_units=units.Pa)

        @blk.Constraint()
        def SEC_equation(blk):
            return blk.SEC == units.convert(
                (vol_flow_feed * dP / vol_flow_perm), to_units=units.kWh / units.m**3
            )

        @blk.Constraint()
        def capital_cost_constraint(blk):
            return blk.capital_cost == units.convert(
                (blk.membrane_cost * blk.membrane_area),
                to_units=blk.costing_package.base_currency,
            )

        @blk.Constraint()
        def fixed_operating_cost_constraint(blk):
            return blk.fixed_operating_cost == units.convert(
                (
                    blk.factor_membrane_replacement
                    * blk.membrane_cost
                    * blk.membrane_area
                ),
                to_units=blk.costing_package.base_currency
                / blk.costing_package.base_period,
            ) + units.convert(
                (blk.SEC * vol_flow_perm * blk.costing_package.electricity_cost),
                to_units=blk.costing_package.base_currency
                / blk.costing_package.base_period,
            )

    def cost_pump(blk, inlet_pressure, outlet_pressure, inlet_vol_flow):
        """
        pump:
        assume (for now) there is just one pump for the diafiltrate
        capital cost assumes centrifugal pump
            The cost calculation is based on Perry's handbook.
            The equation is (5) in https://doi.org/10.1016/j.memsci.2015.04.065
        operating cost comes from the energy needed to power the pumps
            Assumptions:
                - centrifugal pumps
                - average pressure of 10 bar (145 psi) for NF
                - the fluid is dilute enough that the specific gravity and density are that of water
                - the pump efficiency is 70%
                - the unit cost of electricity is $0.168/kWh (from Ref [3] below)
                - there are 8760 hours in a year
                    TODO: update this for a reasonable operating time of the year
            References:
            [1] Volk, Michael. Pump characteristics and applications. CRC Press, 2013.
            [2] Moran, Seán. "Pump Sizing: Bridging the Gap Between Theory and Practice."
                The Best of Equipment Series (2016): 3.
            [3] https://www.bls.gov/regions/midwest/data/averageenergyprices_selectedareas_table.htm
        """
        blk.pump_correlation_factor = Param(
            initialize=622.59,
            doc="Pump correlation factor (constant)",
            units=units.USD_1996 / (units.kPa * units.m**3 / units.hr) ** 0.39,
        )
        blk.pump_exponential_factor = Param(
            initialize=0.39,
            doc="Pump correlation factor (exponential)",
            units=units.dimensionless,
        )
        blk.pump_head_factor = Param(
            initialize=2.31,
            doc="Pump head factor",
            units=units.ft / units.psi,
        )
        blk.pump_power_factor = Param(
            initialize=3.6 * 10 ** (6),
            doc="Pump power factor",
            units=units.dimensionless,
        )
        blk.pump_efficiency = Param(
            initialize=0.7,
            doc="Pump efficiency",
            units=units.dimensionless,
        )

        # create the capital and operating cost variables
        make_capital_cost_var(blk)
        make_fixed_operating_cost_var(blk)

        # calculate the pump head: pump Ref [1] Eqn 1.1
        blk.pump_head = Var(
            initialize=10,
            doc="Pump head in meters",
            units=units.m,
        )

        @blk.Constraint()
        def pump_head_equation(blk):
            return blk.pump_head == units.convert(
                (
                    outlet_pressure
                    * blk.pump_head_factor
                    / blk.costing_package.specific_gravity
                ),
                to_units=units.m,
            )

        # calculate the pump power: pump Ref [2] Eqn 7
        blk.pump_power = Var(
            initialize=10,
            doc="Pump power in kWh required for the operational period",
            units=units.kWh,
        )

        grav_constant = units.convert(
            Constants.acceleration_gravity, to_units=units.m / units.hr**2
        )

        @blk.Constraint()
        def pump_power_equation(blk):
            return blk.pump_power == units.convert(
                (
                    units.convert(
                        (
                            inlet_vol_flow
                            * blk.costing_package.density
                            * grav_constant
                            * blk.pump_head
                            / blk.pump_power_factor
                            / blk.pump_efficiency
                        ),
                        to_units=units.kW,
                    )
                    * blk.costing_package.operating_time  # per one year
                ),
                to_units=units.kWh,
            )

        @blk.Constraint()
        def capital_cost_constraint(blk):
            return blk.capital_cost == units.convert(
                blk.pump_correlation_factor
                * (inlet_vol_flow * inlet_pressure) ** blk.pump_exponential_factor,
                to_units=blk.costing_package.base_currency,
            )

        @blk.Constraint()
        def fixed_operating_cost_constraint(blk):
            return blk.fixed_operating_cost == units.convert(
                blk.pump_power
                * blk.costing_package.electricity_cost
                / blk.costing_package.operating_time,  # per one year
                to_units=blk.costing_package.base_currency
                / blk.costing_package.base_period,
            )