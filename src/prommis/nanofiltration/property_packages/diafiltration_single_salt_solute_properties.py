#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Property package for the multi-salt diafiltration membrane.

Author: Molly Dougher
"""

from pyomo.common.config import ConfigValue
from pyomo.environ import Param, Var, units

from idaes.core import (
    Component,
    MaterialFlowBasis,
    Phase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
)
from idaes.core.util.initialization import fix_state_vars


@declare_process_block_class("SoluteParameter")
class SoluteParameterData(PhysicalParameterBlock):
    """
    Property Package for the single-salt diafiltration membrane.

    Congfig arguments:
        single_salt_system: chosen salt name

    Currently includes the following solutes:
        Li+ (lithium ion)
        Co2+ (cobalt ion)
        Al3+ (aluminum ion)

        Cl- (chloride ion)

    Thus, the following salt are supported:
        lithium_chloride
        cobalt_chloride
        aluminum_chloride
    """

    CONFIG = PhysicalParameterBlock.CONFIG()

    CONFIG.declare(
        "single_salt_system",
        ConfigValue(
            default="lithium_chloride",
            doc="Name of the salt system to be modeled",
        ),
    )

    def build(self):
        super().build()

        self.liquid = Phase()

        # add cation
        self.cation = Component()

        # add anion
        self.anion = Component()

        # ion valence
        charge_dict = {
            "Li": 1,
            "Co": 2,
            "Al": 3,
            "Cl": -1,
        }

        # single solute diffusion coefficient
        diffusion_coefficient_dict = {
            "Li": 3.71,  # mm2/h
            "Co": 2.64,  # mm2/h
            "Al": 2.01,  # mm2/h
            "Cl": 7.31,  # mm2/h
        }

        # thermal reflection coefficient, related to solute rejection
        sigma_dict = {
            "Li": 1,
            "Co": 1,
            "Al": 1,
            "Cl": 1,
        }

        # add partition coefficient
        # currently H,Li is based on https://doi.org/10.1021/acs.iecr.4c04763
        # H,Cl arbitrarily chosen to be the same value
        partition_coefficient_dict = {
            "retentate": {
                "Li": 0.3,
                "Co": 0.03,
                "Al": 0.003,
                "Cl": 0.3,
            },
            "permeate": {
                "Li": 0.6,
                "Co": 0.6,
                "Al": 0.6,
                "Cl": 0.6,
            },
        }

        num_solutes_dict = {
            "lithium_chloride": {
                "Li": 1,
                "Cl": 1,
            },
            "cobalt_chloride": {
                "Co": 1,
                "Cl": 2,
            },
            "aluminum_chloride": {
                "Al": 1,
                "Cl": 3,
            },
        }

        if self.config.single_salt_system == "lithium_chloride":
            cation = "Li"
            anion = "Cl"
        elif self.config.single_salt_system == "cobalt_chloride":
            cation = "Co"
            anion = "Cl"
        elif self.config.single_salt_system == "aluminum_chloride":
            cation = "Al"
            anion = "Cl"
        else:
            # TODO write exception
            pass

        self.charge = Param(
            self.component_list,
            units=units.dimensionless,
            initialize={
                "cation": charge_dict[cation],
                "anion": charge_dict[anion],
            },
        )

        self.diffusion_coefficient = Param(
            self.component_list,
            units=units.mm**2 / units.h,
            initialize={
                "cation": diffusion_coefficient_dict[cation],
                "anion": diffusion_coefficient_dict[anion],
            },
        )

        self.sigma = Param(
            self.component_list,
            units=units.dimensionless,
            initialize={
                "cation": sigma_dict[cation],
                "anion": sigma_dict[anion],
            },
        )

        self.partition_coefficient_retentate = Param(
            self.component_list,
            units=units.dimensionless,
            initialize={
                "cation": partition_coefficient_dict["retentate"][cation],
                "anion": partition_coefficient_dict["retentate"][anion],
            },
        )

        self.partition_coefficient_permeate = Param(
            self.component_list,
            units=units.dimensionless,
            initialize={
                "cation": partition_coefficient_dict["permeate"][cation],
                "anion": partition_coefficient_dict["permeate"][anion],
            },
        )

        self.num_solutes = Param(
            self.component_list,
            units=units.dimensionless,
            initialize={
                "cation": num_solutes_dict[self.config.single_salt_system][cation],
                "anion": num_solutes_dict[self.config.single_salt_system][anion],
            },
            doc="Moles of ions dissociated in solution per mole of lithium and cobalt chloride",
        )

        self._state_block_class = SoluteStateBlock

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "flow_vol": {"method": None},
                "conc_mol_comp": {"method": None},
                "flow_mol_comp": {"method": None},
            }
        )
        obj.add_default_units(
            {
                "time": units.hour,
                "length": units.m,
                "mass": units.kg,
                "amount": units.mol,
                "temperature": units.K,
            }
        )


class _SoluteStateBlock(StateBlock):
    def fix_initialization_states(self):
        """
        Fixes state variables for state blocks.

        Returns:
            None
        """
        fix_state_vars(self)


@declare_process_block_class("SoluteStateBlock", block_class=_SoluteStateBlock)
class SoluteStateBlockData(StateBlockData):
    """
    State block for multi-salt diafiltration membrane
    """

    def build(self):
        super().build()

        self.flow_vol = Var(
            units=units.m**3 / units.h,
            initialize=10,
            bounds=(1e-20, None),
        )
        self.conc_mol_comp = Var(
            self.component_list,
            units=units.mol / units.m**3,
            initialize=1e-5,
            bounds=(1e-20, None),
        )

    def get_material_flow_terms(self, p, j):
        return self.flow_vol * self.conc_mol_comp[j]

    def get_material_flow_basis(self):
        return MaterialFlowBasis.mole

    def define_state_vars(self):
        return {"flow_vol": self.flow_vol, "conc_mol_comp": self.conc_mol_comp}
