#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""
Property package for the multi-component diafiltration membrane.

Author: Molly Dougher
"""

from pyomo.common.config import ConfigValue, ListOf
from pyomo.environ import Param, Set, Var, units

from idaes.core import (
    MaterialFlowBasis,
    Phase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
)
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.initialization import fix_state_vars


@declare_process_block_class("MultiComponentDiafiltrationSoluteParameter")
class MultiComponentDiafiltrationSoluteParameterData(PhysicalParameterBlock):
    """
    Property Package for the multi-component diafiltration membrane.

    Currently includes the following solutes:
        K (potassium ion, +)
        Na (sodium ion, +)
        Li (lithium ion, +)
        Ca (calcium ion, 2+)
        Co (cobalt ion, 2+)
        Al (aluminum ion, 3+)
        La (lanthanum ion, 3+)
        Cl (chloride ion, -)
    """

    CONFIG = PhysicalParameterBlock.CONFIG()

    CONFIG.declare(
        "cation_list",
        ConfigValue(
            domain=ListOf(str),
            default=["Li", "Co"],
            doc="List of cations present in the system",
        ),
    )
    CONFIG.declare(
        "anion_list",
        ConfigValue(
            domain=ListOf(str),
            default=["Cl"],
            doc="List of anions present in the system",
        ),
    )

    def build(self):
        super().build()

        if len(self.config.anion_list) > 1:
            raise ConfigurationError(
                "The multi-component diafiltration unit model only supports systems with a common anion"
            )

        self.liquid = Phase()

        self.component_list = Set(
            initialize=self.config.cation_list + self.config.anion_list
        )

        # ion valence
        charge_dict = {
            "K": 1,
            "Na": 1,
            "Li": 1,
            "Ca": 2,
            "Co": 2,
            "Al": 3,
            "La": 3,
            "Cl": -1,
        }

        # infinite dilution solute diffusion coefficient
        # source: https://www.aqion.de/site/diffusion-coefficients
        boundary_layer_diffusion_coefficient_dict = {
            "K": 7.06,  # mm2 / h
            "Na": 4.79,  # mm2 / h
            "Li": 3.71,  # mm2 / h
            "Ca": 2.85,  # mm2 / h
            "Co": 2.64,  # mm2 / h
            "Al": 2.01,  # mm2 / h
            "La": 2.23,  # mm2 / h # TODO: verify
            "Cl": 7.31,  # mm2 / h
        }
        membrane_diffusion_coefficient_dict = {
            "K": 7.06 * 0.01,  # mm2 / h
            "Na": 4.79 * 0.01,  # mm2 / h
            "Li": 3.71 * 0.01,  # mm2 / h
            "Ca": 2.85 * 0.01,  # mm2 / h
            "Co": 2.64 * 0.01,  # mm2 / h
            "Al": 2.01 * 0.01,  # mm2 / h
            "La": 2.23 * 0.01,  # mm2 / h # TODO: verify
            "Cl": 7.31 * 0.01,  # mm2 / h
        }

        # thermal reflection coefficient, related to solute rejection
        sigma_dict = {
            "K": 1,
            "Na": 1,
            "Li": 1,
            "Ca": 1,
            "Co": 1,
            "Al": 1,
            "La": 1,
            "Cl": 1,
        }

        # steric component of partition coefficient at the solution-membrane interfaces
        # (1 - (radius_ion/radius_pore))**2
        # using hydrated radius
        # steric_partition_coefficient_dict = {
        #     "K": 0.1,
        #     "Na": 0.08,
        #     "Li": 0.06,
        #     "Ca": 0.03,
        #     "Co": 0.02,
        #     "Al": 0.003,
        #     "La": 0.009,
        #     "Cl": 0.1,
        # }
        # using stokes radius
        steric_partition_coefficient_dict = {
            "K": 0.6,
            "Na": 0.4,
            "Li": 0.3,
            "Ca": 0.1,
            "Co": 0.1,
            "Al": 0.01,
            "La": 0.04,
            "Cl": 0.6,
        }

        # using hydrated radius
        # dielectric_partition_coefficient_dict = {
        #     "K": 0.4,
        #     "Na": 0.4,
        #     "Li": 0.4,
        #     "Ca": 0.04,
        #     "Co": 0.04,
        #     "Al": 0.002,
        #     "La": 0.001,
        #     "Cl": 0.4,
        # }
        # using stokes radius
        dielectric_partition_coefficient_dict = {
            "K": 0.2,
            "Na": 0.2,
            "Li": 0.06,
            "Ca": 0.01,
            "Co": 0.02,
            "Al": 0.0009,
            "La": 0.0004,
            "Cl": 0.06,
        }

        if self.config.cation_list == ["K"]:
            salt_system = "K_Cl"
        elif self.config.cation_list == ["Na"]:
            salt_system = "Na_Cl"
        elif self.config.cation_list == ["Li"]:
            salt_system = "Li_Cl"
        elif self.config.cation_list == ["Ca"]:
            salt_system = "Ca_Cl2"
        elif self.config.cation_list == ["Co"]:
            salt_system = "Co_Cl2"
        elif self.config.cation_list == ["Al"]:
            salt_system = "Al_Cl3"
        elif self.config.cation_list == ["La"]:
            salt_system = "La_Cl3"
        elif self.config.cation_list == ["Li", "Co"]:
            salt_system = "Li_Co_Cl3"
        elif self.config.cation_list == ["Li", "Al"]:
            salt_system = "Li_Al_Cl4"
        elif self.config.cation_list == ["Co", "Al"]:
            salt_system = "Co_Al_Cl5"
        elif self.config.cation_list == ["Li", "Co", "Al"]:
            salt_system = "Li_Co_Al_Cl6"

        num_solutes_dict = {
            "K_Cl": {
                "K": 1,
                "Cl": 1,
            },
            "Na_Cl": {
                "Na": 1,
                "Cl": 1,
            },
            "Li_Cl": {
                "Li": 1,
                "Cl": 1,
            },
            "Ca_Cl2": {
                "Ca": 1,
                "Cl": 2,
            },
            "Co_Cl2": {
                "Co": 1,
                "Cl": 2,
            },
            "Al_Cl3": {
                "Al": 1,
                "Cl": 3,
            },
            "La_Cl3": {
                "La": 1,
                "Cl": 3,
            },
            "Li_Co_Cl3": {
                "Li": 1,
                "Co": 1,
                "Cl": 3,
            },
            "Li_Al_Cl4": {
                "Li": 1,
                "Al": 1,
                "Cl": 4,
            },
            "Co_Al_Cl5": {
                "Co": 1,
                "Al": 1,
                "Cl": 5,
            },
            "Li_Co_Al_Cl6": {
                "Li": 1,
                "Co": 1,
                "Al": 1,
                "Cl": 6,
            },
        }

        # create subset of property dictionaries to initialize parameters
        def _subset(mapping_dict):
            return {ion: mapping_dict[ion] for ion in self.component_list}

        initialize_charge_dict = _subset(charge_dict)
        initialize_boundary_layer_diffusion_coefficient_dict = _subset(
            boundary_layer_diffusion_coefficient_dict
        )
        initialize_membrane_diffusion_coefficient_dict = _subset(
            membrane_diffusion_coefficient_dict
        )
        initialize_sigma_dict = _subset(sigma_dict)
        initialize_steric_partition_coefficient_dict = _subset(
            steric_partition_coefficient_dict
        )
        initialize_dielectric_partition_coefficient_dict = _subset(
            dielectric_partition_coefficient_dict
        )
        initialize_num_solutes_dict = _subset(num_solutes_dict[salt_system])

        # initialize properties
        self.charge = Param(
            self.component_list,
            units=units.dimensionless,
            initialize=initialize_charge_dict,
        )

        self.boundary_layer_diffusion_coefficient = Param(
            self.component_list,
            units=units.mm**2 / units.h,
            initialize=initialize_boundary_layer_diffusion_coefficient_dict,
        )

        self.membrane_diffusion_coefficient = Param(
            self.component_list,
            units=units.mm**2 / units.h,
            initialize=initialize_membrane_diffusion_coefficient_dict,
        )

        self.sigma = Param(
            self.component_list,
            units=units.dimensionless,
            initialize=initialize_sigma_dict,
        )

        self.steric_partition_coefficient = Param(
            self.component_list,
            units=units.dimensionless,
            initialize=initialize_steric_partition_coefficient_dict,
        )

        self.dielectric_partition_coefficient = Param(
            self.component_list,
            units=units.dimensionless,
            initialize=initialize_dielectric_partition_coefficient_dict,
        )

        self.num_solutes = Param(
            self.component_list,
            units=units.dimensionless,
            initialize=initialize_num_solutes_dict,
            doc="Moles of ions dissociated in solution per mole of salt(s)",
        )

        self._state_block_class = MultiComponentDiafiltrationSoluteStateBlock

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


class _MultiComponentDiafiltrationSoluteStateBlock(StateBlock):
    def fix_initialization_states(self):
        """
        Fixes state variables for state blocks.

        Returns:
            None
        """
        fix_state_vars(self)


@declare_process_block_class(
    "MultiComponentDiafiltrationSoluteStateBlock",
    block_class=_MultiComponentDiafiltrationSoluteStateBlock,
)
class MultiComponentDiafiltrationSoluteStateBlockData(StateBlockData):
    """
    State block for multi-component diafiltration membrane
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
