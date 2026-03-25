#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
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
        Li (lithium ion, +)
        Co (cobalt ion, 2+)
        Al (aluminum ion, 3+)
        Cl (chloride ion, -)
        SO4 (sulfate ion, 2-)

    The partition coefficients and hindered diffusion coefficients assume
    a negatively charged membrane.
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
            "Li": 1,
            "Co": 2,
            "Al": 3,
            "Cl": -1,
            "SO4": -2,
        }

        # infinite dilution solute diffusion coefficient
        # source: https://www.aqion.de/site/diffusion-coefficients
        # assumption: no hindered transport within the boundary layer (D_water = D_bl)
        boundary_layer_diffusion_coefficient_dict = {
            "Li": 3.71,  # mm2 / h
            "Co": 2.64,  # mm2 / h
            "Al": 2.01,  # mm2 / h
            "Cl": 7.31,  # mm2 / h
            "SO4": 3.85,  # mm2 / h
        }
        # use a hindered diffusion coefficient to estimate membrane diffusion coefficient
        # factor = D_solution / D_membrane
        # estimated from: https://www.science.org/doi/epdf/10.1126/sciadv.adu8302
        if self.config.anion_list[0] == "Cl":
            hindered_diffusion_coefficient_dict = {
                "Li": 0.0003,
                "Co": 0.0003,
                "Al": 0.0003,
                "Cl": 0.008,
                "SO4": 0,
            }
        elif self.config.anion_list[0] == "SO4":
            hindered_diffusion_coefficient_dict = {
                "Li": 0.0007,
                "Co": 0.0007,
                "Al": 0.0007,
                "Cl": 0,
                "SO4": 0.05,
            }

        membrane_diffusion_coefficient_dict = {
            "Li": hindered_diffusion_coefficient_dict["Li"]
            * boundary_layer_diffusion_coefficient_dict["Li"],  # mm2 / h
            "Co": hindered_diffusion_coefficient_dict["Co"]
            * boundary_layer_diffusion_coefficient_dict["Co"]
            / 10,  # mm2 / h
            "Al": hindered_diffusion_coefficient_dict["Al"]
            * boundary_layer_diffusion_coefficient_dict["Al"]
            / 100,  # mm2 / h
            "Cl": hindered_diffusion_coefficient_dict["Cl"]
            * boundary_layer_diffusion_coefficient_dict["Cl"],  # mm2 / h
            "SO4": hindered_diffusion_coefficient_dict["SO4"]
            * boundary_layer_diffusion_coefficient_dict["SO4"],  # mm2 / h
        }

        # thermal reflection coefficient, related to solute rejection
        sigma_dict = {
            "Li": 1,
            "Co": 1,
            "Al": 1,
            "Cl": 1,
            "SO4": 1,
        }

        # partition coefficient at the solution-membrane interfaces
        # estimated from: https://doi.org/10.1126/sciadv.adu8302
        # while H on the retentate and permeate sides can differ, we assume them to be equal for now
        if self.config.anion_list[0] == "Cl":
            ion_partition_coefficient_dict = {
                "retentate": {
                    "Li": 0.3,
                    "Co": 0.4,
                    "Al": 0.5,
                    "Cl": 0.02,
                    "SO4": 0,
                },
                "permeate": {
                    "Li": 0.3,
                    "Co": 0.4,
                    "Al": 0.5,
                    "Cl": 0.02,
                    "SO4": 0,
                },
            }
        elif self.config.anion_list[0] == "SO4":
            ion_partition_coefficient_dict = {
                "retentate": {
                    "Li": 0.06,
                    "Co": 0.06,
                    "Al": 0.06,
                    "Cl": 0,
                    "SO4": 0.003,
                },
                "permeate": {
                    "Li": 0.06,
                    "Co": 0.06,
                    "Al": 0.06,
                    "Cl": 0,
                    "SO4": 0.003,
                },
            }

        partition_coefficient_dict = {
            "retentate": {
                "Li": ion_partition_coefficient_dict["retentate"]["Li"],
                "Co": ion_partition_coefficient_dict["retentate"]["Co"],
                "Al": ion_partition_coefficient_dict["retentate"]["Al"],
                "Cl": ion_partition_coefficient_dict["retentate"]["Cl"],
                "SO4": ion_partition_coefficient_dict["retentate"]["SO4"],
            },
            "permeate": {
                "Li": ion_partition_coefficient_dict["permeate"]["Li"],
                "Co": ion_partition_coefficient_dict["permeate"]["Co"],
                "Al": ion_partition_coefficient_dict["permeate"]["Al"],
                "Cl": ion_partition_coefficient_dict["permeate"]["Cl"],
                "SO4": ion_partition_coefficient_dict["permeate"]["SO4"],
            },
        }

        if self.config.anion_list[0] == "Cl":
            if self.config.cation_list == ["Li"]:
                salt_system = "Li_Cl"
            elif self.config.cation_list == ["Co"]:
                salt_system = "Co_Cl2"
            elif self.config.cation_list == ["Al"]:
                salt_system = "Al_Cl3"
            elif self.config.cation_list == ["Li", "Co"]:
                salt_system = "Li_Co_Cl3"
            elif self.config.cation_list == ["Li", "Al"]:
                salt_system = "Li_Al_Cl4"
            elif self.config.cation_list == ["Co", "Al"]:
                salt_system = "Co_Al_Cl5"
            elif self.config.cation_list == ["Li", "Co", "Al"]:
                salt_system = "Li_Co_Al_Cl6"
        elif self.config.anion_list[0] == "SO4":
            if self.config.cation_list == ["Li"]:
                salt_system = "Li2_SO4"
            elif self.config.cation_list == ["Co"]:
                salt_system = "Co_SO4"
            elif self.config.cation_list == ["Al"]:
                salt_system = "Al2_(SO4)3"
            elif self.config.cation_list == ["Li", "Co"]:
                salt_system = "Li2_Co_(SO4)2"
            elif self.config.cation_list == ["Li", "Al"]:
                salt_system = "Li2_Al2_(SO4)4"
            elif self.config.cation_list == ["Co", "Al"]:
                salt_system = "Co_Al2_(SO4)4"
            elif self.config.cation_list == ["Li", "Co", "Al"]:
                salt_system = "Li2_Co_Al2_(SO4)5"

        num_solutes_dict = {
            "Li_Cl": {
                "Li": 1,
                "Cl": 1,
            },
            "Co_Cl2": {
                "Co": 1,
                "Cl": 2,
            },
            "Al_Cl3": {
                "Al": 1,
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
            "Li2_SO4": {
                "Li": 2,
                "SO4": 1,
            },
            "Co_SO4": {
                "Co": 1,
                "SO4": 1,
            },
            "Al2_(SO4)3": {
                "Al": 2,
                "SO4": 3,
            },
            "Li2_Co_(SO4)2": {
                "Li": 2,
                "Co": 1,
                "SO4": 2,
            },
            "Li2_Al2_(SO4)4": {
                "Li": 2,
                "Al": 2,
                "SO4": 4,
            },
            "Co_Al2_(SO4)4": {
                "Co": 1,
                "Al": 2,
                "SO4": 4,
            },
            "Li2_Co_Al2_(SO4)5": {
                "Li": 2,
                "Co": 1,
                "Al": 2,
                "SO4": 5,
            },
        }

        # initialize dictionaries for a single cation
        cat_1 = self.config.cation_list[0]
        a0 = self.config.anion_list[0]
        initialize_charge_dict = {
            cat_1: charge_dict[cat_1],
            a0: charge_dict[a0],
        }
        initialize_boundary_layer_diffusion_coefficient_dict = {
            cat_1: boundary_layer_diffusion_coefficient_dict[cat_1],
            a0: boundary_layer_diffusion_coefficient_dict[a0],
        }
        initialize_membrane_diffusion_coefficient_dict = {
            cat_1: membrane_diffusion_coefficient_dict[cat_1],
            a0: membrane_diffusion_coefficient_dict[a0],
        }
        initialize_sigma_dict = {
            cat_1: sigma_dict[cat_1],
            a0: sigma_dict[a0],
        }
        initialize_partition_coefficient_retentate_dict = {
            cat_1: partition_coefficient_dict["retentate"][cat_1],
            a0: partition_coefficient_dict["retentate"][a0],
        }
        initialize_partition_coefficient_permeate_dict = {
            cat_1: partition_coefficient_dict["permeate"][cat_1],
            a0: partition_coefficient_dict["permeate"][a0],
        }
        initialize_num_solutes_dict = {
            cat_1: num_solutes_dict[salt_system][cat_1],
            a0: num_solutes_dict[salt_system][a0],
        }

        # add additional cations to dictionaries
        cation_list = self.config.cation_list
        i = 1
        while i < len(cation_list):
            initialize_charge_dict.update({cation_list[i]: charge_dict[cation_list[i]]})
            initialize_boundary_layer_diffusion_coefficient_dict.update(
                {
                    cation_list[i]: boundary_layer_diffusion_coefficient_dict[
                        cation_list[i]
                    ]
                }
            )
            initialize_membrane_diffusion_coefficient_dict.update(
                {cation_list[i]: membrane_diffusion_coefficient_dict[cation_list[i]]}
            )
            initialize_sigma_dict.update({cation_list[i]: sigma_dict[cation_list[i]]})
            initialize_partition_coefficient_retentate_dict.update(
                {
                    cation_list[i]: partition_coefficient_dict["retentate"][
                        cation_list[i]
                    ]
                }
            )
            initialize_partition_coefficient_permeate_dict.update(
                {cation_list[i]: partition_coefficient_dict["permeate"][cation_list[i]]}
            )
            initialize_num_solutes_dict.update(
                {cation_list[i]: num_solutes_dict[salt_system][cation_list[i]]}
            )
            i += 1

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

        self.partition_coefficient_retentate = Param(
            self.component_list,
            units=units.dimensionless,
            initialize=initialize_partition_coefficient_retentate_dict,
        )

        self.partition_coefficient_permeate = Param(
            self.component_list,
            units=units.dimensionless,
            initialize=initialize_partition_coefficient_permeate_dict,
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
