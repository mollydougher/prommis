#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
r"""
Multi-Component Diafiltration Unit Model
========================================

Author: Molly Dougher

This membrane unit model is for the multi-component diafiltration of a multi-salt system with a common anion. Currently, the model and property packages support one, two, and three salt systems; however, the model can be extended to :math:`n` salts by supplying the appropriate properties and arguments (see below). The membrane is designed for use in a diafiltration cascade, i.e., the model represents one spiral-wound membrane module piece within a cascade of several membranes.

Configuration Arguments
-----------------------

The Multi-Component Diafiltration unit model requires a property package that provides the valency (:math:`z_i`), infinite dilution diffusion coefficient (:math:`D_i`) in :math:`\mathrm{mm}^2 \, \mathrm{h}^{-1}`, thermodynamic reflection coefficient (:math:`\sigma_i`), partition coefficients (:math:`H_{i,r}` and :math:`H_{i,p}`) at the retentate-membrane and membrane-permeate interfaces, and number of dissolved species (:math:`n_i`) for each ion :math:`i` in solution. When used in a flowsheet, the user can provide separate property packages for the feed and product streams.

There are four required arguments:

#. ``cation_list`` (list of cations present in the system)

    ``default=["Li", "Co"]``

#. ``anion_list`` (list of anions present in the system)

    ``default=["Cl"]``

#. ``NFE_module_length`` (the desired number of finite elements across the width of the membrane (i.e., the module length))
#. ``NFE_membrane_thickness`` (the desired number of finite elements across the thickness of the membrane)

Degrees of Freedom
------------------

The Multi-Component Diafiltration unit model has :math:`5+2n` degrees of freedom, where :math:`n` is the number of cations in the system:

#. the length of the membrane module (``total_module_length``)
#. the length of the membrane (``total_membrane_length``)
#. the pressure applied to the membrane system (``applied_pressure``)
#. the volumetric flow rate of the feed (``feed_flow_volume``)
#. the cation concentration in the feed (``feed_conc_mol_comp[t,k]``)
#. the volumetric flow rate of the diafiltrate (``diafiltrate_flow_volume``)
#. the cation concentration in the diafiltrate (``diafiltrate_conc_mol_comp[t,k]``)

Model Structure
---------------

There are three phases in the Multi-Component Diafiltration model: the retentate, the membrane, and the permeate. The retentate and the permeate are only discretized with respect to :math:`x` (parallel to the membrane surface), while the membrane is discretized with respect to both :math:`x` and :math:`z` (perpendicular to the membrane surface). The resulting system of partial differential algebraic equations is solved by discretizing with the backward finite difference method.

Assumptions
-----------

The membrane module dimensions, maximum applied pressure, and inlet flow rates assume that one tube (one instance of this model) consists of 4 NF270-440 membranes in series.

The partitioning relationships, which describe how the solutes transition (partition) across the solution-membrane interfaces, are derived assuming Donnan equilibrium. The partitioning coefficients incorporate both steric and Donnan effects.

The default value for the membrane's surface charge (:math:`-44 \, \mathrm{mM}`), was calculated using zeta potential measurements for NF270 membranes. (See `this reference <https://doi.org/10.1021/acs.iecr.4c04763>`_). Currently, the default property package only supports negatively charged membranes.

The membrane is assumed to be :math:`100 \, \mathrm{nm}` thick.

The default value for the membrane permeability (:math:`0.01 \, \mathrm{m} \, \mathrm{h}^{-1} \, \mathrm{bar}^{-1}`) is based off of parameter estimation results from `this reference <https://doi.org/10.1021/acs.iecr.4c04763>`_ for NF270 membranes.

The formation of a boundary layer at the membrane surface due to concentration polarization is neglected for mathematical simplicity.

The dominating transport mechanism within the bulk/retentate solution is convection in the :math:`x`-direction (parallel to the membrane surface). The dominating transport mechanism within the permeate solution is convection in the :math:`z`-direction (perpendicular to the membrane surface).

The transport mechanisms modeled within the membrane are convection, diffusion, and electromigration. Diffusion within the membrane that is normal to the pore walls is ignored, meaning the concentration gradient of ion :math:`i` within the membrane only has a :math:`z`-component (perpendicular to the membrane surface).

Sets
----

The Multi-Component Diafiltration model defines the following discrete sets for solutes and cations in the system, respectively:

.. math:: \mathcal{I}=\{\mathrm{cation_1, cation_2, ..., cation_n, anion}\}
.. math:: \mathcal{K}=\{\mathrm{cation_1, cation_2, ..., cation_n}\}

where :math:`n` is the desired number of cations.

There are 2 continuous sets for each length dimension: ``dimensionless_module_length`` (in the :math:`x`-direction parallel to the membrane surface) and ``dimensionless_membrane_thickness`` (in the :math:`z`-direction perpendicular to the membrane surface). :math:`x` and :math:`z` are non-dimensionalized (denoted as :math:`\bar{x}` and :math:`\bar{z}`, respectively) using the module length (:math:`w`) and membrane thickness (:math:`l`), respectively, to improve numerical stability.

.. math:: \bar{x} \in \mathbb{R} \| 0 \leq \bar{x} \leq 1
.. math:: \bar{z} \in \mathbb{R} \| 0 \leq \bar{z} \leq 1

Some variables have a time domain to be compatible with the property package, even though this is not a dynamic model. Thus, the following set is defined for time.

.. math:: t \in [0]

Default Model Parameters
------------------------

The Multi-Component Diafiltration model has the following parameters.

================ =============================================== ============================ ============= ==========================================================
Parameter        Description                                     Name                         Default Value Units
================ =============================================== ============================ ============= ==========================================================
:math:`\epsilon` numerical tolerance for zero values             ``numerical_zero_tolerance`` 1e-10
:math:`l`        thickness of the membrane                       ``total_membrane_thickness`` 1e-07         :math:`\mathrm{m}`
:math:`L_p`      hydraulic permeability of the membrane          ``membrane_permeability``    0.01          :math:`\mathrm{m} \, \mathrm{h}^{-1} \, \mathrm{bar}^{-1}`
:math:`T`        temperature of the system                       ``temperature``              298           :math:`\mathrm{K}`
:math:`\chi`     concentration of surface charge on the membrane ``membrane_fixed_charge``    -140          :math:`\mathrm{mol} \, \mathrm{m}^{-3}`
================ =============================================== ============================ ============= ==========================================================

Variables
---------

The Multi-Component Diafiltration model adds the following variables.

=========================== ============================================================== ================================================= =========================================================================== =====================================================================================================
Variable                    Description                                                    Name                                              Units                                                                       Indexed over
=========================== ============================================================== ================================================= =========================================================================== =====================================================================================================
:math:`c_{i,d}`             ion concentration in the diafiltrate                           ``diafiltrate_conc_mol_comp``                     :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                                     :math:`t` and :math:`i \in \mathcal{I}`
:math:`c_{i,f}`             ion concentration in the feed                                  ``feed_conc_mol_comp``                            :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                                     :math:`t` and :math:`i \in \mathcal{I}`
:math:`c_{i,m}`             ion concentration in the membrane                              ``membrane_conc_mol_comp``                        :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                                     :math:`t`, :math:`\bar{x}`, :math:`\bar{z}`, amd :math:`i \in \mathcal{I}`
:math:`c_{i,p}`             ion concentration in the permeate                              ``permeate_conc_mol_comp``                        :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                                     :math:`t`, :math:`\bar{x}`, amd :math:`i \in \mathcal{I}`
:math:`c_{i,r}`             ion concentration in the retentate                             ``retentate_conc_mol_comp``                       :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                                     :math:`t`, :math:`\bar{x}`, amd :math:`i \in \mathcal{I}`
:math:`\tilde{D}`           diffusion & convection coefficient denominator in the membrane ``membrane_D_tilde``                              :math:`\mathrm{mm}^2 \, \mathrm{h}^{-1} \, \mathrm{mol} \, \mathrm{m}^{-3}` :math:`t`, :math:`\bar{x}`, and :math:`\bar{z}`
:math:`D_{kj}^{bilinear}`   bilinear cross-diffusion coefficient in the membrane           ``membrane_cross_diffusion_coefficient_bilinear`` :math:`\mathrm{mm}^4 \, \mathrm{h}^{-2} \, \mathrm{mol} \, \mathrm{m}^{-3}` :math:`t`, :math:`\bar{x}`, :math:`\bar{z}`, :math:`k \in \mathcal{K}`, and :math:`j \in \mathcal{K}`
:math:`\alpha_k^{bilinear}` bilinear convection coefficient in the membrane                ``membrane_convection_coefficient_bilinear``      :math:`\mathrm{mm}^2 \, \mathrm{h}^{-1} \, \mathrm{mol} \, \mathrm{m}^{-3}` :math:`t`, :math:`\bar{x}`, :math:`\bar{z}`, and :math:`k \in \mathcal{K}`
:math:`D_{kj}`              cross-diffusion coefficient in the membrane                    ``membrane_cross_diffusion_coefficient``          :math:`\mathrm{mm}^2 \, \mathrm{h}^{-1}`                                    :math:`t`, :math:`\bar{x}`, :math:`\bar{z}`, :math:`k \in \mathcal{K}`, and :math:`j \in \mathcal{K}`
:math:`\alpha_k`            convection coefficient in the membrane                         ``membrnane_convection_coefficient``              :math:`\mathrm{dimensionless}`                                              :math:`t`, :math:`\bar{x}`, :math:`\bar{z}`, and :math:`k \in \mathcal{K}`
:math:`j_i`                 molar flux of ions across the membrane                         ``molar_ion_flux``                                :math:`\mathrm{mol} \, \mathrm{m}^{-2} \, \mathrm{h}^{-1}`                  :math:`t`, :math:`\bar{x}`, amd :math:`i \in \mathcal{I}`
:math:`J_w`                 water flux across the membrane                                 ``volume_flux_water``                             :math:`\mathrm{m}^3 \, \mathrm{m}^{-2} \, \mathrm{h}^{-1}`                  :math:`t` and :math:`\bar{x}`
:math:`L`                   length of the membrane                                         ``total_membrane_length``                         :math:`\mathrm{m}`
:math:`\Delta \pi`          osmotic pressure of feed-side fluid                            ``osmotic_pressure``                              :math:`\mathrm{bar}`                                                        :math:`t` and :math:`\bar{x}`
:math:`\Delta P`            applied pressure to the membrane                               ``applied_pressure``                              :math:`\mathrm{bar}`                                                        :math:`t`
:math:`q_d`                 volumetric flow rate of the diafiltrate                        ``diafiltrate_flow_volume``                       :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}`                                     :math:`t`
:math:`q_f`                 volumetric flow rate of the feed                               ``feed_flow_volume``                              :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}`                                     :math:`t`
:math:`q_p`                 volumetric flow rate of the permeate                           ``permeate_flow_volume``                          :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}`                                     :math:`t` and :math:`\bar{x}`
:math:`q_r`                 volumetric flow rate of the retentate                          ``retentate_flow_volume``                         :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}`                                     :math:`t` and :math:`\bar{x}`
:math:`w`                   length of the membrane module                                  ``total_module_length``                           :math:`\mathrm{m}`
=========================== ============================================================== ================================================= =========================================================================== =====================================================================================================

Derivative Variables
--------------------

The Multi-Component Diafiltration model adds the following derivative variables.

=================================================== =========================================== ================================= ======================================= ==========================================================================
Variable                                            Description                                 Name                              Units                                   Indexed over
=================================================== =========================================== ================================= ======================================= ==========================================================================
:math:`\frac{\mathrm{d}c_{k,r}}{\mathrm{d}\bar{x}}` ion concentration gradient in the retentate ``d_retentate_conc_mass_comp_dx`` :math:`\mathrm{kg} \, \mathrm{m}^{-3}`  :math:`t`, :math:`\bar{x}`, and :math:`k \in \mathcal{K}`
:math:`\frac{\mathrm{d}q_r}{\mathrm{d}\bar{x}}`     retentate flow rate gradient                ``d_retentate_flow_volume_dx``    :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}` :math:`t` and :math:`\bar{x}`
:math:`\frac{\partial c_{k,m}}{\partial \bar{z}}`   ion concentration gradient in the membrane  ``d_membrane_conc_mass_comp_dz``  :math:`\mathrm{kg} \, \mathrm{m}^{-3}`  :math:`t`, :math:`\bar{x}`, :math:`\bar{z}`, and :math:`k \in \mathcal{K}`
=================================================== =========================================== ================================= ======================================= ==========================================================================

Constraints
-----------

**Differential mole balances:**

.. math:: \frac{\mathrm{d}q_r(\bar{x})}{\mathrm{d}\bar{x}} = - J_w(\bar{x}) wL  \qquad \forall \, \bar{x} \in (0, 1]
.. math:: q_r(\bar{x}) \frac{\mathrm{d}c_{k,r}(\bar{x})}{\mathrm{d}\bar{x}} = wL (J_w(\bar{x}) c_{k,r}(\bar{x}) - j_{k}(\bar{x}))  \qquad \forall \, \bar{x} \in (0, 1], \, k \in \mathcal{K}

**Bulk flux balances:**

.. math:: q_p(\bar{x}) = \bar{x} wL J_w(\bar{x}) \qquad \forall \, \bar{x} \in (0, 1]
.. math:: j_{k}(\bar{x}) = c_{k,p}(\bar{x}) J_w(\bar{x}) \qquad \forall \, \bar{x} \in (0, 1], \, k \in \mathcal{K}

**Overall water flux through the membrane:**

.. math:: J_w (\bar{x}) = L_p (\Delta P - \Delta \pi (\bar{x})) \qquad \forall \, \bar{x} \in (0, 1]
.. math:: \Delta \pi (\bar{x}) = \mathrm{R} \mathrm{T} \sum_{i \in \mathcal{I}} n_i \sigma_i (c_{i,r}(\bar{x})-c_{i,p}(\bar{x})) \qquad \forall \, \bar{x} \in (0, 1]

**Cation flux through the membrane:**

*Derived from the extended Nernst-Planck equation*

.. math:: j_k(\bar{x}) = \alpha_k(\bar{x},\bar{z}) c_{k,m}(\bar{x},\bar{z}) J_w(\bar{x}) + \frac{1}{l} \sum_{j \in \mathcal{K}} \left(D_{kj} (\hat{x},\hat{z}) \nabla c_{j,m} (\hat{x},\hat{z}) \right) \qquad \forall \, \bar{x} \in (0, 1], \, k \in \mathcal{K}

where

.. math:: \alpha_k(\bar{x},\bar{z}) = 1 + \dfrac{z_k D_k \chi}{\tilde{D} (\hat{x},\hat{z})}
.. math:: 
    D_{kj}(\bar{x},\bar{z}) = 
    \begin{cases}
        \dfrac{(z_k z_j D_k D_j - z_k z_j D_k D_a)c_{k,m} (\hat{x},\hat{z})}{\tilde{D} (\hat{x},\hat{z})},& \text{if } k \neq j \\
        \dfrac{\sum_{t \in \mathcal{C}} \left((z_t z_a D_k D_a - \beta_{kt})c_{t,m} (\hat{x},\hat{z}) \right) + z_a D_k D_a \chi}{\tilde{D} (\hat{x},\hat{z})} ,& \text{if } k=j \\
    \end{cases}
.. math::
    \beta_{kt} = 
    \begin{cases}
        z_t^2 D_t D_k ,& \text{if } k\neq t \\
        z_t^2 D_t D_a ,& \text{if } k=t \\
    \end{cases}
.. math:: \tilde{D} (\hat{x},\hat{z}) = \sum_{j \in \mathcal{K}} \left((z_j^2 D_j - z_j z_a D_a)c_{j,m} (\hat{x},\hat{z}) \right) - z_a D_a \chi
.. math:: \nabla c_{k,m} (\hat{x},\hat{z})= \dfrac{\partial c_{k,m}(\hat{x},\hat{z})}{\partial \hat{z}}

where the subscript :math:`a` represents the anion in solution.

The diffusion and convection coefficients are reformulated to bilinear constraints:

.. math:: \alpha_k^{bilinear}(\bar{x},\bar{z}) = \alpha_k(\bar{x},\bar{z}) \tilde{D}(\bar{x},\bar{z}) = \tilde{D}(\bar{x},\bar{z}) + z_k D_k \chi
.. math:: D_{kj}^{bilinear}(\bar{x},\bar{z}) = D_{kj}(\bar{x},\bar{z}) \tilde{D}(\bar{x},\bar{z})

*Note that the single solute diffusion coefficients are provided in* :math:`\mathrm{mm}^2\ \, \mathrm{h}^{-1}` *to improve numerical stability, but the diffusion coefficients in the Nernst-Planck equations must be converted to* :math:`\mathrm{m}^2\ \, \mathrm{h}^{-1}`.

**No applied potential on the system:**

.. math:: 0 = \sum_{i \in \mathcal{I}} z_i j_i(\bar{x}) \qquad \forall \, \bar{x} \in (0, 1]

**Electroneutrality:**

.. math:: 0 = \sum_{i \in \mathcal{I}} z_i c_{i,r}(\bar{x})
.. math:: 0 = \chi + \sum_{i \in \mathcal{I}} z_i c_{i,m}(\bar{x},\bar{z}) \qquad \forall \, \bar{x} \in (0, 1]
.. math:: 0 = \sum_{i \in \mathcal{I}} z_i c_{i,p}(\bar{x}) \qquad \forall \, \bar{x} \in (0, 1]

**Partitioning:**

At the the retentate-membrane interface:

.. math:: H_k^{-z_a} H_a^{z_k} = \left(\frac{c_{k,m} (\hat{x},\hat{z}=0)}{c_{k,r} (\hat{x})}\right)^{-z_a} \left(\frac{c_{a,m} (\hat{x},\hat{z}=0)}{c_{a,r}(\hat{x})}\right)^{z_k} \qquad \forall \, \bar{x} \in (0, 1], \, k \in \mathcal{K}

At the membrane-permeate interface:

.. math:: H_k^{-z_a} H_a^{z_k} = \left(\frac{c_{k,m} (\hat{x},\hat{z}=1)}{c_{k,p} (\hat{x})}\right)^{-z_a} \left(\frac{c_{a,m} (\hat{x},\hat{z}=1)}{c_{a,p}(\hat{x})}\right)^{z_k} \qquad \forall \, \bar{x} \in (0, 1], \, k \in \mathcal{K}

**Boundary conditions:**

.. math:: q_r(\bar{x}=0) = q_f + q_d
.. math:: c_{k,r}(\bar{x}=0) = \frac{q_f c_{k,f} + q_d c_{k,d}}{q_f + q_d} \qquad \forall \, k \in \mathcal{K}
.. math:: c_{k,m} (\bar{x}=0,\bar{z}) = 0 \qquad \forall \, \bar{z}, \, k \in \mathcal{K}

The following constraints (which are expected to be zero) are enforced to improve numerical stability (with the appropriate constraints deactivated as described above):

.. math:: q_p(\bar{x}=0) = \epsilon
.. math:: c_{i,p}(\bar{x}=0) = \epsilon \qquad \forall \, i \in \mathcal{I}
.. math:: \frac{\mathrm{d}q_r(\bar{x})}{\mathrm{d}\bar{x}}(\bar{x}=0)=\epsilon
.. math:: \frac{\mathrm{d}c_{k,r}(\bar{x})}{\mathrm{d}\bar{x}}(\bar{x}=0)=\epsilon \qquad \forall \, k \in \mathcal{K}
.. math:: J_w(\bar{x}=0) = \epsilon
.. math:: j_i(\bar{x}=0) = \epsilon \qquad \forall \, i \in \mathcal{I}
"""

from pyomo.common.config import ConfigBlock, ConfigValue, ListOf
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import (
    Constraint,
    Param,
    Reference,
    Set,
    Suffix,
    TransformationFactory,
    Var,
    units,
    value,
)
from pyomo.network import Port

from idaes.core import UnitModelBlockData, declare_process_block_class, useDefault
from idaes.core.initialization import BlockTriangularizationInitializer
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.constants import Constants
from idaes.core.util.exceptions import ConfigurationError


class MultiComponentDiafiltrationInitializer(BlockTriangularizationInitializer):
    """
    Multi-Component Diafiltration Initializer Class.
    """

    @staticmethod
    def _safe_positive(val, floor):
        return max(floor, val)

    @staticmethod
    def _backward_difference(curr, prev, curr_pt, prev_pt):
        delta = curr_pt - prev_pt
        if abs(delta) <= 1e-12:
            return 0
        return (curr - prev) / delta

    def _anion_from_electroneutrality(self, model, cation_conc, include_fixed_charge=False):
        props = model.config.property_package
        anion = model.config.anion_list[0]
        charge_balance = sum(
            value(props.charge[k]) * cation_conc[k] for k in model.cations
        )
        if include_fixed_charge:
            charge_balance += value(model.membrane_fixed_charge)
        anion_charge = value(props.charge[anion])
        return -charge_balance / anion_charge

    def _project_membrane_interface_cations(self, model, cation_conc, floor):
        props = model.config.property_package
        required_charge = max(floor, -value(model.membrane_fixed_charge) + floor)
        current_charge = sum(
            value(props.charge[k]) * cation_conc[k] for k in model.cations
        )

        if current_charge >= required_charge:
            return cation_conc

        if current_charge <= floor:
            total_charge = sum(value(props.charge[k]) for k in model.cations)
            charge_share = {
                k: value(props.charge[k]) / total_charge for k in model.cations
            }
            return {
                k: max(
                    floor,
                    required_charge * charge_share[k] / value(props.charge[k]),
                )
                for k in model.cations
            }

        scale = required_charge / current_charge
        return {k: max(floor, cation_conc[k] * scale) for k in model.cations}

    def _sieving_guess(self, model, ion):
        charge = abs(value(model.config.property_package.charge[ion]))
        return max(0.08, min(0.85, 0.65 / charge))

    def _compute_osmotic_pressure(self, model, retentate_conc, permeate_conc):
        props = model.config.property_package
        return value(
            units.convert(
                Constants.gas_constant
                * model.temperature
                * sum(
                    value(props.num_solutes[j])
                    * value(props.sigma[j])
                    * (
                        (retentate_conc[j] - permeate_conc[j])
                        * units.mol
                        / units.m**3
                    )
                    for j in model.solutes
                ),
                to_units=units.bar,
            )
        )

    def _initialize_transport_coefficients(self, model, t, x, z, floor):
        props = model.config.property_package
        anion = model.config.anion_list[0]
        anion_charge = value(props.charge[anion])
        anion_diffusivity = value(props.diffusion_coefficient[anion])
        fixed_charge = value(model.membrane_fixed_charge)

        d_tilde = sum(
            (
                (
                    value(props.charge[k]) ** 2 * value(props.diffusion_coefficient[k])
                    - value(props.charge[k]) * anion_charge * anion_diffusivity
                )
                * value(model.membrane_conc_mol_comp[t, x, z, k])
            )
            for k in model.cations
        ) - (anion_charge * anion_diffusivity * fixed_charge)

        if abs(d_tilde) <= floor:
            d_tilde = floor

        model.membrane_D_tilde[t, x, z].set_value(d_tilde)

        for k in model.cations:
            k_charge = value(props.charge[k])
            k_diffusivity = value(props.diffusion_coefficient[k])

            convection_bilinear = d_tilde + (k_charge * k_diffusivity * fixed_charge)
            model.membrane_convection_coefficient_bilinear[t, x, z, k].set_value(
                convection_bilinear
            )
            model.membrane_convection_coefficient[t, x, z, k].set_value(
                convection_bilinear / d_tilde
            )

            for j in model.cations:
                j_charge = value(props.charge[j])
                j_diffusivity = value(props.diffusion_coefficient[j])

                if k != j:
                    bilinear = (
                        (k_charge * j_charge * k_diffusivity * j_diffusivity)
                        - (k_charge * j_charge * k_diffusivity * anion_diffusivity)
                    ) * value(model.membrane_conc_mol_comp[t, x, z, k])
                else:
                    bilinear = 0
                    for i in model.cations:
                        i_charge = value(props.charge[i])
                        i_diffusivity = value(props.diffusion_coefficient[i])

                        if k != i:
                            bilinear += (
                                (
                                    i_charge
                                    * anion_charge
                                    * k_diffusivity
                                    * anion_diffusivity
                                )
                                - (i_charge**2 * i_diffusivity * k_diffusivity)
                            ) * value(model.membrane_conc_mol_comp[t, x, z, i])
                        else:
                            bilinear += (
                                (
                                    i_charge
                                    * anion_charge
                                    * k_diffusivity
                                    * anion_diffusivity
                                )
                                - (i_charge**2 * i_diffusivity * anion_diffusivity)
                            ) * value(model.membrane_conc_mol_comp[t, x, z, i])

                    bilinear += (
                        anion_charge
                        * k_diffusivity
                        * anion_diffusivity
                        * fixed_charge
                    )

                model.membrane_cross_diffusion_coefficient_bilinear[
                    t, x, z, k, j
                ].set_value(bilinear)
                model.membrane_cross_diffusion_coefficient[t, x, z, k, j].set_value(
                    bilinear / d_tilde
                )

    def initialization_routine(self, model):
        """
        Initializes the retentate, permeate, and membrane states using a
        staged heuristic consistent with electroneutrality, osmotic pressure,
        and interfacial partitioning.

        Method then calls the block triangularization initializer method.
        """

        props = model.config.property_package
        floor = value(model.numerical_zero_tolerance)
        anion = model.config.anion_list[0]
        x_points = list(model.dimensionless_module_length)
        z_points = list(model.dimensionless_membrane_thickness)

        for t in model.time:
            feed_flow = value(model.feed_flow_volume[t])
            diafiltrate_flow = value(model.diafiltrate_flow_volume[t])
            inlet_flow = feed_flow + diafiltrate_flow
            membrane_area = value(model.total_membrane_length * model.total_module_length)
            pressure = value(model.applied_pressure[t])

            inlet_retentate_cations = {}
            for k in model.cations:
                inlet_retentate_cations[k] = (
                    feed_flow * value(model.feed_conc_mol_comp[t, k])
                    + diafiltrate_flow * value(model.diafiltrate_conc_mol_comp[t, k])
                ) / inlet_flow
            inlet_retentate_anion = self._safe_positive(
                self._anion_from_electroneutrality(model, inlet_retentate_cations),
                floor,
            )

            previous_retentate_flow = inlet_flow
            previous_retentate = {
                **inlet_retentate_cations,
                anion: inlet_retentate_anion,
            }

            for x_index, x in enumerate(x_points):
                x_float = float(x)

                if x_index == 0:
                    model.retentate_flow_volume[t, x].set_value(inlet_flow)
                    model.permeate_flow_volume[t, x].set_value(floor)
                    model.volume_flux_water[t, x].set_value(floor)
                    model.osmotic_pressure[t, x].set_value(floor)

                    for k in model.cations:
                        model.retentate_conc_mol_comp[t, x, k].set_value(
                            inlet_retentate_cations[k]
                        )
                    model.retentate_conc_mol_comp[t, x, anion].set_value(
                        inlet_retentate_anion
                    )

                    for j in model.solutes:
                        model.permeate_conc_mol_comp[t, x, j].set_value(floor)
                        model.molar_ion_flux[t, x, j].set_value(floor)
                        model.d_retentate_conc_mol_comp_dx[t, x, j].set_value(floor)

                    model.d_retentate_flow_volume_dx[t, x].set_value(floor)

                    for z in z_points:
                        for k in model.cations:
                            model.membrane_conc_mol_comp[t, x, z, k].set_value(floor)
                            model.d_membrane_conc_mol_comp_dz[t, x, z, k].set_value(0)
                        model.membrane_conc_mol_comp[t, x, z, anion].set_value(floor)
                        model.d_membrane_conc_mol_comp_dz[t, x, z, anion].set_value(0)
                        model.membrane_D_tilde[t, x, z].set_value(floor)
                        for k in model.cations:
                            model.membrane_convection_coefficient_bilinear[
                                t, x, z, k
                            ].set_value(floor)
                            model.membrane_convection_coefficient[t, x, z, k].set_value(
                                1
                            )
                            for j in model.cations:
                                model.membrane_cross_diffusion_coefficient_bilinear[
                                    t, x, z, k, j
                                ].set_value(floor)
                                model.membrane_cross_diffusion_coefficient[
                                    t, x, z, k, j
                                ].set_value(floor)
                    continue

                retentate_cations = {}
                permeate_cations = {}
                previous_x = x_points[x_index - 1]
                delta_x = x_float - float(previous_x)
                previous_qr = self._safe_positive(previous_retentate_flow, floor)
                provisional_permeate_cations = {}

                for k in model.cations:
                    provisional_permeate_cations[k] = self._safe_positive(
                        previous_retentate[k] * self._sieving_guess(model, k),
                        floor,
                    )

                provisional_permeate_anion = self._safe_positive(
                    self._anion_from_electroneutrality(
                        model,
                        provisional_permeate_cations,
                    ),
                    floor,
                )
                provisional_permeate_state = {
                    **provisional_permeate_cations,
                    anion: provisional_permeate_anion,
                }
                provisional_osmotic_pressure = self._safe_positive(
                    self._compute_osmotic_pressure(
                        model,
                        previous_retentate,
                        provisional_permeate_state,
                    ),
                    floor,
                )
                water_flux = self._safe_positive(
                    value(model.membrane_permeability)
                    * (pressure - provisional_osmotic_pressure),
                    floor,
                )

                for k in model.cations:
                    sieving = self._sieving_guess(model, k)
                    previous_ck = previous_retentate[k]
                    dc_dx = (
                        membrane_area
                        * water_flux
                        * previous_ck
                        * (1 - sieving)
                        / previous_qr
                    )
                    retentate_cations[k] = self._safe_positive(
                        previous_ck + delta_x * dc_dx,
                        floor,
                    )
                    permeate_cations[k] = self._safe_positive(
                        retentate_cations[k] * sieving,
                        floor,
                    )

                retentate_anion = self._safe_positive(
                    self._anion_from_electroneutrality(model, retentate_cations),
                    floor,
                )
                permeate_anion = self._safe_positive(
                    self._anion_from_electroneutrality(model, permeate_cations),
                    floor,
                )

                retentate_state = {**retentate_cations, anion: retentate_anion}
                permeate_state = {**permeate_cations, anion: permeate_anion}

                osmotic_pressure = self._safe_positive(
                    self._compute_osmotic_pressure(
                        model, retentate_state, permeate_state
                    ),
                    floor,
                )
                water_flux = self._safe_positive(
                    value(model.membrane_permeability) * (pressure - osmotic_pressure),
                    floor,
                )
                permeate_flow = self._safe_positive(
                    x_float * membrane_area * water_flux,
                    floor,
                )
                retentate_flow = self._safe_positive(inlet_flow - permeate_flow, floor)

                model.retentate_flow_volume[t, x].set_value(retentate_flow)
                model.permeate_flow_volume[t, x].set_value(permeate_flow)
                model.volume_flux_water[t, x].set_value(water_flux)
                model.osmotic_pressure[t, x].set_value(osmotic_pressure)

                for j in model.solutes:
                    model.retentate_conc_mol_comp[t, x, j].set_value(retentate_state[j])
                    model.permeate_conc_mol_comp[t, x, j].set_value(permeate_state[j])
                    model.molar_ion_flux[t, x, j].set_value(
                        self._safe_positive(permeate_state[j] * water_flux, floor)
                    )

                model.d_retentate_flow_volume_dx[t, x].set_value(
                    self._backward_difference(
                        retentate_flow,
                        previous_retentate_flow,
                        x_float,
                        float(previous_x),
                    )
                )

                for j in model.solutes:
                    model.d_retentate_conc_mol_comp_dx[t, x, j].set_value(
                        self._backward_difference(
                            retentate_state[j],
                            previous_retentate[j],
                            x_float,
                            float(previous_x),
                        )
                    )

                retentate_membrane_interface = {}
                permeate_membrane_interface = {}
                for k in model.cations:
                    retentate_membrane_interface[k] = self._safe_positive(
                        value(props.partition_coefficient_retentate[k])
                        * retentate_state[k],
                        floor,
                    )
                    permeate_membrane_interface[k] = self._safe_positive(
                        value(props.partition_coefficient_permeate[k])
                        * permeate_state[k],
                        floor,
                    )

                retentate_membrane_interface = self._project_membrane_interface_cations(
                    model,
                    retentate_membrane_interface,
                    floor,
                )
                permeate_membrane_interface = self._project_membrane_interface_cations(
                    model,
                    permeate_membrane_interface,
                    floor,
                )

                retentate_membrane_interface[anion] = self._safe_positive(
                    self._anion_from_electroneutrality(
                        model,
                        retentate_membrane_interface,
                        include_fixed_charge=True,
                    ),
                    floor,
                )
                permeate_membrane_interface[anion] = self._safe_positive(
                    self._anion_from_electroneutrality(
                        model,
                        permeate_membrane_interface,
                        include_fixed_charge=True,
                    ),
                    floor,
                )

                for z in z_points:
                    z_float = float(z)
                    for j in model.solutes:
                        membrane_value = self._safe_positive(
                            retentate_membrane_interface[j]
                            + z_float
                            * (
                                permeate_membrane_interface[j]
                                - retentate_membrane_interface[j]
                            ),
                            floor,
                        )
                        model.membrane_conc_mol_comp[t, x, z, j].set_value(
                            membrane_value
                        )

                for z_index, z in enumerate(z_points):
                    z_float = float(z)
                    if z_index == 0:
                        next_z = z_points[min(1, len(z_points) - 1)]
                    else:
                        previous_z = z_points[z_index - 1]

                    for j in model.solutes:
                        if z_index == 0 and len(z_points) > 1:
                            derivative = self._backward_difference(
                                value(model.membrane_conc_mol_comp[t, x, next_z, j]),
                                value(model.membrane_conc_mol_comp[t, x, z, j]),
                                float(next_z),
                                z_float,
                            )
                        elif z_index == 0:
                            derivative = 0
                        else:
                            derivative = self._backward_difference(
                                value(model.membrane_conc_mol_comp[t, x, z, j]),
                                value(
                                    model.membrane_conc_mol_comp[t, x, previous_z, j]
                                ),
                                z_float,
                                float(previous_z),
                            )
                        model.d_membrane_conc_mol_comp_dz[t, x, z, j].set_value(
                            derivative
                        )

                    self._initialize_transport_coefficients(model, t, x, z, floor)

                previous_retentate_flow = retentate_flow
                previous_retentate = retentate_state

        super().initialization_routine(model)


@declare_process_block_class("MultiComponentDiafiltration")
class MultiComponentDiafiltrationData(UnitModelBlockData):
    """
    Multi-Component Diafiltration Unit Model Class.
    """

    # Set default initializer
    default_initializer = MultiComponentDiafiltrationInitializer

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for membrane system",
            doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PhysicalParameterObject** - a PhysicalParameterBlock object.}
""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {see property package for documentation}
""",
        ),
    )
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
    CONFIG.declare(
        "NFE_module_length",
        ConfigValue(
            doc="Number of discretization points in the x-direction (across module length)",
        ),
    )
    CONFIG.declare(
        "NFE_membrane_thickness",
        ConfigValue(
            doc="Number of discretization points in the z-direction (across membrane thickness)",
        ),
    )

    def build(self):
        """
        Build method for the multi-component diafiltration unit model.
        """
        super().build()

        if len(self.config.anion_list) > 1:
            raise ConfigurationError(
                "The multi-component diafiltration unit model only supports systems with a common anion"
            )

        self.add_mutable_parameters()
        self.add_variables()
        self.add_constraints()
        self.discretize_model()
        self.deactivate_unnecessary_objects()
        self.add_scaling_factors()
        self.add_ports()

    def add_mutable_parameters(self):
        """
        Adds default parameters for the multi-component diafiltration unit model.

        Values can be changed by the user during implementation.

        Assumes membrane thickness of 100 nm.

        Membrane permeability and fixed charged are estimated from:
        Liu, Xinhong, et al. (2025) https://doi.org/10.1021/acs.iecr.4c04763
        """
        self.numerical_zero_tolerance = Param(
            initialize=1e-10,
            mutable=True,
            doc="Numerical tolerance for zero values in the model",
        )
        self.total_membrane_thickness = Param(
            initialize=1e-7,
            mutable=True,
            units=units.m,
            doc="Thickness of membrane (z-direction)",
        )
        self.membrane_fixed_charge = Param(
            initialize=-44,
            mutable=True,
            units=units.mol / units.m**3,  # mM
            doc="Fixed charge on the membrane",
        )
        self.membrane_permeability = Param(
            initialize=0.01,
            mutable=True,
            units=units.m / units.h / units.bar,
            doc="Hydraulic permeability coefficient",
        )
        self.temperature = Param(
            initialize=298,
            mutable=True,
            units=units.K,
            doc="System temperature",
        )

    def add_variables(self):
        """
        Adds variables for the multi-component diafiltration unit model.

        Membrane module dimensions and maximum flowrate (17 m3/h) are
        estimated from NF270-440 modules.

        Assumes 4 modules in series.
        """
        # define length scales
        self.dimensionless_module_length = ContinuousSet(bounds=(0, 1))
        self.dimensionless_membrane_thickness = ContinuousSet(bounds=(0, 1))

        # add a time index since the property package variables are indexed over time
        self.time = Set(initialize=[0])

        # add components
        self.solutes = Set(initialize=self.config.cation_list + self.config.anion_list)
        self.cations = Set(initialize=self.config.cation_list)

        # add global variables
        self.total_module_length = Var(
            initialize=4,  # 4 tubes that are ~1m long each (NF270-440)
            units=units.m,
            bounds=[1e-11, None],
            doc="Width of the membrane (x-direction)",
        )
        self.total_membrane_length = Var(
            initialize=41,  # 41 m of length in each tube (NF270-440)
            units=units.m,
            bounds=[1e-11, None],
            doc="Length of the membrane, wound radially",
        )
        self.applied_pressure = Var(
            self.time,
            initialize=10,
            units=units.bar,
            bounds=[1e-11, 41],  # maximum operating presssure (NF270-440)
            doc="Pressure applied to membrane",
        )
        self.feed_flow_volume = Var(
            self.time,
            initialize=12.5,
            units=units.m**3 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric flow rate of the feed",
        )

        def initialize_feed_conc_mol_comp(m, t, j):
            vals = {
                self.config.cation_list[k]: 200
                for k in range(len(self.config.cation_list))
            }
            vals.update({self.config.anion_list[0]: 600})
            return vals[j]

        self.feed_conc_mol_comp = Var(
            self.time,
            self.solutes,
            initialize=initialize_feed_conc_mol_comp,
            units=units.mol / units.m**3,  # mM
            bounds=[1e-11, None],
            doc="Mole concentration of solutes in the feed",
        )
        self.diafiltrate_flow_volume = Var(
            self.time,
            initialize=3.75,
            units=units.m**3 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric flow rate of the diafiltrate",
        )

        def initialize_diafiltrate_conc_mol_comp(m, t, j):
            vals = {
                self.config.cation_list[k]: 10
                for k in range(len(self.config.cation_list))
            }
            vals.update({self.config.anion_list[0]: 30})
            return vals[j]

        self.diafiltrate_conc_mol_comp = Var(
            self.time,
            self.solutes,
            initialize=initialize_diafiltrate_conc_mol_comp,
            units=units.mol / units.m**3,  # mM
            bounds=[1e-11, None],
            doc="Mole concentration of solutes in the diafiltrate",
        )

        # add variables dependent on dimensionless_module_length
        self.volume_flux_water = Var(
            self.time,
            self.dimensionless_module_length,
            initialize=0.06,
            units=units.m**3 / units.m**2 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric water flux of water across the membrane",
        )

        def initialize_molar_ion_flux(m, t, w, j):
            vals = {
                self.config.cation_list[k]: 10
                for k in range(len(self.config.cation_list))
            }
            vals.update({self.config.anion_list[0]: 30})
            return vals[j]

        self.molar_ion_flux = Var(
            self.time,
            self.dimensionless_module_length,
            self.solutes,
            initialize=initialize_molar_ion_flux,
            units=units.mol / units.m**2 / units.h,
            bounds=[1e-11, None],
            doc="Mole flux of solutes across the membrane (z-direction, x-dependent)",
        )
        self.retentate_flow_volume = Var(
            self.time,
            self.dimensionless_module_length,
            initialize=6.75,
            units=units.m**3 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric flow rate of the retentate, x-dependent",
        )

        def initialize_retentate_conc_mol_comp(m, t, w, j):
            vals = {
                i: 0.95 * initialize_feed_conc_mol_comp(m, t, i) for i in self.solutes
            }
            return vals[j]

        self.retentate_conc_mol_comp = Var(
            self.time,
            self.dimensionless_module_length,
            self.solutes,
            initialize=initialize_retentate_conc_mol_comp,
            units=units.mol / units.m**3,  # mM
            bounds=[1e-11, None],
            doc="Mole concentration of solutes in the retentate, x-dependent",
        )
        self.permeate_flow_volume = Var(
            self.time,
            self.dimensionless_module_length,
            initialize=10,
            units=units.m**3 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric flow rate of the permeate, x-dependent",
        )

        def initialize_permeate_conc_mol_comp(m, t, w, j):
            vals = {
                i: 0.75 * initialize_feed_conc_mol_comp(m, t, i) for i in self.solutes
            }
            return vals[j]

        self.permeate_conc_mol_comp = Var(
            self.time,
            self.dimensionless_module_length,
            self.solutes,
            initialize=initialize_permeate_conc_mol_comp,
            units=units.mol / units.m**3,  # mM
            bounds=[1e-11, None],
            doc="Mole concentration of solutes in the permeate, x-dependent",
        )
        self.osmotic_pressure = Var(
            self.time,
            self.dimensionless_module_length,
            initialize=4,
            units=units.bar,
            bounds=[1e-11, None],
            doc="Osmostic pressure difference across the membrane",
        )

        # add variables dependent on dimensionless_module_length and dimensionless_membrane_thickness
        def initialize_membrane_conc_mol_comp(m, t, w, l, j):
            vals = {
                i: 0.1 * initialize_feed_conc_mol_comp(m, t, i) for i in self.solutes
            }
            return vals[j]

        self.membrane_conc_mol_comp = Var(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.solutes,
            initialize=initialize_membrane_conc_mol_comp,
            units=units.mol / units.m**3,  # mM
            bounds=[1e-11, None],
            doc="Mole concentration of solutes in the membrane, x- and z-dependent",
        )
        self.membrane_D_tilde = Var(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            initialize=620,
            units=(units.mm**2 / units.hr) * (units.mol / units.m**3),  # D * c
            doc="Denominator of diffusion and convection coefficients in membrane",
        )

        def initialize_membrane_cross_diffusion_coefficient_bilinear(m, t, w, l, j, k):
            vals = {
                self.config.cation_list[k]: {
                    self.config.cation_list[j]: -3000
                    for j in range(len(self.config.cation_list))
                }
                for k in range(len(self.config.cation_list))
            }
            return vals[j][k]

        self.membrane_cross_diffusion_coefficient_bilinear = Var(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            self.cations,
            initialize=initialize_membrane_cross_diffusion_coefficient_bilinear,
            units=(units.mm**2 / units.h)
            * (units.mm**2 / units.h * units.mol / units.m**3),  # D * D,tilde
            doc="Bi-linear cross diffusion coefficient for cations in membrane",
        )

        def initialize_membrane_convection_coefficient_bilinear(m, t, w, l, j):
            vals = {
                self.config.cation_list[k]: 100
                for k in range(len(self.config.cation_list))
            }
            return vals[j]

        self.membrane_convection_coefficient_bilinear = Var(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            initialize=initialize_membrane_convection_coefficient_bilinear,
            units=(units.mm**2 / units.hr) * (units.mol / units.m**3),  # D,tilde
            doc="Convection coefficient for cations in membrane",
        )

        def initialize_membrane_cross_diffusion_coefficient(m, t, w, l, j, k):
            vals = {
                self.config.cation_list[k]: {
                    self.config.cation_list[j]: -5
                    for j in range(len(self.config.cation_list))
                }
                for k in range(len(self.config.cation_list))
            }
            return vals[j][k]

        self.membrane_cross_diffusion_coefficient = Var(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            self.cations,
            initialize=initialize_membrane_cross_diffusion_coefficient,
            units=units.mm**2 / units.h,
            doc="Cross diffusion coefficient for cations in membrane",
        )

        def initialize_membrane_convection_coefficient(m, t, w, l, j):
            vals = {
                self.config.cation_list[k]: 0.2
                for k in range(len(self.config.cation_list))
            }
            return vals[j]

        self.membrane_convection_coefficient = Var(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            initialize=initialize_membrane_convection_coefficient,
            units=units.dimensionless,
            doc="Convection coefficient for cations in membrane",
        )

        # define the (partial) derivative variables
        self.d_retentate_conc_mol_comp_dx = DerivativeVar(
            self.retentate_conc_mol_comp,
            wrt=self.dimensionless_module_length,
            units=units.mol / units.m**3,  # mM
            doc="Solute concentration gradient in the retentate",
        )
        self.d_retentate_flow_volume_dx = DerivativeVar(
            self.retentate_flow_volume,
            wrt=self.dimensionless_module_length,
            units=units.m**3 / units.h,
            doc="Volume flow gradient in the retentate",
        )
        self.d_membrane_conc_mol_comp_dz = DerivativeVar(
            self.membrane_conc_mol_comp,
            wrt=self.dimensionless_membrane_thickness,
            units=units.mol / units.m**3,  # mM
            doc="Solute concentration gradient wrt membrane thickness",
        )

    def add_constraints(self):
        """
        Adds model constraints for the multi-component diafiltration unit model.
        """

        # mol balance constraints
        def _overall_mol_balance(blk, t, x):
            if x == 0:
                return Constraint.Skip
            return blk.d_retentate_flow_volume_dx[t, x] == (
                -blk.volume_flux_water[t, x]
                * blk.total_membrane_length
                * blk.total_module_length
            )

        self.overall_mol_balance = Constraint(
            self.time, self.dimensionless_module_length, rule=_overall_mol_balance
        )

        def _cation_mol_balance(blk, t, x, k):
            if x == 0:
                return Constraint.Skip
            return (
                blk.retentate_flow_volume[t, x]
                * blk.d_retentate_conc_mol_comp_dx[t, x, k]
            ) == (
                (
                    blk.volume_flux_water[t, x] * blk.retentate_conc_mol_comp[t, x, k]
                    - blk.molar_ion_flux[t, x, k]
                )
                * blk.total_membrane_length
                * blk.total_module_length
            )

        self.cation_mol_balance = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.cations,
            rule=_cation_mol_balance,
        )

        # bulk flux balance constraints
        def _overall_bulk_flux_equation(blk, t, x):
            if x == 0:
                return Constraint.Skip
            return (
                blk.permeate_flow_volume[t, x]
                == blk.volume_flux_water[t, x]
                * x
                * blk.total_membrane_length
                * blk.total_module_length
            )

        self.overall_bulk_flux_equation = Constraint(
            self.time,
            self.dimensionless_module_length,
            rule=_overall_bulk_flux_equation,
        )

        def _cation_bulk_flux_equation(blk, t, x, k):
            if x == 0:
                return Constraint.Skip
            return blk.molar_ion_flux[t, x, k] == (
                blk.permeate_conc_mol_comp[t, x, k] * blk.volume_flux_water[t, x]
            )

        self.cation_bulk_flux_equation = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.cations,
            rule=_cation_bulk_flux_equation,
        )

        # transport constraints (first principles)
        def _lumped_water_flux(blk, t, x):
            if x == 0:
                return Constraint.Skip
            return blk.volume_flux_water[t, x] == (
                blk.membrane_permeability
                * (blk.applied_pressure[t] - blk.osmotic_pressure[t, x])
            )

        self.lumped_water_flux = Constraint(
            self.time, self.dimensionless_module_length, rule=_lumped_water_flux
        )

        def _membrane_D_tilde_calculation(blk, t, x, z):
            if x == 0:
                return Constraint.Skip
            return blk.membrane_D_tilde[t, x, z] == (
                sum(
                    (
                        (
                            (
                                (blk.config.property_package.charge[k] ** 2)
                                * blk.config.property_package.diffusion_coefficient[k]
                            )
                            - (
                                blk.config.property_package.charge[k]
                                * blk.config.property_package.charge[
                                    self.config.anion_list[0]
                                ]
                                * blk.config.property_package.diffusion_coefficient[
                                    self.config.anion_list[0]
                                ]
                            )
                        )
                        * blk.membrane_conc_mol_comp[t, x, z, k]
                    )
                    for k in blk.cations
                )
                - (
                    blk.config.property_package.charge[self.config.anion_list[0]]
                    * blk.config.property_package.diffusion_coefficient[
                        self.config.anion_list[0]
                    ]
                    * blk.membrane_fixed_charge
                )
            )

        self.membrane_D_tilde_calculation = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_membrane_D_tilde_calculation,
        )

        def _membrane_cross_diffusion_coefficient_bilinear_calculation(
            blk, t, x, z, k, j
        ):
            if x == 0:
                return Constraint.Skip
            return (
                blk.membrane_cross_diffusion_coefficient_bilinear[t, x, z, k, j]
                == blk.membrane_cross_diffusion_coefficient[t, x, z, k, j]
                * blk.membrane_D_tilde[t, x, z]
            )

        self.membrane_cross_diffusion_coefficient_bilinear_calculation = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            self.cations,
            rule=_membrane_cross_diffusion_coefficient_bilinear_calculation,
        )

        def _membrane_convection_coefficient_bilinear_calculation(blk, t, x, z, k):
            if x == 0:
                return Constraint.Skip
            return (
                blk.membrane_convection_coefficient_bilinear[t, x, z, k]
                == blk.membrane_convection_coefficient[t, x, z, k]
                * blk.membrane_D_tilde[t, x, z]
            )

        self.membrane_convection_coefficient_bilinear_calculation = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            rule=_membrane_convection_coefficient_bilinear_calculation,
        )

        def _membrane_cross_diffusion_coefficient_calculation(blk, t, x, z, k, j):
            if x == 0:
                return Constraint.Skip
            # off-diagonal
            if k != j:
                return blk.membrane_cross_diffusion_coefficient_bilinear[
                    t, x, z, k, j
                ] == (
                    (
                        (
                            blk.config.property_package.charge[k]
                            * blk.config.property_package.charge[j]
                            * blk.config.property_package.diffusion_coefficient[k]
                            * blk.config.property_package.diffusion_coefficient[j]
                        )
                        - (
                            blk.config.property_package.charge[k]
                            * blk.config.property_package.charge[j]
                            * blk.config.property_package.diffusion_coefficient[k]
                            * blk.config.property_package.diffusion_coefficient[
                                self.config.anion_list[0]
                            ]
                        )
                    )
                    * blk.membrane_conc_mol_comp[t, x, z, k]
                )
            # diagonal
            if k == j:
                return blk.membrane_cross_diffusion_coefficient_bilinear[
                    t, x, z, k, j
                ] == (
                    sum(
                        (
                            (
                                (
                                    blk.config.property_package.charge[i]
                                    * blk.config.property_package.charge[
                                        self.config.anion_list[0]
                                    ]
                                    * blk.config.property_package.diffusion_coefficient[
                                        k
                                    ]
                                    * blk.config.property_package.diffusion_coefficient[
                                        self.config.anion_list[0]
                                    ]
                                )
                                - (
                                    blk.config.property_package.charge[i] ** 2
                                    * blk.config.property_package.diffusion_coefficient[
                                        i
                                    ]
                                    * blk.config.property_package.diffusion_coefficient[
                                        k
                                    ]
                                )
                            )
                            * blk.membrane_conc_mol_comp[t, x, z, i]
                        )
                        for i in blk.cations
                        if k != i
                    )
                    + sum(
                        (
                            (
                                (
                                    blk.config.property_package.charge[i]
                                    * blk.config.property_package.charge[
                                        self.config.anion_list[0]
                                    ]
                                    * blk.config.property_package.diffusion_coefficient[
                                        k
                                    ]
                                    * blk.config.property_package.diffusion_coefficient[
                                        self.config.anion_list[0]
                                    ]
                                )
                                - (
                                    blk.config.property_package.charge[i] ** 2
                                    * blk.config.property_package.diffusion_coefficient[
                                        i
                                    ]
                                    * blk.config.property_package.diffusion_coefficient[
                                        self.config.anion_list[0]
                                    ]
                                )
                            )
                            * blk.membrane_conc_mol_comp[t, x, z, i]
                        )
                        for i in blk.cations
                        if k == i
                    )
                    + blk.config.property_package.charge[self.config.anion_list[0]]
                    * blk.config.property_package.diffusion_coefficient[k]
                    * blk.config.property_package.diffusion_coefficient[
                        self.config.anion_list[0]
                    ]
                    * blk.membrane_fixed_charge
                )

        self.membrane_cross_diffusion_coefficient_calculation = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            self.cations,
            rule=_membrane_cross_diffusion_coefficient_calculation,
        )

        def _membrane_convection_coefficient_calculation(blk, t, x, z, k):
            if x == 0:
                return Constraint.Skip
            return blk.membrane_convection_coefficient_bilinear[t, x, z, k] == (
                blk.membrane_D_tilde[t, x, z]
                + (
                    blk.config.property_package.charge[k]
                    * blk.config.property_package.diffusion_coefficient[k]
                    * blk.membrane_fixed_charge
                )
            )

        self.membrane_convection_coefficient_calculation = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            rule=_membrane_convection_coefficient_calculation,
        )

        def _cation_flux_membrane(blk, t, x, z, k):
            if x == 0:
                return Constraint.Skip
            return blk.molar_ion_flux[t, x, k] == (
                (
                    blk.membrane_convection_coefficient[t, x, z, k]
                    * blk.membrane_conc_mol_comp[t, x, z, k]
                    * blk.volume_flux_water[t, x]
                )
                + sum(
                    (
                        units.convert(
                            blk.membrane_cross_diffusion_coefficient[t, x, z, k, i],
                            to_units=units.m**2 / units.h,
                        )
                        / blk.total_membrane_thickness
                        * blk.d_membrane_conc_mol_comp_dz[t, x, z, i]
                    )
                    for i in blk.cations
                )
            )

        self.cation_flux_membrane = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            self.cations,
            rule=_cation_flux_membrane,
        )

        def _anion_flux_membrane(blk, t, x):
            if x == 0:
                return Constraint.Skip
            return 0 == sum(
                blk.config.property_package.charge[j] * blk.molar_ion_flux[t, x, j]
                for j in blk.solutes
            )

        self.anion_flux_membrane = Constraint(
            self.time, self.dimensionless_module_length, rule=_anion_flux_membrane
        )

        # other physical constraints
        def _osmotic_pressure_calculation(blk, t, x):
            if x == 0:
                return Constraint.Skip
            return blk.osmotic_pressure[t, x] == units.convert(
                (
                    Constants.gas_constant  # J / mol / K
                    * blk.temperature
                    * sum(
                        (
                            blk.config.property_package.num_solutes[j]
                            * blk.config.property_package.sigma[j]
                            * (
                                blk.retentate_conc_mol_comp[t, x, j]
                                - blk.permeate_conc_mol_comp[t, x, j]
                            )
                        )
                        for j in blk.solutes
                    )
                ),
                to_units=units.bar,
            )

        self.osmotic_pressure_calculation = Constraint(
            self.time,
            self.dimensionless_module_length,
            rule=_osmotic_pressure_calculation,
        )

        def _electroneutrality_retentate(blk, t, x):
            return 0 == sum(
                blk.config.property_package.charge[j]
                * blk.retentate_conc_mol_comp[t, x, j]
                for j in blk.solutes
            )

        self.electroneutrality_retentate = Constraint(
            self.time,
            self.dimensionless_module_length,
            rule=_electroneutrality_retentate,
        )

        def _electroneutrality_membrane(blk, t, x, z):
            if x == 0:
                return Constraint.Skip
            return 0 == (
                sum(
                    blk.config.property_package.charge[j]
                    * blk.membrane_conc_mol_comp[t, x, z, j]
                    for j in blk.solutes
                )
                + blk.membrane_fixed_charge
            )

        self.electroneutrality_membrane = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_electroneutrality_membrane,
        )

        def _electroneutrality_permeate(blk, t, x):
            if x == 0:
                return Constraint.Skip
            return 0 == sum(
                blk.config.property_package.charge[j]
                * blk.permeate_conc_mol_comp[t, x, j]
                for j in blk.solutes
            )

        self.electroneutrality_permeate = Constraint(
            self.time,
            self.dimensionless_module_length,
            rule=_electroneutrality_permeate,
        )

        # partitioning equations
        def _cation_equilibrium_retentate_membrane_interface(blk, t, x, k):
            if x == 0:
                return Constraint.Skip
            return (
                (
                    blk.config.property_package.partition_coefficient_retentate[k]
                    ** (-blk.config.property_package.charge[self.config.anion_list[0]])
                )
                * (
                    blk.config.property_package.partition_coefficient_retentate[
                        self.config.anion_list[0]
                    ]
                    ** blk.config.property_package.charge[k]
                )
                * (
                    blk.retentate_conc_mol_comp[t, x, k]
                    ** (-blk.config.property_package.charge[self.config.anion_list[0]])
                )
                * (
                    blk.retentate_conc_mol_comp[t, x, self.config.anion_list[0]]
                    ** blk.config.property_package.charge[k]
                )
            ) == (
                (
                    blk.membrane_conc_mol_comp[t, x, 0, k]
                    ** (-blk.config.property_package.charge[self.config.anion_list[0]])
                )
                * (
                    blk.membrane_conc_mol_comp[t, x, 0, self.config.anion_list[0]]
                    ** blk.config.property_package.charge[k]
                )
            )

        self.cation_equilibrium_retentate_membrane_interface = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.cations,
            rule=_cation_equilibrium_retentate_membrane_interface,
        )

        def _cation_equilibrium_membrane_permeate_interface(blk, t, x, k):
            if x == 0:
                return Constraint.Skip
            return (
                (
                    blk.config.property_package.partition_coefficient_permeate[k]
                    ** (-blk.config.property_package.charge[self.config.anion_list[0]])
                )
                * (
                    blk.config.property_package.partition_coefficient_permeate[
                        self.config.anion_list[0]
                    ]
                    ** blk.config.property_package.charge[k]
                )
                * (
                    blk.permeate_conc_mol_comp[t, x, k]
                    ** (-blk.config.property_package.charge[self.config.anion_list[0]])
                )
                * (
                    blk.permeate_conc_mol_comp[t, x, self.config.anion_list[0]]
                    ** blk.config.property_package.charge[k]
                )
            ) == (
                (
                    blk.membrane_conc_mol_comp[t, x, 1, k]
                    ** (-blk.config.property_package.charge[self.config.anion_list[0]])
                )
                * (
                    blk.membrane_conc_mol_comp[t, x, 1, self.config.anion_list[0]]
                    ** blk.config.property_package.charge[k]
                )
            )

        self.cation_equilibrium_membrane_permeate_interface = Constraint(
            self.time,
            self.dimensionless_module_length,
            self.cations,
            rule=_cation_equilibrium_membrane_permeate_interface,
        )

        # boundary conditions
        def _retentate_flow_volume_boundary_condition(blk, t):
            return (
                blk.retentate_flow_volume[t, 0]
                == blk.feed_flow_volume[t] + blk.diafiltrate_flow_volume[t]
            )

        self.retentate_flow_volume_boundary_condition = Constraint(
            self.time, rule=_retentate_flow_volume_boundary_condition
        )

        def _retentate_conc_mol_comp_boundary_condition(blk, t, k):
            return blk.retentate_conc_mol_comp[t, 0, k] == (
                (
                    blk.feed_flow_volume[t] * blk.feed_conc_mol_comp[t, k]
                    + blk.diafiltrate_flow_volume[t]
                    * blk.diafiltrate_conc_mol_comp[t, k]
                )
                / (blk.feed_flow_volume[t] + blk.diafiltrate_flow_volume[t])
            )

        self.retentate_conc_mol_comp_boundary_condition = Constraint(
            self.time, self.cations, rule=_retentate_conc_mol_comp_boundary_condition
        )

        def _membrane_conc_mol_comp_boundary_condition(blk, t, z, k):
            return (
                blk.membrane_conc_mol_comp[t, 0, z, k]
                == self.numerical_zero_tolerance * units.mol / units.m**3
            )

        self.membrane_conc_mol_comp_boundary_condition = Constraint(
            self.time,
            self.dimensionless_membrane_thickness,
            self.cations,
            rule=_membrane_conc_mol_comp_boundary_condition,
        )

        # constraints to improve numerical stability
        def _permeate_flow_volume_boundary_condition(blk, t):
            return (
                blk.permeate_flow_volume[t, 0]
                == self.numerical_zero_tolerance * units.m**3 / units.h
            )

        self.permeate_flow_volume_boundary_condition = Constraint(
            self.time, rule=_permeate_flow_volume_boundary_condition
        )

        def _permeate_conc_mol_comp_boundary_condition(blk, t, j):
            return (
                blk.permeate_conc_mol_comp[t, 0, j]
                == self.numerical_zero_tolerance * units.mol / units.m**3
            )

        self.permeate_conc_mol_comp_boundary_condition = Constraint(
            self.time, self.solutes, rule=_permeate_conc_mol_comp_boundary_condition
        )

        def _d_retentate_flow_volume_dx_boundary_condition(blk, t):
            return (
                blk.d_retentate_flow_volume_dx[t, 0]
                == self.numerical_zero_tolerance * units.m**3 / units.h
            )

        self.d_retentate_flow_volume_dx_boundary_condition = Constraint(
            self.time, rule=_d_retentate_flow_volume_dx_boundary_condition
        )

        def _d_retentate_conc_mol_comp_dx_boundary_condition(blk, t, k):
            return (
                blk.d_retentate_conc_mol_comp_dx[t, 0, k]
                == self.numerical_zero_tolerance * units.mol / units.m**3
            )

        self.d_retentate_conc_mol_comp_dx_boundary_condition = Constraint(
            self.time,
            self.cations,
            rule=_d_retentate_conc_mol_comp_dx_boundary_condition,
        )

        def _volume_flux_water_boundary_condition(blk, t):
            return (
                blk.volume_flux_water[t, 0]
                == self.numerical_zero_tolerance * units.m / units.h
            )

        self.volume_flux_water_boundary_condition = Constraint(
            self.time, rule=_volume_flux_water_boundary_condition
        )

        def _molar_ion_flux_boundary_condition(blk, t, j):
            return (
                blk.molar_ion_flux[t, 0, j]
                == self.numerical_zero_tolerance * units.mol / units.m**2 / units.h
            )

        self.molar_ion_flux_boundary_condition = Constraint(
            self.time, self.solutes, rule=_molar_ion_flux_boundary_condition
        )

    def discretize_model(self):
        discretizer = TransformationFactory("dae.finite_difference")
        discretizer.apply_to(
            self,
            wrt=self.dimensionless_module_length,
            nfe=self.config.NFE_module_length,
            scheme="BACKWARD",
        )
        discretizer.apply_to(
            self,
            wrt=self.dimensionless_membrane_thickness,
            nfe=self.config.NFE_membrane_thickness,
            scheme="BACKWARD",
        )

    def deactivate_unnecessary_objects(self):
        """
        Deactivates variables and constraints not needed in the multi-component
        diafiltration unit model.
        """
        for t in self.time:
            for x in self.dimensionless_module_length:
                # anion concentration gradient in retentate variable is created by default but
                # is not needed in model; fix to reduce number of variables
                self.d_retentate_conc_mol_comp_dx[t, x, self.config.anion_list[0]].fix(
                    value(self.numerical_zero_tolerance)
                )
                # associated discretization equation not needed in model
                if x != 0:
                    self.d_retentate_conc_mol_comp_dx_disc_eq[
                        t, x, self.config.anion_list[0]
                    ].deactivate()

                for z in self.dimensionless_membrane_thickness:
                    # anion concentration gradient in membrane variable is created by default but
                    # is not needed in model; fix to reduce number of variables
                    self.d_membrane_conc_mol_comp_dz[
                        t, x, z, self.config.anion_list[0]
                    ].fix(value(self.numerical_zero_tolerance))
                    # associated discretization equation not needed in model
                    if z != 0:
                        self.d_membrane_conc_mol_comp_dz_disc_eq[
                            t, x, z, self.config.anion_list[0]
                        ].deactivate()

    def add_scaling_factors(self):
        """
        Assigns scaling factors to certain variables and constraints to
        improve solver performance.
        """
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        self.scaling_factor[self.volume_flux_water] = 1e2
        self.scaling_factor[self.membrane_D_tilde] = 1e-1
        self.scaling_factor[self.membrane_cross_diffusion_coefficient_bilinear] = 1e-2
        self.scaling_factor[self.membrane_convection_coefficient_bilinear] = 1e-1
        self.scaling_factor[self.membrane_cross_diffusion_coefficient] = 1e1
        self.scaling_factor[self.membrane_convection_coefficient] = 1e1

        if len(self.config.cation_list) >= 2:
            for t in self.time:
                for x in self.dimensionless_module_length:
                    if x != 0:
                        self.scaling_factor[self.lumped_water_flux[t, x]] = 1e3

    def add_ports(self):
        self.feed_inlet = Port(doc="Feed Inlet Port")
        self._feed_flow_volume_ref = Reference(self.feed_flow_volume)
        self.feed_inlet.add(self._feed_flow_volume_ref, "flow_vol")
        self._feed_conc_mol_comp_ref = Reference(self.feed_conc_mol_comp)
        self.feed_inlet.add(self._feed_conc_mol_comp_ref, "conc_mol_comp")

        self.diafiltrate_inlet = Port(doc="Diafiltrate Inlet Port")
        self._diafiltrate_flow_volume_ref = Reference(self.diafiltrate_flow_volume)
        self.diafiltrate_inlet.add(self._diafiltrate_flow_volume_ref, "flow_vol")
        self._diafiltrate_conc_mol_comp_ref = Reference(self.diafiltrate_conc_mol_comp)
        self.diafiltrate_inlet.add(self._diafiltrate_conc_mol_comp_ref, "conc_mol_comp")

        self.retentate_outlet = Port(doc="Retentate Outlet Port")
        self._retentate_flow_volume_ref = Reference(
            self.retentate_flow_volume[:, self.dimensionless_module_length.last()]
        )
        self.retentate_outlet.add(self._retentate_flow_volume_ref, "flow_vol")
        self._retentate_conc_mol_comp_ref = Reference(
            self.retentate_conc_mol_comp[:, self.dimensionless_module_length.last(), :]
        )
        self.retentate_outlet.add(self._retentate_conc_mol_comp_ref, "conc_mol_comp")

        self.permeate_outlet = Port(doc="Permeate Outlet Port")
        self._permeate_flow_volume_ref = Reference(
            self.permeate_flow_volume[:, self.dimensionless_module_length.last()]
        )
        self.permeate_outlet.add(self._permeate_flow_volume_ref, "flow_vol")
        self._permeate_conc_mol_comp_ref = Reference(
            self.permeate_conc_mol_comp[:, self.dimensionless_module_length.last(), :]
        )
        self.permeate_outlet.add(self._permeate_conc_mol_comp_ref, "conc_mol_comp")
