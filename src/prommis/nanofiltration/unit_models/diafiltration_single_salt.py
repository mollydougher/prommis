#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
r"""
Single-Salt Diafiltration Unit Model
=================================

Author: Molly Dougher

This membrane unit model is for the multi-component diafiltration of a single-salt system. The membrane is designed for use in a diafiltration cascade, i.e., the model represents one spiral-wound membrane module.

Configuration Arguments
-----------------------

The Single-Salt Diafiltration unit model requires a property package that provides the valency (:math:`z_i`), reflection coefficient (:math:`\sigma_i`), partition coefficient (:math:`H_i`), and number of dissolved species (:math:`n_i`) for each ion :math:`i` in solution. When used in a flowsheet, the user can provide separate property packages for the feed and product streams.

There are two required arguments, ``NFE_module_length`` and ``NFE_membrane_thickness``, to specify the desired number of finite elements across the width and thickness of the membrane, respectively. There is one optional argument, ``charged_membrane``, which is a Boolean (default = ``True``) to specify if the membrane has a fixed surface charge.

Degrees of Freedom
------------------

The Single-Salt Diafiltration unit model has three degrees of freedom (variable names and default values are provided in parentheses):

#. the length of the membrane module (``total_module_length``; :math:`4 \, \mathrm{m}`)
#. the length of the membrane (``total_membrane_length``; :math:`40 \, \mathrm{m}`)
#. the pressure applied to the membrane system (``applied_pressure``; :math:`15 \, \mathrm{bar}`)

To run a simulation (with zero degrees of freedom) in a flowsheet, the following variables must be fixed to obtain zero degrees of freedom (variable names and default values are provided in parentheses):

#. the volumetric flow rate of the feed (``feed_flow_volume``; :math:`100 \, \mathrm{m}^3 \, \mathrm{h}^{-1}`)
#. the lithium concentration in the feed (``feed_conc_mol_comp[t,"Li"]``; :math:`245 \, \mathrm{mol} \, \mathrm{m}^{-3}`)
#. the cobalt concentration in the feed (``feed_conc_mol_comp[t,"Co"]``; :math:`288 \, \mathrm{mol} \, \mathrm{m}^{-3}`)
#. the volumetric flow rate of the diafiltrate (``diafiltrate_flow_volume``; :math:`30 \, \mathrm{m}^3 \, \mathrm{h}^{-1}`)
#. the lithium concentration in the diafiltrate (``diafiltrate_conc_mol_comp[t,"Li"]``; :math:`14 \, \mathrm{mol} \, \mathrm{m}^{-3}`)
#. the cobalt concentration in the diafiltrate (``diafiltrate_conc_mol_comp[t,"Co"]``; :math:`3 \, \mathrm{mol} \, \mathrm{m}^{-3}`)

Model Structure
---------------

There are three phases in the Single-Salt Diafiltration model: the retentate, the membrane, and the permeate. The retentate and the permeate are only discretized with respect to :math:`x` (parallel to the membrane surface), while the membrane is discretized with respect to both :math:`x` and :math:`z` (perpendicular to the membrane surface). The resulting system of partial differential algebraic equations is solved by discretizing with the backward finite difference method.

Assumptions
-----------

The partition coefficients, which describe how the solutes partition across the solution-membrane interfaces, are derived considering steric and electrostatic hindrance.

The membrane has a negative surface charge (:math:`-140 \, \mathrm{mM}`), which was calculated using zeta potential measurements for NF270 membranes. (See `this reference <https://doi.org/10.1021/acs.iecr.4c04763>`_).

The formation of a boundary layer at the membrane surface due to concentration polarization is neglected for mathematical simplicity.

The dominating transport mechanism within the bulk/retentate solution is convection in the :math:`x`-direction (parallel to the membrane surface). The dominating transport mechanism within the permeate solution is convection in the :math:`z`-direction (perpendicular to the membrane surface).

The transport mechanisms modeled within the membrane are convection, diffusion, and electromigration. Diffusion within the membrane that is normal to the pore walls is ignored, meaning the concentration gradient of ion :math:`i` within the membrane only has a :math:`z`-component (perpendicular to the membrane surface).

Sets
----

The Single-Salt Diafiltration model defines the following discrete sets for ions in the system.

.. math:: \mathcal{I}=\{\mathrm{Li^+,Co^{2+},Cl^-}\}

There are 2 continuous sets for each length dimension: ``dimensionless_module_length`` (in the :math:`x`-direction parallel to the membrane surface) and ``dimensionless_membrane_thickness`` (in the :math:`z`-direction perpendicular to the membrane surface). :math:`x` and :math:`z` are non-dimensionalized (denoted as :math:`\bar{x}` and :math:`\bar{z}`, respectively) using the module length or (:math:`w`) and membrane thickness (:math:`l`), respectively, to improve numerics.

.. math:: \bar{x} \in \mathbb{R} \| 0 \leq \bar{x} \leq 1
.. math:: \bar{z} \in \mathbb{R} \| 0 \leq \bar{z} \leq 1

Some variables have a time domain to be compatible with the property package, even though this is not a dynamic model. Thus, the following set is defined for time.

.. math:: t \in [0]

Default Model Parameters
------------------------

The Single-Salt Diafiltration model has the following parameters.

================ =============================================== ============================ ============= ==========================================================
Parameter        Description                                     Name                         Default Value Units
================ =============================================== ============================ ============= ==========================================================
:math:`\epsilon` numerical tolerance for zero values             ``numerical_zero_tolerance`` 1e-10
:math:`l`        thickness of the membrane                       ``total_membrane_thickness`` 1e-07         :math:`\mathrm{m}`
:math:`L_p`      hydraulic permeability of the membrane          ``membrane_permeability``    0.03          :math:`\mathrm{m} \, \mathrm{h}^{-1} \, \mathrm{bar}^{-1}`
:math:`T`        temperature of the system                       ``temperature``              298           :math:`\mathrm{K}`
:math:`\chi`     concentration of surface charge on the membrane ``membrane_fixed_charge``    -140          :math:`\mathrm{mol} \, \mathrm{m}^{-3}`
================ =============================================== ============================ ============= ==========================================================

Variables
---------

The Single-Salt Diafiltration model adds the following variables.

==================================== =============================================== ====================================================== ========================================================== ====================================================
Variable                             Description                                     Name                                                   Units                                                      Notes
==================================== =============================================== ====================================================== ========================================================== ====================================================
:math:`c_{\mathrm{Co^{2+}},d}`       concentration of cobalt ion in the diafiltrate  ``diafiltrate_conc_mol_comp[t, "Co"]``                 :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t`
:math:`c_{\mathrm{Cl^-},d}`          concentration of chloride in the diafiltrate    ``diafiltrate_conc_mol_comp[t, "Cl"]``                 :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t`
:math:`c_{\mathrm{Li^+},d}`          concentration of lithium ion in the diafiltrate ``diafiltrate_conc_mol_comp[t, "Li"]``                 :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t`
:math:`c_{\mathrm{Co^{2+}},f}`       concentration of cobalt ion in the feed         ``feed_conc_mol_comp[t, "Co"]``                        :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t`
:math:`c_{\mathrm{Cl^-},f}`          concentration of chloride in the feed           ``feed_conc_mol_comp[t, "Cl"]``                        :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t`
:math:`c_{\mathrm{Li^+},f}`          concentration of lithium ion in the feed        ``feed_conc_mol_comp[t, "Li"]``                        :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t`
:math:`c_{\mathrm{Cl^-},m}`          concentration of chloride in the membrane       ``membrane_conc_mol_chloride``                         :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`c_{\mathrm{Co^{2+}},m}`       concentration of cobalt ion in the membrane     ``membrane_conc_mol_cobalt``                           :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`c_{\mathrm{Li^+},m}`          concentration of lithium ion in the membrane    ``membrane_conc_mol_lithium``                          :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`c_{\mathrm{Cl^-},p}`          concentration of chloride in the membrane       ``permeate_conc_mol_comp[t, "Cl", :math:`\bar{x}`]``   :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t` and :math:`\bar{x}`
:math:`c_{\mathrm{Co^{2+}},p}`       concentration of cobalt ion in the permeate     ``permeate_conc_mol_comp[t, "Co", :math:`\bar{x}`]``   :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t` and :math:`\bar{x}`
:math:`c_{\mathrm{Li^+},p}`          concentration of lithium ion in the permeate    ``permeate_conc_mol_comp[t, "Li", :math:`\bar{x}`]``   :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t` and :math:`\bar{x}`
:math:`c_{\mathrm{Cl^-},r}`          concentration of chloride in the retentate      ``retentate_conc_mol_comp[t, "Cl", :math:`\bar{x}`]``  :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t` and :math:`\bar{x}`
:math:`c_{\mathrm{Co^{2+}},r}`       concentration of cobalt ion in the retentate    ``retentate_conc_mol_comp[t, "Co", :math:`\bar{x}`]``  :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t` and :math:`\bar{x}`
:math:`c_{\mathrm{Li^+},r}`          concentration of lithium ion in the retentate   ``retentate_conc_mol_comp[t, "Li", :math:`\bar{x}`]``  :math:`\mathrm{mol} \, \mathrm{m}^{-3}`                    discretized over :math:`t` and :math:`\bar{x}`
:math:`\alpha_{Li^+}`                linearized convection coefficient (Li)          ``convection_coefficient_lithium``                     :math:`\mathrm{dimensionless}`                             discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`\alpha_{Co^{2+}}`             linearized convection coefficient (Co)          ``convection_coefficient_cobalt``                      :math:`\mathrm{dimensionless}`                             discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`D_{\mathrm{Li^+,Li^+}}`       linearized cross-diffusion coefficient (Li,Li)  ``D_lithium_lithium``                                  :math:`\mathrm{m}^2 \, \mathrm{h}^{-1}`                    discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`D_{\mathrm{Li^+,Co^{2+}}}`    linearized cross-diffusion coefficient (Li,Co)  ``D_lithium_cobalt``                                   :math:`\mathrm{m}^2 \, \mathrm{h}^{-1}`                    discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`D_{\mathrm{Co^{2+},Li^+}}`    linearized cross-diffusion coefficient (Co,Li)  ``D_cobalt_lithium``                                   :math:`\mathrm{m}^2 \, \mathrm{h}^{-1}`                    discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`D_{\mathrm{Co^{2+},Co^{2+}}}` linearized cross-diffusion coefficient (Co,Co)  ``D_cobalt_cobalt``                                    :math:`\mathrm{m}^2 \, \mathrm{h}^{-1}`                    discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`j_{\mathrm{Cl^-}}`            molar flux of chloride across the membrane      ``mol_flux_chloride``                                  :math:`\mathrm{mol} \, \mathrm{m}^{-2} \, \mathrm{h}^{-1}` discretized over :math:`\bar{x}`
:math:`j_{\mathrm{Co^{2+}}}`         molar flux of cobalt ion across the membrane    ``mol_flux_cobalt``                                    :math:`\mathrm{mol} \, \mathrm{m}^{-2} \, \mathrm{h}^{-1}` discretized over :math:`\bar{x}`
:math:`j_{\mathrm{Li^+}}`            molar flux of lithium ion across the membrane   ``mol_flux_lithium``                                   :math:`\mathrm{mol} \, \mathrm{m}^{-2} \, \mathrm{h}^{-1}` discretized over :math:`\bar{x}`
:math:`J_w`                          water flux across the membrane                  ``volume_flux_water``                                  :math:`\mathrm{m}^3 \, \mathrm{m}^{-2} \, \mathrm{h}^{-1}` discretized over :math:`\bar{x}`
:math:`L`                            length of the membrane                          ``total_membrane_length``                              :math:`\mathrm{m}`
:math:`\Delta \pi`                   osmotic pressure of feed-side fluid             ``osmotic_pressure``                                   :math:`\mathrm{bar}`                                       discretized over :math:`\bar{x}`
:math:`\Delta P`                     applied pressure to the membrane                ``applied_pressure``                                   :math:`\mathrm{bar}`
:math:`q_d`                          volumetric flow rate of the diafiltrate         ``diafiltrate_flow_volume``                            :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}`                    discretized over :math:`t`
:math:`q_f`                          volumetric flow rate of the feed                ``feed_flow_volume``                                   :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}`                    discretized over :math:`t`
:math:`q_p`                          volumetric flow rate of the permeate            ``permeate_flow_volume``                               :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}`                    discretized over :math:`t` and :math:`\bar{x}`
:math:`q_r`                          volumetric flow rate of the retentate           ``retentate_flow_volume``                              :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}`                    discretized over :math:`t` and :math:`\bar{x}`
:math:`w`                            length of the membrane module                   ``total_module_length``                                :math:`\mathrm{m}`
==================================== =============================================== ====================================================== ========================================================== ====================================================

Derivative Variables
--------------------

The Single-Salt Diafiltration model adds the following derivative variables.

================================================================== =============================================== =========================================================== ======================================= ====================================================
Variable                                                           Description                                     Name                                                        Units                                   Notes
================================================================== =============================================== =========================================================== ======================================= ====================================================
:math:`\frac{\mathrm{d}c_{\mathrm{Co^{2+}},r}}{\mathrm{d}\bar{x}}` cobalt ion concentration gradient in retentate  ``d_retentate_conc_mass_comp_dx[t, "Co", :math:`\bar{x}`]`` :math:`\mathrm{kg} \, \mathrm{m}^{-3}`  discretized over :math:`t` and :math:`\bar{x}`
:math:`\frac{\mathrm{d}c_{\mathrm{Li^+},r}}{\mathrm{d}\bar{x}}`    lithium ion concentration gradient in retentate ``d_retentate_conc_mass_comp_dx[t, "Li", :math:`\bar{x}`]`` :math:`\mathrm{kg} \, \mathrm{m}^{-3}`  discretized over :math:`t` and :math:`\bar{x}`
:math:`\frac{\mathrm{d}q_r}{\mathrm{d}\bar{x}}`                    retentate flow rate gradient                    ``d_retentate_flow_volume_dx``                              :math:`\mathrm{m}^3 \, \mathrm{h}^{-1}` discretized over :math:`t` and :math:`\bar{x}`
:math:`\frac{\partial c_{\mathrm{Co^{2+}},m}}{\partial \bar{z}}`   cobalt ion concentration gradient in membrane   ``d_membrane_conc_mass_cobalt_dz``                          :math:`\mathrm{kg} \, \mathrm{m}^{-3}`  discretized over :math:`\bar{x}` and :math:`\bar{z}`
:math:`\frac{\partial c_{\mathrm{Li^+},m}}{\partial \bar{z}}`      lithium ion concentration gradient in membrane  ``d_membrane_conc_mass_lithium_dz``                         :math:`\mathrm{kg} \, \mathrm{m}^{-3}`  discretized over :math:`\bar{x}` and :math:`\bar{z}`
================================================================== =============================================== =========================================================== ======================================= ====================================================

Constraints
-----------

Differential mass balances in the retentate:

.. math:: \frac{\mathrm{d}q_r(\bar{x})}{\mathrm{d}\bar{x}} = - J_w(\bar{x}) wL  \qquad \forall \, \bar{x} \neq 0
.. math:: q_r(\bar{x}) \frac{\mathrm{d}c_{\mathrm{Li^+},r}(\bar{x})}{\mathrm{d}\bar{x}} = wL (J_w(\bar{x}) c_{\mathrm{Li^+},r}(\bar{x}) - j_{\mathrm{Li^+}}(\bar{x}))  \qquad \forall \, \bar{x} \neq 0
.. math:: q_r(\bar{x}) \frac{\mathrm{d}c_{\mathrm{Co^{2+}},r}(\bar{x})}{\mathrm{d}\bar{x}} = wL (J_w(\bar{x}) c_{\mathrm{Co^{2+}},r}(\bar{x}) - j_{\mathrm{Co^{2+}}}(\bar{x}))  \qquad \forall \, \bar{x} \neq 0

Electroneutrality in the retentate:

.. math:: 0 = z_{\mathrm{Li^+}} c_{\mathrm{Li^+},r}(\bar{x}) + z_{\mathrm{Co^{2+}}} c_{\mathrm{Co^{2+}},r}(\bar{x}) + z_{\mathrm{Cl^-}} c_{\mathrm{Cl^-},r}(\bar{x})

Overall water flux through the membrane:

.. math:: J_w (\bar{x}) = L_p (\Delta P - \Delta \pi (\bar{x})) \qquad \forall \, \bar{x} \neq 0

Osmotic pressure:

.. math:: \Delta \pi (\bar{x}) = \mathrm{R} \mathrm{T} \sum_{i \in \mathcal{I}} n_i \sigma_i (c_{i,r}(\bar{x})-c_{i,p}(\bar{x}))

Nernst-Plank equations for the ion flux through the membrane:

.. math:: j_{\mathrm{Li^+}}(\bar{x}) = \alpha_{Li^+}(\bar{x},\bar{z}) c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) J_w(\bar{x}) + \frac{D_{\mathrm{Li^+,Li^+}}(\bar{x},\bar{z})}{l} \frac{\partial c_{\mathrm{Li^+},m}(\bar{x},\bar{z})}{\partial \bar{z}} + \frac{D_{\mathrm{Li^+,Co^{2+}}}(\bar{x},\bar{z})}{l} \frac{\partial c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})}{\partial \bar{z}} \qquad \forall \, \bar{z} \neq 0
.. math:: j_{\mathrm{Co^{2+}}}(\bar{x}) = \alpha_{Li^+}(\bar{x},\bar{z}) c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z}) J_w(\bar{x}) + \frac{D_{\mathrm{Co^{2+},Li^+}}(\bar{x},\bar{z})}{l} \frac{\partial c_{\mathrm{Li^+},m}(\bar{x},\bar{z})}{\partial \bar{z}} + \frac{D_{\mathrm{Co^{2+},Co^{2+}}}(\bar{x},\bar{z})}{l} \frac{\partial c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})}{\partial \bar{z}} \qquad \forall \, \bar{z} \neq 0

with the convection and cross-diffusion coefficients:

.. math:: \tilde{D}(\bar{x},\bar{z}) = z_{\mathrm{Li^+}}(z_{\mathrm{Li^+}} D_{\mathrm{Li^+}} - z_{\mathrm{Cl^-}} D_{\mathrm{Cl^-}})c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + z_{\mathrm{Co^{2+}}}(z_{\mathrm{Co^{2+}}} D_{\mathrm{Co^{2+}}} - z_{\mathrm{Cl^-}} D_{\mathrm{Cl^-}})c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z}) - z_{\mathrm{Cl^-}} D_{\mathrm{Cl^-}} \chi
.. math:: \alpha_{\mathrm{Li^+}}(\bar{x},\bar{z}) = 1 + \frac{z_{\mathrm{Li^+}} D_{\mathrm{Li^+}} \chi}{\tilde{D}(\bar{x},\bar{z})}
.. math:: \alpha_{\mathrm{Co^{2+}}}(\bar{x},\bar{z}) = 1 + \frac{z_{\mathrm{Co^{2+}}} D_{\mathrm{Co^{2+}}} \chi}{\tilde{D}(\bar{x},\bar{z})}
.. math:: D_{\mathrm{Li^+,Li^+}}(\bar{x},\bar{z}) = \frac{z_{\mathrm{Li^+}} D_{\mathrm{Li^+}} D_{\mathrm{Cl^-}}(z_{\mathrm{Cl^-}} - z_{\mathrm{Li^+}})c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + z_{\mathrm{Co^{2+}}} D_{\mathrm{Li^+}}(z_{\mathrm{Cl^-}} D_{\mathrm{Cl^-}} - z_{\mathrm{Co^{2+}}} D_{\mathrm{Co^{2+}}})c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z}) + z_{\mathrm{Cl^-}} D_{\mathrm{Li^+}} D_{\mathrm{Cl^-}} \chi}{\tilde{D}(\bar{x},\bar{z})}
.. math:: D_{\mathrm{Li^+,Co^{2+}}}(\bar{x},\bar{z}) = \frac{z_{\mathrm{Li^+}} z_{\mathrm{Co^{2+}}} D_{\mathrm{Li^+}}(D_{\mathrm{Co^{2+}}} - D_{\mathrm{Cl^-}})c_{\mathrm{Li^+},m}(\bar{x},\bar{z})}{\tilde{D}(\bar{x},\bar{z})}
.. math:: D_{\mathrm{Co^{2+},Li^+}}(\bar{x},\bar{z}) = \frac{z_{\mathrm{Li^+}} z_{\mathrm{Co^{2+}}} D_{\mathrm{Co^{2+}}}(D_{\mathrm{Li^+}} - D_{\mathrm{Cl^-}})c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})}{\tilde{D}(\bar{x},\bar{z})}
.. math:: D_{\mathrm{Co^{2+},Co^{2+}}}(\bar{x},\bar{z}) = \frac{z_{\mathrm{Li^+}} D_{\mathrm{Co^{2+}}} (z_{\mathrm{Cl^-}} D_{\mathrm{Cl^-}} - z_{\mathrm{Li^+}} D_{\mathrm{Li^+}})c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + z_{\mathrm{Co^{2+}}} D_{\mathrm{Co^{2+}}} D_{\mathrm{Cl^-}} (z_{\mathrm{Cl^-}} - z_{\mathrm{Co^{2+}}})c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z}) + z_{\mathrm{Cl^-}} D_{\mathrm{Co^{2+}}} D_{\mathrm{Cl^-}} \chi}{\tilde{D}(\bar{x},\bar{z})}

For numerical stability, the convection and cross-diffusion coefficients are linearized:

.. math:: \alpha_{Li^+}(\bar{x},\bar{z}) = \omega_0 + \omega_1 c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + \omega_2 c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})
.. math:: \alpha_{Co^{2+}}(\bar{x},\bar{z}) = \omega_3 + \omega_4 c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + \omega_5 c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})

.. math:: D_{\mathrm{Li^+,Li^+}}(\bar{x},\bar{z}) = \beta_0 + \beta_1 c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + \beta_2 c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})
.. math:: D_{\mathrm{Li^+,Co^{2+}}}(\bar{x},\bar{z}) = \beta_3 + \beta_4 c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + \beta_5 c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})
.. math:: D_{\mathrm{Co^{2+},Li^+}}(\bar{x},\bar{z}) = \beta_6 + \beta_7 c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + \beta_8 c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})
.. math:: D_{\mathrm{Co^{2+},Co^{2+}}}(\bar{x},\bar{z}) = \beta_9 + \beta_{10} c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + \beta_{11} c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z})

and the the following regressed parameter values are obtained when :math:`\chi = 0 \, \mathrm{mM}` (valid for membrane lithium and cobalt concentration range of :math:`50-80 \, \mathrm{mM}`) and when :math:`\chi = -140 \, \mathrm{mM}` (valid for membrane lithium concentration range of :math:`50-80 \, \mathrm{mM}` and membrane cobalt concentration range of :math:`80-110 \, \mathrm{mM}`):

================== =========================== ============================== ============================================================
Parameter          Value (:math:`\chi = 0 mM`) Value (:math:`\chi = -140 mM`) Units
================== =========================== ============================== ============================================================
:math:`\omega_0`   :math:`1`                   :math:`0.365`                  :math:`\mathrm{dimensionless}`
:math:`\omega_1`   :math:`0`                   :math:`0.00137`                :math:`\mathrm{mol} \, \mathrm{m}^{-3}`
:math:`\omega_2`   :math:`0`                   :math:`0.00309`                :math:`\mathrm{mol} \, \mathrm{m}^{-3}`
:math:`\omega_3`   :math:`1`                   :math:`0.0945`                 :math:`\mathrm{dimensionless}`
:math:`\omega_4`   :math:`0`                   :math:`0.00195`                :math:`\mathrm{mol} \, \mathrm{m}^{-3}`
:math:`\omega_5`   :math:`0`                   :math:`0.00441`                :math:`\mathrm{mol} \, \mathrm{m}^{-3}`
:math:`\beta_0`    :math:`-4.07e-06`           :math:`-4.33e-06`              :math:`\mathrm{m}^2 \, \mathrm{h}^{-1}`
:math:`\beta_1`    :math:`-3.96e-09`           :math:`-4.21e-09`              :math:`\mathrm{m}^5 \, \mathrm{mol}^{-1} \, \mathrm{h}^{-1}`
:math:`\beta_2`    :math:`3.98e-09`            :math:`5.10e-09`               :math:`\mathrm{m}^5 \, \mathrm{mol}^{-1} \, \mathrm{h}^{-1}`
:math:`\beta_3`    :math:`-9.63e-07`           :math:`-1.63e-06`              :math:`\mathrm{m}^2 \, \mathrm{h}^{-1}`
:math:`\beta_4`    :math:`-1.03e-08`           :math:`-1.09e-08`              :math:`\mathrm{m}^5 \, \mathrm{mol}^{-1} \, \mathrm{h}^{-1}`
:math:`\beta_5`    :math:`1.03e-08`            :math:`1.32e-08`               :math:`\mathrm{m}^5 \, \mathrm{mol}^{-1} \, \mathrm{h}^{-1}`
:math:`\beta_6`    :math:`-5.23e-07`           :math:`-1.31e-06`              :math:`\mathrm{m}^2 \, \mathrm{h}^{-1}`
:math:`\beta_7`    :math:`2.47e-09`            :math:`4.67e-09`               :math:`\mathrm{m}^5 \, \mathrm{mol}^{-1} \, \mathrm{h}^{-1}`
:math:`\beta_8`    :math:`-2.49e-09`           :math:`1.43e-09`               :math:`\mathrm{m}^5 \, \mathrm{mol}^{-1} \, \mathrm{h}^{-1}`
:math:`\beta_9`    :math:`-4.00e-06`           :math:`-6.03e-06`              :math:`\mathrm{m}^2 \, \mathrm{h}^{-1}`
:math:`\beta_{10}` :math:`6.40e-09`            :math:`1.21e-08`               :math:`\mathrm{m}^5 \, \mathrm{mol}^{-1} \, \mathrm{h}^{-1}`
:math:`\beta_{11}` :math:`-6.44e-09`           :math:`3.69e-09`               :math:`\mathrm{m}^5 \, \mathrm{mol}^{-1} \, \mathrm{h}^{-1}`
================== =========================== ============================== ============================================================

No applied potential on the system:

.. math:: 0 = z_{\mathrm{Li^+}} j_{\mathrm{Li^+}}(\bar{x}) + z_{\mathrm{Co^{2+}}} j_{\mathrm{Co^{2+}}}(\bar{x}) + z_{\mathrm{Cl^-}} j_{\mathrm{Cl^-}}(\bar{x}) \qquad \forall \, \bar{x} \neq 0

Electroneutrality in the membrane:

.. math:: 0 = z_{\mathrm{Li^+}} c_{\mathrm{Li^+},m}(\bar{x},\bar{z}) + z_{\mathrm{Co^{2+}}} c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z}) + z_{\mathrm{Cl^-}} c_{\mathrm{Cl^-},m}(\bar{x},\bar{z}) + \chi \qquad \forall \, \bar{z} \neq 0

Mass balance (via convection) on the permeate outlet:

.. math:: q_p(\bar{x}) = \bar{x} wL J_w(\bar{x}) \qquad \forall \, \bar{x} \neq 0
.. math:: j_{\mathrm{Li^+}}(\bar{x}) = c_{\mathrm{Li^+},p}(\bar{x}) J_w(\bar{x}) \qquad \forall \, \bar{x} \neq 0
.. math:: j_{\mathrm{Co^{2+}}}(\bar{x}) = c_{\mathrm{Co^{2+}},p}(\bar{x}) J_w(\bar{x}) \qquad \forall \, \bar{x} \neq 0

Electroneutrality in the retentate:

.. math:: 0 = z_{\mathrm{Li^+}} c_{\mathrm{Li^+},p}(\bar{x}) + z_{\mathrm{Co^{2+}}} c_{\mathrm{Co^{2+}},p}(\bar{x}) + z_{\mathrm{Cl^-}} c_{\mathrm{Cl^-},p}(\bar{x})

Partitioning at the retentate-membrane interface:

.. math:: H_{Li^+} H_{Cl^-} c_{\mathrm{Li^+},r}(\bar{x}) c_{\mathrm{Cl^-},r}(\bar{x}) = c_{\mathrm{Li^+},m}(\bar{x},\bar{z}=0) c_{\mathrm{Cl^-},m}(\bar{x},\bar{z}=0) \qquad \forall \, \bar{x} \neq 0
.. math:: H_{Co^{2+}} H_{Cl^-}^{z_{Co^{2+}}} c_{\mathrm{Co^{2+}},r}(\bar{x}) c_{\mathrm{Cl^-},r}(\bar{x})^{z_{Co^{2+}}} =c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z}=0) c_{\mathrm{Cl^-},m}(\bar{x},\bar{z}=0)^{z_{Co^{2+}}} \qquad \forall \, \bar{x} \neq 0

Partitioning at the membrane-permeate interface:

.. math:: H_{Li^+} H_{Cl^-} c_{\mathrm{Li^+},p}(\bar{x}) c_{\mathrm{Cl^-},p}(\bar{x}) = c_{\mathrm{Li^+},m}(\bar{x},\bar{z}=1) c_{\mathrm{Cl^-},m}(\bar{x},\bar{z}=1) \qquad \forall \, \bar{x} \neq 0
.. math:: H_{Co^{2+}} H_{Cl^-}^{z_{Co^{2+}}} c_{\mathrm{Co^{2+}},p}(\bar{x}) c_{\mathrm{Cl^-},p}(\bar{x})^{z_{Co^{2+}}} =c_{\mathrm{Co^{2+}},m}(\bar{x},\bar{z}=1) c_{\mathrm{Cl^-},m}(\bar{x},\bar{z}=1)^{z_{Co^{2+}}} \qquad \forall \, \bar{x} \neq 0

The following initial conditions are fixed to complete the model:

.. math:: q_r(\bar{x}=0) = q_f + q_d
.. math:: c_{\mathrm{Li^+},r}(\bar{x}=0) = \frac{q_f c_{\mathrm{Li^+},f} + q_d c_{\mathrm{Li^+},d}}{q_f + q_d}
.. math:: c_{\mathrm{Co^{2+}},r}(\bar{x}=0) = \frac{q_f c_{\mathrm{Co^{2+}},f} + q_d c_{\mathrm{Co^{2+}},d}}{q_f + q_d}
.. math:: q_p(\bar{x}=0) = \epsilon
.. math:: c_{\mathrm{Li^+},p}(\bar{x}=0) = \epsilon
.. math:: c_{\mathrm{Co^{2+}},p}(\bar{x}=0) = \epsilon
.. math:: c_{\mathrm{Li^+},m} (\bar{x}=0,\bar{z}) = \epsilon \qquad \forall \, \bar{z}
.. math:: c_{\mathrm{Co^{2+}},m} (\bar{x}=0,\bar{z}) = \epsilon \qquad \forall \, \bar{z}
.. math:: c_{\mathrm{Cl^-},m} (\bar{x}=0,\bar{z}) = \epsilon \qquad \forall \, \bar{z}
.. math:: \frac{\mathrm{d}q_r(\bar{x})}{\mathrm{d}\bar{x}}(\bar{x}=0)=\epsilon
.. math:: \frac{\mathrm{d}c_{\mathrm{Li^+},r}(\bar{x})}{\mathrm{d}\bar{x}}(\bar{x}=0)=\epsilon
.. math:: \frac{\mathrm{d}c_{\mathrm{Co^{2+}},r}(\bar{x})}{\mathrm{d}\bar{x}}(\bar{x}=0)=\epsilon

The following initial conditions are fixed to improve numerical stability (with the appropriate constraints deactivated as described above):

.. math:: J_w(\bar{x}=0) = \epsilon
.. math:: j_{\mathrm{Li^+}}(\bar{x}=0) = \epsilon
.. math:: j_{\mathrm{Co^{2+}}}(\bar{x}=0) = \epsilon
.. math:: j_{\mathrm{Cl^-}}(\bar{x}=0) = \epsilon

"""
# TODO: finish updating documentation

from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import (
    Constraint,
    Expression,
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

from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.constants import Constants
from idaes.core.util.scaling import constraint_autoscale_large_jac


@declare_process_block_class("SingleSaltDiafiltration")
class SingleSaltDiafiltrationData(UnitModelBlockData):
    """
    Single-Salt Diafiltration Unit Model Class.
    """

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
        "NFE_module_length",
        ConfigValue(
            doc="Number of discretization points in the x-direction",
        ),
    )
    CONFIG.declare(
        "NFE_membrane_thickness",
        ConfigValue(
            doc="Number of discretization points in the z-direction",
        ),
    )
    CONFIG.declare(
        "charged_membrane",
        ConfigValue(
            default=True, doc="Boolean argument if the membrane has a surface charge"
        ),
    )

    def build(self):
        """
        Build method for the single salt diafiltration unit model
        """
        super().build()

        # TODO: generalize to any 2 cations and 1 anion

        self.add_mutable_parameters()
        self.add_variables()
        self.add_constraints()
        self.discretize_model()
        self.fix_initial_values()
        self.add_helpful_expressions()
        self.add_scaling_factors()
        self.add_ports()

    def add_mutable_parameters(self):
        """
        Adds parameters for the single salt diafiltration unit model.

        Assigns default values that can be changed by the user during implementation.
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
        if self.config.charged_membrane:
            self.membrane_fixed_charge = Param(
                initialize=-140,
                mutable=True,
                units=units.mol / units.m**3,  # mM
                doc="Fixed charge on the membrane",
            )
        else:
            self.membrane_fixed_charge = Param(
                initialize=0,
                mutable=True,
                units=units.mol / units.m**3,  # mM
                doc="Fixed charge on the membrane",
            )
        self.membrane_permeability = Param(
            initialize=0.003,
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
        Adds variables for the single salt diafiltration unit model.
        """
        # define length scales
        self.dimensionless_module_length = ContinuousSet(bounds=(0, 1))
        self.dimensionless_membrane_thickness = ContinuousSet(bounds=(0, 1))

        # add a time index since the property package variables are indexed over time
        self.time = Set(initialize=[0])

        # add components
        self.solutes = Set(initialize=["cation", "anion"])
        self.cations = Set(initialize=["cation"])

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
            initialize=15,
            units=units.bar,
            bounds=[1e-11, None],
            doc="Pressure applied to membrane",
        )
        self.feed_flow_volume = Var(
            self.time,
            initialize=10,
            units=units.m**3 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric flow rate of the feed",
        )

        def initialize_feed_conc_mol_comp(m, t, j):
            vals = {"cation": 150, "anion": 150}
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
            initialize=3,
            units=units.m**3 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric flow rate of the diafiltrate",
        )

        def initialize_diafiltrate_conc_mol_comp(m, t, j):
            vals = {"cation": 14, "anion": 14}
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
            self.dimensionless_module_length,
            initialize=0.003,
            units=units.m**3 / units.m**2 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric water flux of water across the membrane",
        )
        self.mol_flux_cation = Var(
            self.dimensionless_module_length,
            initialize=7,
            units=units.mol / units.m**2 / units.h,
            bounds=[1e-11, None],
            doc="Mole flux of cation across the membrane (z-direction, x-dependent)",
        )
        self.mol_flux_anion = Var(
            self.dimensionless_module_length,
            initialize=7,
            units=units.mol / units.m**2 / units.h,
            bounds=[1e-11, None],
            doc="Mole flux of anion across the membrane (z-direction, x-dependent)",
        )
        self.retentate_flow_volume = Var(
            self.time,
            self.dimensionless_module_length,
            initialize=7.5,
            units=units.m**3 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric flow rate of the retentate, x-dependent",
        )

        def initialize_retentate_conc_mol_comp(m, t, p, j):
            vals = {"cation": 48, "anion": 48}
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
            initialize=5.5,
            units=units.m**3 / units.h,
            bounds=[1e-11, None],
            doc="Volumetric flow rate of the permeate, x-dependent",
        )

        def initialize_permeate_conc_mol_comp(m, t, p, j):
            vals = {"cation": 48, "anion": 48}
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
            self.dimensionless_module_length,
            initialize=1,
            units=units.bar,
            bounds=[1e-11, None],
            doc="Osmostic pressure of the feed-side fluid",
        )

        # add variables dependent on dimensionless_module_length and dimensionless_membrane_thickness
        self.membrane_conc_mol_cation = Var(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            initialize=75,
            units=units.mol / units.m**3,  # mM
            bounds=[1e-11, None],
            doc="Mole concentration of cation in the membrane, x- and z-dependent",
        )
        self.membrane_conc_mol_anion = Var(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            initialize=75,
            units=units.mol / units.m**3,  # mM
            bounds=[1e-11, None],
            doc="Mole concentration of anion in the membrane, x- and z-dependent",
        )

        self.D_tilde = Var(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            initialize=1e3,
            units=units.mm**2 / units.h * units.mol / units.m**3,
            # bounds=[1e-20, None],
            doc="Denominator of diffusion and convection coefficients",
        )
        self.D_cation_cation_bilinear = Var(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            initialize=-1e3,
            units=units.mm**2
            / units.h
            * (units.mm**2 / units.h * units.mol / units.m**3),
            # bounds=[1e-20, None],
            doc="Bi-linear cross diffusion coefficient for cation-cation",
        )
        self.alpha_cation_bilinear = Var(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            initialize=1e2,
            units=units.mm**2 / units.h * units.mol / units.m**3,
            # bounds=[1e-20, None],
            doc="Bi-linear cross diffusion coefficient for cation",
        )
        self.D_cation_cation = Var(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            initialize=-5,
            units=units.mm**2 / units.h,
            # bounds=[1e-20, None],
            doc="Cross diffusion coefficient for cation-cation",
        )
        self.convection_coefficient_cation = Var(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            initialize=0.6,
            units=units.dimensionless,
            doc="Convection coefficient for cation",
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
        self.d_membrane_conc_mol_cation_dz = DerivativeVar(
            self.membrane_conc_mol_cation,
            wrt=self.dimensionless_membrane_thickness,
            units=units.mol / units.m**3,  # mM
            doc="Lithium concentration gradient wrt z in the membrane",
        )

    def add_constraints(self):
        """
        Adds model constraints for the single salt diafiltration unit model.
        """

        # differential mol balance constraints
        def _overall_mol_balance(blk, x):
            if x == 0:
                return Constraint.Skip
            return blk.d_retentate_flow_volume_dx[0, x] == (
                -blk.volume_flux_water[x]
                * blk.total_membrane_length
                * blk.total_module_length
            )

        self.overall_mol_balance = Constraint(
            self.dimensionless_module_length, rule=_overall_mol_balance
        )

        def _cation_mol_balance(blk, x):
            if x == 0:
                return Constraint.Skip
            return (
                blk.retentate_flow_volume[0, x]
                * blk.d_retentate_conc_mol_comp_dx[0, x, "cation"]
            ) == (
                (
                    blk.volume_flux_water[x]
                    * blk.retentate_conc_mol_comp[0, x, "cation"]
                    - blk.mol_flux_cation[x]
                )
                * blk.total_membrane_length
                * blk.total_module_length
            )

        self.cation_mol_balance = Constraint(
            self.dimensionless_module_length, rule=_cation_mol_balance
        )

        # bulk flux balance constraints
        def _bulk_flux_balance_overall(blk, x):
            if x == 0:
                return Constraint.Skip
            return (
                blk.permeate_flow_volume[0, x]
                == blk.volume_flux_water[x]
                * x
                * blk.total_membrane_length
                * blk.total_module_length
            )

        self.bulk_flux_balance_overall = Constraint(
            self.dimensionless_module_length, rule=_bulk_flux_balance_overall
        )

        def _bulk_flux_balance_cation(blk, x):
            if x == 0:
                return Constraint.Skip
            return blk.mol_flux_cation[x] == (
                blk.permeate_conc_mol_comp[0, x, "cation"] * blk.volume_flux_water[x]
            )

        self.bulk_flux_balance_cation = Constraint(
            self.dimensionless_module_length, rule=_bulk_flux_balance_cation
        )

        # transport constraints (first principles)
        def _lumped_water_flux(blk, x):
            if x == 0:
                return Constraint.Skip
            return blk.volume_flux_water[x] == (
                blk.membrane_permeability
                * (blk.applied_pressure - blk.osmotic_pressure[x])
            )

        self.lumped_water_flux = Constraint(
            self.dimensionless_module_length, rule=_lumped_water_flux
        )

        def _D_tilde_calculation(blk, x, z):
            return blk.D_tilde[x, z] == (
                (
                    (
                        (
                            (blk.config.property_package.charge["cation"] ** 2)
                            * blk.config.property_package.diffusion_coefficient[
                                "cation"
                            ]
                        )
                        - (
                            blk.config.property_package.charge["cation"]
                            * blk.config.property_package.charge["anion"]
                            * blk.config.property_package.diffusion_coefficient["anion"]
                        )
                    )
                    * blk.membrane_conc_mol_cation[x, z]
                )
                - (
                    blk.config.property_package.charge["anion"]
                    * blk.config.property_package.diffusion_coefficient["anion"]
                    * blk.membrane_fixed_charge
                )
            )

        self.D_tilde_calculation = Constraint(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_D_tilde_calculation,
        )

        def _D_cation_cation_bilinear_calculation(blk, x, z):
            return (
                blk.D_cation_cation_bilinear[x, z]
                == blk.D_cation_cation[x, z] * blk.D_tilde[x, z]
            )

        self.D_cation_cation_bilinear_calculation = Constraint(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_D_cation_cation_bilinear_calculation,
        )

        def _alpha_cation_bilinear_calculation(blk, x, z):
            return (
                blk.alpha_cation_bilinear[x, z]
                == blk.convection_coefficient_cation[x, z] * blk.D_tilde[x, z]
            )

        self.alpha_cation_bilinear_calculation = Constraint(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_alpha_cation_bilinear_calculation,
        )

        def _D_cation_cation_calculation(blk, x, z):
            return blk.D_cation_cation_bilinear[x, z] == (
                (
                    (
                        blk.config.property_package.charge["cation"]
                        * blk.config.property_package.charge["anion"]
                        * blk.config.property_package.diffusion_coefficient["cation"]
                        * blk.config.property_package.diffusion_coefficient["anion"]
                    )
                    - (
                        (blk.config.property_package.charge["cation"] ** 2)
                        * blk.config.property_package.diffusion_coefficient["cation"]
                        * blk.config.property_package.diffusion_coefficient["anion"]
                    )
                )
                * blk.membrane_conc_mol_cation[x, z]
                + (
                    blk.config.property_package.charge["anion"]
                    * blk.config.property_package.diffusion_coefficient["cation"]
                    * blk.config.property_package.diffusion_coefficient["anion"]
                    * blk.membrane_fixed_charge
                )
            )

        self.D_cation_cation_calculation = Constraint(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_D_cation_cation_calculation,
        )

        def _convection_coefficient_cation_calculation(blk, x, z):
            return blk.alpha_cation_bilinear[x, z] == (
                blk.D_tilde[x, z]
                + (
                    blk.config.property_package.charge["cation"]
                    * blk.config.property_package.diffusion_coefficient["cation"]
                    * blk.membrane_fixed_charge
                )
            )

        self.convection_coefficient_cation_calculation = Constraint(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_convection_coefficient_cation_calculation,
        )

        def _cation_flux_membrane(blk, x, z):
            if x == 0:
                return Constraint.Skip
            return blk.mol_flux_cation[x] == (
                (
                    blk.convection_coefficient_cation[x, z]
                    * blk.membrane_conc_mol_cation[x, z]
                    * blk.volume_flux_water[x]
                )
                + (
                    units.convert(
                        blk.D_cation_cation[x, z], to_units=units.m**2 / units.h
                    )
                    / blk.total_membrane_thickness
                    * blk.d_membrane_conc_mol_cation_dz[x, z]
                )
            )

        self.cation_flux_membrane = Constraint(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_cation_flux_membrane,
        )

        def _anion_flux_membrane(blk, x):
            if x == 0:
                return Constraint.Skip
            return 0 == (
                (blk.config.property_package.charge["cation"] * blk.mol_flux_cation[x])
                + (blk.config.property_package.charge["anion"] * blk.mol_flux_anion[x])
            )

        self.anion_flux_membrane = Constraint(
            self.dimensionless_module_length, rule=_anion_flux_membrane
        )

        # other physical constraints
        def _osmotic_pressure_calculation(blk, x):
            if x == 0:
                return Constraint.Skip
            return blk.osmotic_pressure[x] == units.convert(
                (
                    Constants.gas_constant  # J / mol / K
                    * blk.temperature
                    * (
                        (
                            blk.config.property_package.num_solutes["cation"]
                            * blk.config.property_package.sigma["cation"]
                            * (
                                blk.retentate_conc_mol_comp[0, x, "cation"]
                                - blk.permeate_conc_mol_comp[0, x, "cation"]
                            )
                        )
                        + (
                            blk.config.property_package.num_solutes["anion"]
                            * blk.config.property_package.sigma["anion"]
                            * (
                                blk.retentate_conc_mol_comp[0, x, "anion"]
                                - blk.permeate_conc_mol_comp[0, x, "anion"]
                            )
                        )
                    )
                ),
                to_units=units.bar,
            )

        self.osmotic_pressure_calculation = Constraint(
            self.dimensionless_module_length, rule=_osmotic_pressure_calculation
        )

        def _electroneutrality_retentate(blk, x):
            return 0 == (
                blk.config.property_package.charge["cation"]
                * blk.retentate_conc_mol_comp[0, x, "cation"]
                + blk.config.property_package.charge["anion"]
                * blk.retentate_conc_mol_comp[0, x, "anion"]
            )

        self.electroneutrality_retentate = Constraint(
            self.dimensionless_module_length, rule=_electroneutrality_retentate
        )

        def _electroneutrality_membrane(blk, x, z):
            if x == 0:
                return Constraint.Skip
            return 0 == (
                blk.config.property_package.charge["cation"]
                * blk.membrane_conc_mol_cation[x, z]
                + blk.config.property_package.charge["anion"]
                * blk.membrane_conc_mol_anion[x, z]
                + blk.membrane_fixed_charge
            )

        self.electroneutrality_membrane = Constraint(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_electroneutrality_membrane,
        )

        def _electroneutrality_permeate(blk, x):
            return 0 == (
                blk.config.property_package.charge["cation"]
                * blk.permeate_conc_mol_comp[0, x, "cation"]
                + blk.config.property_package.charge["anion"]
                * blk.permeate_conc_mol_comp[0, x, "anion"]
            )

        self.electroneutrality_permeate = Constraint(
            self.dimensionless_module_length, rule=_electroneutrality_permeate
        )

        # boundary conditions
        def _retentate_membrane_interface_cation(blk, x):
            if x == 0:
                return Constraint.Skip
            return (
                blk.config.property_package.partition_coefficient_retentate["cation"]
                ** (-blk.config.property_package.charge["anion"])
                * blk.config.property_package.partition_coefficient_retentate["anion"]
                ** blk.config.property_package.charge["cation"]
                * blk.retentate_conc_mol_comp[0, x, "cation"]
                ** (-blk.config.property_package.charge["anion"])
                * blk.retentate_conc_mol_comp[0, x, "anion"]
                ** blk.config.property_package.charge["cation"]
            ) == (
                blk.membrane_conc_mol_cation[x, 0]
                ** (-blk.config.property_package.charge["anion"])
                * blk.membrane_conc_mol_anion[x, 0]
                ** blk.config.property_package.charge["cation"]
            )

        self.retentate_membrane_interface_cation = Constraint(
            self.dimensionless_module_length, rule=_retentate_membrane_interface_cation
        )

        def _membrane_permeate_interface_cation(blk, x):
            if x == 0:
                return Constraint.Skip
            return (
                blk.config.property_package.partition_coefficient_permeate["cation"]
                ** (-blk.config.property_package.charge["anion"])
                * blk.config.property_package.partition_coefficient_permeate["anion"]
                ** blk.config.property_package.charge["cation"]
                * blk.permeate_conc_mol_comp[0, x, "cation"]
                ** (-blk.config.property_package.charge["anion"])
                * blk.permeate_conc_mol_comp[0, x, "anion"]
                ** blk.config.property_package.charge["cation"]
            ) == (
                blk.membrane_conc_mol_cation[x, 1]
                ** (-blk.config.property_package.charge["anion"])
                * blk.membrane_conc_mol_anion[x, 1]
                ** blk.config.property_package.charge["cation"]
            )

        self.membrane_permeate_interface_cation = Constraint(
            self.dimensionless_module_length, rule=_membrane_permeate_interface_cation
        )

        # initial conditions
        def _initial_retentate_flow_volume(blk):
            return (
                blk.retentate_flow_volume[0, 0]
                == blk.feed_flow_volume[0] + blk.diafiltrate_flow_volume[0]
            )

        self.initial_retentate_flow_volume = Constraint(
            rule=_initial_retentate_flow_volume
        )

        def _initial_retentate_conc_mol_cation(blk):
            return blk.retentate_conc_mol_comp[0, 0, "cation"] == (
                (
                    blk.feed_flow_volume[0] * blk.feed_conc_mol_comp[0, "cation"]
                    + blk.diafiltrate_flow_volume[0]
                    * blk.diafiltrate_conc_mol_comp[0, "cation"]
                )
                / (blk.feed_flow_volume[0] + blk.diafiltrate_flow_volume[0])
            )

        self.initial_retentate_conc_mol_cation = Constraint(
            rule=_initial_retentate_conc_mol_cation
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

    def fix_initial_values(self):
        """
        Fix initial values for the single salt diafiltration unit model to improve numerics.
        """
        for x in self.dimensionless_module_length:
            # anion concentration gradient in retentate variable is created by default but
            # is not needed in model; fix to reduce number of variables
            self.d_retentate_conc_mol_comp_dx[0, x, "anion"].fix(
                value(self.numerical_zero_tolerance)
            )
            # associated discretization equation not needed in model
            if x != 0:
                self.d_retentate_conc_mol_comp_dx_disc_eq[0, x, "anion"].deactivate()

        # set "zero" initial values to a sufficiently small value (expected to be 0)
        self.permeate_flow_volume[0, 0].fix(value(self.numerical_zero_tolerance))
        self.permeate_conc_mol_comp[0, 0, "cation"].fix(
            value(self.numerical_zero_tolerance)
        )
        self.volume_flux_water[0].fix(value(self.numerical_zero_tolerance))
        self.mol_flux_cation[0].fix(value(self.numerical_zero_tolerance))
        self.mol_flux_anion[0].fix(value(self.numerical_zero_tolerance))

        # initial values for the membrane cocnentration partial derivatives
        for z in self.dimensionless_membrane_thickness:
            self.membrane_conc_mol_cation[0, z].fix(
                value(self.numerical_zero_tolerance)
            )

    def add_helpful_expressions(self):
        def _peclet_number_cation(blk, x, z):
            if x == 0:
                return Expression.Skip
            return abs(
                blk.convection_coefficient_cation[x, z]
                * blk.membrane_conc_mol_cation[x, z]
                * blk.volume_flux_water[x]
            ) / (
                abs(
                    (
                        blk.D_cation_cation[x, z]
                        / blk.total_membrane_thickness
                        * blk.d_membrane_conc_mol_cation_dz[x, z]
                    )
                )
            )

        self.peclet_number_cation = Expression(
            self.dimensionless_module_length,
            self.dimensionless_membrane_thickness,
            rule=_peclet_number_cation,
        )

        def _sieving_coefficient_cation(blk, x):
            if x == 0:
                return Expression.Skip
            return (
                blk.permeate_conc_mol_comp[0, x, "cation"]
                / blk.retentate_conc_mol_comp[0, x, "cation"]
            )

        self.sieving_coefficient_cation = Expression(
            self.dimensionless_module_length,
            rule=_sieving_coefficient_cation,
        )

    def add_scaling_factors(self):
        """
        Assigns scaling factors to certain variables and constraints to
        improve solver performance.
        """
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Add scaling factors for poorly scaled variables
        # for x in self.dimensionless_module_length:
        #     for z in self.dimensionless_membrane_thickness:
        #         self.scaling_factor[self.D_11[x, z]] = 1e6
        #         self.scaling_factor[self.D_12[x, z]] = 1e7
        #         self.scaling_factor[self.D_21[x, z]] = 1e7
        #         self.scaling_factor[self.D_22[x, z]] = 1e6

        # Add scaling factors for poorly scaled constraints
        constraint_autoscale_large_jac(self)

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
