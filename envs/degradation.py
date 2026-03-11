import numpy as np

# DEGRADATION MODEL: Cortés-Arcos et al. (2020), Applied Sciences, 10(15), 5330
# https://doi.org/10.3390/app10155330
#
# Version 2 Cyclic — Equation 23 (Section 2.3.2):
#   ε_cyc = ΔQ / Q(C_loss = 20%)
#   Cost_deg = ε_cyc × Cost_bat
#
# Simplified to a per-step marginal throughput cost:
#   Cost_deg = κ × |P_act_MW| × dt_hours
#   where κ = Cost_bat / Q_lifetime_MWh  [£/MWh]
#
# The paper recommends Version 2 for optimization problems (Section 4):
# "versions #2 and #3 are good candidates for estimating battery degradation
#  costs in problems where deterministic models are needed"
#
# --- κ DERIVATION for CATL EnerOne 1P416S LFP ---
# κ = Cost_bat / Q_lifetime_MWh
#
# Parameters:
#   E_nominal    = 372.7 kWh       (CATL EnerOne datasheet: 280 Ah × 1331.2 V)
#   DoD          = 0.80            (SoC_max=0.9 − SoC_min=0.1)
#   N_cycles     = 3,500           (CATL LFP minimum cycle life at 1C, to 80% capacity)
#   Cost_bat     = £74,540         (£200/kWh × 372.7 kWh; BEIS Energy Storage Capital
#                                   Cost Report 2023: £150–250/kWh for installed LFP BESS)
#
# Q_lifetime_MWh = N_cycles × 2 × DoD × E_nominal / 1000
#                = 3500 × 2 × 0.80 × 372.7 / 1000
#                = 2,087 MWh  (bidirectional: abs(P_act) counts both charge and discharge)
#
# κ = £74,540 / 2,087 = £35.7/MWh  →  κ = 35.0 £/MWh

class ThroughputDegradation:
    def __init__(self, kappa: float):
        self.kappa = kappa

    def calculate_costs(self, P_act_MW: float, dt_hours: float) -> float:
        """
        Marginal degradation cost per timestep (£).

        Implements Cortés-Arcos et al. (2020) Eq. 23, Version 2 Cyclic:
            Cost_deg = κ · |P_act_MW| · dt_hours
            κ = Cost_bat / Q_lifetime_MWh  [£/MWh]
        """
        return self.kappa * abs(P_act_MW) * dt_hours
