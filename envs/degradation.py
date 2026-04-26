import numpy as np

# DEGRADATION MODEL: Cortés-Arcos et al. (2020), Applied Sciences, 10(15), 5330
# https://doi.org/10.3390/app10155330
#
# Version 2 (cyclic) — Equation 23 (Section 2.3.2):
#   ε_cyc    = ΔQ / Q(C_loss = 20%)
#   Cost_deg = ε_cyc × Cost_bat
#
# For optimisation this is linearised as a marginal cost per unit throughput:
#
#   Cost_deg = κ × E_throughput
#
# where:
#   κ              = Cost_bat / Q_lifetime_MWh   [£/MWh]
#   Q_lifetime_MWh = lifetime energy throughput to 20% capacity loss
#
# In a power-based MDP with timestep dt_hours and active power P_act_MW:
#
#   E_throughput = |P_act_MW| × dt_hours
#   ⟹ Cost_deg  = κ × |P_act_MW| × dt_hours
#
# Cortés-Arcos et al. explicitly recommend Version 2 and 3 as suitable
# for deterministic optimisation problems (Section 4).
#
# κ DERIVATION for CATL EnerOne 1P416S LFP (updated 2025-03)
#
# Parameters (modern utility-scale LFP, EnerOne cabinet):
#   E_nominal  = 372.7 kWh  (CATL EnerOne datasheet: 280 Ah × 1331.2 V)
#   DoD        = 0.80  (SoC_max = 0.9, SoC_min = 0.1 → 80% usable window)
#   N_cycles   = 8,000 (LFP at ~0.5C, 80% DoD, to 80% retained capacity;
#                modern LFP cells: 6,000–10,000 cycles at this operating window)
#   Cost_bat   ≈ £44,724  (≈ £120/kWh × 372.7 kWh; consistent with 2024–2025
#                utility-scale pack costs from NREL ATB 2024, converted to GBP)
#
# Lifetime energy throughput (bidirectional):
#
#   Q_lifetime_MWh = N_cycles × 2 × DoD × E_nominal / 1000
#                  = 8000 × 2 × 0.80 × 372.7 / 1000
#                  ≈ 4,770 MWh
#
# Marginal degradation cost:
#
#   κ = Cost_bat / Q_lifetime_MWh
#     ≈ £44,724 / 4,770 MWh
#     ≈ £9.38/MWh  →  κ ≈ 10.0 £/MWh (rounded up for conservative purpopses)
#
# The previous value (κ = 35 £/MWh) used 2022-era assumptions:
#   Cost_bat ≈ £200/kWh, N_cycles ≈ 3,500 at 80% DoD (1C)
# giving Q_lifetime ≈ 2,087 MWh and κ ≈ 35.7 £/MWh. That higher κ made
# cycling unprofitable except during very high-spread periods and is retained
# as a conservative legacy scenario in ablation studies.


class ThroughputDegradation:
    """Linear throughput-cost degradation model (Cortés-Arcos et al., 2020)."""

    def __init__(self, kappa: float):
        """Initialise the degradation model.

        Args:
            kappa: Marginal degradation cost coefficient κ (£/MWh).
        """
        self.kappa = kappa

    def calculate_costs(self, P_act_MW: float, dt_hours: float) -> float:
        """Compute the marginal degradation cost for one timestep (£).

        Implements Cortés-Arcos et al. (2020) Eq. 23, Version 2 cyclic:
            Cost_deg = κ · |P_act_MW| · dt_hours

        Args:
            P_act_MW: Actual applied power in MW. Positive = discharge.
            dt_hours: Timestep duration in hours.

        Returns:
            Degradation cost in GBP.
        """
        return self.kappa * abs(P_act_MW) * dt_hours
