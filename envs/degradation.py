import numpy as np

# DEGRADATION MODEL: Cortés-Arcos et al. (2020), Applied Sciences, 10(15), 5330
# https://doi.org/10.3390/app10155330
#
# Version 2 (cyclic) — Equation 23 (Section 2.3.2):
#   ε_cyc   = ΔQ / Q(C_loss = 20%)
#   Cost_deg = ε_cyc × Cost_bat
#
# For optimisation, this can be linearised as a marginal cost per unit
# throughput:
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
#   ⇒ Cost_deg   = κ × |P_act_MW| × dt_hours
#
# Cortés-Arcos et al. explicitly recommend Version 2 and 3 as suitable
# for deterministic optimisation problems (Section 4).
#
# --- κ DERIVATION for CATL EnerOne 1P416S LFP (updated 2025‑03) ---
#
# We follow the standard linear throughput-cost approach used in
# Cortés‑Arcos (2020) and Xu et al. (2017, arXiv:1707.04567) and plug in
# 2024/2025 LFP economics for a single CATL EnerOne cabinet:
#
#   κ = Cost_bat / Q_lifetime_MWh
#
# Parameters (modern utility‑scale LFP, EnerOne cabinet):
#   E_nominal  = 372.7 kWh
#                (CATL EnerOne datasheet: 280 Ah × 1331.2 V, rated energy 372.7 kWh)
#                e.g. EnerOne product sheets list 372.7 kWh with 280 Ah cells.[web:149][web:249][web:251]
#   DoD        = 0.80
#                (SoC_max = 0.9, SoC_min = 0.1 → 80% usable window)
#   N_cycles   = 8,000
#                (LFP cell cycle life at ~0.5C, 80% DoD, to 80% retained capacity;
#                 modern LFP storage cells are typically rated 6,000–10,000 cycles
#                 at this operating window.[web:241][web:182][web:149][web:246])
#   Cost_bat   ≈ £44,724
#                (≈ £120/kWh × 372.7 kWh; consistent with 2024–2025 utility‑scale
#                 battery‑pack costs of 120–150 $/kWh from NREL ATB 2024 and
#                 Cole & Karmakar’s cost projections, converted to GBP.[web:201][web:206])
#
# Lifetime energy throughput (bidirectional, counting both charge and discharge):
#
#   Q_lifetime_MWh = N_cycles × 2 × DoD × E_nominal / 1000
#                   = 8000 × 2 × 0.80 × 372.7 / 1000
#                   ≈ 4,770 MWh
#
# Marginal degradation cost:
#
#   κ = Cost_bat / Q_lifetime_MWh
#     ≈ £44,724 / 4,770 MWh
#     ≈ £9.38/MWh  →  κ ≈ 10.0 £/MWh (rounded)
#
# This κ is consistent with:
#   • Modern LFP cabinet costs from NREL ATB 2024 and similar studies.[web:201][web:206]
#   • Cycle‑life ranges (6,000–10,000 cycles at 80% DoD) reported for LFP.[web:241][web:182][web:149]
#   • The linearised throughput‑cost framework of Cortés‑Arcos (2020) and
#     Xu et al. (2017).[web:234][web:208][web:235]
#
# Previous value (κ = 35 £/MWh) used 2022‑era assumptions:
#   • Cost_bat ≈ £200/kWh
#   • N_cycles ≈ 3,500 at 80% DoD (1C)
# giving Q_lifetime ≈ 2,087 MWh and κ ≈ 35.7 £/MWh. That higher κ made
# cycling unprofitable except during very high‑spread periods and is now
# treated as a conservative legacy scenario in ablation studies.


class ThroughputDegradation:
    def __init__(self, kappa: float):
        self.kappa = kappa

    def calculate_costs(self, P_act_MW: float, dt_hours: float) -> float:
        """
        Marginal degradation cost per timestep (£).

        Implements Cortés-Arcos et al. (2020) Eq. 23, Version 2 cyclic:
            Cost_deg = κ · |P_act_MW| · dt_hours
            with κ = Cost_bat / Q_lifetime_MWh  [£/MWh],
        where Q_lifetime_MWh is the bidirectional lifetime energy throughput
        to 20% capacity loss at the chosen operating window.
        """
        return self.kappa * abs(P_act_MW) * dt_hours
