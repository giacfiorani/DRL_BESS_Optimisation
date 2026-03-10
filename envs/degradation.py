import numpy as np

# DEGRADATION MODEL: Cortés-Arcos et al. (2020)
# Version 2 Cyclic Degradation Cost — Equation 23
# cost = κ · |P_act| · dt
# where κ = C_bat / E_lifetime (£/MWh throughput)
class ThroughputDegradation:
    def __init__(self, kappa: float):
        self.kappa = kappa

    def calculate_costs(self, P_act_MW: float, dt_hours: float) -> float:
        return self.kappa * abs(P_act_MW) * dt_hours
