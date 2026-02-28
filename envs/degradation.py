import numpy as np

# DEGRADATION MODEL: Cortés-Arcos et al. (2020)
# Linear Marginal Throughput Cost (Equation 21 & 23)
class SocWeightedDegradation:
    def __init__(self, kappa:float, alpha:float):
        self.kappa = kappa
        self.alpha = alpha
    def calculate_costs(self, P_act_MW:float, dt_hours:float):
        cost = self.deg_kappa * abs(P_act_MW) * self.dt_hours
        return cost

