import numpy as np

class SocWeightedDegradation:
    def __init__(self, kappa:float, alpha:float):
        self.kappa = kappa
        self.alpha = alpha
    def calculate_costs(self, P_act_MW:float, dt_hours:float, soc_t:float):
        Cost = self.kappa * abs(P_act_MW) * dt_hours * (1 + self.alpha * ((soc_t - 0.5)**2))
        return Cost