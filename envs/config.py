# ========
# BESS HARDWARE SPECIFICATIONS (CATL EnerOne, Liquid-Cooled Rack)
# ========
N_cells = 416  # 1P416S: 416 cells in series
Q_cell = 280  # Ah (CATL 280Ah LFP prismatic cells)
Q_cell_C = 280 * 3600  # C (Coulombs) for Coulomb counting: Q = 280 Ah × 3600 s/h
V_nominal = 1331.2  # V (nominal pack voltage)
E_nominal = 372.7  # kWh (nominal energy)
E_max = 0.3727  # MWh (same as E_nominal, converted to MWh)

# ========
# OPERATIONAL PARAMETERS
# ========
SoC_min = 0.2  # Minimum SoC (fraction)
SoC_max = 0.9  # Maximum SoC (fraction)
SoC_initial = 0.5  # Initial SoC (fraction) - set to mid-range for safety

# ========
# EFFICIENCY PARAMETERS
# ========
eff_dis = 1.0  # Discharge efficiency (η_discharge = 1 per MDP)
eff_ch = 0.995  # Charge efficiency (η_charge = 0.995 per MDP)
self_dis = 0.0  # Self-discharge rate per timestep (set to 0 for simplicity)

# ========
# POWER AND TIMING PARAMETERS
# ========
C_rate = 0.5  # C-rate (0.5C = 2-hour discharge)
P_step = C_rate * E_max  # MW (power step: P_step = 0.5C × E_nominal = 0.18635 MW)
P_max = P_step  # MW (maximum power, equal to P_step for discrete actions)
dt = 0.5  # hours (timestep = 30 minutes)

# ========
# REWARD FUNCTION PARAMETERS
# ========
lambda_ci = 0.9  # Carbon penalty weight (λ) - tunable scalar
