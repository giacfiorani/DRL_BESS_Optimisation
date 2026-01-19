import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium import Env, spaces
import env_config
from data import traini

class BatteryEnv(Env):
    def __init__(self, env_config):
        super().__init__()

        # ========
        # 1. HARDWARE & MDP PARAMETERS
        # ========
        # CATL EnerOne 1P416S
        self.N_cells = env_config.N_cells
        self.Q_cell_Ah = env_config.Q_cell
        # Pack charge (Coulombs): 280 Ah * 3600 s/h
        self.Q_pack_C = env_config.Q_cell_C  # 1,008,000 C
        self.V_nominal = env_config.V_nominal  # V
        self.E_nominal = env_config.E_nominal  # kWh

        # Operational limits (from your MDP)
        self.SoC_min = env_config.SoC_min
        self.SoC_max = env_config.SoC_max
        self.SoC_initial = float(env_config.SoC_initial)

        # Time step: 30 minutes
        self.dt_hours = env_config.dt
        self.dt_seconds = self.dt_hours * 3600.0

        # Efficiencies (from your MDP)
        self.eta_ch = env_config.eff_ch   # charge efficiency
        self.eta_dis = env_config.eff_dis    # discharge efficiency

        # Power / action scaling
        self.P_max_MW = env_config.P_max_MW
        self.n_power_levels = env_config.n_power_levels

        # Reward env_config: carbon weight λ
        self.lambda_ci = float(env_config.lambda_ci)
        
        # ECM R-int model parameter (internal resistance)
        R_cell_mOhm = 0.4  # mΩ per cell (from Product Specification Sheet)
        self.R_sys = (R_cell_mOhm / 1000.0) * self.N_cells  # Ω (pack resistance)

        # ========
        # Read from OCV Lookup Table and Interpolate to get OCV-SOC Curve
        # ========
        self.ocv_soc_points, self.ocv_cell_volts = env_config.ocv_lookup_table()
        self.ocv_table_size = env_config.ocv_table_size
        self.soc_grid = np.linspace(0.0, 1.0, self.ocv_table_size, dtype=np.float32)
        self.ocv_cell_table = np.interp(
            self.soc_grid, self.ocv_soc_points, self.ocv_cell_volts
        ).astype(np.float32)
        self.ocv_pack_table = (self.ocv_cell_table * self.N_cells).astype(np.float32)

        # ========
        # 3. LOAD DATA
        # ========
        training_data = pd.read_parquet("data/training_data.parquet")

        # Price dataset (£/MWh)
        self.power_price = training_data["price_gbp_mwh"].to_numpy(dtype=np.float32)  # Wholesale Power Price £/MWh
        
        # Carbon intensity dataset (gCO2/kWh)
        self.ci_data = training_data["carbon_gco2_kwh"].to_numpy(dtype=np.float32)
        
        # Time encoding τ_t = hour * 2 + minute // 30
        self.timestamp = pd.to_datetime(training_data["timestamp"])
        self.tau_data = training_data["tau"].to_numpy(dtype=np.int32)

        self.max_steps = len(self.power_price)

        # ========
        # 4. INTERNAL ENV VARIABLES
        # ========
        self.current_step = 0
        self.soc = self.SoC_initial
        self.p_prev = 0.0
        self.reward = 0.0
        self.actions = []  

        # ========
        # 5. ACTION SPACE (DISCRETE)
        # ========
        self.power_levels = np.linspace(-self.P_max_MW, self.P_max_MW, self.n_power_levels)
        self.action_space = gym.spaces.Discrete(env_config.n_power_levels)

        # ========
        # 6. OBSERVATION SPACE
        # ========
        # [SoC_t, price_t, CI_t, τ_t, P_{t-1}]
        low = np.array(
            [0.0, -300.0, 0.0, 0.0, -self.P_max_MW],
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 2500.0, 1000.0, 49.0, self.P_max_MW],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # =========
    # OCV + PHYSICS HELPERS
    # =========
    def _get_ocv_pack(self, soc: float) -> float:
        """Get pack open-circuit voltage from SoC using lookup table."""
        soc = float(np.clip(soc, 0.0, 1.0))
        # Use floor to avoid index out of bounds, then clip to valid range
        idx = int(np.clip(soc * (self.ocv_table_size - 1), 0, self.ocv_table_size - 1))
        return float(self.ocv_pack_table[idx])


    def _compute_current_from_power(self, P_grid_W: float, V_oc_pack: float) -> float:
        """
        Solve P = I * (V_oc - I * R) for I, using R-int ECM:
            R I^2 - V_oc I + P = 0
        Use root:
            I = (V_oc - sqrt(V_oc^2 - 4 R P)) / (2 R)
        """
        if P_grid_W == 0.0:
            return 0.0

        discriminant = V_oc_pack**2 - 4.0 * self.R_sys * P_grid_W
        if discriminant < 0.0:
            discriminant = 0.0  # numerical guard

        I = (V_oc_pack - np.sqrt(discriminant)) / (2.0 * self.R_sys)
        return float(I)
    
    def _apply_soc_protection(self, P_requested_MW: float, soc: float) -> tuple[float, float, float]:
        """
        Protection function: clamp power to prevent overcharging/overdischarging.
        
        Returns (P_applied_MW, I_applied_A, V_oc_pack_V) such that SoC won't violate limits
        over this timestep.
        """
        soc = float(soc)

        # 1) actuator limit on requested power
        P_req_MW = float(np.clip(P_requested_MW, -self.P_max_MW, self.P_max_MW))
        if P_req_MW == 0.0:
            V_oc = self._get_ocv_pack(soc)
            return 0.0, 0.0, V_oc

        # 2) compute requested current from requested power
        V_oc = self._get_ocv_pack(soc)
        I_req = self._compute_current_from_power(P_req_MW * 1e6, V_oc)

        # 3) allowable SoC movement this step
        delta_soc_up_max   = max(self.SoC_max - soc, 0.0)  # room to charge
        delta_soc_down_max = max(soc - self.SoC_min, 0.0)  # room to discharge

        # If you’re at a limit, forbid current that would push further
        if delta_soc_up_max == 0.0 and I_req < 0.0:
            return 0.0, 0.0, V_oc
        if delta_soc_down_max == 0.0 and I_req > 0.0:
            return 0.0, 0.0, V_oc

        dt = self.dt_seconds
        Q  = self.Q_pack_C

        # Your SoC update rule is:
        # charge (I<0): ΔSoC = -(I dt / Q) * η_ch
        # discharge(I>0): ΔSoC = -(I dt / Q) / η_dis
        # So invert those to get current bounds:
        I_min = -(delta_soc_up_max   * Q) / (dt * self.eta_ch)   # most negative allowed
        I_max =  (delta_soc_down_max * Q * self.eta_dis) / dt    # most positive allowed

        # 4) clamp current, then compute corresponding applied power
        I_applied = float(np.clip(I_req, I_min, I_max))
        P_applied_W  = I_applied * (V_oc - I_applied * self.R_sys)
        P_applied_MW = float(np.clip(P_applied_W / 1e6, -self.P_max_MW, self.P_max_MW))

        # Important: ensure I and P are consistent after the final clip
        I_applied = self._compute_current_from_power(P_applied_MW * 1e6, V_oc)

        return P_applied_MW, float(I_applied), V_oc
    
    def _get_obs(self):
        """
        Observation: [SoC, Price, CI, tau, P_prev]
        """
        idx = min(self.current_step, self.max_steps - 1)

        return np.array(
            [
                self.soc,
                self.power_price[idx],
                self.ci_data[idx],
                self.tau_data[idx],
                self.p_prev,
            ],
            dtype=np.float32,
        )

    # =========
    # GYM API
    # =========
    def step(self, action_idx):

        # Check if action index is valid
        if not (0 <= action_idx < self.n_power_levels):
            raise ValueError(f"Invalid action index {action_idx}, must be between 0 and {self.n_power_levels-1}.")

        # Grid-side power in MW and W
        P_requested_MW = self.power_levels[action_idx]
        
        # Protection: clamp power to prevent overcharging/overdischarging
        P_applied_MW, I_applied, V_oc_pack = self._apply_soc_protection(P_requested_MW, self.soc)
        
        # 3. SoC update (Coulomb counting with efficiencies)
        if I_applied < 0.0:
            # Charging (I < 0) → SoC increases, scaled by η_ch
            delta_soc = -(I_applied * self.dt_seconds / self.Q_pack_C) * self.eta_ch
        elif I_applied > 0.0:
            # Discharging (I > 0) → SoC decreases, scaled by 1/η_dis
            delta_soc = -(I_applied * self.dt_seconds / self.Q_pack_C) / self.eta_dis
        else:  # idle (I == 0)
            delta_soc = 0.0

        # Update SoC (protection function should prevent violations)
        self.soc = self.soc + float(delta_soc)
        
        # 4. Reward calculation (profit + carbon penalty)
        idx = self.current_step
        price = float(self.power_price[idx])
        ci_t = float(self.ci_data[idx])

        # Energy traded (MWh): E = P [MW] * dt [h]
        E_MWh = P_applied_MW * self.dt_hours

        E_import_kWh = max(-E_MWh, 0) * 1000  # charging (energy imported from grid)
        E_export_kWh = max(E_MWh, 0) * 1000   # discharging (energy exported to grid)

        # Profit component
        if P_applied_MW > 0.0:      # discharge → sell
            profit = E_MWh * price
        elif P_applied_MW < 0.0:    # charge → buy
            profit = E_MWh * price
        else:
            profit = 0.0

        # Carbon penalty: λ (E_imports - E_exports) · CI_t
        # Charging (E_import > 0): positive penalty (penalizes carbon imports)
        # Discharging (E_export > 0): negative penalty (rewards carbon exports)
        carbon_penalty = self.lambda_ci * (E_import_kWh - E_export_kWh) * ci_t

        reward = profit - carbon_penalty
        self.reward = float(reward)

        # 5. Advance time & build next observation
        self.p_prev = float(P_applied_MW)
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()
        return obs, self.reward, terminated, truncated, {
            "P_requested_MW" : P_requested_MW,
            "P_applied_MW" : P_applied_MW
        }

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.soc = self.SoC_initial
        self.p_prev = 0.0
        self.reward = 0.0
        
        obs = self._get_obs()
        return obs, {}