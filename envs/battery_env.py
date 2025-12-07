import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium import Env, spaces
import env_config


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
        # P_step ≈ 0.5C * E_nominal ≈ 0.186 MW (can be refined)
        self.P_step_MW = env_config.P_step

        # Reward env_config: carbon weight λ
        self.lambda_ci = float(env_config.lambda_ci)
        
        # ECM R-int model parameter (internal resistance)
        R_cell_mOhm = 0.4  # mΩ per cell (from Product Specification Sheet)
        self.R_sys = (R_cell_mOhm / 1000.0) * self.N_cells  # Ω (pack resistance)

        # ========
        # 2. OCV LOOKUP TABLE (The DC OCV-SOC Curve from Spec Sheet @25°C)
        # ========
        self.ocv_soc_points = np.array(
            [0.00, 0.05, 0.10, 0.15, 0.20,
             0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70,
             0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
            dtype=np.float32,
        )
        self.ocv_cell_volts = np.array(
            [2.893, 3.182, 3.205, 3.230, 3.250,
             3.264, 3.283, 3.288, 3.288, 3.289, 
             3.290, 3.293, 3.303, 3.327, 3.329,
             3.329, 3.330, 3.330, 3.331, 3.332, 3.386],
            dtype=np.float32,
        )

        # ========
        # 3. LOAD DATA
        # ========
        merged_data = pd.read_parquet("data/merged_data.parquet")

        # Price datasets (£/MWh)
        self.ssp_data = merged_data["ssp"].to_numpy(dtype=np.float32)  # Settlement Sell Price
        self.sbp_data = merged_data["sbp"].to_numpy(dtype=np.float32)  # Settlement Buy Price
        
        # Carbon intensity dataset (gCO2/kWh)
        self.ci_data = merged_data["carbon_gco2_kwh"].to_numpy(dtype=np.float32)
        
        # Time encoding τ_t = hour * 2 + minute // 30
        self.timestamp = pd.to_datetime(merged_data["timestamp"])
        self.tau_data = (
            self.timestamp.dt.hour * 2
            + (self.timestamp.dt.minute // 30)
        ).to_numpy(dtype=np.int32)

        self.max_steps = len(self.ssp_data)

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
        self.action_space = gym.spaces.Discrete(3)  # {0,1,2} → {-1,0,+1}

        # ========
        # 6. OBSERVATION SPACE
        # ========
        # [SoC_t, price_t, CI_t, τ_t, P_{t-1}]
        low = np.array(
            [0.0, -300.0, 0.0, 0.0, -self.P_step_MW],
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 500.0, 1000.0, 49.0, self.P_step_MW],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # =========
    # OCV + PHYSICS HELPERS
    # =========
    def _get_ocv_pack(self, soc: float) -> float:
        """Get pack open-circuit voltage from SoC using lookup table."""
        soc_clipped = float(
            np.clip(soc, self.ocv_soc_points[0], self.ocv_soc_points[-1])
        )
        v_cell = np.interp(soc_clipped, self.ocv_soc_points, self.ocv_cell_volts)
        return float(v_cell * self.N_cells)

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
    
    def _clamp_power_by_soc_limits(self, P_grid_MW: float, soc: float) -> float:
        """
        Protection function: clamp power to prevent overcharging/overdischarging.
        
        Prevents SoC from exceeding [SoC_min, SoC_max] by limiting power command.
        Uses simplified energy-based calculation to estimate maximum safe power.
        
        Parameters
        ----------
        P_grid_MW : float
            Requested grid-side power (MW). P > 0 = discharge, P < 0 = charge
        soc : float
            Current state of charge (fraction)
        
        Returns
        -------
        float
            Clamped power (MW) that won't violate SoC limits
        """
        if P_grid_MW == 0.0:
            return 0.0
        
        # Get OCV for current SoC to estimate pack energy capacity
        V_oc_pack = self._get_ocv_pack(soc)
        E_nominal_MWh = self.E_nominal / 1000.0
        
        if P_grid_MW < 0.0:  # Charging
            # Maximum SoC increase allowed
            soc_room = self.SoC_max - soc
            if soc_room <= 0.0:
                return 0.0  # Already at max, cannot charge
            
            # Maximum energy we can add (MWh)
            E_room_MWh = soc_room * E_nominal_MWh
            
            # Maximum charge power that fits in available room
            # Account for charge efficiency: energy stored = P * dt * eta_ch
            P_max_charge_MW = E_room_MWh / (self.eta_ch * self.dt_hours)
            
            # Clamp: P_grid_MW is negative, so we want max(negative, -positive) = less negative
            P_clamped = max(P_grid_MW, -P_max_charge_MW)
            
        else: #Discharging
            # Maximum SoC decrease allowed
            soc_available = soc - self.SoC_min
            if soc_available <= 0.0:
                return 0.0  # Already at min, cannot discharge
            
            # Maximum energy we can extract (MWh)
            E_available_MWh = soc_available * E_nominal_MWh
            
            # Maximum discharge power
            # Account for discharge efficiency: energy delivered = P * dt / eta_dis
            P_max_discharge_MW = (E_available_MWh * self.eta_dis) / self.dt_hours
            
            # Clamp to prevent overdischarging
            P_clamped = min(P_grid_MW, P_max_discharge_MW)
    
        return float(P_clamped)
    
    def _get_obs(self):
        """
        Observation: [SoC, price_indicator, CI, tau, P_prev]
        price_indicator uses SSP as a visible grid price feature.
        """
        idx = min(self.current_step, self.max_steps - 1)

        return np.array(
            [
                self.soc,
                self.ssp_data[idx],
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
        # 1. Map discrete action → direction {-1, 0, +1}
        action_map = {0: -1, 1: 0, 2: 1}
        if action_idx not in action_map:
            raise ValueError(f"Invalid action index {action_idx}, must be 0,1,2.")
        direction = action_map[action_idx]

        # Grid-side power in MW and W
        P_grid_MW_requested = direction * self.P_step_MW
        
        # Protection: clamp power to prevent overcharging/overdischarging
        P_grid_MW = self._clamp_power_by_soc_limits(P_grid_MW_requested, self.soc)
        P_grid_W = P_grid_MW * 1e6

        # 2. Physics: compute current from power and OCV
        V_oc_pack = self._get_ocv_pack(self.soc)
        current_I = self._compute_current_from_power(P_grid_W, V_oc_pack)

        # 3. SoC update (Coulomb counting with efficiencies)
        if current_I < 0.0:
            # Charging (I < 0) → SoC increases, scaled by η_ch
            delta_soc = -(current_I * self.dt_seconds / self.Q_pack_C) * self.eta_ch
        else:
            # Discharging or idle (I >= 0) → SoC decreases, scaled by 1/η_dis
            delta_soc = -(current_I * self.dt_seconds / self.Q_pack_C) / self.eta_dis

        # Update SoC (protection function should prevent violations, but clip as safety backup)
        # self.soc = float(np.clip(self.soc + delta_soc, self.SoC_min, self.SoC_max))

        # 4. Reward calculation (profit + carbon penalty)
        idx = self.current_step
        ssp = float(self.ssp_data[idx])
        sbp = float(self.sbp_data[idx])
        ci_t = float(self.ci_data[idx])

        # Energy traded (MWh): E = P [MW] * dt [h]
        E_MWh = P_grid_MW * self.dt_hours

        # Profit component
        if P_grid_MW > 0.0:      # discharge → sell at SSP
            profit = E_MWh * ssp
        elif P_grid_MW < 0.0:    # charge → buy at SBP
            profit = E_MWh * sbp
        else:
            profit = 0.0

        # Carbon penalty: -λ (P_t · CI_t · Δt)  (literal MDP form)
        carbon_penalty = -self.lambda_ci * (P_grid_MW * ci_t * self.dt_hours)

        reward = profit + carbon_penalty
        self.reward = float(reward)

        # 5. Advance time & build next observation
        self.p_prev = P_grid_MW
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()
        return obs, self.reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.soc = self.SoC_initial
        self.p_prev = 0.0
        self.reward = 0.0
        
        obs = self._get_obs()
        return obs, {}