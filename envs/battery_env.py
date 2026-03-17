import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium import Env, spaces
from pathlib import Path
from envs.degradation import ThroughputDegradation

class BatteryEnv(Env):

    """
    BatteryEnv — Day-Ahead Planning + Intraday Rebalancing Environment

    This environment models a grid-scale battery operating in the UK power market
    with two market layers:

    • Day-Ahead (DA): a forward commitment schedule for the next delivery day,
    published on trade day D-1 after a fixed publish time.
    • Intraday (ID / MID): a short-term market used to rebalance deviations
    during the actual delivery day.

    The agent:
    • builds a DA commitment for tomorrow once the DA curve is published,
    • executes real-time dispatch during delivery,
    • earns DA revenue on committed energy,
    • earns/pays MID revenue on deviations from the DA plan,
    • is physically constrained by battery SoC via a safety shield.

    Time resolution: 30-minute settlement periods (UK SPs).
    One environment step = one delivery half-hour.
    """
    
    metadata = {"render_modes": []}

    def __init__(
        self,
        config,
        publish_hour: int = 12,
        episode_days: int = 7,
        randomize_init_soc: bool = True,
        randomize_start: bool = True,
        lambda_ci: float | None = None,
        init_soc_low: float = 0.3,
        init_soc_high: float = 0.7,
        seed: int | None = 42,
        split: str = "train",       # "train" | "val" | "test"
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        # env initialisation
        super().__init__()

        # -----------------
        # Config
        # -----------------
        self.publish_hour = int(publish_hour)
        self.episode_days = int(episode_days)
        self.randomize_init_soc = bool(randomize_init_soc)
        self.randomize_start    = bool(randomize_start)
        self.np_random = np.random.default_rng(seed)

        # ========
        # 1. HARDWARE & MDP PARAMETERS
        # ========
        # CATL EnerOne 1P416S
        self.N_cells = config.N_cells
        # Pack charge (Coulombs): 280 Ah * 3600 s/h
        self.Q_pack_C = config.Q_cell_C # 1,008,000 C - Since 1P416S Q_cell = Q_pack
        self.V_nominal = config.V_nominal  # V
        self.E_nominal = config.E_nominal  # kWh

        # Operational limits (from your MDP)
        self.SoC_min = config.SoC_min
        self.SoC_max = config.SoC_max
        self.SoC_initial = float(config.SoC_initial)

        # Time step: 30 minutes
        self.dt_hours = config.dt
        self.dt_seconds = self.dt_hours * 3600.0

        # Efficiencies
        self.eta_ch = config.eff_ch   # charge efficiency
        self.eta_dis = config.eff_dis    # discharge efficiency

        # Power / action scaling
        self.P_max_MW = config.P_max_MW
        self.n_power_levels = config.n_power_levels

        # Reward config: carbon weight λ
        self.lambda_ci = lambda_ci
        
        # ECM R-int model parameter (internal resistance)
        self.R_cell_mOhm = config.R_cell_mOhm  # mΩ per cell
        self.R_sys = (self.R_cell_mOhm / 1000.0) * self.N_cells  # Ω (pack resistance)

        # Init SoC range defaults to your operational limits
        self.init_soc_low = float(self.SoC_min if init_soc_low is None else init_soc_low)
        self.init_soc_high = float(self.SoC_max if init_soc_high is None else init_soc_high)

        #Scaling Factors for Profit and Carbon Penalty rewards
        self.profit_scale = float(config.S_profit)
        self.carbon_scale = float(config.S_carbon_gbp)
        self.price_scale = float(config.S_price)
        self.ci_scale = float(config.S_ci)

        #Degradation parameters
        self.deg_kappa = float(config.deg_kappa)

        # Instantiate Degradation Model — Cortés-Arcos et al. (2020) Eq. 23
        self.degradation_model = ThroughputDegradation(self.deg_kappa)

        # ========
        # Read from OCV Lookup Table and Interpolate to get OCV-SOC Curve
        # ========
        self.ocv_soc_points, self.ocv_cell_volts = config.ocv_lookup_table()
        
        # ========
        # 2. LOAD DATA
        # ========
        # trade_ts: time at which DA information becomes available (trade day D-1)
        # delivery_ts: physical electricity delivery half-hour (delivery day D)
        # The agent steps forward in delivery_ts, not trade_ts

        ROOT_DIR = Path(__file__).resolve().parents[1]
        DATA_PATH = ROOT_DIR / "data" / "data.parquet"
        df = pd.read_parquet(DATA_PATH).copy()

        df["trade_ts"] = pd.to_datetime(df["trade_ts"])
        df["delivery_ts"] = pd.to_datetime(df["delivery_ts"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["delivery_date"] = pd.to_datetime(df["delivery_date"]).dt.floor("D")

        # Index on delivery timestamp
        df = df.sort_values("delivery_ts").reset_index(drop=True)

        self.df = df

        # Arrays for fast access
        self.trade_ts = df["trade_ts"].to_numpy(dtype="datetime64[ns]")
        self.delivery_ts = df["delivery_ts"].to_numpy(dtype="datetime64[ns]")
        self.trade_date = df["trade_date"].to_numpy(dtype="datetime64[ns]")
        self.delivery_date = df["delivery_date"].to_numpy(dtype="datetime64[ns]")       
        self.tau = df["tau"].to_numpy(dtype=np.int32)  # 1..48
        self.id_price = df["mid_price_gbp_mwh"].to_numpy(dtype=np.float32) # intraday prices data
        self.da_price = df["da_price_gbp_mwh"].to_numpy(dtype=np.float32) # day ahead prices data
        self.ci = df["ci_actual_gco2_kwh"].to_numpy(dtype=np.float32) # carbon intensity data (actual)
        self.ci_forecast = df["ci_forecast_gco2_kwh"].to_numpy(dtype=np.float32) # forecaste Carbon intensity data
        self.mef = df["mef_gco2_kwh"].to_numpy(dtype=np.float32) # Marginal Emissions Factor data
        self.carbon_price = df["uka_gbp_tco2"].to_numpy(dtype=np.float32) # carbon price


        # Precomputed Datetime Arrays
        self.tomorrow_day_arr = self.delivery_date.astype('datetime64[D]') + np.timedelta64(1, 'D')
        publish_ts_arr = self.tomorrow_day_arr - np.timedelta64(1,'D') + np.timedelta64(self.publish_hour, 'h')
        self.da_avail_arr = (self.delivery_ts >= publish_ts_arr)

        #====
        # Group environment episodes by DELIVERY day (not trade day):
        # • DA commitments apply to an entire delivery day (48 SPs)
        # • MID prices and carbon intensity are realised at delivery time
        # • Each episode day = one physical delivery day
        #===
        # inside __init__ after grouping
        day_to_idx = df.groupby("delivery_date").indices

        valid_days = []
        day_indices = {}

        for d, idxs in day_to_idx.items():
            idxs = np.array(sorted(idxs), dtype=np.int64)

            if len(idxs) != 48:
                continue

            taus = df.loc[idxs, "tau"].to_numpy()
            if set(taus.tolist()) != set(range(1, 49)):
                continue

            ts_day = df.loc[idxs, "delivery_ts"].to_numpy(dtype="datetime64[ns]")
            deltas = np.diff(ts_day).astype("timedelta64[m]").astype(int)
            if not np.all(deltas == 30):
                continue

            # CANONICAL KEY: datetime64[D]
            d64 = np.datetime64(pd.Timestamp(d).date(), "D")
            valid_days.append(d64)
            day_indices[d64] = idxs

        self.valid_days = np.array(sorted(valid_days), dtype="datetime64[D]")
        self.day_indices = day_indices
        self.day_pos = {d: i for i, d in enumerate(self.valid_days)}

        # ========
        # Chronological 70 / 15 / 15 train / val / test split
        # Never shuffle — this is time-series data.
        # ========
        n = len(self.valid_days)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        self.valid_days_train = self.valid_days[:n_train]
        self.valid_days_val   = self.valid_days[n_train : n_train + n_val]
        self.valid_days_test  = self.valid_days[n_train + n_val :]

        _split_map = {"train": self.valid_days_train,
                      "val":   self.valid_days_val,
                      "test":  self.valid_days_test}
        if split not in _split_map:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {split!r}")
        self.split = split
        self.active_valid_days = _split_map[split]

        # ========
        # 3) ACTION SPACE
        # ========
        # Action space:
        # [dispatch_idx, plan_idx, plan_slot]
        #
        # dispatch_idx:
        #   Real-time physical dispatch request for the current delivery SP.
        #
        # plan_idx:
        #   Power level to commit in the DA plan for a future delivery day.
        #
        # plan_slot:
        #   Target settlement period (0..47) of tomorrow's DA plan to update.
        #
        # This allows the agent to gradually construct a full 48-slot DA schedule
        # after the DA curve is published, while still dispatching in real time.

        self.power_levels = np.linspace(-self.P_max_MW, self.P_max_MW, self.n_power_levels).astype(np.float32)
        
        # MultiDiscrete action: [dispatch_idx, plan_idx, plan_slot]
        self.action_space = gym.spaces.MultiDiscrete([self.n_power_levels, self.n_power_levels, 48])
        
        # ========
        # 4) OBSERVATION SPACE
        # ========
        # obs = [SoC, spot_price_now, CI_now, tau_now, P_prev, da_available] + tomorrow_DA_curve_48
        low_main  = np.array([0.0, -500.0, 0.0, 1.0, -self.P_max_MW, 0.0, -self.P_max_MW], dtype=np.float32)
        high_main = np.array([1.0, 7000.0, 1000.0, 48.0,  self.P_max_MW, 1.0,  self.P_max_MW], dtype=np.float32)

        # these define the min/max limits for each of the 48 entries of the “tomorrow DA price curve” that are included in the observation.
        low_da_curve = np.full((48,), -500.0, dtype=np.float32)
        high_da_curve = np.full((48,), 7000.0, dtype=np.float32)

        # tomorrow CI curve bounds (gCO2/kWh)
        low_ci_da_curve  = np.full((48,), 0.0, dtype=np.float32)
        high_ci_da_curve = np.full((48,), 1000.0, dtype=np.float32)

        # Total Obs length = 48 + 7 + 48 = 103 floats
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low_main, low_da_curve, low_ci_da_curve]),
            high=np.concatenate([high_main, high_da_curve, high_ci_da_curve]),
            dtype=np.float32,
        )

        # ========
        # 5) INTERNAL ENV VARIABLES
        # ========
        self.soc = float(self.SoC_initial)
        self.p_prev = 0.0

        self.days_done = 0
        self.current_day = None
        self.current_day_idxs = None
        self.slot0 = 0  # 0..47

        # Rolling plans 
        self.today_plan = np.full(48, -1, dtype=np.int32)  # -1 means “no commitment”  - commitments for current day
        self.tomorrow_plan = np.full(48, -1, dtype=np.int32) # being built for next day
        

    # =========
    # OCV + PHYSICS HELPERS
    # =========
    def _get_ocv_pack(self, soc: float) -> float:
        """Get pack open-circuit voltage from SoC using lookup table."""
        soc = float(np.clip(soc, 0.0, 1.0)) #safety net to keep SoC between 0 and 1

        #interpolate given SoC
        ocv_cell = np.interp(
            soc, self.ocv_soc_points, self.ocv_cell_volts
        )
        ocv_pack = ocv_cell * self.N_cells
        return float(ocv_pack)


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
    # Safety shield:
    # Physical infeasible actions are clipped to SoC limits.
    # This avoids explicit imbalance settlement (SBP/SSP) while ensuring
    # the agent never violates battery constraints.
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

        # The SoC update rule is:
        # charge (I<0): ΔSoC = -(I dt / Q) * η_ch
        # discharge(I>0): ΔSoC = -(I dt / Q) * (1 / η_dis)
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

    # ========
    # DA availability + tomorrow curve
    # ========
    
    def _get_tomorrow_day_from_delivery_ts(self, idx: int) -> np.datetime64:
        return self.tomorrow_day_arr[idx]

    # check if Day Ahead (DA) prices are published
    def _da_available_now(self, idx: int) -> bool:
        """
        At decision time = current delivery_ts, do we have tomorrow's DA curve?
        DA for delivery day D is published at (D-1) publish_hour.
        """
        return bool(self.da_avail_arr[idx])

    # Retrieve DA prices for all 48 sp
    def _get_da_curve_for_delivery_day(self, delivery_day: np.datetime64) -> np.ndarray:
        idxs = self.day_indices[delivery_day]
        return self.da_price[idxs].astype(np.float32)   

    def _get_tomorrow_da_curve(self, idx: int) -> np.ndarray:
        tomorrow_day = self._get_tomorrow_day_from_delivery_ts(idx)
        if tomorrow_day not in self.day_indices:
            return np.zeros(48, dtype=np.float32)
        return self._get_da_curve_for_delivery_day(tomorrow_day)

    def _get_ci_curve_for_delivery_day(self, delivery_day: np.datetime64) -> np.ndarray:
        idxs = self.day_indices[delivery_day]
        return self.ci_forecast[idxs].astype(np.float32)
    
    def _get_tomorrow_ci_curve(self, idx: int) -> np.ndarray:
        tomorrow_day = self._get_tomorrow_day_from_delivery_ts(idx)
        if tomorrow_day not in self.day_indices:
            return np.zeros(48, dtype=np.float32)
        return self._get_ci_curve_for_delivery_day(tomorrow_day)
    
    # Observation consists of:
    # • current SoC
    # • intraday (MID) price for this delivery half-hour
    # • carbon intensity
    # • settlement period index (tau)
    # • previous applied power
    # • DA availability flag
    # • planned DA power for this slot (if any)
    # • full DA price curve for tomorrow (48 values), visible only after publish
    # • full Forecasted Carbon Intensity curve for tomorrow (48 values), visible only after publish
    
    def _get_obs(self) -> np.ndarray:

        idx = int(self.current_day_idxs[self.slot0]) # row index of current settlement period
        tau0 = int(self.tau[idx]) - 1

        # convert plan index to Power
        planned_idx = int(self.today_plan[tau0])
        if planned_idx < 0 or planned_idx >= self.n_power_levels:
            planned_idx = -1 
        planned_power_now = float(self.power_levels[planned_idx]) if planned_idx >= 0 else 0.0

        # 1. The Indicator Flag: Checks if it is past the 12:00 publish time
        da_avail = 1.0 if self._da_available_now(idx) else 0 # 1 if DA can be used for planningn right now

        # 2. The Zero-Masking Logic
        # tomorrow curves only visible after publish, and only if tomorrow exists
        if da_avail == 1.0:
            tomorrow_da = self._get_tomorrow_da_curve(idx)
            tomorrow_ci = self._get_tomorrow_ci_curve(idx)
        else:
            tomorrow_da = np.zeros(48, dtype=np.float32)
            tomorrow_ci = np.zeros(48, dtype=np.float32)
        
        # --- Normalise ---
        soc_n           = float(self.soc)                                        # [0,1]
        price_n         = float(np.tanh(self.id_price[idx] / self.price_scale))  # [-1,1]
        ci_n            = float(np.tanh(self.ci[idx] / self.ci_scale))           # [-1,1]
        tau_n           = float(self.tau[idx]) / 48.0                            # [0,1]
        p_prev_n        = float(self.p_prev / self.P_max_MW)                     # [-1,1]
        da_avail_n      = da_avail                                               # {0,1}
        planned_power_n = float(planned_power_now / self.P_max_MW)               # [-1,1]

        tomorrow_da_n = np.tanh(tomorrow_da / self.price_scale).astype(np.float32)
        tomorrow_ci_n = np.tanh(tomorrow_ci / self.ci_scale).astype(np.float32)
            
        # 3. Construct the fixed-size state vector
        main = np.array(
            [soc_n, price_n, ci_n, tau_n, p_prev_n, da_avail_n, planned_power_n],
            dtype=np.float32,
        )
        return np.concatenate([main, tomorrow_da_n, tomorrow_ci_n], axis=0)

    
    # -----------------
    # Time advance
    # -----------------
    def _advance_one_slot(self):
        self.slot0 += 1
        if self.slot0 < 48:
            return True  # still same day

        # day rollover
        self.slot0 = 0
        self.days_done += 1

        # shift plans: tomorrow plan becomes today's commitment
        self.today_plan = self.tomorrow_plan.copy()
        self.tomorrow_plan[:] = -1 # reset tomorrow plan buffer

        # advance to next delivery day in dataset
        pos = self.day_pos[self.current_day]
        nxt_pos = pos + 1
        if nxt_pos >= len(self.valid_days):
            return False  # dataset end

        self.current_day = self.valid_days[nxt_pos]
        self.current_day_idxs = self.day_indices[self.current_day]
        return True


    # =========
    # GYM API
    # =========
    def step(self, action):
        # actions
        dispatch_idx = int(action[0])
        plan_idx     = int(action[1])
        plan_slot    = int(action[2])

        dispatch_idx = int(np.clip(dispatch_idx, 0, self.n_power_levels - 1))
        plan_idx     = int(np.clip(plan_idx,     0, self.n_power_levels - 1))
        plan_slot    = int(np.clip(plan_slot,    0, 47))

        # --- current row first ---
        idx = int(self.current_day_idxs[self.slot0])
        decision_ts = self.delivery_ts[idx] # 1..48
        tau0 = int(self.tau[idx]) - 1  # 0..47

        planned_idx = int(self.today_plan[tau0])
        if planned_idx < 0 or planned_idx >= self.n_power_levels:
            planned_idx = -1

        # --- availability / tomorrow existence ---
        da_avail = self._da_available_now(idx)
        tomorrow_day = self._get_tomorrow_day_from_delivery_ts(idx)
        tomorrow_exists = tomorrow_day in self.day_indices

        # --- execute dispatch (agent can deviate) ---
        dispatch_idx_eff = dispatch_idx
        P_req_MW = float(self.power_levels[dispatch_idx_eff])
        P_applied_MW, I_applied, V_oc_pack = self._apply_soc_protection(P_req_MW, self.soc)

        
        # --- SoC update ---
        if I_applied < 0.0:
            delta_soc = -(I_applied * self.dt_seconds / self.Q_pack_C) * self.eta_ch
        elif I_applied > 0.0:
            delta_soc = -(I_applied * self.dt_seconds / self.Q_pack_C) / self.eta_dis
        else:
            delta_soc = 0.0
        self.soc = float(self.soc + delta_soc)

        # --- planned vs actual power (for reward split) ---
        P_plan_MW = float(self.power_levels[planned_idx]) if planned_idx >= 0 else 0.0
        P_act_MW  = float(P_applied_MW)
        P_dev_MW  = P_act_MW - P_plan_MW #deviations of agent from initial plan

        da_price_now = float(self.da_price[idx])
        id_price_now = float(self.id_price[idx])
        carbon_price_now = float(self.carbon_price[idx])
        ci_now = float(self.ci[idx])
        mef_now = float(self.mef[idx])


        # --- DA + ID revenue ---
        # energy
        E_plan_MWh = P_plan_MW * self.dt_hours
        E_dev_MWh  = P_dev_MW  * self.dt_hours

        # DEGRADATION COST — Cortés-Arcos et al. (2020) Eq. 23
        deg_cost = self.degradation_model.calculate_costs(P_act_MW, dt_hours=self.dt_hours)

        # Raw profits (£)
        R_DA = E_plan_MWh * da_price_now
        R_ID = E_dev_MWh  * id_price_now
        R_total = R_DA + R_ID - deg_cost

        # Carbon cashflow (£) using UKA price
        E_act_MWh = P_act_MW * self.dt_hours
        E_import_kWh = max(-E_act_MWh, 0.0) * 1000.0
        E_export_kWh = max(E_act_MWh, 0.0) * 1000.0

        net_tCO2 = (E_import_kWh * ci_now - E_export_kWh * mef_now) / 1e6  # g -> tCO2
        carbon_cashflow_gbp = carbon_price_now * net_tCO2

        # Normalisation
        profit_norm = np.tanh(R_total / self.profit_scale)
        carbon_norm = np.tanh(carbon_cashflow_gbp / self.carbon_scale)

        

        # Reward decomposition (Plan + Adjust):
        #
        # • DA revenue:
        #   Energy committed in the DA plan is settled at the DA price.
        #
        # • ID revenue:
        #   Any deviation between actual dispatch and DA plan is settled
        #   at the intraday (MID) price.
        #
        # • Degradation cost:
        #   Subtracted directly from the gross market revenue to calculate Net Profit.
        #   Modeled as a SoC-weighted marginal cost of energy throughput.
        #
        # • Carbon penalty:
        #   Applied to actual energy imported/exported, based on realised
        #   carbon intensity.
        #
        # This mirrors a realistic DA commitment with intraday rebalancing,
        # while physical infeasibility is prevented by SoC safety shielding.
        reward = profit_norm - self.lambda_ci * carbon_norm

        self.p_prev = float(P_applied_MW)

        # Tomorrow plan update:
        # • The agent may update tomorrow's DA plan only after DA publish time.
        # • The plan applies to the NEXT delivery day.
        # • The current delivery day's plan is frozen and cannot be changed.
        if da_avail and tomorrow_exists:
            self.tomorrow_plan[plan_slot] = int(plan_idx)

        # 3) ADVANCE TIME
        # At delivery-day rollover:
        # • tomorrow_plan becomes today_plan (fixed DA commitment)
        # • a fresh tomorrow_plan buffer is initialised
        ok = self._advance_one_slot()

        terminated = (self.days_done >= self.episode_days)
        truncated = (not ok) and (not terminated)

        obs = self._get_obs()

        info = {
           # --- Essential Timestamps / Indices ---
            "delivery_ts": str(self.delivery_ts[idx]),
            "tau": int(tau0 + 1),
            "idx": int(idx),
            "days_done": int(self.days_done),

            # --- Agent Action Tracking ---
            "dispatch_idx_agent": int(action[0]),
            "plan_idx_agent": int(action[1]),
            "plan_slot_agent": int(plan_slot),
            "planned_idx_today" : int(planned_idx),
            "tomorrow_plan_value_written": int(self.tomorrow_plan[plan_slot]) if da_avail else -999,
            "da_available": bool(da_avail),

            # --- Physical Battery Physics ---
            "P_req_MW": float(P_req_MW),
            "P_planned_MW" : float(P_plan_MW),     # (Also acts as DA_dispatched_MW)
            "P_dev_MW" : float(P_dev_MW),          # (Also acts as ID_dispatched_MW)
            "P_applied_MW": float(P_applied_MW),
            "soc": float(self.soc),
            "delta_soc": float(delta_soc),

            # --- Market & Environment Variables ---
            "da_price_now": float(da_price_now),
            "id_price_now": float(id_price_now),
            "ci_now": float(ci_now),
            "mef_now": float(mef_now),
            "carbon_price_now": float(carbon_price_now),

            # --- Episode Accumulators (For TensorBoard) ---
            "Planned_Profit": float(R_DA),
            "Intraday_Profit": float(R_ID),
            "degradation_cost_gbp": float(deg_cost), 
            "carbon_cashflow": float(carbon_cashflow_gbp),
            "net_carbon_tCO2": float(net_tCO2),      

            # --- Neural Network Normalisation ---
            "profit_norm": float(profit_norm),
            "carbon_penalty_norm": float(carbon_norm),
        }

        return obs, float(reward), bool(terminated), bool(truncated), info
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # SoC reset at the start of an episode (training episodes) - so it doesnt become biased from starting at 0.5
        if self.randomize_init_soc:
            soc0 = float(self.np_random.uniform(self.init_soc_low, self.init_soc_high))
            self.soc = float(np.clip(soc0, self.SoC_min, self.SoC_max))
        else:
            self.soc = float(np.clip(self.SoC_initial, self.SoC_min, self.SoC_max))

        self.p_prev = 0.0
        self.days_done = 0

        # Choose starting day within the active split, with enough room for episode_days
        if options is not None and options.get("delivery_day") is not None:
            start_day = np.datetime64(pd.to_datetime(options["delivery_day"]).floor("D"))
            if start_day not in set(self.valid_days.tolist()):
                raise ValueError("Requested delivery_day not in valid_days.")
            start_pos = self.day_pos[start_day]
        else:
            if self.randomize_start:
                max_start = max(len(self.active_valid_days) - self.episode_days, 1)
                local_pos = int(self.np_random.integers(0, max_start))
            else:
                local_pos = 0  # fixed start — debugging/verification only
            start_pos = self.day_pos[self.active_valid_days[local_pos]]

        self.current_day = self.valid_days[start_pos]
        self.current_day_idxs = self.day_indices[self.current_day]
        self.slot0 = 0

        # Reset rolling plans
        self.today_plan[:] = -1
        self.tomorrow_plan[:] = -1

        obs = self._get_obs()
        info = {
            "start_day": str(self.current_day),
            "publish_hour": int(self.publish_hour),
            "episode_days": int(self.episode_days),
            "init_soc": float(self.soc),
        }
        return obs, info
    