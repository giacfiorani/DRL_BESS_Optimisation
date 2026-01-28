import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium import Env, spaces
import env_config
from pathlib import Path



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
        env_config,
        publish_hour: int = 12,
        episode_days: int = 30,
        randomize_init_soc: bool = True,
        init_soc_low: float = 0.3,   # if None -> use SoC_min
        init_soc_high: float = 0.7,  # if None -> use SoC_max
        seed: int | None = None,
    ):
        # env initialisation
        super().__init__()

        # -----------------
        # Config
        # -----------------
        self.publish_hour = int(publish_hour)
        self.episode_days = int(episode_days)
        self.randomize_init_soc = bool(randomize_init_soc)
        self.np_random = np.random.default_rng(seed)

        # ========
        # 1. HARDWARE & MDP PARAMETERS
        # ========
        # CATL EnerOne 1P416S
        self.N_cells = env_config.N_cells
        # Pack charge (Coulombs): 280 Ah * 3600 s/h
        self.Q_pack_C = env_config.Q_cell_C # 1,008,000 C - Since 1P416S Q_cell = Q_pack
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

        # Reward config: carbon weight λ
        self.lambda_ci = float(env_config.lambda_ci)
        
        # ECM R-int model parameter (internal resistance)
        self. R_cell_mOhm = env_config.R_cell_mOhm  # mΩ per cell
        self.R_sys = (self.R_cell_mOhm / 1000.0) * self.N_cells  # Ω (pack resistance)

        # Init SoC range defaults to your operational limits
        self.init_soc_low = float(self.SoC_min if init_soc_low is None else init_soc_low)
        self.init_soc_high = float(self.SoC_max if init_soc_high is None else init_soc_high)

        # ========
        # Read from OCV Lookup Table and Interpolate to get OCV-SOC Curve
        # ========
        self.ocv_soc_points, self.ocv_cell_volts = env_config.ocv_lookup_table()
        
        # ========
        # 2. LOAD DATA
        # ========
        # trade_ts: time at which DA information becomes available (trade day D-1)
        # delivery_ts: physical electricity delivery half-hour (delivery day D)
        # The agent steps forward in delivery_ts, not trade_ts

        ROOT_DIR = Path(__file__).resolve().parents[1]
        DATA_PATH = ROOT_DIR / "data" / "training_data.parquet"
        df = pd.read_parquet(DATA_PATH).copy()

        df["trade_ts"] = pd.to_datetime(df["trade_ts"]) # Trade Day Timestamp
        df["delivery_ts"] = pd.to_datetime(df["delivery_ts"])  # Delivery Day Timestamp
        df["trade_date"] = pd.to_datetime(df["trade_date"])  # Trade day (auction day)
        df["delivery_date"] = pd.to_datetime(df["delivery_date"])  # Delivery day for each row (the day electricity is delivered)

        #index on delivery timestamp 
        df = df.sort_values("delivery_ts").reset_index(drop=True)

        # DA publish timestamp:
        # DA prices for delivery day D become available on trade day D-1
        # at a fixed publish hour (e.g. 12:00).
        df["da_publish_ts"] = df["trade_date"] + pd.Timedelta(hours=self.publish_hour)

        self.df = df

        # Arrays for fast access
        self.trade_ts = df["trade_ts"].to_numpy(dtype="datetime64[ns]")
        self.delivery_ts = df["delivery_ts"].to_numpy(dtype="datetime64[ns]")
        self.trade_date = df["trade_date"].to_numpy(dtype="datetime64[ns]")
        self.delivery_date = df["delivery_date"].to_numpy(dtype="datetime64[ns]")
        self.da_publish_ts = df["da_publish_ts"].to_numpy(dtype="datetime64[ns]") 
        self.tau = df["tau"].to_numpy(dtype=np.int32)  # 1..48
        self.id_price = df["mid_price_gbp_mwh"].to_numpy(dtype=np.float32) # intraday prices data
        self.da_price = df["da_price_gbp_mwh"].to_numpy(dtype=np.float32) # day ahead prices data
        self.ci = df["carbon_gco2_kwh"].to_numpy(dtype=np.float32) # carbon intensity data

        #====
        # Group environment episodes by DELIVERY day (not trade day):
        # • DA commitments apply to an entire delivery day (48 SPs)
        # • MID prices and carbon intensity are realised at delivery time
        # • Each episode day = one physical delivery day
        #===
        day_to_idx = df.groupby("delivery_date").indices
        valid_days = []
        day_indices = {}

        #loop through each day 
        for d, idxs in day_to_idx.items():
            idxs = np.array(sorted(idxs), dtype=np.int64)
            
            # Must be exactly 48 rows
            if len(idxs) != 48:
                continue
            
            # Must contain tau 1..48
            taus = df.loc[idxs, "tau"].to_numpy()
            if set(taus.tolist()) != set(range(1, 49)):
                continue
            
            # Must be exactly 30-min cadence
            ts_day = df.loc[idxs, "delivery_ts"].to_numpy(dtype="datetime64[ns]")
            deltas = np.diff(ts_day).astype("timedelta64[m]").astype(int)
            if not np.all(deltas == 30):
                continue
            #store the day 
            d64 = np.datetime64(pd.Timestamp(d).floor("D"))
            valid_days.append(d64)
            day_indices[d64] = idxs
        
        #ensure dataset has valid complete days
        self.valid_days = np.array(sorted(valid_days), dtype="datetime64[ns]")
        self.day_indices = day_indices
        if len(self.valid_days) == 0:
            raise ValueError("No valid 48-slot trade days found in training_data.parquet.")

        # Map day -> position in valid_days for fast “next day” stepping
        self.day_pos = {d: i for i, d in enumerate(self.valid_days)}

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
        low_main  = np.array([0.0, -300.0, 0.0, 1.0, -self.P_max_MW, 0.0, -self.P_max_MW], dtype=np.float32)
        high_main = np.array([1.0, 2500.0, 1000.0, 48.0,  self.P_max_MW, 1.0,  self.P_max_MW], dtype=np.float32)

        # these define the min/max limits for each of the 48 entries of the “tomorrow DA price curve” that are included in the observation.
        low_curve = np.full((48,), -300.0, dtype=np.float32)
        high_curve = np.full((48,), 2500.0, dtype=np.float32)

        # Total Obs length = 48 + 7 = 55 floats
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low_main, low_curve]),
            high=np.concatenate([high_main, high_curve]),
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

    # ========
    # DA availability + tomorrow curve
    # ========

    # check if Day Ahead (DA) prices are published
    def _da_available_now(self, idx: int) -> bool:
        return self.trade_ts[idx] >= self.da_publish_ts[idx]

    # Retrieve DA prices for all 48 sp
    def _get_da_curve_for_delivery_day(self, delivery_day: np.datetime64) -> np.ndarray:
        idxs = self.day_indices[delivery_day]
        return self.da_price[idxs].astype(np.float32)

    def _get_tomorrow_da_curve(self) -> np.ndarray:
        pos = self.day_pos[self.current_day]
        if pos + 1 >= len(self.valid_days):
            return np.zeros(48, dtype=np.float32)
        tomorrow_day = self.valid_days[pos + 1]
        return self._get_da_curve_for_delivery_day(tomorrow_day)
    
    # Observation consists of:
    # • current SoC
    # • intraday (MID) price for this delivery half-hour
    # • carbon intensity
    # • settlement period index (tau)
    # • previous applied power
    # • DA availability flag
    # • planned DA power for this slot (if any)
    # • full DA price curve for tomorrow (48 values), visible only after publish
    def _get_obs(self) -> np.ndarray:

        idx = int(self.current_day_idxs[self.slot0]) # row index of current settlement period

        tau0 = int(self.tau[idx]) - 1
        planned_idx = int(self.today_plan[tau0]) # convert plan index to Power
        planned_power_now = float(self.power_levels[planned_idx]) if planned_idx >= 0 else 0.0

        tau_now = float(self.tau[idx])
        da_avail = 1.0 if self._da_available_now(idx) else 0.0 # 1 if DA can be used for planningn right now
        
        #only reveal tommorrow's DA curve if DA is published and tommorrow exists in database
        pos = self.day_pos[self.current_day]
        tomorrow_exists = (pos + 1) < len(self.valid_days)
        tomorrow_curve = self._get_tomorrow_da_curve() if (da_avail == 1.0 and tomorrow_exists) else np.zeros(48, dtype=np.float32)

        # observation = main state + optional tommorrow curve 
        main = np.array(
            [self.soc, self.id_price[idx], self.ci[idx], tau_now, self.p_prev, da_avail, planned_power_now],
            dtype=np.float32,
        )
        return np.concatenate([main, tomorrow_curve], axis=0)

    
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

        # advance to next trade day in dataset
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

        # --- current row first ---
        idx = int(self.current_day_idxs[self.slot0])
        now_ts = self.trade_ts[idx] # 1..48
        tau0 = int(self.tau[idx]) - 1  # 0..47

        # --- availability / tomorrow existence ---
        da_avail = self._da_available_now(idx)
        pos = self.day_pos[self.current_day]
        tomorrow_exists = (pos + 1) < len(self.valid_days)

        # --- planned index for this delivery slot ---
        planned_idx = int(self.today_plan[tau0])

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
        ci_now       = float(self.ci[idx])

        # --- DA + ID revenue ---
        # energy
        E_plan_MWh = P_plan_MW * self.dt_hours
        E_dev_MWh  = P_dev_MW  * self.dt_hours

        R_DA = E_plan_MWh * da_price_now # revenue from planned action
        R_ID = E_dev_MWh  * id_price_now # revenue from deviation action

        # --- carbon penalty on actual ---
        E_act_MWh = P_act_MW * self.dt_hours
        E_import_kWh = max(-E_act_MWh, 0.0) * 1000.0
        E_export_kWh = max(E_act_MWh, 0.0) * 1000.0
        carbon_penalty = self.lambda_ci * (E_import_kWh - E_export_kWh) * ci_now
        profit = R_DA + R_ID

        # Reward decomposition (Plan + Adjust):
        #
        # • DA revenue:
        #   Energy committed in the DA plan is settled at the DA price.
        #
        # • ID revenue:
        #   Any deviation between actual dispatch and DA plan is settled
        #   at the intraday (MID) price.
        #
        # • Carbon penalty:
        #   Applied to actual energy imported/exported, based on realised
        #   carbon intensity.
        #
        # This mirrors a realistic DA commitment with intraday rebalancing,
        # while physical infeasibility is prevented by SoC safety shielding.
        reward = profit - carbon_penalty

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
            "trade_ts": str(pd.Timestamp(now_ts)),
            "trade_date": str(pd.Timestamp(self.trade_date[idx])),
            "delivery_date": str(pd.Timestamp(self.delivery_date[idx])),
            "tau": int(tau0 + 1),
            "decision_ts": str(pd.Timestamp(self.trade_ts[idx])),
		    "delivery_ts": str(pd.Timestamp(self.delivery_ts[idx])),

            "dispatch_idx_exec": int(dispatch_idx_eff),
            "dispatch_idx_agent": int(action[0]),
            "plan_idx_agent": int(action[1]),
            "planned_idx_today" : int(planned_idx),
            "plan_slot_agent": int(plan_slot),
            "tomorrow_plan_value_written": int(self.tomorrow_plan[plan_slot]) if da_avail else -999,

            "P_req_MW": float(P_req_MW),
            "P_applied_MW": float(P_applied_MW),
            "delta_soc": float(delta_soc),

            "id_price_now": float(id_price_now),
            "da_price_now": float(da_price_now),
            "ci_now": float(ci_now),
            "profit": float(profit),
            "carbon_penalty": float(carbon_penalty),

            "da_available": bool(da_avail),
            "days_done": int(self.days_done),
            "soc": float(self.soc),
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

        # Choose starting day (enough room to run episode_days)
        if options is not None and options.get("delivery_day") is not None:
            start_day = np.datetime64(pd.to_datetime(options["delivery_day"]).floor("D"))
            if start_day not in set(self.valid_days.tolist()):
                raise ValueError("Requested delivery_day not in valid_days.")
            start_pos = self.day_pos[start_day]
        else:
            max_start = max(len(self.valid_days) - self.episode_days, 1)
            start_pos = int(self.np_random.integers(0, max_start))

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