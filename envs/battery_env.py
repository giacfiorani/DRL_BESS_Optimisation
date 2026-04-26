import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium import Env, spaces
from pathlib import Path
from envs.degradation import ThroughputDegradation
from envs.reward_scaling import compute_scales

class BatteryEnv(Env):
    """Day-Ahead Planning and Intraday Rebalancing environment for a grid-scale BESS.

    Models a battery operating in the UK power market across two market layers:

    - Day-Ahead (DA): a forward commitment schedule for the next delivery day,
      published on trade day D-1 after a fixed publish time.
    - Intraday (ID / MID): a short-term market used to rebalance deviations
      during the actual delivery day.

    The agent builds a DA commitment for tomorrow once the DA curve is published,
    executes real-time dispatch during delivery, earns DA revenue on committed
    energy, earns or pays MID revenue on deviations from the DA plan, and is
    physically constrained by battery SoC via a safety shield.

    Time resolution: 30-minute settlement periods (UK SPs).
    One environment step corresponds to one delivery half-hour.
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
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        train_start: str | None = None,
        continuous_action: bool = False,
        precomputed_scales: dict | None = None,
    ):
        """Initialise the BatteryEnv.

        Args:
            config: Environment configuration object (see env_config.py).
            publish_hour: Hour at which tomorrow's DA curve becomes available.
            episode_days: Number of delivery days per episode.
            randomize_init_soc: If True, sample initial SoC uniformly from
                [init_soc_low, init_soc_high] on each reset.
            randomize_start: If True, sample episode start day uniformly from
                the active split.
            lambda_ci: Carbon-intensity penalty weight. Overrides config if set.
            init_soc_low: Lower bound for initial SoC randomisation.
            init_soc_high: Upper bound for initial SoC randomisation.
            seed: NumPy random seed for reproducibility.
            split: Dataset partition to sample episodes from. One of
                ``"train"``, ``"val"``, or ``"test"``.
            train_ratio: Fraction of valid days allocated to the training split.
            val_ratio: Fraction of valid days allocated to the validation split.
            train_start: Optional ISO date string. Data before this date is
                excluded before computing reward scales and data splits.
            continuous_action: If True, use a 3-D continuous action space;
                otherwise use a MultiDiscrete discrete space.
            precomputed_scales: Optional pre-computed reward/observation scaling
                constants. If None, scales are derived from the training split.
        """
        super().__init__()

        self.publish_hour = int(publish_hour)
        self.episode_days = int(episode_days)
        self.randomize_init_soc = bool(randomize_init_soc)
        self.randomize_start    = bool(randomize_start)
        self.np_random = np.random.default_rng(seed)
        self.continuous_action = continuous_action
        self.train_start = train_start

        # Hardware and physics (scaled for N_cabinets parallel cabinets)
        self.N_cells = config.N_cells
        self.Q_pack_C = config.Q_cell_C
        self.V_nominal = config.V_nominal
        self.E_nominal = config.E_nominal
        self.E_max = config.E_max

        self.SoC_min = config.SoC_min
        self.SoC_max = config.SoC_max
        self.SoC_initial = float(config.SoC_initial)

        self.dt_hours = config.dt
        self.dt_seconds = self.dt_hours * 3600.0
        self.eta_ch = config.eff_ch
        self.eta_dis = config.eff_dis

        self.P_max_MW = config.P_max_MW
        self.n_power_levels = config.n_power_levels
        self.lambda_ci = lambda_ci

        # R-int ECM: series resistance of the parallel-connected cabinet stack.
        # R_cabinet = R_cell × N_cells (series), R_sys = R_cabinet / N_cabinets (parallel).
        self.R_cell_mOhm = config.R_cell_mOhm
        self.N_cabinets = config.N_cabinets
        R_cabinet_ohm = (self.R_cell_mOhm / 1000.0) * self.N_cells
        self.R_sys = R_cabinet_ohm / self.N_cabinets

        self.deg_kappa = float(config.deg_kappa)
        self.degradation_model = ThroughputDegradation(self.deg_kappa)

        self.alpha_thresh = config.alpha_thresh
        self.monthly_budget = config.monthly_budget
        self.scale_numeric = config.scale_numeric

        self.init_soc_low = float(init_soc_low if init_soc_low is not None else self.SoC_min)
        self.init_soc_high = float(init_soc_high if init_soc_high is not None else self.SoC_max)

        # Reward scaling and data loading
        if precomputed_scales is not None:
            scales = precomputed_scales
        else:
            scales = compute_scales(train_ratio=train_ratio, train_start=self.train_start)

        self.profit_scale = float(scales["S_profit"])
        self.carbon_scale = float(scales["S_carbon_gbp"])
        self.price_scale = float(scales["S_price"])
        self.ci_scale = float(scales["S_ci"])

        self.scale_universal = self.profit_scale / self.E_max

        ROOT_DIR = Path(__file__).resolve().parents[1]
        DATA_PATH = ROOT_DIR / "data" / "data.parquet"
        df = pd.read_parquet(DATA_PATH).copy()

        df["delivery_ts"] = pd.to_datetime(df["delivery_ts"])
        df["delivery_date"] = pd.to_datetime(df["delivery_date"]).dt.floor("D")
        df = df.sort_values("delivery_ts").reset_index(drop=True)

        # Map to NumPy arrays for high-speed index-based access during stepping.
        self.id_price    = df["mid_price_gbp_mwh"].to_numpy(dtype=np.float32)
        self.da_price    = df["da_price_gbp_mwh"].to_numpy(dtype=np.float32)
        self.ci          = df["ci_actual_gco2_kwh"].to_numpy(dtype=np.float32)
        self.ci_forecast = df["ci_forecast_gco2_kwh"].to_numpy(dtype=np.float32)
        self.mef         = df["mef_gco2_kwh"].to_numpy(dtype=np.float32)
        self.carbon_price = df["uka_gbp_tco2"].to_numpy(dtype=np.float32)
        self.delivery_ts  = df["delivery_ts"].to_numpy(dtype="datetime64[ns]")
        self.delivery_date = df["delivery_date"].to_numpy(dtype="datetime64[ns]")
        self.tau          = df["tau"].to_numpy(dtype=np.int32)

        # Precompute DA availability flag for all rows.
        self.tomorrow_day_arr = self.delivery_date.astype('datetime64[D]') + np.timedelta64(1, 'D')
        publish_ts_arr = (
            self.tomorrow_day_arr - np.timedelta64(1, 'D')
            + np.timedelta64(self.publish_hour, 'h')
        )
        self.da_avail_arr = (self.delivery_ts >= publish_ts_arr)

        # Build day index mapping; discard incomplete days (fewer than 48 SPs).
        day_to_idx = df.groupby("delivery_date").indices
        valid_days = []
        day_indices = {}

        for d, idxs in day_to_idx.items():
            if len(idxs) == 48:
                d64 = np.datetime64(pd.Timestamp(d).date(), "D")
                valid_days.append(d64)
                day_indices[d64] = np.array(sorted(idxs), dtype=np.int64)

        self.valid_days  = np.array(sorted(valid_days), dtype="datetime64[D]")
        self.day_indices = day_indices
        self.day_pos     = {d: i for i, d in enumerate(self.valid_days)}

        # Chronological train / validation / test split.
        n       = len(self.valid_days)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        self.valid_days_train = self.valid_days[:n_train]
        self.valid_days_val   = self.valid_days[n_train : n_train + n_val]
        self.valid_days_test  = self.valid_days[n_train + n_val :]

        _split_map = {
            "train": self.valid_days_train,
            "val":   self.valid_days_val,
            "test":  self.valid_days_test,
        }

        if split not in _split_map:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {split!r}")

        self.split = split
        self.active_valid_days = _split_map[split]

        # Action and observation spaces.
        self.power_levels = np.linspace(
            -self.P_max_MW, self.P_max_MW, self.n_power_levels
        ).astype(np.float32)

        if self.continuous_action:
            # 3-D action: [dispatch_power, da_planning_power_frac, da_aggressiveness]
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.MultiDiscrete([self.n_power_levels, self.n_power_levels, 48])

        # Observation space: 7 scalar features + 48-slot DA curve + 48-slot CI forecast = 103 floats.
        low_main  = np.array([0.0, -500.0, 0.0, 1.0, -self.P_max_MW, 0.0, -self.P_max_MW], dtype=np.float32)
        high_main = np.array([1.0, 7000.0, 1000.0, 48.0, self.P_max_MW, 1.0, self.P_max_MW], dtype=np.float32)
        low_da    = np.full((48,), -500.0, dtype=np.float32)
        high_da   = np.full((48,), 7000.0, dtype=np.float32)
        low_ci    = np.full((48,), 0.0, dtype=np.float32)
        high_ci   = np.full((48,), 1000.0, dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low_main, low_da, low_ci]),
            high=np.concatenate([high_main, high_da, high_ci]),
            dtype=np.float32,
        )

        self.ocv_soc_points, self.ocv_cell_volts = config.ocv_lookup_table()

        self.soc      = float(self.SoC_initial)
        self.p_prev   = 0.0
        self.days_done = 0
        self.current_day = None
        self.today_plan    = np.full(48, -1, dtype=np.int32)
        self.tomorrow_plan = np.full(48, -1, dtype=np.int32)
        self.today_plan_continuous    = np.zeros(48, dtype=np.float32)
        self.tomorrow_plan_continuous = np.zeros(48, dtype=np.float32)
        self.tomorrow_plan_locked     = False


    def _get_ocv_pack(self, soc: float) -> float:
        """Return pack open-circuit voltage interpolated from the OCV-SoC lookup table.

        Args:
            soc: State of charge in [0, 1].

        Returns:
            Pack OCV in volts.
        """
        soc = float(np.clip(soc, 0.0, 1.0))
        ocv_cell = np.interp(soc, self.ocv_soc_points, self.ocv_cell_volts)
        return float(ocv_cell * self.N_cells)


    def _compute_current_from_power(self, P_grid_W: float, V_oc_pack: float) -> float:
        """Solve P = I(V_oc - IR) for current using the R-int ECM.

        The quadratic R I² - V_oc I + P = 0 is solved for the physically
        meaningful root:
            I = (V_oc - sqrt(V_oc² - 4RP)) / (2R)

        Args:
            P_grid_W: Requested power at the grid connection point (W).
                Positive = discharge, negative = charge.
            V_oc_pack: Pack open-circuit voltage at the current SoC (V).

        Returns:
            Current in amperes. Positive = discharge, negative = charge.
        """
        if P_grid_W == 0.0:
            return 0.0

        discriminant = V_oc_pack**2 - 4.0 * self.R_sys * P_grid_W
        if discriminant < 0.0:
            discriminant = 0.0  # numerical guard against floating-point rounding

        I = (V_oc_pack - np.sqrt(discriminant)) / (2.0 * self.R_sys)
        return float(I)

    # Physically infeasible actions are clipped to SoC limits rather than
    # triggering explicit imbalance settlement (SBP/SSP). This safety shield
    # ensures the agent never violates hard battery constraints.
    def _apply_soc_protection(
        self, P_requested_MW: float, soc: float
    ) -> tuple[float, float, float]:
        """Clamp requested power to the range feasible without violating SoC limits.

        Args:
            P_requested_MW: Agent-requested power in MW. Positive = discharge.
            soc: Current state of charge in [0, 1].

        Returns:
            Tuple of (P_applied_MW, I_applied_A, V_oc_pack_V).
        """
        soc = float(soc)

        P_req_MW = float(np.clip(P_requested_MW, -self.P_max_MW, self.P_max_MW))
        if P_req_MW == 0.0:
            V_oc = self._get_ocv_pack(soc)
            return 0.0, 0.0, V_oc

        V_oc  = self._get_ocv_pack(soc)
        I_req = self._compute_current_from_power(P_req_MW * 1e6, V_oc)

        delta_soc_up_max   = max(self.SoC_max - soc, 0.0)
        delta_soc_down_max = max(soc - self.SoC_min, 0.0)

        if delta_soc_up_max == 0.0 and I_req < 0.0:
            return 0.0, 0.0, V_oc
        if delta_soc_down_max == 0.0 and I_req > 0.0:
            return 0.0, 0.0, V_oc

        dt = self.dt_seconds
        Q  = self.Q_pack_C

        # Invert the SoC update equations to derive current bounds:
        #   charge (I<0):     ΔSoC = -(I dt / Q) η_ch    ⟹ I_min = -(Δmax_up Q)  / (dt η_ch)
        #   discharge (I>0):  ΔSoC = -(I dt / Q) / η_dis ⟹ I_max =  (Δmax_down Q η_dis) / dt
        I_min = -(delta_soc_up_max   * Q) / (dt * self.eta_ch)
        I_max =  (delta_soc_down_max * Q * self.eta_dis) / dt

        I_applied    = float(np.clip(I_req, I_min, I_max))
        P_applied_W  = I_applied * (V_oc - I_applied * self.R_sys)
        P_applied_MW = float(np.clip(P_applied_W / 1e6, -self.P_max_MW, self.P_max_MW))

        # Re-derive current from the clipped power to keep I and P mutually consistent.
        I_applied = self._compute_current_from_power(P_applied_MW * 1e6, V_oc)

        return P_applied_MW, float(I_applied), V_oc


    def _get_tomorrow_day_from_delivery_ts(self, idx: int) -> np.datetime64:
        return self.tomorrow_day_arr[idx]

    def _da_available_now(self, idx: int) -> bool:
        """Return True if tomorrow's DA curve has been published at the current timestamp.

        DA prices for delivery day D are published at (D-1) publish_hour.

        Args:
            idx: Row index into the flat delivery-timestamp arrays.

        Returns:
            True if the DA curve for tomorrow is available.
        """
        return bool(self.da_avail_arr[idx])

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


    def _expand_da_plan(self, aggressiveness_raw: float, power_frac_raw: float, idx: int):
        """Expand a 2-D DA intent into a 48-slot MW schedule using rank-based selection.

        Args:
            aggressiveness_raw: Action dimension 2 in [-1, 1]. Maps to commitment
                fraction α ∈ [0, 1], where α = 0 means no DA commitment and
                α = 1 means all slots activated.
            power_frac_raw: Action dimension 1 in [-1, 1]. Maps to planning
                power P_plan_base ∈ [0, P_max_MW].
            idx: Row index of the current settlement period.

        The top α/2 fraction of DA price slots are designated for discharge at
        +P_plan_base MW; the bottom α/2 fraction for charge at -P_plan_base MW.
        Slot activation is subject to a forward SoC-feasibility simulation.
        """
        alpha       = (float(np.clip(aggressiveness_raw, -1.0, 1.0)) + 1.0) / 2.0
        P_plan_base = (float(np.clip(power_frac_raw, -1.0, 1.0)) + 1.0) / 2.0 * self.P_max_MW

        da_prices = self._get_tomorrow_da_curve(idx)

        # Rank slots: 0.0 = cheapest, 1.0 = most expensive.
        order = np.argsort(np.argsort(da_prices))
        ranks = order.astype(np.float32) / max(len(da_prices) - 1, 1)

        discharge_mask = ranks > (1.0 - alpha / 2.0)
        charge_mask    = ranks < (alpha / 2.0)

        plan          = np.zeros(48, dtype=np.float32)
        sim_soc       = self.soc
        delta_per_slot = (P_plan_base * self.dt_hours) / self.E_max

        for s in range(48):
            if discharge_mask[s] and sim_soc - delta_per_slot >= self.SoC_min:
                plan[s]  = P_plan_base
                sim_soc -= delta_per_slot
            elif charge_mask[s] and sim_soc + delta_per_slot <= self.SoC_max:
                plan[s]  = -P_plan_base
                sim_soc += delta_per_slot

        self.tomorrow_plan_continuous[:] = plan


    def _get_obs(self) -> np.ndarray:
        """Construct the 103-dimensional observation vector for the current step.

        Components (in order):
            - SoC (unnormalised, [0, 1])
            - tanh-normalised MID price
            - tanh-normalised carbon intensity
            - normalised settlement period index τ/48
            - normalised previous applied power P_prev / P_max
            - DA availability flag {0, 1}
            - normalised planned DA power for the current slot
            - 48-slot tanh-normalised DA price curve (zero-masked before publish time)
            - 48-slot tanh-normalised CI forecast curve (zero-masked before publish time)

        Returns:
            Float32 array of shape (103,).
        """
        idx  = int(self.current_day_idxs[self.slot0])
        tau0 = int(self.tau[idx]) - 1

        if self.continuous_action:
            planned_power_now = float(self.today_plan_continuous[tau0])
        else:
            planned_idx = int(self.today_plan[tau0])
            if planned_idx < 0 or planned_idx >= self.n_power_levels:
                planned_idx = -1
            planned_power_now = float(self.power_levels[planned_idx]) if planned_idx >= 0 else 0.0

        # Tomorrow curves are zero-masked until the DA publish time.
        da_avail = 1.0 if self._da_available_now(idx) else 0.0
        if da_avail == 1.0:
            tomorrow_da = self._get_tomorrow_da_curve(idx)
            tomorrow_ci = self._get_tomorrow_ci_curve(idx)
        else:
            tomorrow_da = np.zeros(48, dtype=np.float32)
            tomorrow_ci = np.zeros(48, dtype=np.float32)

        soc_n           = float(self.soc)                                        # [0, 1]
        price_n         = float(np.tanh(self.id_price[idx] / self.price_scale))  # [-1, 1]
        ci_n            = float(np.tanh(self.ci[idx]       / self.ci_scale))     # [-1, 1]
        tau_n           = float(self.tau[idx]) / 48.0                            # [0, 1]
        p_prev_n        = float(self.p_prev / self.P_max_MW)                     # [-1, 1]
        da_avail_n      = da_avail                                               # {0, 1}
        planned_power_n = float(planned_power_now / self.P_max_MW)               # [-1, 1]

        tomorrow_da_n = np.tanh(tomorrow_da / self.price_scale).astype(np.float32)
        tomorrow_ci_n = np.tanh(tomorrow_ci / self.ci_scale).astype(np.float32)

        main = np.array(
            [soc_n, price_n, ci_n, tau_n, p_prev_n, da_avail_n, planned_power_n],
            dtype=np.float32,
        )
        return np.concatenate([main, tomorrow_da_n, tomorrow_ci_n], axis=0)


    def _advance_one_slot(self):
        self.slot0 += 1
        if self.slot0 < 48:
            return True

        self.slot0 = 0
        self.days_done += 1

        # Roll tomorrow's DA plan into today's committed schedule.
        self.today_plan = self.tomorrow_plan.copy()
        self.tomorrow_plan[:] = -1

        self.today_plan_continuous[:] = self.tomorrow_plan_continuous
        self.tomorrow_plan_continuous[:] = 0.0
        self.tomorrow_plan_locked = False

        pos     = self.day_pos[self.current_day]
        nxt_pos = pos + 1
        if nxt_pos >= len(self.valid_days):
            return False

        self.current_day      = self.valid_days[nxt_pos]
        self.current_day_idxs = self.day_indices[self.current_day]
        return True


    def step(self, action):
        """Execute one settlement-period step.

        Args:
            action: For discrete mode, a length-3 array [dispatch_idx,
                plan_idx, plan_slot]. For continuous mode, a length-3 float
                array [dispatch_frac, plan_power_frac, aggressiveness].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info) following the
            Gymnasium API. ``reward`` is the linearly-scaled net profit per
            unit of battery capacity.
        """
        idx  = int(self.current_day_idxs[self.slot0])
        tau0 = int(self.tau[idx]) - 1

        da_avail       = self._da_available_now(idx)
        tomorrow_day   = self._get_tomorrow_day_from_delivery_ts(idx)
        tomorrow_exists = tomorrow_day in self.day_indices

        if self.continuous_action:
            # action[0]: real-time dispatch       [-1, 1] → [-P_max, P_max]
            # action[1]: DA planning power frac   [-1, 1] → [0, P_max]
            # action[2]: DA aggressiveness        [-1, 1] → α ∈ [0, 1]
            P_req_MW  = float(np.clip(action[0], -1.0, 1.0)) * self.P_max_MW
            P_plan_MW = float(self.today_plan_continuous[tau0])

            # Commit the DA plan once per day on the first DA-available step.
            if da_avail and tomorrow_exists and not self.tomorrow_plan_locked:
                self._expand_da_plan(action[2], action[1], idx)
                self.tomorrow_plan_locked = True

            dispatch_idx = plan_idx = plan_slot = planned_idx = -1
        else:
            dispatch_idx = int(np.clip(action[0], 0, self.n_power_levels - 1))
            plan_idx     = int(np.clip(action[1], 0, self.n_power_levels - 1))
            plan_slot    = int(np.clip(action[2], 0, 47))
            planned_idx  = int(self.today_plan[tau0])
            if planned_idx < 0 or planned_idx >= self.n_power_levels:
                planned_idx = -1
            P_req_MW  = float(self.power_levels[dispatch_idx])
            P_plan_MW = float(self.power_levels[planned_idx]) if planned_idx >= 0 else 0.0
            if da_avail and tomorrow_exists:
                self.tomorrow_plan[plan_slot] = int(plan_idx)

        P_applied_MW, I_applied, V_oc_pack = self._apply_soc_protection(P_req_MW, self.soc)

        if I_applied < 0.0:
            delta_soc = -(I_applied * self.dt_seconds / self.Q_pack_C) * self.eta_ch
        elif I_applied > 0.0:
            delta_soc = -(I_applied * self.dt_seconds / self.Q_pack_C) / self.eta_dis
        else:
            delta_soc = 0.0
        self.soc = float(self.soc + delta_soc)

        P_act_MW = float(P_applied_MW)
        P_dev_MW = P_act_MW - P_plan_MW

        da_price_now     = float(self.da_price[idx])
        id_price_now     = float(self.id_price[idx])
        carbon_price_now = float(self.carbon_price[idx])
        ci_now           = float(self.ci[idx])
        mef_now          = float(self.mef[idx])

        E_plan_MWh = P_plan_MW * self.dt_hours
        E_dev_MWh  = P_dev_MW  * self.dt_hours

        # Degradation cost — Cortés-Arcos et al. (2020) Eq. 23
        deg_cost = self.degradation_model.calculate_costs(P_act_MW, dt_hours=self.dt_hours)

        R_DA = E_plan_MWh * da_price_now
        R_ID = E_dev_MWh  * id_price_now

        E_act_MWh    = P_act_MW * self.dt_hours
        E_import_kWh = max(-E_act_MWh, 0.0) * 1000.0
        E_export_kWh = max(E_act_MWh,  0.0) * 1000.0

        # Carbon cashflow: AEF on imports, MEF on exports (asymmetric attribution).
        net_tCO2            = (E_import_kWh * ci_now - E_export_kWh * mef_now) / 1e6
        carbon_cashflow_gbp = carbon_price_now * net_tCO2

        steps_per_month = 48 * 30
        e_th = self.monthly_budget / steps_per_month

        emissions_tco2 = E_import_kWh * ci_now / 1e6
        P_thresh       = 0.0  # retained for info-dict backward compatibility
        R_total_gbp    = R_DA + R_ID - deg_cost - (self.lambda_ci or 0.0) * carbon_cashflow_gbp

        r_mwh   = R_total_gbp / self.E_max
        r_step  = r_mwh / self.scale_universal
        reward  = float(r_step)
        actual_reward = float(R_total_gbp)

        self.p_prev = float(P_applied_MW)

        ok         = self._advance_one_slot()
        terminated = (self.days_done >= self.episode_days)
        truncated  = (not ok) and (not terminated)

        obs = self._get_obs()

        info = {
            "delivery_ts":               str(self.delivery_ts[idx]),
            "tau":                        int(tau0 + 1),
            "idx":                        int(idx),
            "days_done":                  int(self.days_done),
            "dispatch_idx_agent":         int(dispatch_idx),
            "plan_idx_agent":             int(plan_idx),
            "plan_slot_agent":            int(plan_slot),
            "planned_idx_today":          int(planned_idx),
            "tomorrow_plan_value_written": (int(self.tomorrow_plan[plan_slot])
                                            if (da_avail and not self.continuous_action)
                                            else -999),
            "da_available":               bool(da_avail),
            "P_req_MW":                   float(P_req_MW),
            "P_planned_MW":               float(P_plan_MW),
            "P_dev_MW":                   float(P_dev_MW),
            "P_applied_MW":               float(P_applied_MW),
            "soc":                        float(self.soc),
            "delta_soc":                  float(delta_soc),
            "da_price_now":               float(da_price_now),
            "id_price_now":               float(id_price_now),
            "ci_now":                     float(ci_now),
            "mef_now":                    float(mef_now),
            "carbon_price_now":           float(carbon_price_now),
            "actual_reward":              float(actual_reward),
            "Planned_Profit":             float(R_DA),
            "Intraday_Profit":            float(R_ID),
            "degradation_cost_gbp":       float(deg_cost),
            "carbon_cashflow":            float(carbon_cashflow_gbp),
            "net_carbon_tCO2":            float(net_tCO2),
            "emissions_tco2":             float(emissions_tco2),
            "P_thresh_gbp":               float(P_thresh),
            "R_total_gbp":                float(R_total_gbp),
            "r_mwh":                      float(r_mwh),
            "reward_raw":                 float(reward),
        }

        return obs, float(reward), bool(terminated), bool(truncated), info


    def reset(self, seed=None, options=None):
        """Reset the environment to the start of a new episode.

        Args:
            seed: Optional integer seed passed to ``super().reset()``.
            options: Optional dict. If ``options["delivery_day"]`` is set to
                an ISO date string, the episode starts on that specific day.

        Returns:
            Tuple of (obs, info).
        """
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Randomise initial SoC to prevent the policy from overfitting to a
        # fixed starting state of charge.
        if self.randomize_init_soc:
            soc0 = float(self.np_random.uniform(self.init_soc_low, self.init_soc_high))
            self.soc = float(np.clip(soc0, self.SoC_min, self.SoC_max))
        else:
            self.soc = float(np.clip(self.SoC_initial, self.SoC_min, self.SoC_max))

        self.p_prev    = 0.0
        self.days_done = 0

        if options is not None and options.get("delivery_day") is not None:
            start_day = np.datetime64(options["delivery_day"], "D")
            if not (self.valid_days == start_day).any():
                raise ValueError("Requested delivery_day not in valid_days.")
            start_pos = self.day_pos[start_day]
        else:
            if self.randomize_start:
                max_start = max(len(self.active_valid_days) - self.episode_days, 1)
                local_pos = int(self.np_random.integers(0, max_start))
            else:
                local_pos = 0
            start_pos = self.day_pos[self.active_valid_days[local_pos]]

        self.current_day      = self.valid_days[start_pos]
        self.current_day_idxs = self.day_indices[self.current_day]
        self.slot0            = 0

        self.today_plan[:]    = -1
        self.tomorrow_plan[:] = -1
        self.today_plan_continuous[:]    = 0.0
        self.tomorrow_plan_continuous[:] = 0.0
        self.tomorrow_plan_locked = False

        obs  = self._get_obs()
        info = {
            "start_day":    str(self.current_day),
            "publish_hour": int(self.publish_hour),
            "episode_days": int(self.episode_days),
            "init_soc":     float(self.soc),
        }
        return obs, info
