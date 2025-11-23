import sys

import gymnasium as gym
import pandas as pd
from gymnasium import Env, spaces
from pandas._config import detect_console_encoding
from pandas._libs.tslibs import delta_to_nanoseconds
from pandas.io import parquet
import config
import numpy as np
import matplotlib.pyplot as plt



class BatteryEnv(Env):
    def __init__(self, config):

        # ========
        # LOAD CONFIG PARAMETERS
        # ========

        self.power_max = config.P_max
        self.energy_max = config.E_max
        self.SoC_min = config.SoC_min
        self.SoC_max = config.SoC_max
        self.eff_dis = config.eff_dis
        self.eff_ch = config.eff_ch
        self.time_step = config.dt
        self.SoC_initial = config.SoC_initial
        self.self_discharge = config.self_dis
        C_rate= config.C_rate
        self.P_step = C_rate * self.energy_max
        self.lambda_ci = config.lambda_ci  # Store as instance variable for use in step()

        # ========
        # LOAD DATA
        # ========
        #Load the data that was processed in data folder
        merged_data = pd.read_parquet("data/merged_data.parquet")   

        #Price dataset
        self.sell_price_data = merged_data['ssp'].to_numpy(dtype=np.float32)
        self.buy_price_data = merged_data['sbp'].to_numpy(dtype=np.float32)
        #Carbon Intensity dataset
        self.carbon_intensity_data = merged_data['carbon_gco2_kwh'].to_numpy(dtype=np.float32)
        
        #Timestamp data - convert to useful feature tau_t (0-47)
        self.timestamp = pd.to_datetime(merged_data['timestamp'])
        self.tau_data = (self.timestamp.dt.hour * 2 + (self.timestamp.dt.minute // 30)).to_numpy()
        
        # ========
        # INTERNAL ENV VARIABLES
        # ========
        
        self.current_timestep = 0
        self.SoC = self.SoC_initial
        self.reward = 0
        self.actions = []

        # ========
        # ACTION SPACE (DISCRETE)
        # ========
        # -1 (charge), 0 (idle), +1 (discharge)
        # Agent sees 0/1/2 and we map internally
        self.action_space = gym.spaces.Discrete(3)

        # ========
        # OBSERVATION SPACE 
        # ========
        # State = [SoC, price, carbon intensity, tau]
        #these values can be changed accordingly but are high and low values for the agent to understand what the range is
        low  = np.array([0.0,    -300,   0.0,   0    ], dtype=np.float32)
        high = np.array([1.0,  500, 1000.0, 48    ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # ========
        # INITIAL STATE 
        # ========
        
        self.state = np.array([
            self.SoC,
            self.sell_price_data[0],
            self.carbon_intensity_data[0],
            self.tau_data[0]
        ], dtype =np.float32)
    
    #NEED TO MAKE SURE THIS IS CORRECT
    def _get_obs(self):
        """
        Convert the current internal state of the environment into the
        observation vector expected by the agent.

        Having this helper keeps the logic DRY because both `reset` and `step`
        can simply call `_get_obs()` after mutating the internal members
        (SoC, timestep, etc.) instead of duplicating array construction code.
        """
        idx = int(np.clip(self.current_timestep, 0, len(self.sell_price_data) - 1))

        obs = np.array(
            [
                float(self.SoC),
                float(self.sell_price_data[idx]),
                float(self.carbon_intensity_data[idx]),
                float(self.tau_data[idx]),
            ],
            dtype=np.float32,
        )

        # Keep `self.state` in sync so any legacy code reading it directly
        # still works, but return a copy to avoid unintentional mutations.
        self.state = obs
        return obs.copy()

    def step(self, action):
        """
        Simulate one time step in the environment.

        Parameters
        ------
        action: float
            The action to be taken, -1, 0 or 1.

        Returns
        ------
        tuple
            a tuple containing the new state, reward, done flag, and additional info.
        """
        # --- Unpack the current observation for downstream calculations ---
        SoC, price, carbon_intensity, tau = self.state

        # --- Actions that can be taken by Agent - Mapping Discrete action to Power ---
        if action == 0: #charge
            P_raw = -self.P_step # grid -> battery
        elif  action == 1: #idle
            P_raw = 0.0
        elif action == 2: #discharging
            P_raw = +self.P_step #battery -> grid
        else:
            raise ValueError("Invalid action")

        # --- Clamping Power by feasibility (SoC limits) ---

        if P_raw < 0:
            E_room = (self.SoC_max - self.SoC) * self.energy_max
            E_step = abs(P_raw) * self.eff_ch * self.time_step
            if E_step > E_room:
                P_t = - E_room/(self.eff_ch * self.time_step)
            else:
                P_t = P_raw 
        elif P_raw > 0:
            E_room = (self.SoC - self.SoC_min) * self.energy_max  # Fixed: missing closing parenthesis
            E_step = abs(P_raw) * self.time_step / self.eff_dis
            if E_step > E_room:
                P_t = (E_room * self.eff_dis) / self.time_step  # Fixed: E-room → E_room
            else:
                P_t = P_raw 
        else:
            P_t = 0

        # --- Updating SoC using update equation ---

        if P_t < 0: #charging
            dSoC = (self.eff_ch * abs(P_t) * self.time_step) / self.energy_max  # Fixed: self.config.eff_ch → self.eff_ch
            self.SoC = self.SoC *(1 - self.self_discharge) + dSoC
        elif P_t > 0: #discharging
            dSoC = (abs(P_t) * self.time_step) / (self.energy_max * self.eff_dis)
            self.SoC = self.SoC*(1 - self.self_discharge) - dSoC
        else: #idle
            self.SoC = self.SoC*(1 - self.self_discharge) #self discharge BUT at the moment its at 0

        #clip the SoC, so that it doesnt go out of the min/max boundaries
        self.SoC = float(np.clip(self.SoC, self.SoC_min, self.SoC_max))


        # --- Computing Reward ---
        #Reward = profit - lambda * carbon_cost

        #energy traded this step:
        delta_E = P_t * self.time_step #MWh

        #Profit term
        #when charging -> P_t < 0 -> you pay for electricity 
        #when dicharging -> P_t > 0 -> you sell electricity 

        buy_price = self.buy_price_data[self.current_timestep]
        sell_price = self.sell_price_data[self.current_timestep]  # Fixed: self.self_price_data → self.sell_price_data

        if P_t > 0: #discharging - you are selling 
            profit_t = sell_price * delta_E
        elif P_t < 0: #charging - paying for electricity
            profit_t = buy_price * delta_E
        else: #staying idle
            profit_t = 0

    
        #Carbon Cost Term
        ci_t = self.carbon_intensity_data[self.current_timestep]
        lambda_ci = self.lambda_ci  # Fixed: use instance variable instead of config.lambda_ci

        # convert gCO2/kWh → tCO2/MWh = (g/kWh) × (1e-6)
        emission_intensity_t = ci_t * 1e-6

        carbon_cost_t = lambda_ci * emission_intensity_t * (-delta_E)
        #NOTE - we need to put a negative in front of the Delta_E as:
        #Charging -> Delta_E < 0 -> We should decrease reward so increase value of Carbon Cost
        #Discharging -> Delta_E > 0 -> We should increase reward so decrease value of Carbon Cost

        #REWARD EQUATION
        reward_t = profit_t - carbon_cost_t
        self.reward = float(reward_t)
    
        #increment timestep
        self.current_timestep += 1

        #check if the end of the data is reached
        if self.current_timestep >= len(self.sell_price_data):
            terminated = True
        else:
            terminated = False
        
        truncated = False
        
        next_obs = self._get_obs()

        #Returning the gymnasium step tuple
        return next_obs, self.reward, terminated, truncated, {}


    def close(self):
        pass