import sys

import gymnasium
import pandas as pd
from gymnasium import Env, spaces
from pandas.io import parquet
import config
import numpy as np
import matplotlib.pyplot as plt



class BatteryEnv(Env):
    def __init__(self, config):

        # ========
        # LOAD CONFIG PARAMETERS
        # ========


        #Setting all parameters of the environment
        self.power_max = config.P_max
        self.energy_max = config.E_max
        self.SoC_min = config.SoC_min
        self.SoC_max = config.SoC_max
        self.eff_dis = config.eff_dis
        self.eff_ch = config.eff_ch
        self.time_step = config.dt
        self.SoC_initial = config.SoC_initial

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
        self.action_space = spaces.Discrete(3)

        # ========
        # OBSERVATION SPACE 
        # ========
        # State = [SoC, price, carbon intensity, tau]
        #these values can be changed accordingly but are high and low values for the agent to understand what the range is
        low  = np.array([0.0,    -300,   0.0,   0    ], dtype=np.float32)
        high = np.array([1.0,  500, 1000.0, 48    ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ========
        # INITIAL STATE 
        # ========
        
        self.state = np.array([
            self.SoC,
            self.sell_price_data[0],
            self.carbon_intensity_data[0],
            self.tau_data[0]
        ], dtype =np.float32)
        

    def render(self):
        #create 
    
    def reset(self):


    def step(self, action):

    
    def close(self):