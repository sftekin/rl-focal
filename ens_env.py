import numpy as np
import gym
from gym import spaces

class EnsembleEnv(gym.Env):
    def __init__(self, data, num_models, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.data = data  # Historical stock prices
        self.initial_balance = initial_balance
        self.current_step = 0
        self.cash = initial_balance
        self.stock_owned = 0
        self.total_value = initial_balance
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(data.shape[1],), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_balance
        self.stock_owned = 0
        self.total_value = self.initial_balance
        return self._get_observation()
    
    def step(self, action):
        current_price = self.data[self.current_step]
        reward = 0

        # Execute action
        if action == 1:  # Buy
            shares_to_buy = self.cash // current_price
            self.stock_owned += shares_to_buy
            self.cash -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.cash += self.stock_owned * current_price
            self.stock_owned = 0
        
        # Update portfolio value
        self.total_value = self.cash + self.stock_owned * current_price
        reward = self.total_value - self.initial_balance  # Reward as profit/loss
        self.current_step += 1
        
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        return self.data[self.current_step]
