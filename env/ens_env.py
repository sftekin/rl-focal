import numpy as np
import gym
from gym import spaces

from env.ens_metrics import fitness_function

class HistData:
    def __init__(self, data_dict):
        self.data_dict = data_dict
    
    def __getitem__(self, key):
        # Check if the key is a slice
        if isinstance(key, slice):
            # Create a new dictionary with sliced values
            sliced_dict = {k: v[key] for k, v in self.data_dict.items()}
            return HistData(sliced_dict)
        elif key in self.data_dict:
            return self.data_dict[key]
        else:
            raise KeyError(f"Key '{key}' not found in the data dictionary.")

    def __repr__(self):
        return repr(self.data_dict)


class EnsembleEnv(gym.Env):
    def __init__(self, data, num_models):
        super(EnsembleEnv, self).__init__()
        self.data = data  # Historical model performances
        self.current_step = int(len(data) * .25)
        self.num_models = num_models

        # parse the data
        self.hist_data = self._parse_data()

        # create initial model pool randomly
        self.init_model_pool = np.random.randint(high=2, low=0, size=(num_models))

        # set the initial states:
        self.current_score = self._get_observation(self.init_model_pool)
        # - initial model repr.
        # - initial model fitness score.

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(data.shape[1],), dtype=np.float32)

    def _parse_data(self):
        labels = self.data[:, -1]
        labels_arr = np.repeat(labels[:, None], self.num_models, axis=1)
        probs = self.data[:, :-1]
        indv_probs = np.split(probs, self.num_models, axis=1)
        bin_preds = np.concatenate([a.argmax(axis=1)[:, None]  for a in indv_probs], axis=1)
        error_arr = (bin_preds == labels_arr).astype(int)
        data_dict = {
            "error_arr": error_arr,
            "pred_arr": bin_preds,
            "label_arr": labels,
        }
        return HistData(data_dict)

    def _get_observation(self, model_pool):
        hist_data = self.hist_data[:self.current_step]
        score = fitness_function(model_pool, [0.5, 0.5], hist_data)
        return score

    def reset(self):
        self.current_step = 0
        self.stock_owned = 0
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
    
