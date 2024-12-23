import numpy as np
import gym
from gym import spaces

from env.ens_methods import voting
from env.ens_metrics import fitness_function

class HistData:
    def __init__(self, data_dict):
        self.data_dict = data_dict
    
    def __getitem__(self, key):
        # Check if the key is a slice
        if isinstance(key, slice) or isinstance(key, int):
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
        self.num_models = num_models
        self.window_size = 500

        # parse the data
        self.hist_data = self._parse_data()
        self.total_len = len(data)

        # create initial model pool randomly
        self.init_model_pool = np.random.randint(high=2, low=0, size=(num_models))
        self.initial_step = int(len(data) * .25)
        self.current_step = self.initial_step
        self.current_model_pool = self.init_model_pool

        # set the initial states:
        self.current_state = self._get_observation()
        self.current_reward = self._evaluate_pool()

        # set the space lengths
        self.obsv_space_len = len(self.current_state)
        self.ac_space_len = num_models


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

    def _get_observation(self):
        hist_data = self.hist_data[:self.current_step]
        score = fitness_function(self.current_model_pool, [0.5, 0.5], hist_data)
        observation = np.append(self.current_model_pool, score)
        return observation

    def _evaluate_pool(self):
        if sum(self.current_model_pool) < 2:
            return 0
        current_data = self.hist_data[self.current_step]
        comb_idx = self.current_model_pool.astype(bool)
        x, y = current_data["pred_arr"][comb_idx], current_data["label_arr"].astype(int)

        # todo: make better prediction
        ens_pred = voting(x[None, :], method="plurality")
        if  y == ens_pred:
            reward = 1
        else:
            reward = 0
        return reward

    def reset(self):
        self.current_step = self.initial_step
        self.current_model_pool = np.random.randint(
            high=2, low=0, size=(self.num_models))
        self.current_reward = 0
        return self._get_observation()

    def step(self, action):
        self.current_model_pool = action.astype(int)

        # Execute action
        self.current_step += 1
        reward = self._evaluate_pool()
        
        # Update reward
        reward = self.current_reward + reward
        self.current_reward = reward
     
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, {}
    
    def get_current_progress(self):
        return int(self.current_step / self.total_len * 100)
