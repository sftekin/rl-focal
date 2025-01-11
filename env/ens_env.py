import torch
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
    def __init__(self, data, num_models, window_size=-1, device="cuda"):
        super(EnsembleEnv, self).__init__()
        self.data = data  # Historical model performances
        self.num_models = num_models
        self.window_size = window_size
        self.device = device

        assert self.window_size < len(self.data)

        # parse the data
        self.hist_data = self._parse_data()

        # create initial model pool randomly
        self.init_model_pool = np.random.randint(high=2, low=0, size=(num_models))
        self.initial_step = self.window_size if window_size > 0 else int(len(data) * 0.1)
        self.current_step = self.initial_step
        self.current_model_pool = self.init_model_pool
        self.total_len = len(data) - self.initial_step

        # set the initial states:
        self.current_state = self._get_observation()
        # self.current_reward = self._evaluate_pool()

        # set the space lengths
        self.obsv_space_len = len(self.current_state)
        self.ac_space_len = num_models

    def _parse_data(self):
        labels = self.data[:, -1]
        labels_arr = np.repeat(labels[:, None], self.num_models, axis=1)
        probs = self.data[:, :-1]
        indv_probs = np.split(probs, self.num_models, axis=1)
        bin_preds = np.concatenate([a.argmax(axis=1)[:, None] for a in indv_probs], axis=1)
        error_arr = (bin_preds == labels_arr).astype(int)
        data_dict = {
            "prob_arr": probs,
            "error_arr": error_arr,
            "pred_arr": bin_preds,
            "label_arr": labels,
        }
        return HistData(data_dict)

    def _get_observation(self):
        start_idx = self.current_step - self.window_size if self.window_size > 0 else 0
        hist_data = self.hist_data[start_idx:self.current_step]
        score = fitness_function(self.current_model_pool, [0.5, 0.5], hist_data)
        observation = np.append(self.current_model_pool, score)
        observation = np.append(observation, self.current_model_pool.sum())
        return observation

    def _evaluate_pool(self, prediction_policy):
        current_data = self.hist_data[self.current_step]
        comb_idx = self.current_model_pool.astype(bool)
        x, y = current_data["pred_arr"], current_data["label_arr"].astype(int)

        if prediction_policy is None:            
            if sum(self.current_model_pool) < 2:
                return 0, None
            ens_pred = voting(x[comb_idx][None, :], method="plurality")
            pred_probs = None
        else:
            mask = np.repeat(comb_idx, 4)
            x = current_data["prob_arr"] * mask.astype(int)
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            pred_probs = prediction_policy(x)
            # ens_pred = pred_probs.argmax().item()
            # pred_probs = pred_probs[ens_pred]

            m = torch.distributions.Categorical(pred_probs)
            ens_pred = m.sample()
            pred_probs = m.log_prob(ens_pred)

        if y == ens_pred:
            reward = 1
        elif prediction_policy is not None:
            reward = -1
        else:
            reward = -1 - (comb_idx.sum() / len(comb_idx))

        return reward, pred_probs

    def reset(self):
        self.current_step = self.initial_step
        # self.current_model_pool = np.random.randint(
        #     high=2, low=0, size=(self.num_models))
        self.current_model_pool = np.ones(self.num_models).astype(int)
        return self._get_observation()

    def step(self, action, prediction_policy=None):
        self.current_model_pool = action.astype(int)

        # Execute action
        self.current_step += 1
        reward, pred_probs = self._evaluate_pool(prediction_policy)
     
        return self._get_observation(), reward, pred_probs
