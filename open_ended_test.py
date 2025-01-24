import torch
import tqdm
import argparse
import numpy as np
import gym
from env.ens_metrics import fitness_function
from focal_agent import PolicyNetwork, MLP, REINFORCE


device = "cuda"

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
    def __init__(self, data, num_models, window_size=-1, device="cuda", space_size=4):
        super(EnsembleEnv, self).__init__()
        self.data = data  # Historical model performances
        self.num_models = num_models
        self.window_size = window_size
        self.device = device
        self.space_size = space_size

        assert self.window_size < len(self.data)

        # parse the data
        self.hist_data = self._parse_data()

        # create initial model pool randomly
        self.init_model_pool = np.random.randint(high=2, low=0, size=(num_models))
        self.initial_step = self.window_size if window_size > 0 else 0
        self.current_step = self.initial_step + 1
        self.current_model_pool = self.init_model_pool
        self.total_len = len(data) - self.initial_step

        # set the initial states:
        self.current_state = self._get_observation()
        # self.current_reward = self._evaluate_pool()

        # set the space lengths
        self.obsv_space_len = len(self.current_state)
        self.ac_space_len = num_models

    def _parse_data(self):
        error_arr = (self.data <= 0.5).astype(int)
        data_dict = {
            "error_arr": error_arr,
            "pred_arr": self.data,
        }
        return HistData(data_dict)

    def _get_observation(self):
        start_idx = self.current_step - self.window_size if self.window_size > 0 else 0
        hist_data = self.hist_data[start_idx:self.current_step]
        # score = fitness_function(self.current_model_pool, [0.5, 0.5], hist_data)
        prev_rewards = hist_data[self.current_step - 1]["pred_arr"]
        # observation = np.append(self.current_model_pool, score)
        observation = np.append(self.current_model_pool, prev_rewards)
        return observation

    def _evaluate_pool(self):
        current_data = self.hist_data[self.current_step]
        comb_idx = self.current_model_pool.astype(bool)
        x = current_data["pred_arr"]
        reward = x[comb_idx][0]
        return reward

    def reset(self):
        self.current_step = self.initial_step + 1
        self.current_model_pool = np.ones(self.num_models).astype(int)
        return self._get_observation()

    def step(self, action, prediction_policy=None):
        actions = np.zeros(self.num_models)
        actions[action] = 1
        self.current_model_pool = actions

        # Execute action
        self.current_step += 1
        reward = self._evaluate_pool()
     
        return self._get_observation(), reward


def step_policy(train_env, select_args, num_models, ep_count, update_freq):
    select_policy = select_args["policy"] 
    select_agent = select_args["agent"]

    state = train_env.reset()
    action_count = np.zeros(num_models)
    episode_reward = 0
    # for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 2)):
    for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 3)):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action_prob = select_policy(state)
        m = torch.distributions.Categorical(action_prob)
        action = m.sample()
        pred_prob = m.log_prob(action)
        action_count[action.item()] += 1

        next_state, reward = train_env.step(action.item())

        select_agent.store_outcome([pred_prob], reward)

        state = next_state
        episode_reward += np.max([reward, 0])

        if count % update_freq == 0 and count > 0:
            select_agent.update_policy()

    total_acc = (episode_reward / count) * 100

    if ep_count % 5 == 0:
        print(f"Episode: {ep_count}, Total Acc: {total_acc:.2f}%, Action Counts: {action_count}")

    return total_acc


def train_loop(train_env, n_episodes, select_args, num_models, max_tolerance=50, update_freq=10):
    best_reward, tol = 0, 0
    agent_rw = []
    best_select_policy_dict = select_args["policy"].state_dict()
    for episode in range(n_episodes):
        episode_reward = step_policy(train_env, select_args, num_models, ep_count=episode, update_freq=update_freq)
        agent_rw.append(episode_reward)

        if best_reward < episode_reward:
            tol = 0
            best_select_policy_dict = select_args["policy"].state_dict()
            best_reward = episode_reward
        else:
            tol += 1

        if tol >= max_tolerance:
            print("reached max tolerance breaking...")
            break

    select_args["policy"].load_state_dict(best_select_policy_dict)
    select_args["agent"].update_policy_params(select_args["policy"])

    return agent_rw, select_args


def create_data():
    train_scores = np.load("results/checkpoints/open_ended/train_scores.npy")
    test_scores = np.load("results/checkpoints/open_ended/test_scores.npy")
    return train_scores, test_scores

def main(args):
    num_models = 3
    train_data, test_data = create_data()
    train_env = EnsembleEnv(train_data, num_models=num_models, device=args.device, window_size=args.window_size)
    policy = MLP(train_env.obsv_space_len, [100, 100], 3).to(args.device)
    agent = REINFORCE(policy, clip_epsilon=0.2, aggregate_loss="mean")

    select_args = {
        "agent": agent,
        "policy": policy,
        "update": True
    }

    select_agent_rw, select_args = train_loop(train_env, args.sel_episodes, select_args,
                                            num_models, max_tolerance=args.max_tolerance,
                                                    update_freq=args.update_freq)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train and test script for the rl-focal')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_names", type=str, default="all")
    parser.add_argument("--dataset_type", type=str, default="lang", choices=["lang", "vision"])
    parser.add_argument("--task_name", type=str, default="mmlu_hf", choices=["gsm8k", "mmlu_hf", "bbh", "gpqa", "musr"])
    parser.add_argument("--sel_episodes", type=int, default=25)
    parser.add_argument("--ens_episodes", type=int, default=25)
    parser.add_argument("--window_size", type=int, default=-1)
    parser.add_argument("--max_tolerance", type=int, default=150)
    parser.add_argument("--update_freq", type=int, default=1000)
    arguments = parser.parse_args()
    main(arguments)