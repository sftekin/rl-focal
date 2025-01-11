import os
import argparse
import tqdm
import torch
import numpy as np
import glob
from plotting import plot_actions, plot_rewards
import matplotlib.pyplot as plt

from env.ens_env import EnsembleEnv
from focal_agent import PolicyNetwork, MLP, REINFORCE
from data_loader import DataCreator
from config import RESULTS_DIR


device = "cuda"

def get_last_checkpoint_dirname(checkpoint_dir):
    cur_dirs = [f for f in glob.glob(os.path.join(checkpoint_dir, "exp_*"))]
    dir_name = os.path.join(checkpoint_dir, f"exp_{len(cur_dirs)}")
    return dir_name


def save_arr(arr, file_name):
    with open(file_name, "wb") as f:
        np.save(f, arr)


def load_arr(file_name):
    with open(file_name, "rb") as f:
        ret_arr = np.load(f)
    return ret_arr



def step_policy(train_env, select_args, ens_args, num_models, ep_count):
    select_policy = select_args["policy"] 
    select_agent = select_args["agent"]
    select_update = select_args["update"]

    ens_policy = ens_args["policy"] 
    ens_agent = ens_args["agent"]
    ens_update = ens_args["update"]

    state = train_env.reset()
    action_count = np.zeros(num_models)
    episode_reward = 0
    for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 2)):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs = select_policy(state)
        dists = [torch.distributions.Categorical(prob) for prob in action_probs]
        action = [dist.sample() for dist in dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, action)]
        action = np.array([a.detach().item() for a in action])
        # action = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        action_count += action

        if ens_update:
            next_state, reward, pred_prob = train_env.step(action, ens_policy)
        else:
            next_state, reward, pred_prob = train_env.step(action)

        if select_update:
            select_agent.store_outcome(log_probs, reward)

        if ens_update:
            ens_agent.store_outcome([pred_prob], reward)

        state = next_state
        episode_reward += np.max([reward, 0])

        if count % 100 == 0 and count > 0:
            if ens_update:
                ens_agent.update_policy()

            if select_update:
                select_agent.update_policy()

    total_acc = (episode_reward / count) * 100
    print(f"Episode: {ep_count}, Total Acc: {total_acc:.2f}%")
    print(action_count)
    return total_acc



def train(train_data, test_data, num_models, n_episodes=100):
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
    dir_name = get_last_checkpoint_dirname(checkpoint_dir)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    train_env = EnsembleEnv(train_data, num_models, device=device, window_size=500)
    policy1 = PolicyNetwork(train_env.obsv_space_len, 
                            np.ones(train_env.ac_space_len).astype(int) * 2).to(device)
    policy2 = MLP(num_models * 4, [100, 100], 4).to(device)

    agent1 = REINFORCE(policy1, aggregate_loss="mean")
    agent2 = REINFORCE(policy2, gamma=0.5, aggregate_loss="sum")

    select_args = {
        "agent": agent1,
        "policy": policy1,
        "update": True
    }

    ens_args = {
        "agent": agent2,
        "policy": policy2,
        "update": False
    }

    prev_reward = 0
    tol = 0
    max_tolerance = 5
    select_agent_rw = []
    ens_agent_rw = []
    for episode in range(n_episodes):
       
        if episode < n_episodes // 2:
            episode_reward = step_policy(train_env, select_args, ens_args, num_models, ep_count=episode)
            select_agent_rw.append(episode_reward)
        else:
            select_args["update"] = False
            ens_args["update"] = True
            episode_reward = step_policy(train_env, select_args, ens_args, num_models, ep_count=episode)
            ens_agent_rw.append(episode_reward)

        if prev_reward < episode_reward:
            print("better reward")
            tol = 0
        else:
            tol += 1
        prev_reward = episode_reward

        if tol >= max_tolerance:
            print("reached max tolerance breaking...")
            break

    select_agent_rw = np.array(select_agent_rw)
    ens_agent_rw = np.array(ens_agent_rw)

    select_arr_path = os.path.join(dir_name, "train_select_agent_rewards.npy")
    save_arr(select_agent_rw, select_arr_path)

    ens_arr_path = os.path.join(dir_name, "train_ens_agent_rewards.npy")
    save_arr(ens_agent_rw, ens_arr_path)

    print("Data is saved...")
    agents = (agent1, agent2)
    policies = (policy1, policy2)

    print("Starting testing")
    ens_args["update"] = True
    select_args["update"] = True
    test_env = EnsembleEnv(test_data, num_models, device=device, window_size=0)
    test_reward = step_policy(train_env, select_args, ens_args, num_models, ep_count=episode)
    print(f"Test Acc: {test_reward:.2f}%")

    return agents, policies


def main(args):
    dataset_name="mmlu_hf"
    m_names = ['Mistral-7B-Instruct-v0.2', 'Mixtral-8x7B-v0.1', 
               'gemma-2b', 'gemma-7b', 'Llama-2-13b-hf', 'phi-2',
                 'Llama-2-70b-hf', 'Llama-2-7b-hf']

    datacreator = DataCreator(dataset_name, model_names=m_names, task_type="lang")
    train_data, test_data, num_models = datacreator.create()

    train(train_data, test_data, num_models, n_episodes=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train and test script for the rl-focal')
    parser.add_argument("--model_names", type=str, default="all")
    parser.add_argument("--dataset_type", type=str, default="lang", choices=["lang", "vision"])
    arguments = parser.parse_args()
    main(arguments)
