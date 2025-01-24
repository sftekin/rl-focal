import os
import argparse
import tqdm
import torch
import numpy as np
import glob
from plotting.plot import plot_actions, plot_rewards
import matplotlib.pyplot as plt

from env.ens_env import EnsembleEnv
from focal_agent import PolicyNetwork, MLP, REINFORCE
from data_generator.data_loader import DataCreator
from config import RESULTS_DIR


torch.autograd.set_detect_anomaly(True)
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


def step_policy(train_env, select_args, ens_args, num_models, ep_count, update_freq):
    select_policy = select_args["policy"] 
    select_agent = select_args["agent"]
    select_update = select_args["update"]

    ens_policy = ens_args["policy"] 
    ens_agent = ens_args["agent"]
    ens_update = ens_args["update"]

    state = train_env.reset()
    action_count = np.zeros(num_models)
    episode_reward = 0
    # for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 2)):
    for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 2)):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs = select_policy(state)
        dists = [torch.distributions.Categorical(prob) for prob in action_probs]
        action = [dist.sample() for dist in dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, action)]
        action = np.array([a.detach().item() for a in action])
        # action = np.array([0, 1, 1, 1, 0, 1, 1])
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

        if count % update_freq == 0 and count > 0:
            if ens_update:
                ens_agent.update_policy()

            if select_update:
                select_agent.update_policy()

    total_acc = (episode_reward / count) * 100

    if ep_count % 5 == 0:
        print(f"Episode: {ep_count}, Total Acc: {total_acc:.2f}%, Action Counts: {action_count}")

    return total_acc


def train_loop(train_env, n_episodes, select_args, ens_args, num_models, max_tolerance=50, update_freq=10):
    best_reward, tol = 0, 0
    agent_rw = []
    best_ens_policy_dict = ens_args["policy"].state_dict()
    best_select_policy_dict = select_args["policy"].state_dict()
    for episode in range(n_episodes):
        episode_reward = step_policy(train_env, select_args, ens_args, num_models, ep_count=episode, update_freq=update_freq)
        agent_rw.append(episode_reward)

        if best_reward < episode_reward:
            tol = 0
            best_ens_policy_dict = ens_args["policy"].state_dict()
            best_select_policy_dict = select_args["policy"].state_dict()
            best_reward = episode_reward
        else:
            tol += 1

        if tol >= max_tolerance:
            print("reached max tolerance breaking...")
            break

    ens_args["policy"].load_state_dict(best_ens_policy_dict)
    ens_args["agent"].update_policy_params(ens_args["policy"])
    
    select_args["policy"].load_state_dict(best_select_policy_dict)
    select_args["agent"].update_policy_params(select_args["policy"])

    return agent_rw, select_args, ens_args


def train_test(train_data, test_data, num_models, args):
    space_size = (train_data.shape[1] - 1) // num_models
    train_env = EnsembleEnv(train_data, num_models, device=args.device, window_size=args.window_size, space_size=space_size)
    policy1 = PolicyNetwork(train_env.obsv_space_len, 
                            np.ones(train_env.ac_space_len).astype(int) * 2).to(args.device)
    policy2 = MLP(num_models * space_size, [100, 100], space_size).to(args.device)

    agent1 = REINFORCE(policy1, clip_epsilon=0.2, aggregate_loss="mean")
    agent2 = REINFORCE(policy2, lr=0.001, clip_epsilon=0.2, gamma=0.5, aggregate_loss="sum")

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

    # train select agent
    select_agent_rw, select_args, ens_args = train_loop(train_env, args.sel_episodes, select_args,
                                                        ens_args, num_models, max_tolerance=args.max_tolerance,
                                                        update_freq=args.update_freq)
    
    # train ensemble agent
    select_args["update"] = False
    ens_args["update"] = True
    ens_agent_rw, select_args, ens_args = train_loop(train_env, args.ens_episodes, select_args,
                                                    ens_args, num_models, max_tolerance=args.max_tolerance,
                                                    update_freq=args.update_freq)

    select_agent_rw = np.array(select_agent_rw)
    ens_agent_rw = np.array(ens_agent_rw)

    print("Starting testing")
    ens_args["update"] = True
    select_args["update"] = True
    test_env = EnsembleEnv(test_data, num_models, device=args.device, window_size=0, space_size=space_size)
    test_reward = step_policy(test_env, select_args, ens_args, num_models, ep_count=0, update_freq=args.update_freq)

    return select_agent_rw, ens_agent_rw, test_reward


def save_exp(select_agent_rw, ens_agent_rw, test_reward, task_name):
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints", task_name)
    dir_name = get_last_checkpoint_dirname(checkpoint_dir)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f"Test Acc: {test_reward:.2f}%")
    score_path = os.path.join(dir_name, "test_score.txt")
    with open(score_path, "w") as file:
        file.write(f"Test Acc: {test_reward:.2f}%\n")
    

    select_arr_path = os.path.join(dir_name, "train_select_agent_rewards.npy")
    save_arr(select_agent_rw, select_arr_path)

    ens_arr_path = os.path.join(dir_name, "train_ens_agent_rewards.npy")
    save_arr(ens_agent_rw, ens_arr_path)

    print("Data is saved...")


def main(args):
    datacreator = DataCreator(args.task_name, model_names=args.model_names, task_type="lang")

    select_agent_rewards, ens_agent_rewards, test_rewards = [], [], []
    for i, (train_data, test_data, num_models, ds_name) in enumerate(datacreator.load()):
        print(f"Dataset-{i}: {ds_name}")
        select_agent_rw, ens_agent_rw, test_reward = train_test(train_data, test_data, num_models, args)
        select_agent_rewards.append(select_agent_rw)
        ens_agent_rewards.append(ens_agent_rw)
        test_rewards.append(test_reward)
        print("\n")
    
    select_agent_rewards = np.mean(select_agent_rewards, axis=0)
    ens_agent_rewards = np.mean(ens_agent_rewards, axis=0)
    test_rewards = np.mean(test_rewards, axis=0)

    save_exp(select_agent_rewards, ens_agent_rewards, test_rewards, args.task_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train and test script for the rl-focal')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_names", type=str, default="all")
    parser.add_argument("--dataset_type", type=str, default="lang", choices=["lang", "vision"])
    parser.add_argument("--task_name", type=str, default="mmlu_hf", choices=["gsm8k", "mmlu_hf", "bbh", "gpqa", "musr"])
    parser.add_argument("--sel_episodes", type=int, default=25)
    parser.add_argument("--ens_episodes", type=int, default=25)
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--max_tolerance", type=int, default=150)
    parser.add_argument("--update_freq", type=int, default=1000)
    arguments = parser.parse_args()
    main(arguments)
