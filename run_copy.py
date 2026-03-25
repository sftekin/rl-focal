import os
import argparse
import tqdm
import torch
import numpy as np
import glob
import time
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


def perform_test(ens_args, select_args, test_data, args, num_models, space_size):
    ens_args["update"] = True
    select_args["update"] = True
    test_env = EnsembleEnv(test_data, num_models, device=args.device, window_size=0, space_size=space_size)
    test_reward, test_errors = step_policy(test_env, select_args, ens_args, num_models, ep_count=0, update_freq=args.update_freq)
    print(f"Interim Test score: {test_reward}")

def step_policy(train_env, select_args, ens_args, num_models, ep_count, update_freq, verbose=False):
    select_policy = select_args["policy"] 
    select_agent = select_args["agent"]
    select_update = select_args["update"]

    ens_policy = ens_args["policy"] 
    ens_agent = ens_args["agent"]
    ens_update = ens_args["update"]

    state = train_env.reset()
    action_count = np.zeros(num_models)
    episode_result = 0
    average_time = 0
    
    # for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 2)):
    errors = []
    # for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 2)):
    for count in range(train_env.total_len - train_env.window_size - 2):
        start_time = time.time()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs = select_policy(state)
        try:
            # dists = [torch.distributions.Categorical(prob) for prob in action_probs]
            dists = [torch.distributions.Bernoulli(prob) for prob in action_probs]
        except ValueError as e:
            print(e)
            continue
        action = [dist.sample() for dist in dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, action)]
        action = np.array([a.detach().item() for a in action])
        action_count += action

        if ens_update:
            next_state, reward, pred_prob, ep_result = train_env.step(action, ens_policy)
        else:
            next_state, reward, pred_prob, ep_result = train_env.step(action)

        if select_update:
            select_agent.store_outcome(log_probs, reward)

        if ens_update:
            ens_agent.store_outcome([pred_prob], reward)

        state = next_state
        episode_result += ep_result

        if count % update_freq == 0 and count > 0:
            if ens_update:
                ens_agent.update_policy()

            if select_update:
                select_agent.update_policy()
        average_time += time.time() - start_time
        errors.append(ep_result)
    total_acc = (episode_result / count) * 100
    average_time /= count
    if ep_count % 1 == 0 and verbose:
        print(f"Episode: {ep_count}, Total Acc: {total_acc:.2f}%, Action Counts: {action_count}, Avg Inf Time: {average_time:.4}s")

    return total_acc, errors


def train_loop(train_env, n_episodes, select_args, ens_args, num_models, max_tolerance=50,
               update_freq=10, perform_test_flag=False, test_data=None, args=None, space_size=None):
    best_reward, tol = 0, 0
    agent_rw = []
    best_ens_policy_dict = ens_args["policy"].state_dict()
    best_select_policy_dict = select_args["policy"].state_dict()
    for episode in range(n_episodes):
        episode_reward, ep_errors = step_policy(train_env, select_args, ens_args, num_models, ep_count=episode, update_freq=update_freq, verbose=True)
        agent_rw.append(episode_reward)

        if best_reward < episode_reward:
            tol = 0
            best_ens_policy_dict = ens_args["policy"].state_dict()
            best_select_policy_dict = select_args["policy"].state_dict()
            best_reward = episode_reward
            
            # if perform_test_flag:
            #     perform_test(ens_args.copy(), select_args.copy(), test_data, args, num_models, space_size)
            
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
    div_metric_weights = np.array([
        args.focal_div_weight,
        args.plurality_voting_weight,
        args.fleiss_kappa_weight,
        args.q_stats_weight,
        args.corr_coef_weight,
        args.binary_disagreement_weight,
        args.kappa_stats_weight,
        args.binary_entropy_weight])
    train_env = EnsembleEnv(data=train_data, 
                            num_models=num_models,
                            div_metric_weights=div_metric_weights,
                            device=args.device,
                            window_size=args.window_size, 
                            space_size=space_size,
                            task_name=args.task_name, 
                            alpha=args.alpha)

    policy1 = PolicyNetwork(input_dim=train_env.obsv_space_len, 
                            action_dims=np.ones(train_env.ac_space_len).astype(int) * 2)
    policy2 = MLP(input_dim=num_models * space_size, 
                  hidden_dim=[100, 100], 
                  output_dim=space_size)
    policy1 = policy1.to(args.device)
    policy2 = policy2.to(args.device)

    agent1 = REINFORCE(policy_network=policy1,
                       lr=args.select_agent_lr,
                       clip_epsilon=args.select_agent_clip_epsilon,
                       gamma=args.select_agent_gamma,
                       aggregate_loss="mean")
    agent2 = REINFORCE(policy2,
                       lr=args.ens_agent_lr,
                       clip_epsilon=args.ens_agent_clip_epsilon,
                       gamma=args.ens_agent_gamma,
                       aggregate_loss="sum")

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
    select_agent_rw, select_args, ens_args = train_loop(train_env, args.select_agent_epoch, select_args,
                                                        ens_args, num_models, max_tolerance=args.max_tolerance,
                                                        update_freq=args.update_freq)

    # train ensemble agent
    select_args["update"] = False
    ens_args["update"] = True
    ens_agent_rw, select_args, ens_args = train_loop(train_env, args.ensemble_agent_epoch, select_args,
                                                    ens_args, num_models, max_tolerance=args.max_tolerance,
                                                    update_freq=args.update_freq, perform_test_flag=True,
                                                    test_data=test_data, args=args, space_size=space_size)

    print("Starting testing")
    exp_results = {
        "select_agent_rw": np.array(select_agent_rw),
        "ens_agent_rw": np.array(ens_agent_rw),
        "select_policy": policy1.state_dict(),
        "ens_policy": policy2.state_dict()
    }

    ens_args["update"] = False
    select_args["update"] = True
    # ens_args["policy"] = None
    test_env = EnsembleEnv(test_data, num_models, device=args.device, window_size=0, space_size=space_size)
    test_reward, test_errors = step_policy(test_env, select_args, ens_args, num_models, ep_count=0, update_freq=args.update_freq, verbose=True)
    exp_results["test_reward"] = test_reward
    exp_results["test_errors"] = test_errors

    return exp_results


def save_exp(exp_results, task_name):
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints", task_name)
    dir_name = get_last_checkpoint_dirname(checkpoint_dir)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    select_agent_rewards, ens_agent_rewards, test_rewards = [], [], []
    for i, exp_res in enumerate(exp_results):
        torch.save(exp_res["select_policy"], os.path.join(dir_name, f'select_policy_{i}.pth'))
        torch.save(exp_res["ens_policy"], os.path.join(dir_name, f'ens_policy_{i}.pth'))

        select_agent_rewards.append(exp_res["select_agent_rw"])
        ens_agent_rewards.append(exp_res["ens_agent_rw"])
        test_rewards.append(exp_res["test_reward"])

    select_agent_rewards = np.mean(select_agent_rewards, axis=0)
    ens_agent_rewards = np.mean(ens_agent_rewards, axis=0)
    test_reward = np.mean(test_rewards, axis=0)

    print(f"Test Acc: {test_reward:.2f}%")
    score_path = os.path.join(dir_name, "test_score.txt")
    with open(score_path, "w") as file:
        file.write(f"Test Acc: {test_reward:.2f}%\n")

    select_arr_path = os.path.join(dir_name, "train_select_agent_rewards.npy")
    save_arr(select_agent_rewards, select_arr_path)

    ens_arr_path = os.path.join(dir_name, "train_ens_agent_rewards.npy")
    save_arr(ens_agent_rewards, ens_arr_path)

    print("Data is saved...")


def main(args):
    test_errors = []
    switched_ds_index = []
    task_names = ["bbh", "gsm8k", "mmlu_hf", "gpqa", "musr"]
    old_epoch_1, old_epoch_2 = args.select_agent_epoch, args.ensemble_agent_epoch
    for task_name in task_names:
        if task_name == "mmlu_hf":
            args.select_agent_epoch = 10
            args.ensemble_agent_epoch = 10
        else:
            args.select_agent_epoch = old_epoch_1
            args.ensemble_agent_epoch = old_epoch_2
            
        switched_ds_index.append(len(test_errors))
        datacreator = DataCreator(task_name, model_names=args.model_names, task_type="lang")
        # exp_results = []
        for i, (train_data, test_data, num_models, ds_name) in enumerate(datacreator.load()):
            print(f"Dataset-{i}: {ds_name}")
            exp_result = train_test(train_data, test_data, num_models, args)
            # exp_results.append(exp_result)
            test_errors += exp_result["test_errors"]
            print("\n")
    np.save("switched_ds_index_3.npy", switched_ds_index)
    np.save("test_errors_3.npy", test_errors)
    test_errors = np.array(test_errors)
    cumulative_acc = np.cumsum(test_errors) / np.arange(1, len(test_errors)+1)
    # save_exp(exp_results, args.task_name)
    # test_rewards = np.mean([exp_res["test_reward"] for exp_res in exp_results], axis=0)

    return cumulative_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train and test script for the rl-focal')
    # experiment arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_names", type=str, default="all")
    parser.add_argument("--dataset_type", type=str, default="lang", choices=["lang", "vision"])
    parser.add_argument("--task_name", type=str, default="musr", choices=["gsm8k", "mmlu_hf", "bbh", "gpqa", "musr"])

    # select agent arguments
    parser.add_argument("--alpha", type=float, default=1.0, help="size penalty constant")
    parser.add_argument("--select_agent_epoch", type=int, default=15)
    parser.add_argument("--select_agent_gamma", type=float, default=0.99)
    parser.add_argument("--select_agent_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--select_agent_lr", type=float, default=0.001)
    parser.add_argument("--focal_div_weight", type=float, default=0.5)
    parser.add_argument("--plurality_voting_weight", type=float, default=0.5)
    parser.add_argument("--fleiss_kappa_weight", type=float, default=0)
    parser.add_argument("--q_stats_weight", type=float, default=0)
    parser.add_argument("--corr_coef_weight", type=float, default=0)
    parser.add_argument("--binary_disagreement_weight", type=float, default=0)
    parser.add_argument("--kappa_stats_weight", type=float, default=0)
    parser.add_argument("--binary_entropy_weight", type=float, default=0)

    # ensemble agent arguments
    parser.add_argument("--ensemble_agent_epoch", type=int, default=30)
    parser.add_argument("--ens_agent_gamma", type=float, default=0.5)
    parser.add_argument("--ens_agent_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--ens_agent_lr", type=float, default=0.0001)

    # common argumentss for both agents
    parser.add_argument("--window_size", type=int, default=0)
    parser.add_argument("--max_tolerance", type=int, default=150)
    parser.add_argument("--update_freq", type=int, default=10)
    arguments = parser.parse_args()
    main(arguments)
