import tqdm
import torch
import numpy as np
from env.ens_env import EnsembleEnv
from focal_agent import PolicyNetwork, MLP, REINFORCE
from data_loader import DataCreator
from plotting import plot_actions, plot_rewards
import matplotlib.pyplot as plt


device = "cuda"

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
        action_count += action

        if ens_update:
            next_state, reward, pred_prob = train_env.step(action, ens_policy)
        else:
            next_state, reward, pred_prob = train_env.step(action)

        if select_update:
            select_agent.store_outcome(log_probs, reward)
        if ens_update:
            ens_agent.store_outcome(torch.log(pred_prob), reward)

        state = next_state
        episode_reward += np.max([reward, 0])

        if count % 100 == 0 and count > 0:
            if ens_update:
                ens_agent.update_policy()

            if select_update:
                select_agent.update_policy()

    print(f"Episode: {ep_count}, Total Acc: {(episode_reward / count) * 100:.2f}%")
    print(action_count)
    return episode_reward



def train(train_data, test_data, num_models, n_episodes=100):
    train_env = EnsembleEnv(train_data, num_models, device=device, window_size=500)
    policy1 = PolicyNetwork(train_env.obsv_space_len, 
                            np.ones(train_env.ac_space_len).astype(int) * 2).to(device)
    policy2 = MLP(num_models * 4, [100, 100], 4).to(device)

    agent1 = REINFORCE(policy1)
    agent2 = REINFORCE(policy2)

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
    exploration_noise = 0.1
    # fig, ax = plt.subplots()
    train_rewards = []
    for episode in range(n_episodes):
       
        if episode < 50:
            episode_reward = step_policy(train_env, select_args, ens_args, num_models, ep_count=episode)
        else:
            select_args["update"] = False
            ens_args["update"] = True
            episode_reward = step_policy(train_env, select_args, ens_args, num_models, ep_count=episode)

        if prev_reward < episode_reward:
            print("better reward")
            # plot_actions(action_count, count, m_names, episode, ax)
        prev_reward = episode_reward
        train_rewards.append(episode_reward)

    train_rewards = np.array(train_rewards)

    plot_rewards(train_rewards, "train_rewards")

    agents = (agent1, agent2)
    policies = (policy1, policy2)

    test(test_data, select_args, ens_args, num_models)

    return agents, policies

def test(test_data, select_args, ens_args, num_models):
    test_env = EnsembleEnv(test_data, num_models, window_size=0)
    select_policy = select_args["policy"] 
    select_agent = select_args["agent"]
    select_update = select_args["update"]

    ens_policy = ens_args["policy"] 
    ens_agent = ens_args["agent"]
    ens_update = ens_args["update"]

    state = test_env.reset()
    episode_reward = 0
    action_count = np.zeros(num_models)
    for count in tqdm.tqdm(range(test_env.total_len - test_env.initial_step - 2)):
        action_probs = select_policy(state)
        dists = [torch.distributions.Categorical(prob) for prob in action_probs]
        action = [dist.sample() for dist in dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, action)]
        action = np.array([a.detach().item() for a in action])
        action_count += action

        if ens_update:
            next_state, reward, pred_prob = test_env.step(action, ens_policy)
        else:
            next_state, reward, pred_prob = test_env.step(action)

        if select_update:
            select_agent.store_outcome(log_probs, reward)
        if ens_update:
            ens_agent.store_outcome(torch.log(pred_prob), reward)

        state = next_state
        episode_reward += np.max([reward, 0])

        if count % 100 == 0:
            if ens_update:
                ens_agent.update_policy()

            if select_update:
                select_agent.update_policy()

    print(f"Test Acc: {(episode_reward / count) * 100:.2f}%")
    print(action_count)


def main():
    dataset_name="mmlu_hf"

    m_names = ['Mistral-7B-Instruct-v0.2', 'Mixtral-8x7B-v0.1', 
               'gemma-2b', 'gemma-7b', 'Llama-2-13b-hf', 'phi-2',
                 'Llama-2-70b-hf', 'Llama-2-7b-hf']

    datacreator = DataCreator(dataset_name, model_names=m_names, task_type="lang")
    train_data, test_data, num_models = datacreator.create()

    agent, policy = train(train_data, test_data, num_models, n_episodes=100)
    # test(test_data, num_models, agent, policy)


if __name__ == "__main__":
    main()
