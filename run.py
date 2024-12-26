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
    exploration_noise = 0.01
    episode_reward = 0
    for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 1)):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs = select_policy(state)
        action = (action_probs + torch.randn_like(action_probs) * exploration_noise > 0.5).int().detach().cpu().numpy()

        action_count += action
        
        if ens_update:
            next_state, reward, pred_prob = train_env.step(action, ens_policy)
        else:
            next_state, reward, pred_prob = train_env.step(action)

        if select_update: select_agent.store_outcome(torch.log(action_probs), reward)
        if ens_update: ens_agent.store_outcome(torch.log(pred_prob), reward)
        state = next_state
        episode_reward += reward

    if select_update:
        select_agent.update_policy()
    
    if ens_update:
        ens_agent.update_policy()

    print(f"Episode {ep_count}, Total Reward: {episode_reward}")
    print(action_count / count)
    return episode_reward




def train(train_data, num_models, n_episodes=100):
    train_env = EnsembleEnv(train_data, num_models, device=device)
    policy1 = PolicyNetwork(input_dim=train_env.obsv_space_len, 
                           output_dim=train_env.ac_space_len).to(device)
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
    exploration_noise = 0.01
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

    return agents, policies

def test(test_data, num_models, agent, policy):
    test_env = EnsembleEnv(test_data, num_models, window_size=0)
    state = test_env.reset()
    episode_reward = 0
    exploration_noise = 0
    action_count = np.zeros(num_models)
    for count in tqdm.tqdm(range(test_env.total_len - test_env.initial_step - 1)):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs = policy(state)
        action = (action_probs + torch.randn_like(action_probs) * exploration_noise > 0.5).int().detach().cpu().numpy()
        # action = action_dist.sample()

        action_count += action
        next_state, reward, pred_prob = test_env.step(action)
        agent.store_outcome(torch.log(action_probs), reward)

        state = next_state
        episode_reward += reward

        agent.update_policy()
    print(f"Test Total Reward: {episode_reward}")
    print(f"Total Acc: {(episode_reward / count) * 100:.2f}%")


def main():
    dataset_name="mmlu_hf"
    model_names = "all"

    m_names = ["Llama-2-13b", "Llama-2-7b", 
               "Mixtral-8x7B", "gemma-7b", "Llama-2-70b", 
               "Mistral-7B", "gemma-2b", "phi-2"]

    datacreator = DataCreator(dataset_name, model_names=model_names, task_type="lang")
    train_data, test_data, num_models = datacreator.create()

    agent, policy = train(train_data, num_models, n_episodes=50)
    # test(test_data, num_models, agent, policy)


if __name__ == "__main__":
    main()
