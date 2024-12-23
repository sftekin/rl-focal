import tqdm
import torch
import numpy as np
from env.ens_env import EnsembleEnv
from focal_agent import PolicyNetwork, REINFORCE
from data_loader import DataCreator
from plotting import plot_actions, plot_rewards
import matplotlib.pyplot as plt


device = "cuda"


def train(train_data, num_models, n_episodes=100):
    train_env = EnsembleEnv(train_data, num_models)
    policy = PolicyNetwork(input_dim=train_env.obsv_space_len, 
                           output_dim=train_env.ac_space_len).to(device)
    agent = REINFORCE(policy)

    prev_reward = 0
    exploration_noise = 0.01
    # fig, ax = plt.subplots()
    train_rewards = []
    for episode in range(n_episodes):
        state = train_env.reset()
        done = False
        episode_reward = 0

        action_count = np.zeros(num_models)
        for count in tqdm.tqdm(range(train_env.total_len - train_env.window_size - 1)):
            state = torch.tensor(state, dtype=torch.float32).to(device)
            action_probs = policy(state)
            action = (action_probs + torch.randn_like(action_probs) * exploration_noise > 0.5).int().detach().cpu().numpy()
            # action = action_dist.sample()

            action_count += action
            next_state, reward, done, _ = train_env.step(action)
            agent.store_outcome(torch.log(action_probs), reward)

            state = next_state
            episode_reward += reward
        agent.update_policy()
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")
        if prev_reward < episode_reward:
            print(action_count / count)
            # plot_actions(action_count, count, m_names, episode, ax)
        prev_reward = episode_reward
        train_rewards.append(episode_reward)

    train_rewards = np.array(train_rewards)

    plot_rewards(train_rewards, "train_rewards")

    return agent, policy

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
        next_state, reward, done, _ = test_env.step(action)
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
    test(test_data, num_models, agent, policy)




if __name__ == "__main__":
    main()
