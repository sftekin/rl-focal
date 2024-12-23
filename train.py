import torch
import numpy as np
from env.ens_env import EnsembleEnv
from focal_agent import PolicyNetwork, REINFORCE
from data_loader import DataCreator
from plotting import plot_actions
import matplotlib.pyplot as plt


def train():
    dataset_name="mmlu_hf"
    model_names = "all"
    device = "cuda"

    m_names = ["Llama-2-13b", "Llama-2-7b", 
               "Mixtral-8x7B", "gemma-7b", "Llama-2-70b", 
               "Mistral-7B", "gemma-2b", "phi-2"]

    datacreator = DataCreator(dataset_name, model_names=model_names, task_type="lang")
    data, num_models = datacreator.create()

    env = EnsembleEnv(data, num_models)
    policy = PolicyNetwork(input_dim=env.obsv_space_len, output_dim=env.ac_space_len).to(device)
    agent = REINFORCE(policy)

    prev_reward = 0
    fig, ax = plt.subplots()
    for episode in range(1000):  # Train for 1000 episodes
        state = env.reset()
        done = False
        episode_reward = 0

        action_count = np.zeros(len(m_names))
        count = 0
        while not done:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            action_probs = policy(state)
            action = (action_probs > 0.5).int().detach().cpu().numpy()
            # action = action_dist.sample()

            action_count += action
            next_state, reward, done, _ = env.step(action)
            agent.store_outcome(torch.log(action_probs), reward)

            state = next_state
            episode_reward += reward
            if env.get_current_progress() % 10 == 0 or count == 0:
                print(f"{env.get_current_progress():.2f}%")
            count += 1


        agent.update_policy()
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")
        if prev_reward < episode_reward:
            plot_actions(action_count, count, m_names, episode, ax)
        prev_reward = episode_reward



if __name__ == "__main__":
    train()
