import torch
from env.ens_env import EnsembleEnv
from focal_agent import PolicyNetwork, REINFORCE
from data_loader import DataCreator


def train():
    dataset_name="mmlu_hf"
    model_names = "all"

    datacreator = DataCreator(dataset_name, model_names=model_names, task_type="lang")
    data, num_models = datacreator.create()

    env = EnsembleEnv(data, num_models)
    policy = PolicyNetwork(input_dim=env.obsv_space_len, output_dim=env.ac_space_len)
    agent = REINFORCE(policy)

    for episode in range(1000):  # Train for 1000 episodes
        state = env.reset()
        done = False
        episode_reward = 0
        
        while env.current_step < 4000:
            state = torch.tensor(state, dtype=torch.float32)
            action_probs = policy(state)
            action = (action_probs > 0.5).int()
            # action = action_dist.sample()

            next_state, reward, done, _ = env.step(action.numpy())
            agent.store_outcome(torch.log(action_probs), reward)

            state = next_state
            episode_reward += reward

        agent.update_policy()
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")



if __name__ == "__main__":
    train()
