import torch
from ens_env import StockTradingEnv
from focal_agent import PolicyNetwork, REINFORCE


def train():
    env = StockTradingEnv(data)
    policy = PolicyNetwork(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
    agent = REINFORCE(policy)

    for episode in range(1000):  # Train for 1000 episodes
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            action_probs = policy(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            agent.store_outcome(action_dist.log_prob(action), reward)
            
            state = next_state
            episode_reward += reward
        
        agent.update_policy()
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")





if __name__ == "__main__":
    train()
