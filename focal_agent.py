import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_dim[1], output_dim)
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.softmax(out, dim=-1)
        return out


class REINFORCE:
    def __init__(self, policy_network, lr=0.001):
        self.policy = policy_network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.5  # Discount factor
        self.log_probs = []
        self.rewards = []

    def store_outcome(self, log_prob, reward, sum_flag=False):
        if sum_flag: self.log_probs.append(log_prob.sum())
        else: self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)  # Normalize
        
        loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            loss.append(-log_prob * reward)
        
        self.optimizer.zero_grad()
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer.step()
        # print(loss)
        
        self.log_probs = []
        self.rewards = []

