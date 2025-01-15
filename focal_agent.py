import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dims):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.action_heads = nn.ModuleList([
            nn.Linear(128, dim) for dim in action_dims
        ])

    def forward(self, x):
        x = F.relu(self.fc(x))
        outputs = [F.softmax(head(x), dim=-1) for head in self.action_heads]
        return outputs


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
    def __init__(self, policy_network, lr=0.001, clip_epsilon=0.2, gamma=0.99, aggregate_loss="mean"):
        self.policy = policy_network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.log_probs = []
        self.old_probs = []
        self.rewards = []
        self.aggregate_loss = aggregate_loss

    def update_policy_params(self, new_network):
        self.policy = new_network
        self.optimizer = optim.Adam(new_network.parameters(), lr=self.lr, eps=1e-5)
        # Clear memory
        self.log_probs = []
        self.rewards = []


    def store_outcome(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def store_old_policy(self):
        for log_probs in self.log_probs:
            temp = [log_prob.detach() for log_prob in log_probs]
            self.old_probs.append(temp)

    def update_policy(self):
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards)
        advantages = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        if len(self.old_probs) == 0:
            self.store_old_policy()

        # Compute PPO loss with clipping
        policy_loss = []
        for log_probs, old_probs, advantage in zip(self.log_probs, self.old_probs, advantages):
            for log_prob, old_prob in zip(log_probs, old_probs):
                ratio = torch.exp(log_prob - log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                policy_loss.append(-torch.min(surr1, surr2))

        # Optimize policy network
        self.optimizer.zero_grad()
        if self.aggregate_loss == "mean":
            policy_loss = torch.stack(policy_loss).mean()
        elif self.aggregate_loss == "sum":
            policy_loss = torch.stack(policy_loss).sum()
        else:
            raise RuntimeError
        policy_loss.backward()
        self.optimizer.step()

        # store old policies
        self.store_old_policy()

        # Clear memory
        self.log_probs = []
        self.rewards = []
