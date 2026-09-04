import random
import os

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n              # 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# -------------------------
# Q network
# -------------------------

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

q = DQN(state_dim, action_dim).to(device)

optimizer = optim.Adam(q.parameters(), lr=1e-3)

gamma = 0.99
epsilon = 1.0

# -------------------------
# epsilon-greedy
# -------------------------

def choose_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

    with torch.no_grad():
        return q(state_tensor).argmax().item()

# -------------------------
# SARSA training
# -------------------------

for episode in range(3000):
    
    state, _ = env.reset()
    action = choose_action(state)
    total_reward = 0

    while True:
    
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if not terminated:
            next_action = choose_action(next_state)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        q_values = q(state_tensor)[action]

        if terminated:
            target = torch.tensor(reward, dtype=torch.float32, device=device)
        else:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
            with torch.no_grad():
                target = reward + gamma * q(next_state_tensor)[next_action]

        # TD loss
        loss = (q_values - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        if not terminated:
            action = next_action

        total_reward += reward

        if done:
            break

    epsilon = max(0.05, epsilon * 0.98)
    
    if (episode + 1) % 100 == 0:
        print(
            f"episode={episode}, "
            f"reward={total_reward}, "
            f"epsilon={epsilon:.2f}"
        )

os.makedirs("checkpoints", exist_ok=True)
torch.save(q.state_dict(), "checkpoints/sarsa_cartpole.pt")
print("model save to checkpoints/sarsa_cartpole.pt")

env.close()
