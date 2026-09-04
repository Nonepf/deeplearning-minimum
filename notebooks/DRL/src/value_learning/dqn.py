import random
from collections import deque
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------
# Environment
# -------------------------

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]  # 4
action_dim = env.action_space.n             # 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# -------------------------
# Q Network
# -------------------------

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


q = DQN(state_dim, action_dim).to(device)
q_target = DQN(state_dim, action_dim).to(device)

q_target.load_state_dict(q.state_dict())

optimizer = optim.Adam(q.parameters(), lr=1e-3)


# -------------------------
# Replay Buffer
# -------------------------

buffer = deque(maxlen=10000)


# -------------------------
# Hyperparameters
# -------------------------

gamma = 0.99
batch_size = 64
epsilon = 1.0


# -------------------------
# Training
# -------------------------

def choose_action(state):
    # epsilon-greedy
    if random.random() < epsilon:
        return env.action_space.sample()
    
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    with torch.no_grad():
        return q(state_tensor).argmax().item()



for episode in range(300):

    state, _ = env.reset()
    total_reward = 0

    while True:
            
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # save transition
        buffer.append((state, action, reward, next_state, terminated))

        state = next_state
        total_reward += reward

        # train
        if len(buffer) >= batch_size:

            batch = random.sample(buffer, batch_size)

            states, actions, rewards, next_states, terminateds = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            actions = torch.tensor(actions, device=device).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
            terminateds = torch.tensor(terminateds, dtype=torch.float32, device=device)

            # Q(s, a)
            q_values = q(states).gather(1, actions).squeeze()

            # r + gamma max Q_target(s', a')
            with torch.no_grad():
                next_q_values = q_target(next_states).max(1).values
                targets = rewards + gamma * (1 - terminateds) * next_q_values
            loss = ((q_values - targets) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # epsilon decay
    epsilon = max(0.05, epsilon * 0.98)

    # update target network
    if episode % 10 == 0:
        q_target.load_state_dict(q.state_dict())

    print(
        f"episode={episode}, "
        f"reward={total_reward}, "
        f"epsilon={epsilon:.2f}"
    )

os.makedirs("checkpoints", exist_ok=True)
torch.save(q.state_dict(), "checkpoints/dqn_cartpole.pt")
print("model saved to checkpoints/dqn_cartpole.pt")

env.close()
