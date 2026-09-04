import os

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# -------------------------
# Environment
# -------------------------

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n              # 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# -------------------------
# Policy Network
# -------------------------

policy = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim)
).to(device)

optimizer = optim.Adam(policy.parameters(), lr=1e-3)

gamma = 0.99


# -------------------------
# Training
# -------------------------

for episode in range(500):

    state, _ = env.reset()

    log_probs = []
    rewards = []

    while True:

        # 1. policy network outputs logits
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        logits = policy(state_tensor)

        # 2. convert logits to categorical policy
        dist = Categorical(logits=logits)

        # 3. sample action from policy
        action = dist.sample()

        # save log π(a|s)
        log_probs.append(dist.log_prob(action))

        # 4. interact with environment
        next_state, reward, terminated, truncated, _ = env.step(action.item())

        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break

    # -------------------------
    # Compute returns G_t
    # -------------------------

    returns = []

    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)

    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    # optional normalization
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # -------------------------
    # REINFORCE loss
    #
    # L = - sum G_t log π(a_t | s_t)
    # -------------------------

    loss = 0

    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G

    # -------------------------
    # Gradient update
    # -------------------------

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(rewards)
    
    if (episode + 1) % 50 == 0:
        print(
            f"episode={episode}, "
            f"reward={total_reward}"
        )

os.makedirs("checkpoints", exist_ok=True)
torch.save(policy.state_dict(), "checkpoints/reinforce_cartpole.pt")
print("model saved to checkpoints/reinforce_cartpole")

env.close()
