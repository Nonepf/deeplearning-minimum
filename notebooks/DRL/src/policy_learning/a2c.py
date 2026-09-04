# copied from actor-critic.py

import os

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Environment

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n              # 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# --------------------------------
# Actor-Critic Network
# --------------------------------

actor = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim),
).to(device)

critic = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
).to(device)

optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
gamma = 0.99


# --------------------------------
# Training
# --------------------------------

rollout_steps = 5

for episode in range(500):

    state, _ = env.reset()
    total_reward = 0

    while True:

        rewards = []
        log_probs = []
        values = []
        dones = []
        
        for _ in range(rollout_steps):

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            logits, value = actor(state_tensor), critic(state_tensor)

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.squeeze())
            dones.append(done)

            if done:
                break
    


    if (episode + 1) % 50 == 0:
        print(
            f"episode={episode}, "
            f"reward={total_reward}"
        )

os.makedirs("checkpoints", exist_ok=True)
torch.save(actor.state_dict(), "checkpoints/actor_critic_cartpole.pt")
print("saved to checkpoints/actor_critic_cartpole.pt")

env.close()
