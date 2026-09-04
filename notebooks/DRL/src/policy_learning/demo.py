import argparse

import gymnasium as gym
import torch
import torch.nn as nn

# Parse Input

parser = argparse.ArgumentParser()
parser.add_argument("name", help="model name(REINFORCE/AC)")
args = parser.parse_args()

model_path = None
if args.name == "REINFORCE":
    model_path = "checkpoints/reinforce_cartpole.pt"
elif args.name == "AC":
    model_path = "checkpoints/actor_critic_cartpole.pt"
else:
    raise RuntimeError("model name dosn't exist!")

# Preparing Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

env = gym.make("CartPole-v1", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim),
).to(device)
policy.load_state_dict(torch.load(model_path, map_location=device))
policy.eval()

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = policy(state_tensor)
            action = logits.argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(
        f"episode={episode}, "
        f"reward={total_reward}"
    )

env.close()

