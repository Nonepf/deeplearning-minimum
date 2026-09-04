# run dqn.py before you start dqn_demo.py

import argparse

import gymnasium as gym
import torch
import torch.nn as nn

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

parser = argparse.ArgumentParser()
parser.add_argument("name", help="model name(dqn/sarsa)")
args = parser.parse_args()

model_path = None
if args.name == "dqn":
    model_path = "checkpoints/dqn_cartpole.pt"
elif args.name == "sarsa":
    model_path = "checkpoints/sarsa_cartpole.pt"
else:
    raise RuntimeError("model name doesn't exist!")

assert model_path is not None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

env = gym.make("CartPole-v1", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q = DQN(state_dim, action_dim).to(device)
q.load_state_dict(torch.load(model_path, map_location=device))
q.eval()

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            q_values = q(state_tensor)
            action = q_values.argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(
        f"episode={episode}, "
        f"reward={total_reward}"
    )

env.close()
