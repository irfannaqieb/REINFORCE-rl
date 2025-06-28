import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random, numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-2
MAX_EPISODES = 1_000
SOLVED_AT = 475
WINDOW_SIZE = 100


def build_policy() -> nn.Module:
    """
    Builds a policy network for the CartPole environment.
    """
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    return model


policy = build_policy()
optimizer = optim.Adam(policy.parameters(), lr=LR)
print(
    "Policy has, ", sum(p.numel() for p in policy.parameters()), "params"
)  # sanity check


def select_action(state):
    """
    Selects an action based on the current state.
    """
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    logits = policy(state_tensor)
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)

    return action.item(), log_prob.squeeze()


def run_episode(env):
    """
    Runs an episode and returns the log probabilities and rewards.
    """
    state, _ = env.reset(seed=SEED)
    log_probs, rewards = [], []

    while True:
        action, log_p = select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        log_probs.append(log_p)
        rewards.append(reward)

        if terminated or truncated:
            break
    return log_probs, rewards


def compute_returns(rewards, gamma=GAMMA):
    """
    Computes the returns for each step in the episode
    """
    returns = []
    running_sum = 0

    for r in reversed(rewards):
        running_sum = r + gamma * running_sum
        returns.append(running_sum)

    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    return returns


def train():
    env = gym.make(ENV_NAME)
    scores = deque(maxlen=WINDOW_SIZE)

    for episode in range(1, MAX_EPISODES + 1):
        log_probs, rewards = run_episode(env)
        returns = compute_returns(rewards)
        log_probs = torch.stack(log_probs)

        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores.append(sum(rewards))
        if episode % 20 == 0:
            avg = np.mean(scores)
            print(f"Episode {episode:4d} | avg reward {avg:6.2f}")

        if len(scores) == WINDOW_SIZE and np.mean(scores) >= SOLVED_AT:
            print(f"Solved in {episode} episodes")
            break
    env.close()


if __name__ == "__main__":
    train()
