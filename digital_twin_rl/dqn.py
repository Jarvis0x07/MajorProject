"""Double DQN doctor agent trained inside the patient-twin environment.

Invalid actions (safety rules, and the BCQ support filter when enabled) are
masked both when acting and when bootstrapping the Double-DQN target, so the
agent never learns from or proposes a masked regimen.

Targets are n-step returns. Episodes last at most a few treatment lines, so
with the default n the target is the observed return of the rest of the
episode; one-step bootstrapping propagated noisy value estimates through
several stochastic lines and under-learned which first-line regimen is best.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .environment import TwinTreatmentEnv


@dataclass
class DQNConfig:
    episodes: int = 30000
    gamma: float = 0.99
    n_step: int = 8
    lr: float = 5e-4
    batch_size: int = 128
    buffer_size: int = 100_000
    warmup_steps: int = 1000
    target_sync_steps: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.6
    hidden: int = 128
    seed: int = 42


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.discount = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, done, next_mask, discount) -> None:
        i = self.cursor
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.done[i], self.next_mask[i] = next_obs, done, next_mask
        self.discount[i] = discount
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, steps: list[tuple], gamma: float, n_step: int) -> None:
        """steps: (obs, action, reward, next_obs, terminated, next_mask) in order."""
        length = len(steps)
        for t in range(length):
            end = min(t + n_step, length)
            ret = sum(gamma ** k * steps[t + k][2] for k in range(end - t))
            _, _, _, boot_obs, terminated, boot_mask = steps[end - 1]
            self.add(steps[t][0], steps[t][1], ret, boot_obs, float(terminated), boot_mask, gamma ** (end - t))

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {"obs": torch.from_numpy(self.obs[idx]), "action": torch.from_numpy(self.action[idx]),
                "reward": torch.from_numpy(self.reward[idx]), "next_obs": torch.from_numpy(self.next_obs[idx]),
                "done": torch.from_numpy(self.done[idx]), "next_mask": torch.from_numpy(self.next_mask[idx]),
                "discount": torch.from_numpy(self.discount[idx])}


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, q, torch.full_like(q, -1e9)).argmax(dim=-1)


def train_dqn(env: TwinTreatmentEnv, config: DQNConfig, log_every: int = 1000) -> tuple[QNetwork, list[dict]]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np_rng = np.random.default_rng(config.seed)
    env.rng.seed(config.seed)
    online = QNetwork(env.observation_dim, env.action_count, config.hidden)
    target = QNetwork(env.observation_dim, env.action_count, config.hidden)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size, env.observation_dim, env.action_count)
    decay_episodes = max(1, int(config.episodes * config.epsilon_decay_fraction))
    steps, returns, history = 0, [], []

    for episode in range(config.episodes):
        epsilon = max(config.epsilon_end, config.epsilon_start - (config.epsilon_start - config.epsilon_end) * episode / decay_episodes)
        obs, info = env.reset()
        episode_return, done, episode_steps = 0.0, False, []
        while not done:
            mask = info["action_mask"]
            if random.random() < epsilon:
                action = random.choice(np.flatnonzero(mask).tolist())
            else:
                with torch.no_grad():
                    q = online(torch.from_numpy(obs).unsqueeze(0))[0]
                action = int(masked_argmax(q, torch.from_numpy(mask)))
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Truncation (line limit) is not a true terminal state, so it still bootstraps.
            episode_steps.append((obs, action, reward, next_obs, terminated, info["action_mask"]))
            obs, episode_return, done = next_obs, episode_return + reward, terminated or truncated
        buffer.add_episode(episode_steps, config.gamma, config.n_step)
        for _ in episode_steps:
            steps += 1
            if buffer.size >= config.warmup_steps:
                batch = buffer.sample(config.batch_size, np_rng)
                q_taken = online(batch["obs"]).gather(1, batch["action"].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_action = masked_argmax(online(batch["next_obs"]), batch["next_mask"])
                    next_q = target(batch["next_obs"]).gather(1, next_action.unsqueeze(1)).squeeze(1)
                    target_q = batch["reward"] + batch["discount"] * (1.0 - batch["done"]) * next_q
                loss = F.smooth_l1_loss(q_taken, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
            if steps % config.target_sync_steps == 0:
                target.load_state_dict(online.state_dict())
        returns.append(episode_return)
        if (episode + 1) % log_every == 0:
            recent = returns[-log_every:]
            history.append({"episode": episode + 1, "epsilon": round(epsilon, 3),
                            "mean_return": float(np.mean(recent)), "steps": steps})
            print(f"  episode {episode + 1:>6}  epsilon {epsilon:.2f}  mean return {np.mean(recent):.3f}")
    online.eval()
    return online, history


class DQNPolicy:
    def __init__(self, network: QNetwork, name: str = "dqn"):
        self.network, self.name = network, name

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.network(torch.from_numpy(observation).unsqueeze(0))[0].numpy()

    def act(self, observation: np.ndarray, info: dict) -> int:
        if not info["action_mask"].any():
            raise ValueError("Action mask allows no regimen")
        return int(np.argmax(np.where(info["action_mask"], self.q_values(observation), -np.inf)))


def save_agent(path: Path, network: QNetwork, config: DQNConfig, regimens: list[list[str]], extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": network.state_dict(), "config": asdict(config), "regimens": regimens,
                "obs_dim": network.net[0].in_features, "n_actions": network.net[-1].out_features, **extra}, path)


def load_agent(path: Path) -> tuple[QNetwork, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    network = QNetwork(checkpoint["obs_dim"], checkpoint["n_actions"], checkpoint["config"]["hidden"])
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return network, checkpoint
