from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -4.0
LOG_STD_MAX = 0.5
EPS = 1e-8


def layer_init(
    layer: nn.Module, std: float = math.sqrt(2.0), bias_const: float = 0.0
) -> nn.Module:
    """Orthogonal init following the usual PPO recipe."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNGaussianActorCritic(nn.Module):
    def __init__(
        self,
        input_channels: int,
        action_dim: int,
        action_low: np.ndarray | torch.Tensor,
        action_high: np.ndarray | torch.Tensor,
        input_size: int = 84,
        init_log_std: float = -1.0,
    ):
        super().__init__()

        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        if action_low_t.ndim != 1 or action_high_t.ndim != 1:
            raise ValueError(
                "action_low and action_high must be 1D arrays for Box spaces."
            )
        if (
            action_low_t.shape != action_high_t.shape
            or action_low_t.shape[0] != action_dim
        ):
            raise ValueError("Action bounds shape mismatch with action_dim.")

        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
        )

        conv_out_dim = self._get_conv_out((input_channels, input_size, input_size))
        self.fc = layer_init(nn.Linear(conv_out_dim, 512))
        # PPO detail: small std for policy head, std=1 for value head.
        self.mu_head = layer_init(nn.Linear(512, action_dim), std=0.01)
        self.v_head = layer_init(nn.Linear(512, 1), std=1.0)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(init_log_std), dtype=torch.float32)
        )

    def _get_conv_out(self, shape: tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            out = self.conv(dummy)
            return int(np.prod(out.shape[1:]))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input with shape (B,H,W,C), got {tuple(x.shape)}"
            )

        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._encode(x)

        # PPO continuo: Gaussian diagonal policy.
        # Keep the mean inside the action bounds to reduce excessive clipping.
        mu_unscaled = torch.tanh(self.mu_head(encoded))
        mu = mu_unscaled * self.action_scale + self.action_bias

        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mu)
        value = self.v_head(encoded).squeeze(-1)
        return mu, log_std, value


class PPOAgentCont:
    def __init__(
        self,
        action_dim: int,
        action_low: np.ndarray | torch.Tensor,
        action_high: np.ndarray | torch.Tensor,
        input_channels: int,
        input_size: int = 84,
        device: str | torch.device = "cpu",
        lr: float = 1e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = 0.02,
        init_log_std: float = -1.0,
    ):
        self.device = torch.device(device)
        self.model = CNNGaussianActorCritic(
            input_channels=input_channels,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            input_size=input_size,
            init_log_std=init_log_std,
        ).to(self.device)
        # CleanRL/Baselines-style Adam epsilon.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        self.action_dim = int(action_dim)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.initial_lr = float(lr)

    @property
    def action_low(self) -> torch.Tensor:
        return self.model.action_low

    @property
    def action_high(self) -> torch.Tensor:
        return self.model.action_high

    def set_learning_rate(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def _dist_and_value(
        self, obs: torch.Tensor
    ) -> tuple[Normal, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std, value = self.model(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        return dist, mu, log_std, std, value

    def _log_prob_from_action(self, dist: Normal, action: torch.Tensor) -> torch.Tensor:
        return dist.log_prob(action).sum(dim=-1)

    def _entropy(self, dist: Normal) -> torch.Tensor:
        return dist.entropy().sum(dim=-1)

    def act(
        self, obs: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            dist, _, log_std, _, value = self._dist_and_value(obs_tensor)
            policy_action = dist.sample()
            log_prob = self._log_prob_from_action(dist, policy_action)
            env_action = torch.clamp(policy_action, self.action_low, self.action_high)
            action_std = log_std.exp().mean()

        return (
            env_action.squeeze(0).cpu().numpy().astype(np.float32),
            policy_action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
            float(action_std.item()),
        )

    def get_value(self, obs: np.ndarray | torch.Tensor) -> float:
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, _, value = self.model(obs_tensor)
        return float(value.item())

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        terminals: list[bool],
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        values_arr = np.asarray(values + [last_value], dtype=np.float32)
        terminals_arr = np.asarray(terminals, dtype=np.float32)

        advantages = np.zeros_like(rewards_arr, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards_arr))):
            non_terminal = 1.0 - terminals_arr[t]
            delta = (
                rewards_arr[t]
                + self.gamma * values_arr[t + 1] * non_terminal
                - values_arr[t]
            )
            gae = delta + self.gamma * self.lam * non_terminal * gae
            advantages[t] = gae

        returns = advantages + values_arr[:-1]
        return advantages, returns

    def update(
        self, buffer: dict, last_value: float, epochs: int = 4, batch_size: int = 64
    ) -> dict[str, float]:
        if len(buffer["obs"]) == 0:
            return {
                "loss": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
                "action_std": 0.0,
                "early_stop": 0.0,
            }

        obs = torch.as_tensor(
            np.asarray(buffer["obs"]), dtype=torch.float32, device=self.device
        )
        policy_actions = torch.as_tensor(
            np.asarray(buffer["policy_actions"]),
            dtype=torch.float32,
            device=self.device,
        )
        old_logprobs = torch.as_tensor(
            buffer["logprobs"], dtype=torch.float32, device=self.device
        )
        old_values = torch.as_tensor(
            buffer["values"], dtype=torch.float32, device=self.device
        )

        advantages_np, returns_np = self.compute_gae(
            rewards=buffer["rewards"],
            values=buffer["values"],
            terminals=buffer["terminals"],
            last_value=last_value,
        )

        advantages = torch.as_tensor(
            advantages_np, dtype=torch.float32, device=self.device
        )
        returns_raw = torch.as_tensor(
            returns_np, dtype=torch.float32, device=self.device
        )

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + EPS
        )

        # Stabilize the critic when reward shaping makes targets large/variable.
        returns_mean = returns_raw.mean()
        returns_std = returns_raw.std(unbiased=False) + EPS
        returns = (returns_raw - returns_mean) / returns_std

        n = obs.size(0)
        metrics = {
            "loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "action_std": [],
        }
        early_stop = False

        for _ in range(epochs):
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]

                b_obs = obs[batch_idx]
                b_policy_actions = policy_actions[batch_idx]
                b_old_logprobs = old_logprobs[batch_idx]
                b_old_values = old_values[batch_idx]
                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]

                dist, _, log_std, _, values_new = self._dist_and_value(b_obs)
                new_logprobs = self._log_prob_from_action(dist, b_policy_actions)
                entropy = self._entropy(dist).mean()

                log_ratio = new_logprobs - b_old_logprobs
                ratio = log_ratio.exp()

                surr1 = ratio * b_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                b_old_values_norm = (b_old_values - returns_mean) / returns_std
                values_new_norm = (values_new - returns_mean) / returns_std

                value_pred_clipped = b_old_values_norm + (
                    values_new_norm - b_old_values_norm
                ).clamp(-self.clip_eps, self.clip_eps)
                value_loss_unclipped = (values_new_norm - b_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - b_returns).pow(2)
                critic_loss = (
                    0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                )

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                    )
                    action_std = log_std.exp().mean().item()

                metrics["loss"].append(float(loss.item()))
                metrics["actor_loss"].append(float(actor_loss.item()))
                metrics["critic_loss"].append(float(critic_loss.item()))
                metrics["entropy"].append(float(entropy.item()))
                metrics["approx_kl"].append(float(approx_kl))
                metrics["clip_fraction"].append(float(clip_fraction))
                metrics["action_std"].append(float(action_std))

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        returns_var = float(np.var(returns_np))
        if returns_var < EPS:
            explained_variance = 0.0
        else:
            values_np = np.asarray(buffer["values"], dtype=np.float32)
            explained_variance = float(
                1.0 - np.var(returns_np - values_np) / (returns_var + EPS)
            )

        out = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}
        out["explained_variance"] = explained_variance
        out["early_stop"] = float(early_stop)
        return out
