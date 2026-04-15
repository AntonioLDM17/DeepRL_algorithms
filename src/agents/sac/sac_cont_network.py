# sac_cont_network.py
"""
Continuous SAC networks for pixel-based observations.

Key differences from the discrete version:
  - Actor outputs (mu, log_std) → reparameterised sample → tanh squash → rescale
  - Critic takes (state_features, action) as input and outputs a scalar Q-value
  - Log-prob requires the tanh correction: log π(a|s) = log π(u|s) − Σ log(1 − tanh²(u))

Init strategy: orthogonal init (same as PPO) for the conv/fc trunk, then
  small std=0.01 on the mu head to keep initial actions near-zero and avoid
  saturating tanh from the start.  The log_std head uses std=1.0 (standard).
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def layer_init(
    layer: nn.Module, std: float = math.sqrt(2.0), bias_const: float = 0.0
) -> nn.Module:
    """Orthogonal init — same recipe as PPO."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


def _conv_output_size(in_channels: int, image_size: int = 84) -> int:
    """Flattened size after the Nature-DQN conv stack (no ReLU needed for shape)."""
    dummy = torch.zeros(1, in_channels, image_size, image_size)
    conv = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    )
    with torch.no_grad():
        return int(conv(dummy).numel())


class CNNEncoder(nn.Module):
    """
    Nature-DQN encoder.
    Expects uint8 HWC input [B, H, W, C] — permutes and normalises internally,
    same convention as PPO's _encode().
    """

    def __init__(self, in_channels: int, feature_dim: int = 512, image_size: int = 84):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
        )
        flat_size = _conv_output_size(in_channels, image_size)
        self.fc = layer_init(nn.Linear(flat_size, feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C] uint8 or float32."""
        if x.ndim != 4:
            raise ValueError(
                f"CNNEncoder expects [B,H,W,C] input, got shape {tuple(x.shape)}"
            )
        x = x.permute(0, 3, 1, 2).float() / 255.0  # → [B, C, H, W]
        h = self.conv(x).reshape(x.size(0), -1)
        return F.relu(self.fc(h))


class ContinuousActor(nn.Module):
    """
    Outputs a squashed Gaussian policy for continuous action spaces.

    act() returns:
        action      [B, action_dim]  — tanh-squashed, rescaled to [low, high]
        log_prob    [B]              — log π(a|s) with tanh correction

    The tanh correction removes the bias that would arise from naively computing
    log-prob of the pre-squash sample under the Gaussian:
        log π(a|s) = log π(u|s) − Σ_i log(1 − tanh²(u_i) + ε)
    where u is the pre-squash sample and a = tanh(u).
    """

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        action_low: np.ndarray | torch.Tensor,
        action_high: np.ndarray | torch.Tensor,
        feature_dim: int = 512,
        image_size: int = 84,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, feature_dim, image_size)

        self.mu_head = layer_init(nn.Linear(feature_dim, action_dim), std=0.01)
        self.log_std_head = layer_init(nn.Linear(feature_dim, action_dim), std=1.0)

        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu [B,A], log_std [B,A]) — pre-squash parameters."""
        features = self.encoder(x)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action via reparameterisation, apply tanh squash + rescale.

        Returns:
            action    [B, action_dim]  in [action_low, action_high]
            log_prob  [B]
        """
        mu, log_std = self.forward(x)
        std = log_std.exp()

        # Reparameterisation: u = mu + std * ε,  ε ~ N(0,1)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()

        # Squash
        a_tanh = torch.tanh(u)
        action = a_tanh * self.action_scale + self.action_bias

        # Log-prob with tanh correction
        log_prob = dist.log_prob(u).sum(dim=-1)  # [B]
        log_prob -= torch.log(self.action_scale * (1 - a_tanh.pow(2)) + EPS).sum(dim=-1)

        return action, log_prob

    @torch.no_grad()
    def act_deterministic(self, x: torch.Tensor) -> np.ndarray:
        """Greedy action for evaluation — tanh(mu) rescaled, no sampling."""
        mu, _ = self.forward(x)
        return (
            (torch.tanh(mu) * self.action_scale + self.action_bias)
            .squeeze(0)
            .cpu()
            .numpy()
        )


class ContinuousCritic(nn.Module):
    """
    Twin Q-networks for continuous actions.

    Continuous SAC critics receive (state, action) and output a scalar Q-value,
    unlike the discrete version which outputs Q for all actions at once.

    Two separate CNN encoders + heads to ensure independence between Q1 and Q2.
    forward() → (Q1 [B], Q2 [B])
    """

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        feature_dim: int = 512,
        image_size: int = 84,
    ):
        super().__init__()

        # Q1
        self.encoder1 = CNNEncoder(in_channels, feature_dim, image_size)
        self.q1_head = nn.Sequential(
            layer_init(nn.Linear(feature_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )

        # Q2
        self.encoder2 = CNNEncoder(in_channels, feature_dim, image_size)
        self.q2_head = nn.Sequential(
            layer_init(nn.Linear(feature_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )

    def forward(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:      [B, H, W, C]
        action: [B, action_dim]  — should already be in env action space
        Returns: Q1 [B], Q2 [B]
        """
        f1 = self.encoder1(x)
        f2 = self.encoder2(x)
        sa1 = torch.cat([f1, action], dim=-1)
        sa2 = torch.cat([f2, action], dim=-1)
        return self.q1_head(sa1).squeeze(-1), self.q2_head(sa2).squeeze(-1)
