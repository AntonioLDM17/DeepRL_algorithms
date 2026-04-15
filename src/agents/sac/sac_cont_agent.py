# sac_cont_agent.py
"""
Continuous Soft Actor-Critic agent for pixel-based environments.

Differences from the discrete version (sac_agent.py):
  - Actor uses reparameterised Gaussian + tanh squash instead of Categorical
  - Critics take (state, action) → scalar Q instead of state → Q[A]
  - Bellman target uses a sampled next action (MC estimate, not full expectation)
  - No probability-weighted sums anywhere — single-sample estimates throughout
  - Target entropy defaults to -action_dim  (Haarnoja et al. recommendation)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.sac.sac_cont_network import ContinuousActor, ContinuousCritic


class SACAgentCont:
    def __init__(
        self,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        in_channels: int,
        cfg,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim

        feature_dim = getattr(cfg, "SAC_FEATURE_DIM", 512)
        image_size = getattr(cfg, "IMAGE_SIZE", 84)

        # ── Networks ─────────────────────────────────────────────────────
        self.actor = ContinuousActor(
            in_channels=in_channels,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            feature_dim=feature_dim,
            image_size=image_size,
        ).to(device)

        self.critic = ContinuousCritic(
            in_channels=in_channels,
            action_dim=action_dim,
            feature_dim=feature_dim,
            image_size=image_size,
        ).to(device)

        self.critic_target = ContinuousCritic(
            in_channels=in_channels,
            action_dim=action_dim,
            feature_dim=feature_dim,
            image_size=image_size,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.SAC_ACTOR_LR, eps=1e-4
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.SAC_CRITIC_LR, eps=1e-4
        )

        self.auto_entropy = getattr(cfg, "SAC_AUTO_ENTROPY", True)

        self.target_entropy = float(getattr(cfg, "SAC_TARGET_ENTROPY", -action_dim))

        init_alpha = float(getattr(cfg, "SAC_ALPHA_INIT", 0.2))
        self.log_alpha = torch.tensor(
            [math.log(init_alpha)], requires_grad=True, device=device
        )
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha], lr=cfg.SAC_ALPHA_LR, eps=1e-4
        )
        self._fixed_alpha = float(getattr(cfg, "SAC_ALPHA_INIT", 0.2))

        self.gamma = cfg.GAMMA
        self.tau = getattr(cfg, "SAC_TAU", 0.005)
        self.grad_clip = getattr(cfg, "GRAD_CLIP_NORM", 10.0)
        self.train_steps = 0

    @torch.no_grad()
    def act(self, imgs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        imgs: uint8 HWC numpy array [H, W, C]
        Returns: action numpy array [action_dim] in env action space
        """
        x = torch.as_tensor(imgs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            return self.actor.act_deterministic(x)
        action, _ = self.actor.get_action(x)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def learn(self, replay) -> dict[str, float]:
        """
        One gradient update.  replay.sample() must return:
            imgs [B,H,W,C], actions [B,A], rewards [B], next_imgs [B,H,W,C], dones [B]
        """
        imgs, actions, rewards, next_imgs, dones = replay.sample()

        alpha = (
            self.log_alpha.exp().detach() if self.auto_entropy else self._fixed_alpha
        )

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.get_action(
                next_imgs
            )  # [B,A], [B]

            next_q1_t, next_q2_t = self.critic_target(next_imgs, next_actions)
            next_v = torch.min(next_q1_t, next_q2_t) - alpha * next_log_probs  # [B]

            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        q1, q2 = self.critic(imgs, actions)
        qf1_loss = F.mse_loss(q1, target_q)
        qf2_loss = F.mse_loss(q2, target_q)
        critic_loss = 0.5 * (qf1_loss + qf2_loss)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()

        for p in self.critic.parameters():
            p.requires_grad = False

        new_actions, log_probs = self.actor.get_action(imgs)  # [B,A], [B]

        q1_pi, q2_pi = self.critic(imgs, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * log_probs - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        # Unfreeze critic for next iteration
        for p in self.critic.parameters():
            p.requires_grad = True

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_probs.detach() + self.target_entropy)
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        self._soft_update(self.critic, self.critic_target)

        self.train_steps += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "qf1_loss": float(qf1_loss.item()),
            "qf2_loss": float(qf2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha),
        }

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.copy_(self.tau * p_src.data + (1.0 - self.tau) * p_tgt.data)

    def update_target(self):
        """Hard target update — kept for API parity."""
        self.critic_target.load_state_dict(self.critic.state_dict())
