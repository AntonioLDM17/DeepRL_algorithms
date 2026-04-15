import torch
import torch.nn.functional as F
import numpy as np
from src.agents.rainbow.rainbow_network import RainbowDQN
 
class RainbowAgent:
    def __init__(self, num_actions, cfg, device):
        self.cfg = cfg
        self.device = device
        self.num_actions = num_actions
        self.use_distributional = bool(getattr(cfg, "USE_DISTRIBUTIONAL", True))
 
        # C51 support
        self.num_atoms = cfg.NUM_ATOMS
        self.v_min = cfg.V_MIN
        self.v_max = cfg.V_MAX
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
 
        self.online = RainbowDQN(
            num_actions=num_actions,
            num_atoms=self.num_atoms,
            use_dueling=cfg.USE_DUELING,
            use_noisy=cfg.USE_NOISY_NETS,
            noisy_std_init=cfg.NOISY_STD_INIT
        ).to(device)
 
        self.target = RainbowDQN(
            num_actions=num_actions,
            num_atoms=self.num_atoms,
            use_dueling=cfg.USE_DUELING,
            use_noisy=cfg.USE_NOISY_NETS,
            noisy_std_init=cfg.NOISY_STD_INIT
        ).to(device)
 
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
 
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.LR)
 
        self.gamma_n = (cfg.N_STEP_GAMMA ** cfg.N_STEP) if cfg.USE_N_STEP else cfg.GAMMA
 
        self.train_steps = 0
 
    @torch.no_grad()
    def act(self, imgs, epsilon=None):
        """
        For Rainbow with NoisyNets: typically epsilon-greedy can be disabled.
        But we allow epsilon parameter; if None and noisy enabled -> greedy.
        imgs: uint8 HWC stacked
        """
        if epsilon is not None and np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
 
        x = torch.tensor(imgs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        x = x.permute(0, 3, 1, 2)
        q = self.online.q_values(x, self.support)  # [1, A]
        return int(torch.argmax(q, dim=1).item())
 
    def reset_noise(self):
        if self.cfg.USE_NOISY_NETS:
            self.online.reset_noise()
            self.target.reset_noise()
 
    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())
 
    def _projection(self, next_dist, rewards, dones):
        """
        C51 projection of target distribution onto fixed support.
        next_dist: [B, atoms] for chosen next action
        rewards: [B]
        dones: [B] float (1 if done)
        """
        B = rewards.size(0)
        Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * self.gamma_n * self.support.view(1, -1)
        Tz = torch.clamp(Tz, self.v_min, self.v_max)
 
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = l.clamp(0, self.num_atoms - 1)
        u = u.clamp(0, self.num_atoms - 1)
 
        m = torch.zeros(B, self.num_atoms, device=self.device)
 
        # distribute probability mass
        offset = torch.arange(B, device=self.device).unsqueeze(1) * self.num_atoms
        l_idx = (l + offset).view(-1)
        u_idx = (u + offset).view(-1)
 
        next_dist_flat = next_dist.view(-1)
 
        m_flat = m.view(-1)
        eq_mask = (u == l)
        m_flat.index_add_(0, l_idx, (next_dist * (u.float() - b + eq_mask.float())).view(-1))
        m_flat.index_add_(0, u_idx, (next_dist * (b - l.float())).view(-1))
 
        m = m_flat.view(B, self.num_atoms)
        return m
 
    def learn(self, replay):
        """
        Sample from PER + n-step and run either:
          - distributional (C51) update (cfg.USE_DISTRIBUTIONAL=True), or
          - standard TD update on expected Q values (cfg.USE_DISTRIBUTIONAL=False).
        replay.sample() returns:
          imgs [B,C,H,W], actions [B], rewards [B], next_imgs [B,C,H,W], dones [B], weights [B], idxs
        """
        imgs, actions, rewards, next_imgs, dones, weights, idxs = replay.sample()

        if not self.use_distributional:
            q = self.online.q_values(imgs, self.support)  # [B,A]
            q_a = q.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

            with torch.no_grad():
                if self.cfg.USE_DOUBLE_DQN:
                    next_q_online = self.online.q_values(next_imgs, self.support)  # [B,A]
                    next_actions = torch.argmax(next_q_online, dim=1)  # [B]
                    next_q_target = self.target.q_values(next_imgs, self.support)  # [B,A]
                    next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # [B]
                else:
                    next_q_target = self.target.q_values(next_imgs, self.support)
                    next_q = next_q_target.max(dim=1)[0]

                target = rewards + (1.0 - dones) * self.gamma_n * next_q  # [B]

            per_sample_loss = F.smooth_l1_loss(q_a, target, reduction="none")  # [B]
            loss = (weights * per_sample_loss).mean()

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.GRAD_CLIP_NORM)
            self.optim.step()

            td_error = (target - q_a).abs().detach().cpu().numpy()
            new_priorities = td_error + self.cfg.PER_EPS
            replay.update_priorities(idxs, new_priorities)

            self.train_steps += 1

            if self.cfg.USE_NOISY_NETS:
                self.reset_noise()

            return float(loss.item()), float(per_sample_loss.mean().item())
 
        # Current dist for taken actions
        logits = self.online(imgs)  # [B,A,atoms]
        log_probs = F.log_softmax(logits, dim=-1)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_atoms)  # [B,1,atoms]
        log_p_a = log_probs.gather(1, actions).squeeze(1)  # [B,atoms]
 
        with torch.no_grad():
            # Next action selection: Double DQN uses online for argmax on expected Q
            if self.cfg.USE_DOUBLE_DQN:
                next_q = self.online.q_values(next_imgs, self.support)  # [B,A]
                next_actions = torch.argmax(next_q, dim=1)  # [B]
            else:
                next_q = self.target.q_values(next_imgs, self.support)
                next_actions = torch.argmax(next_q, dim=1)
 
            # Next dist from target for those actions
            next_logits_t = self.target(next_imgs)
            next_probs_t = F.softmax(next_logits_t, dim=-1)  # [B,A,atoms]
            na = next_actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_atoms)
            next_dist = next_probs_t.gather(1, na).squeeze(1)  # [B,atoms]
 
            target_dist = self._projection(next_dist, rewards, dones)  # [B,atoms]
            target_dist = torch.clamp(target_dist, 1e-8, 1.0)  # avoid log(0)
 
        # Cross-entropy loss: -sum target * log p
        per_sample_loss = -(target_dist * log_p_a).sum(dim=1)  # [B]
 
        # PER importance weights
        loss = (weights * per_sample_loss).mean()
 
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.GRAD_CLIP_NORM)
        self.optim.step()
 
        # Update priorities: use per-sample loss (proxy for TD error)
        new_priorities = per_sample_loss.detach().cpu().numpy() + self.cfg.PER_EPS
        replay.update_priorities(idxs, new_priorities)
 
        self.train_steps += 1
        
        if self.cfg.USE_NOISY_NETS:
            self.reset_noise()

        return float(loss.item()), float(per_sample_loss.mean().item())
