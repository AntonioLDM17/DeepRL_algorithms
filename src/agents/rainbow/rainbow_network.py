import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agents.rainbow.noisy_layers import NoisyLinear


class RainbowDQN(nn.Module):
    """
    Rainbow CNN encoder (Nature DQN) + (optional) Dueling + Distributional (C51).

    Input:  x [B, C, H, W]  (C = 3 * frame_stack)
    Output: logits [B, num_actions, num_atoms]
    """

    def __init__(
        self,
        num_actions: int,
        num_atoms: int,
        input_channels: int = 12,   # <- IMPORTANT: pass 3*frame_stack
        use_dueling: bool = True,
        use_noisy: bool = True,
        noisy_std_init: float = 0.5,
        image_size: int = 84,       # used only to compute conv_out_size safely
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_atoms = int(num_atoms)
        self.use_dueling = bool(use_dueling)
        self.use_noisy = bool(use_noisy)

        # CNN encoder (same as DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute conv output size dynamically (same pattern as network.DQN)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            conv_out = self.conv(dummy)
            self.conv_out_size = conv_out.view(1, -1).size(1)

        Linear = (lambda a, b: NoisyLinear(a, b, std_init=noisy_std_init)) if self.use_noisy else nn.Linear

        if self.use_dueling:
            # Value stream: outputs distribution over atoms
            self.value = nn.Sequential(
                Linear(self.conv_out_size, 512),
                nn.ReLU(),
                Linear(512, self.num_atoms),
            )
            # Advantage stream: outputs distribution over atoms for each action
            self.advantage = nn.Sequential(
                Linear(self.conv_out_size, 512),
                nn.ReLU(),
                Linear(512, self.num_actions * self.num_atoms),
            )
        else:
            # Single head: distribution over atoms for each action
            self.head = nn.Sequential(
                Linear(self.conv_out_size, 512),
                nn.ReLU(),
                Linear(512, self.num_actions * self.num_atoms),
            )

    def reset_noise(self):
        """Reset noise parameters for all NoisyLinear layers (if enabled)."""
        if not self.use_noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns logits: [B, A, atoms]
        """
        z = self.conv(x)
        z = torch.flatten(z, 1)  # [B, conv_out_size]

        if self.use_dueling:
            v = self.value(z)  # [B, atoms]
            a = self.advantage(z)  # [B, A*atoms]
            a = a.view(-1, self.num_actions, self.num_atoms)  # [B, A, atoms]
            v = v.view(-1, 1, self.num_atoms)  # [B, 1, atoms]
            logits = v + (a - a.mean(dim=1, keepdim=True))  # [B, A, atoms]
        else:
            logits = self.head(z).view(-1, self.num_actions, self.num_atoms)

        return logits

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities [B, A, atoms]."""
        logits = self.forward(x)
        # return F.softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, 1e-6, 1.0)

        return probs


    def q_values(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Expected Q = sum_z p(z) * z
        support: [atoms]
        returns: [B, A]
        """
        probs = self.dist(x)  # [B, A, atoms]
        return torch.sum(probs * support.view(1, 1, -1), dim=-1)