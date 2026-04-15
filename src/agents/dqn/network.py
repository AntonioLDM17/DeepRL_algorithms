import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    DQN estándar según Mnih et al. (2015)
    Entrada: Imagen RGB (con frame stacking opcional)
    Salida: Q-values para cada acción discreta
    """
    def __init__(self, num_actions, input_channels=3):
        super().__init__()
        
        # CNN encoder (Nature DQN architecture)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out = self.conv(dummy)
            self.conv_out_size = conv_out.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)