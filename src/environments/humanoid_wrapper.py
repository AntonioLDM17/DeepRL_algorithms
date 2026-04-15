# humanoid_wrapper.py
import gymnasium as gym
import numpy as np


class DiscreteHumanoidWrapper(gym.ActionWrapper):
    """
    Convierte Humanoid (acción continua Box(-0.4,0.4,(17,))) en un espacio Discrete pequeño.

    - acción 0: vector cero (no-op)
    - para cada dimensión i:
        acciones que SOLO modifican la componente i a valores discretos != 0

    Nº acciones:
        1 + action_dim * (num_bins - 1)
    Con Humanoid: action_dim=17, num_bins=3 -> 1 + 17*(2)=35 acciones.
    """

    def __init__(self, env: gym.Env, num_bins: int = 3, scale: float = 0.4, include_zero: bool = True):
        super().__init__(env)
        assert hasattr(env.action_space, "shape"), "Se esperaba action_space continuo (Box)."

        self.action_dim = int(env.action_space.shape[0])  # Humanoid -> 17
        self.acceptable_actions = 17
        self.num_bins = int(num_bins)
        assert self.num_bins >= 2, "num_bins debe ser >=2"

        self.scale = float(scale)
        self.include_zero = bool(include_zero)

        self._build_action_set()
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def _build_action_set(self):
        bins = np.linspace(-self.scale, self.scale, self.num_bins, dtype=np.float32)
        non_zero_bins = [b for b in bins if abs(float(b)) > 1e-12]

        actions = []
        if self.include_zero:
            actions.append(np.zeros(self.action_dim, dtype=np.float32))

        for i in range(self.acceptable_actions):
            for b in non_zero_bins:
                a = np.zeros(self.action_dim, dtype=np.float32)
                a[i] = np.float32(b)
                actions.append(a)

        self._actions = actions

    def action(self, action_idx: int):
        idx = int(action_idx)
        if idx < 0 or idx >= len(self._actions):
            return np.zeros(self.action_dim, dtype=np.float32)

        a = self._actions[idx].copy()

        # clip al rango real del entorno por seguridad
        if hasattr(self.env.action_space, "low") and hasattr(self.env.action_space, "high"):
            a = np.clip(a, self.env.action_space.low, self.env.action_space.high).astype(np.float32)

        return a

    def get_actions(self):
        """Helper para debug: devuelve la lista de acciones continuas prototipo."""
        return self._actions