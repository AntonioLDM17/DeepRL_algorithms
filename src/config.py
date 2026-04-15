import torch
from pathlib import Path

# =============================================================================
# DEVICE & PATHS
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
class EnvConfig:
    """Environment-specific configurations."""

    WALKER2D = {
        "env_id": "Walker2d-v4",
        "name": "walker",
        "num_discrete_actions": 3,
        "action_scale": 1.0,
        "max_episode_steps": 10_000,
    }

    HALFCHEETAH = {
        "env_id": "HalfCheetah-v4",
        "name": "halfcheetah",
        "num_discrete_actions": 3,
        "action_scale": 0.5,
        "max_episode_steps": 10_000,
    }

    HUMANOID = {
        "env_id": "Humanoid-v4",
        "name": "humanoid",
        "num_discrete_actions": 7,
        "action_scale": 0.4,
        "max_episode_steps": 30_000,
    }

    ACTIVE = WALKER2D


ENV_ID = EnvConfig.ACTIVE["env_id"]
ENV_NAME = EnvConfig.ACTIVE["name"]
NUM_DISCRETE_ACTIONS = EnvConfig.ACTIVE["num_discrete_actions"]
ACTION_SCALE = EnvConfig.ACTIVE["action_scale"]
MAX_EPISODE_STEPS = EnvConfig.ACTIVE["max_episode_steps"]


# =============================================================================
# NETWORK ARCHITECTURE
# =============================================================================
class NetworkConfig:
    """Neural network architecture settings."""

    IMAGE_SIZE = 84
    IMAGE_CHANNELS = 3
    FRAME_STACK = 4

    CNN_CHANNELS = [32, 64, 64]
    CNN_KERNELS = [8, 4, 3]
    CNN_STRIDES = [4, 2, 1]

    FC_HIDDEN_DIM = 512


IMAGE_SIZE = NetworkConfig.IMAGE_SIZE
FRAME_STACK = NetworkConfig.FRAME_STACK


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
class TrainingConfig:
    """Generic training hyperparameters used across algorithms."""

    TOTAL_STEPS = 20_000_000

    BUFFER_SIZE = 100_000
    BATCH_SIZE = 256

    LEARNING_RATE = 1e-4
    GAMMA = 0.99

    TARGET_UPDATE_FREQ = 5_000

    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 100_000

    LEARNING_STARTS = 1_000
    TRAIN_FREQ = 4

    GRAD_CLIP = 10.0
    SEED = 42


TOTAL_STEPS = TrainingConfig.TOTAL_STEPS
BATCH_SIZE = TrainingConfig.BATCH_SIZE
BUFFER_SIZE = TrainingConfig.BUFFER_SIZE
GAMMA = TrainingConfig.GAMMA
LR = TrainingConfig.LEARNING_RATE
TARGET_UPDATE_FREQ = TrainingConfig.TARGET_UPDATE_FREQ
EPS_START = TrainingConfig.EPS_START
EPS_END = TrainingConfig.EPS_END
EPS_DECAY = TrainingConfig.EPS_DECAY
LEARNING_STARTS = TrainingConfig.LEARNING_STARTS
TRAIN_FREQ = TrainingConfig.TRAIN_FREQ
GRAD_CLIP = TrainingConfig.GRAD_CLIP


# =============================================================================
# PPO HYPERPARAMETERS
# =============================================================================
class PPOConfig:
    """Default PPO hyperparameters tuned for more stable continuous control."""

    ROLLOUT_STEPS = 4096
    EPOCHS = 4
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENTROPY_COEF = 1e-3
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.02
    INIT_LOG_STD = -1.0


# =============================================================================
# LOGGING & CHECKPOINTING
# =============================================================================
class LoggingConfig:
    """Logging and saving configurations."""

    SAVE_FREQ = 100_000
    KEEP_LAST_N_CHECKPOINTS = 5

    RECORD_VIDEO_TRAIN = False
    VIDEO_FREQ_TRAIN = 1_000

    RECORD_VIDEO_EVAL = True
    VIDEO_FREQ_EVAL = 1
    NUM_EVAL_EPISODES = 10

    LOG_FREQ = 1
    REWARD_WINDOW = 100

    USE_TENSORBOARD = True
    TENSORBOARD_DIR = RESULTS_DIR / "tensorboard"


SAVE_FREQ = LoggingConfig.SAVE_FREQ
VIDEO_FREQ_TRAIN = LoggingConfig.VIDEO_FREQ_TRAIN
VIDEO_FREQ_EVAL = LoggingConfig.VIDEO_FREQ_EVAL


# =============================================================================
# ALGORITHM VARIANTS
# =============================================================================
class AlgorithmConfig:
    """Selector + flags de variantes."""

    MODE = "sac"  # "dqn", "rainbow", "ppo", "sac"

    USE_DOUBLE_DQN = True
    USE_DUELING = True
    USE_PER = True
    USE_NOISY_NETS = True
    USE_DISTRIBUTIONAL = True

    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 2_000_000
    PER_EPS = 1e-6

    NOISY_STD_INIT = 0.5

    N_STEP = 3
    N_STEP_GAMMA = 0.99
    USE_N_STEP = True

    NUM_ATOMS = 51
    V_MIN = -10.0
    V_MAX = 1200.0

    LR = TrainingConfig.LEARNING_RATE
    GAMMA = TrainingConfig.GAMMA
    GRAD_CLIP_NORM = TrainingConfig.GRAD_CLIP
    SEED = TrainingConfig.SEED

    # -------------------------------------------------------------------
    # Soft Actor-Critic (SAC)
    # -------------------------------------------------------------------

    # Learning rates
    SAC_ACTOR_LR = 3e-4
    SAC_CRITIC_LR = 3e-4
    SAC_ALPHA_LR = 3e-4

    # Network width
    SAC_FEATURE_DIM = 512

    # Entropy temperature
    SAC_AUTO_ENTROPY = True
    SAC_ALPHA_INIT = 0.2
    SAC_TARGET_ENTROPY_RATIO = 0.98

    SAC_TAU = 0.005


_VALID_ALGOS = {"dqn", "rainbow", "ppo", "sac"}


def get_algo_name() -> str:
    m = getattr(AlgorithmConfig, "MODE", "dqn").lower()
    return m if m in _VALID_ALGOS else "dqn"


# =============================================================================
# PATHS
# =============================================================================
def get_checkpoint_path(env_name: str, step=None):
    algo_name = get_algo_name()
    path = CHECKPOINT_DIR / env_name / algo_name
    path.mkdir(parents=True, exist_ok=True)
    if step is not None:
        return path / f"{algo_name}_{env_name}_{step}.pth"
    return path


def get_video_path(env_name: str, mode: str = "train"):
    algo_name = get_algo_name()
    path = VIDEO_DIR / env_name / algo_name / mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_path(env_name: str | None = None):
    selected_env = ENV_NAME if env_name is None else env_name
    path = RESULTS_DIR / selected_env / get_algo_name()
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================
def print_config():
    algo = get_algo_name().upper()

    print("=" * 80)
    print(f"CONFIGURATION SUMMARY - {algo}")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    print("\n[Environment]")
    print(f"  Name: {ENV_NAME}")
    print(f"  ID: {ENV_ID}")
    print(f"  Discrete bins per dim: {NUM_DISCRETE_ACTIONS}")
    print(f"  Action scale: {ACTION_SCALE}")
    print(f"  Max episode steps: {MAX_EPISODE_STEPS}")

    print("\n[Network]")
    print(f"  Image Size: {NetworkConfig.IMAGE_SIZE}x{NetworkConfig.IMAGE_SIZE}")
    print(f"  Frame Stack: {FRAME_STACK}")
    print(f"  Input Channels: {NetworkConfig.IMAGE_CHANNELS * FRAME_STACK}")

    print("\n[Training]")
    print(f"  Total Steps: {TrainingConfig.TOTAL_STEPS:,}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Buffer Size: {BUFFER_SIZE:,}")
    print(f"  Learning Rate: {LR}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Target update freq: {TARGET_UPDATE_FREQ}")
    print(f"  Learning starts: {LEARNING_STARTS}")
    print(f"  Train freq: {TRAIN_FREQ}")
    print(f"  Grad clip: {GRAD_CLIP}")

    if get_algo_name() == "ppo":
        print("\n[PPO defaults]")
        print(f"  Rollout steps: {PPOConfig.ROLLOUT_STEPS}")
        print(f"  PPO epochs: {PPOConfig.EPOCHS}")
        print(f"  Batch size: {PPOConfig.BATCH_SIZE}")
        print(f"  LR: {PPOConfig.LEARNING_RATE}")
        print(f"  Gamma: {PPOConfig.GAMMA}")
        print(f"  GAE lambda: {PPOConfig.GAE_LAMBDA}")
        print(f"  Clip eps: {PPOConfig.CLIP_EPS}")
        print(f"  Entropy coef: {PPOConfig.ENTROPY_COEF}")
        print(f"  Value coef: {PPOConfig.VALUE_COEF}")
        print(f"  Max grad norm: {PPOConfig.MAX_GRAD_NORM}")
        print(f"  Target KL: {PPOConfig.TARGET_KL}")
        print(f"  Init log std: {PPOConfig.INIT_LOG_STD}")

    print("\n[Algorithm flags]")
    print(f"  MODE: {AlgorithmConfig.MODE}")
    print(f"  Double DQN: {AlgorithmConfig.USE_DOUBLE_DQN}")
    print(f"  Dueling: {AlgorithmConfig.USE_DUELING}")
    print(
        f"  PER: {AlgorithmConfig.USE_PER} "
        f"(alpha={AlgorithmConfig.PER_ALPHA}, beta0={AlgorithmConfig.PER_BETA_START})"
    )
    print(
        f"  Noisy Nets: {AlgorithmConfig.USE_NOISY_NETS} (std_init={AlgorithmConfig.NOISY_STD_INIT})"
    )
    print(
        f"  n-step: {AlgorithmConfig.USE_N_STEP} (n={AlgorithmConfig.N_STEP}, gamma={AlgorithmConfig.N_STEP_GAMMA})"
    )
    print(
        f"  Distributional (C51): {AlgorithmConfig.USE_DISTRIBUTIONAL} "
        f"(atoms={AlgorithmConfig.NUM_ATOMS}, v=[{AlgorithmConfig.V_MIN},{AlgorithmConfig.V_MAX}])"
    )
    print("=" * 80)


__all__ = [
    "DEVICE",
    "EnvConfig",
    "NetworkConfig",
    "TrainingConfig",
    "PPOConfig",
    "LoggingConfig",
    "AlgorithmConfig",
    "ENV_ID",
    "ENV_NAME",
    "NUM_DISCRETE_ACTIONS",
    "ACTION_SCALE",
    "MAX_EPISODE_STEPS",
    "IMAGE_SIZE",
    "FRAME_STACK",
    "TOTAL_STEPS",
    "BATCH_SIZE",
    "BUFFER_SIZE",
    "GAMMA",
    "LR",
    "TARGET_UPDATE_FREQ",
    "EPS_START",
    "EPS_END",
    "EPS_DECAY",
    "LEARNING_STARTS",
    "TRAIN_FREQ",
    "GRAD_CLIP",
    "SAVE_FREQ",
    "VIDEO_FREQ_TRAIN",
    "VIDEO_FREQ_EVAL",
    "CHECKPOINT_DIR",
    "VIDEO_DIR",
    "RESULTS_DIR",
    "get_algo_name",
    "get_checkpoint_path",
    "get_video_path",
    "get_results_path",
    "print_config",
]
