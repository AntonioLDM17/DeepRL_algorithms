# Deep RL Labs – Discretized Continuous Control from Pixels

This project is a modular Deep Reinforcement Learning (DRL) framework focused on applying **value-based methods (DQN and Rainbow)** to **continuous-control MuJoCo environments**, using a custom pipeline based on:

- **Manual action discretization**
- **Pixel-based observations (RGB + frame stacking)**
- **Reward shaping**

Instead of using standard continuous-control algorithms (e.g., SAC, TD3, PPO), this repo explores how far we can push **DQN-style methods in continuous domains** by adapting the environment.

---

## 🚀 Project Overview

The core idea is simple but non-trivial:

> Convert a continuous-control problem into a discrete one, and solve it using value-based DRL.

This is achieved through a pipeline where:

1. The MuJoCo environment remains continuous internally
2. A wrapper discretizes the action space into a small set of prototype actions
3. Observations are converted into RGB images
4. Frames are stacked to capture dynamics
5. A reward wrapper reshapes the signal to stabilize learning
6. DQN or Rainbow is trained on top of this

Supported environments:
- `Walker2d-v4`
- `HalfCheetah-v4`
- `Humanoid-v4`

Supported algorithms:
- `dqn`
- `rainbow`

---

## 🧠 Key Design Decisions

### 1. Discretizing Continuous Actions

Instead of using a full combinatorial discretization, we define **prototype actions**:

- One no-op action
- For each joint: a small set of non-zero values applied **only to that joint**

Example (Walker2d):
- Action dimension: 6
- `num_bins = 3`
- Total actions:
```

1 + 6 * (3 - 1) = 13

```

This keeps the action space manageable while still allowing meaningful control via action sequences.

---

### 2. Visual Observations

The agent does **not receive state vectors**. Instead:

- RGB frames from `render()`
- Cropped and resized to `84x84`
- 4 frames stacked → input shape `(84, 84, 12)`

We intentionally keep **RGB (not grayscale)** because color helps distinguish body parts and posture.

---

### 3. Reward Shaping

Reward shaping is critical for stability.

Example (Walker):
```

r = r_original + 0.1 * x_velocity

```

Example (Humanoid):
- Forward velocity (clipped)
- Height bonus
- Survival bonus
- Termination penalty

The goal is to **reinforce useful behavior (moving forward, staying alive)** without changing the task objective too much.

---

## 📁 Repository Structure

```

src/
├── agents/
│   ├── dqn/
│   └── rainbow/
├── environments/
│   ├── walker_wrapper.py
│   ├── humanoid_wrapper.py
│   ├── reward_wrapper.py
│   ├── humanoid_reward_wrapper.py
│   ├── image_wrapper.py
│   └── replay_buffer.py
├── train/
│   ├── train_dqn.py
│   ├── train_rainbow.py
│   └── resume_checkpoint.py
├── evaluate/
├── config.py
├── utils.py
└── test.py

````

### Key modules:

- `config.py`: global configuration and hyperparameters
- `environments/`: all wrappers (discretization, reward, image pipeline)
- `agents/`: DQN and Rainbow implementations
- `train/`: training and checkpointing scripts
- `utils.py`: saving, loading, logging

---

## ⚙️ Configuration

Important defaults:

```python
IMAGE_SIZE = 84
FRAME_STACK = 4
TOTAL_STEPS = 20_000_000
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
TARGET_UPDATE_FREQ = 5000
````

### Rainbow-specific:

```python
USE_DOUBLE_DQN = True
USE_DUELING = True
USE_PER = True
USE_NOISY_NETS = True
USE_DISTRIBUTIONAL = True
USE_N_STEP = True

N_STEP = 3
NUM_ATOMS = 51
V_MIN = -10.0
V_MAX = 1200.0
```

⚠️ Note: `V_MAX` is significantly larger than Atari defaults to match MuJoCo reward scales.

---

## 🏋️ Training

### Train DQN

```bash
python src/train/train_dqn.py --task walker
```

### Train Rainbow

```bash
python src/train/train_rainbow.py --task walker
```

### Resume training

```bash
python src/train/resume_checkpoint.py --task walker --checkpoint path/to/model.pth
```

---

## 📊 Logging & Outputs

### TensorBoard

Tracks:

* `Reward/Episode`
* `Reward/MeanWindow`
* `Loss/TD_Loss` (DQN)
* `Loss/DistributionalLoss` (Rainbow)
* `Loss/MeanSampleLoss` (Rainbow)

### Saved data

```
checkpoints/<env>/<algo>/
videos/<env>/<algo>/
results/<env>/<algo>/
```

---

## 🧪 Replay Buffers

### DQN

* Standard replay buffer
* Stores images as `uint8` for memory efficiency

### Rainbow

* Prioritized Experience Replay (PER)
* N-step returns
* SumTree for efficient sampling

---

## 🧠 Algorithms

### DQN

* CNN encoder
* ε-greedy exploration
* Target network
* Gradient clipping

### Rainbow (full)

* Double DQN
* Dueling architecture
* PER
* Noisy Nets (exploration)
* N-step returns
* Distributional RL (C51)

---

## ⚠️ Known Issues & Pitfalls

### 1. Environment mismatch (VERY IMPORTANT)

Do **not rely on global config** (`EnvConfig.ACTIVE`).

Always check:

* `--task` argument
* Correct wrapper usage
* Correct checkpoint path

Otherwise you may:

* Train Walker but save under Humanoid
* Load incompatible checkpoints
* Get shape mismatches

---

### 2. Rainbow is fragile

Rainbow is much more sensitive than DQN because it combines:

* PER
* C51
* Noisy Nets
* N-step returns

If it doesn’t learn:

* Check C51 projection
* Check `V_MIN / V_MAX`
* Check action space consistency
* Check reward shaping

---

### 3. Humanoid is much harder

* Larger action space (17 dims)
* Discretization is restrictive
* Requires strong reward shaping

Expect:

* Slower learning
* Higher instability
* More tuning required

---

## 🔄 Full Training Pipeline

1. Parse `--task`
2. Build environment (`gym.make`)
3. Apply:

   * Discretization wrapper
   * Reward wrapper
   * Image wrapper
4. Stack frames → `(84,84,12)`
5. Agent selects action:

   * DQN → ε-greedy
   * Rainbow → expectation of distribution
6. Step environment
7. Store transition
8. Sample batch
9. Update network:

   * DQN → TD loss
   * Rainbow → distributional loss + PER
10. Periodically:

* Update target network
* Save checkpoint
* Log to TensorBoard
* Record video

---

## 🧾 Final Notes

This project is not just about comparing DQN vs Rainbow.

The main takeaway is:

> In Deep RL, performance depends as much on the **pipeline design** as on the algorithm itself.

Key factors:

* Action discretization
* Observation preprocessing
* Reward shaping
* Training stability

Rainbow can outperform DQN by a large margin — but only if everything is correctly tuned.

---

## 📌 Future Work

* Run experiments with multiple seeds
* Extend ablation to other Rainbow components
* Improve discretization strategies
* Explore hybrid methods (value-based + policy-based)
* Fully integrate PPO into the training pipeline

---

## 👥 Authors

* Antonio Lorenzo
* Andrés Martínez
* Pablo García

