# 🚀 Lunar Lander — DQN with Stable-Baselines3

Training a Deep Q-Network (DQN) agent to solve OpenAI Gymnasium's `LunarLander-v3` environment.

## Demo Videos

| Episode | Download |
|---------|----------|
| Episode 1 — Early Training | [⬇️ Download Video 1](https://github.com/Divyansh-yecho/Lunar_lander_toy_model/raw/main/Video_1.mp4) |
| Episode 2 — Later Training | [⬇️ Download Video 2](https://github.com/Divyansh-yecho/Lunar_lander_toy_model/raw/main/Video_2.mp4) |

---

## Setup

Open `lunar_lander.ipynb` in **Google Colab** (recommended) or any Jupyter environment.

```bash
pip install gymnasium[box2d] stable-baselines3
```

---

## Training

The agent trains in 3 phases (800k total timesteps):

| Phase | Steps | Purpose |
|-------|-------|---------|
| 1 | 200k | Initial exploration |
| 2 | 300k | Continued learning from checkpoint |
| 3 | 300k | Final refinement |

All verbose output is redirected to `logs/training.log` — the notebook stays clean.

---

## Results

| Metric | Value |
|--------|-------|
| Mean reward (20 episodes) | ~250+ |
| Training time (Colab T4) | ~20 min |

---

## Project Structure

```
Lunar_lander_toy_model/
├── lunar_lander.ipynb         # Main notebook
├── Video_1.mp4                # Training episode 1
├── Video_2.mp4                # Training episode 2
└── logs/
    ├── training.log           # All training output
    └── monitor.*              # Episode rewards
```

---

## Algorithm — DQN Hyperparameters

```python
learning_rate          = 5e-4
buffer_size            = 100_000
learning_starts        = 5_000
batch_size             = 64
gamma                  = 0.99
exploration_fraction   = 0.15
exploration_final_eps  = 0.02
target_update_interval = 1_000
```
