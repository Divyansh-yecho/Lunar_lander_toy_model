# 🚀 Lunar Lander — DQN with Stable-Baselines3

> A Deep Q-Network (DQN) agent trained from scratch to autonomously land a spacecraft in OpenAI Gymnasium's `LunarLander-v3` environment — using experience replay, target networks, and multi-phase training across 800k timesteps.

---

## 🎬 Demo Videos

| Episode | Download |
|---------|----------|
| Episode 1 — Early Training | [⬇️ Download Video 1](https://github.com/Divyansh-yecho/Lunar_lander_toy_model/raw/main/Video_1.mp4) |
| Episode 2 — Later Training | [⬇️ Download Video 2](https://github.com/Divyansh-yecho/Lunar_lander_toy_model/raw/main/Video_2.mp4) |

---

## 📌 Project Overview

The **Lunar Lander** problem is a classic continuous control benchmark in reinforcement learning. The agent controls a lander with 4 discrete actions — fire left engine, fire right engine, fire main engine, or do nothing — and must learn to navigate and land smoothly between two flag posts without crashing or running out of fuel.

This project implements **DQN (Deep Q-Network)**, the algorithm that pioneered deep reinforcement learning, originally introduced by DeepMind. The agent learns entirely from raw environment observations with no human guidance — just rewards and penalties.

**Reward structure:**
- ✅ Landing safely between the flags: +100 to +140
- 🦵 Each leg ground contact: +10
- 🔥 Firing main engine: −0.3 per frame
- 💥 Crashing: −100
- 🏁 Successfully landing to rest: +200

---

## 🧠 Algorithm — DQN

DQN combines Q-learning with a deep neural network to approximate the Q-value function. Key techniques used:

| Technique | Purpose |
|-----------|---------|
| **Experience Replay** | Stores past transitions in a buffer; samples random mini-batches to break correlation between consecutive updates |
| **Target Network** | A separate frozen copy of the Q-network updated every 1000 steps — stabilizes training |
| **ε-Greedy Exploration** | Agent starts exploring randomly (ε=1.0) and gradually shifts to exploiting learned policy (ε=0.02) |
| **MLP Policy** | 2-layer fully connected network mapping 8 observations → 4 action Q-values |

### Hyperparameters

```python
learning_rate          = 5e-4      # Adam optimizer step size
buffer_size            = 100_000   # replay memory capacity
learning_starts        = 5_000     # steps before first gradient update
batch_size             = 64        # mini-batch size per update
gamma                  = 0.99      # discount factor for future rewards
train_freq             = 4         # update every 4 environment steps
target_update_interval = 1_000     # sync target network every 1000 steps
exploration_fraction   = 0.15      # fraction of training spent exploring
exploration_final_eps  = 0.02      # minimum exploration rate
```

---

## 🏋️ Training Strategy

Training was split into **3 phases** totalling **800,000 timesteps** to allow progressive improvement and checkpoint recovery:

| Phase | Timesteps | Cumulative | Notes |
|-------|-----------|------------|-------|
| 1 | 200,000 | 200k | Initial policy formation, heavy exploration |
| 2 | 300,000 | 500k | Loaded from best checkpoint, continued learning |
| 3 | 300,000 | 800k | Final refinement, exploitation-heavy |

Checkpoints are saved every **10,000 steps**, so training can be resumed at any point without starting over.

All SB3 verbose output is silently redirected to `logs/training.log` — keeping the notebook clean while preserving a full audit trail.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Mean reward (20 episodes) | ~250+ |
| Std deviation | ~30 |
| Training time (Colab T4 GPU) | ~20 minutes |
| Total timesteps | 800,000 |

A mean reward above **200** is generally considered a solved environment for LunarLander-v3.

---

## 🛠️ Setup & Usage

### Requirements

```bash
pip install gymnasium[box2d] stable-baselines3
```

### Run in Google Colab
1. Open `lunar_lander.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Mount your Google Drive when prompted
3. Run all cells top to bottom — training, evaluation, and video recording are fully automated

### Run Locally
```bash
git clone https://github.com/Divyansh-yecho/Lunar_lander_toy_model.git
cd Lunar_lander_toy_model
pip install gymnasium[box2d] stable-baselines3 jupyter
jupyter notebook lunar_lander.ipynb
```

---

## 📁 Project Structure

```
Lunar_lander_toy_model/
├── lunar_lander.ipynb         # Main training notebook
├── Video_1.mp4                # Recorded agent — episode 1
├── Video_2.mp4                # Recorded agent — episode 2
├── checkpoints/               # Model saved every 10k steps
│   └── dqn_lunar_XXXXX_steps.zip
├── models/
│   └── final_model.zip        # Best final model weights
└── logs/
    ├── training.log           # Full training output log
    └── monitor.csv            # Per-episode reward tracking
```

---

## 📚 References

- [Playing Atari with Deep Reinforcement Learning — Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

---

## 👤 Author

**Divyansh** · [@Divyansh-yecho](https://github.com/Divyansh-yecho)
