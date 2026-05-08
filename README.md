# TacticRL

Reinforcement Learning for Soccer Tactics & Formation Optimization  
CS 594 — Reinforcement Learning, Spring 2026, UIC  
Team: Data Mavericks (Muneeb, Ansh, Nihal, Taabish)

**Stack**: Google Research Football (gfootball) + StatsBomb + Stable-Baselines3 PPO

---

## Setup

### 1. System Dependencies (run once, before pip)

```bash
chmod +x setup_env.sh && ./setup_env.sh
```

Installs: cmake, SDL2, OpenGL, Boost. Required by gfootball — `pip` alone won't work.

### 2. Python Environment (Python 3.8 or 3.9 ONLY)

```bash
python3.9 -m venv tacticrl_venv
source tacticrl_venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install gfootball==2.10.2   # install last
```

> **Warning**: gfootball does NOT support Python 3.10+.

### 3. Verify Installation

```bash
python verify_install.py
```

---

## Project Structure

```
tacticrl/
├── env/
│   ├── tactical_env.py      # Custom gfootball wrapper (29 discrete actions)
│   ├── obs_wrapper.py       # Flatten + augment observation (115 → 124 dim)
│   └── reward_shaper.py     # Shaped reward function
├── data/
│   ├── statsbomb_loader.py  # StatsBomb free data pipeline
│   └── formation_dist.py    # KL reference distribution (Laplace-smoothed)
├── training/
│   ├── train.py             # Main PPO training entry point
│   ├── curriculum.py        # Self-play opponent snapshot manager
│   └── callbacks.py         # SB3 callbacks (win rate, entropy logging)
├── evaluation/
│   ├── evaluate.py          # Win rate, KL divergence, formation entropy
│   ├── metrics.py           # Standalone metric utilities
│   └── ablation.py          # Curriculum vs no-curriculum comparison
└── visualization/
    ├── formation_heatmap.py # Formation usage by match state
    ├── pressing_map.py      # Pressing intensity timeline
    └── substitution_plot.py # Tactical switch timeline
configs/
├── ppo_config.yaml          # Main config (with curriculum)
└── ppo_no_curriculum.yaml   # Ablation config
```

---

## Training

```bash
# Build StatsBomb reference distribution (run once)
python -c "
from tacticrl.data.statsbomb_loader import load_all_formation_changes
from tacticrl.data.formation_dist import build_formation_distribution, save_distribution
df = load_all_formation_changes()
dist = build_formation_distribution(df)
save_distribution(dist)
"

# Train PPO agent
python -m tacticrl.training.train --config configs/ppo_config.yaml

# Monitor training
tensorboard --logdir runs/
```

---

## Evaluation

```python
from tacticrl.evaluation.evaluate import evaluate_model

results = evaluate_model(
    model_path='checkpoints/final_model',       # SB3 adds .zip automatically
    vec_normalize_path='checkpoints/vec_normalize_final.pkl',
    n_episodes=100
)
print(f"Win rate: {results['win_rate']:.3f}")
print(f"Formation entropy: {results['formation_entropy']:.3f}")
print(f"KL vs StatsBomb: {results['kl_divergence']:.3f}")
```

---

## Action Space

| Action Range | Meaning                        |
|-------------|--------------------------------|
| 0–18        | gfootball base actions (move, pass, shoot, ...) |
| 19–25       | Formation switch (7 formations) |
| 26–28       | Pressing level (low/medium/high) |

**Supported formations**: 4-4-2, 4-3-3, 4-2-3-1, 3-5-2, 4-5-1, 5-3-2, 3-4-3

---

## Common Errors

| Error | Fix |
|-------|-----|
| `CMake Error: Could not find SDL2` | Run `setup_env.sh` |
| `collections has no attribute 'Callable'` | Downgrade to Python 3.9 |
| `too many values to unpack (expected 4)` | Pin `gym==0.21.0` |
| `KL divergence = inf` | Laplace smoothing is already applied — check for empty agent distribution |
| `model.zip not found` | Don't add `.zip` to `model_path` — SB3 handles it |
