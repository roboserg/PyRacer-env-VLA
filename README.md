# PyRacer

Retro 3D-perspective racing game (Pygame) wrapped as a gymnasium-inspired environment for RL and VLA research.

![Trained VLA Agent](gifs/trained_VLA_agent.gif)

## Overview

PyRacer is a research environment for training vision-language-action (VLA) models to drive a retro racing game. The goal is to teach AI agents to drive by learning from human demonstrations and improving through reinforcement learning.

**Pipeline:**
1. **Data Collection**: Record human gameplay with keyboard controls, capturing frames + actions
2. **SFT (Supervised Fine-Tuning)**: Train a SmolVLM model to imitate human driving behavior
3. **GRPO (Group Relative Policy Optimization)**: Refine the model through RL to discover better strategies

**VLA Architecture:**
- Uses HuggingFace `SmolVLM-Instruct` as the base model
- Implements **Chain-of-Thought (CoT)** reasoning: the model generates a reasoning trace before outputting actions
- Special tokens encode actions: `<FWD_0>`, `<BRK_1>`, `<LEFT_2>`, `<RIGHT_3>`
- Example model output: `<thought>Car is drifting left, need to steer right...</thought><action><FWD_1><RIGHT_3></action>`

**Training Data:**
- Saves frames (~10 FPS) and metadata to `recordings/YYYYMMDD_HHMMSS/`
- Generates 20+ text annotations per frame (speed, position, on-road status, etc.)

## Run

```bash
python main.py                               # Human play with full menu system
python scripts/play.py                       # Human play via VLA environment (no menus)
python scripts/play.py --record              # Human play + record dataset
python scripts/eval.py --bot                 # Run rule-based bot
python scripts/eval.py                       # Run trained VLA model
python scripts/train.py                      # Train VLA model on recorded data
python scripts/train_rl.py                   # Train RL agent (GRPO)
```

## Gym-like Interface

```python
from src.gym.env import GameEnvironment
env = GameEnvironment()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # action: 0-15 (4-bit flags)
```

Action bits: `accel=1, brake=2, left=4, right=8`

## Agent Types

- HumanAgent: Keyboard-controlled human player
- BotAgent: Rule-based bot for baseline performance
- VL-Agent: Uses trained SmolVLM model for vision-language action prediction
- RL Agents: GRPO for reinforcement learning

## Data Collection & Training

1. Record gameplay: `python scripts/play.py --record`
   - Saves frames and metadata to `recordings/YYYYMMDD_HHMMSS/`
   - Generates vision-language pairs for VLA training

2. Train VLA model: `python scripts/train.py`
   - Fine-tunes SmolVLM on recorded dataset
   - Outputs to `models/` directory

3. Train RL agents: `python scripts/train_rl.py`
    - Implements GRPO (Group Relative Policy Optimization)
    - Uses recorded datasets for offline RL or online training

## Credits

Originally by [Christian Dueñas](http://www.youtube.com/channel/UCB2mKxxXPK3X8SJkAc-db3A) — inspired by [OneloneCoder's C++ tutorial](https://www.youtube.com/watch?v=KkMZI5Jbf18).
