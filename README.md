# PyRacer

Retro 3D-perspective racing game (Pygame) wrapped as a Gymnasium environment for RL and VLA research.

## Run

```bash
python main.py                               # Human play with full menu system
python scripts/play.py                       # Human play via VLA environment (no menus)
python scripts/play.py --record              # Human play + record dataset
python scripts/eval.py --agent bot           # Run rule-based bot
python scripts/eval.py --agent vla           # Run trained VLA model
python scripts/train.py                      # Train VLA model on recorded data
python scripts/train_rl.py                   # Train RL agent (GRPO, PPO, etc.)
```

## Gym Interface

```python
from vla.env import GameEnvironment
env = GameEnvironment()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # action: 0-15 (4-bit flags)
```

Action bits: `accel=1, brake=2, left=4, right=8`

## Agent Types

- HumanAgent: Keyboard-controlled human player
- BotAgent: Rule-based bot for baseline performance
- RandomAgent: Random actions for exploration
- VL-Agent: Uses trained SmolVLM model for vision-language action prediction
- RL Agents: PPO, GRPO, etc. for reinforcement learning

## Data Collection & Training

1. Record gameplay: `python scripts/play.py --record`
   - Saves frames and metadata to `recordings/YYYYMMDD_HHMMSS/`
   - Generates vision-language pairs for VLA training

2. Train VLA model: `python scripts/train.py`
   - Fine-tunes SmolVLM on recorded dataset
   - Outputs to `models/` directory

3. Train RL agents: `python scripts/train_rl.py`
   - Supports GRPO (Group Relative Policy Optimization) and other RL algorithms
   - Uses recorded datasets for offline RL or online training

## Credits

Originally by [Christian Dueñas](http://www.youtube.com/channel/UCB2mKxxXPK3X8SJkAc-db3A) — inspired by [OneloneCoder's C++ tutorial](https://www.youtube.com/watch?v=KkMZI5Jbf18).
