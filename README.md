# PyRacer

Retro 3D-perspective racing game (Pygame) wrapped as a **Gymnasium environment** for RL and VLA (Vision-Language-Action) research.

## Run

```bash
python main.py                    # Human play
python scripts/play.py            # Human play via gym env
python scripts/record.py          # Record gameplay dataset
python scripts/bot_play.py        # Rule-based bot
python vla/train.py               # Train SmolVLM on recorded data
```

## Gym Interface

```python
from vla.env import GameEnvironment
env = GameEnvironment()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # action: 0-15 (4-bit flags)
```

Action bits: `accel=1, brake=2, left=4, right=8`

## Credits

Originally by [Christian Dueñas](http://www.youtube.com/channel/UCB2mKxxXPK3X8SJkAc-db3A) — inspired by [OneloneCoder's C++ tutorial](https://www.youtube.com/watch?v=KkMZI5Jbf18).
