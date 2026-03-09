# AGENTS.md - PyRacer Development Guide

PyRacer is a retro racing game (Pygame, Python 3.12) with a VLA (Vision-Language-Action) module for collecting training data and training models.

## Running

```bash
python main.py          # Play the game (human, with menus)
python scripts/play.py  # Play via VLA environment
python scripts/record.py          # Record human gameplay dataset
python scripts/bot_play_random.py # Run bot player
python scripts/test_race.py       # Quick test
```

## Project Structure

```
main.py                   # Entry point: initializes Game, runs menu/game loop
data/
  gameFiles/              # Core game logic: game.py, car.py, map.py
  menus/                  # Menu system: main_menu.py, options_menu.py, controls_menu.py, credits_menu.py, menu.py
  util/                   # Utilities: fps.py, controls.py
  images/                 # Image assets: spritesheet.py
vla/
  env.py                  # GameEnvironment (gymnasium Env), Observation dataclass
  agents/
    agent.py              # Abstract Agent base class
    human_agent.py        # Keyboard input agent
    random_agent.py       # Uniformly random action agent
    bot_agent.py          # Rule-based agent (steers toward track center)
    vla_agent.py          # VLA model inference agent
  recorder.py             # Frame capture + JSONL metadata logging
  train.py                # VLA model training
  dataset.py              # Dataset loading/processing
  model.py                # Model wrapper
  play.py                 # VLA environment runner
  print_config.py         # Config printer
scripts/
  play.py                 # Run human play via VLA env
  record.py               # Record gameplay to dataset
  bot_play_random.py      # Run random bot
  test_race.py            # Test script
```

## VLA Module

The `vla/` module wraps the game as a `gymnasium.Env` for data collection and model training.

- **Observation**: frame (pygame.Surface), speed, position, car_x/y, on_road, car_offset_from_center
- **Actions**: `{accel, brake, left, right}` discrete booleans
- **Agents**: subclass `Agent`, implement `predict(observation) -> (action, state)`
- **Recording**: saves frames as JPG + `metadata.jsonl` to `vla/data/recordings/YYYYMMDD_HHMMSS/`
- **Training stack**: PyTorch, transformers (SmolVLM), gymnasium, PIL

## Code Style

- **Classes**: PascalCase | **Functions/vars**: snake_case | **Constants**: UPPER_SNAKE_CASE
- **Indentation**: 4 spaces
- **No type hints** (not used in game code; vla/ uses them)
- **Imports**: stdlib → third-party → local
- **Pygame init**: Use `pygame.display.init()` + `pygame.font.init()` (minimal, fast setup)
- **Delta time**: `self.game.dt` for frame-independent movement
- **Asset paths**: always `os.path.join()`

## Key Patterns

- `Game` object is the central state; pass `self.game` reference through classes
- Game loop: `update()` → `draw()` each frame; `reset()` between sessions
- Agents live in `vla/agents/`, subclass `Agent`, implement `predict(observation) → (action, state)`
- No automated tests — test manually with `python main.py` or relevant script
