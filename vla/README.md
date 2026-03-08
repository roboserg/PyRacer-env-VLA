# PyRacer VLA (Vision-Language-Action) Module

A modular game environment system for PyRacer that enables human gameplay with automatic dataset recording, bot players with customizable policies, non-blocking threaded input, and rich auto-generated text annotations.

---

## 🚀 Quick Start (30 seconds)

### Run Human Player with Recording
```bash
python3 human_play_record.py
```
Recordings saved to: `/vla/data/recordings/YYYYMMDD_HHMMSS/`

### Run Bot Player (No Recording)
```bash
python3 bot_play_random.py
```

### Test Recording
```bash
python3 test_recording.py
```

---

## 📁 Module Structure

### Core Modules (`/vla/`)
- `controller.py` - Abstract Controller base class
- `human_controller.py` - Keyboard input handler
- `bot_controller.py` - Bot player with random/learned policies
- `observation.py` - Game state wrapper + frame capture
- `recorder.py` - Frame capture + JSONL metadata logging
- `environment.py` - Main orchestrator with threading
- `__init__.py` - Module exports

### Example Scripts
- `human_play_record.py` - Human player with automatic recording
- `bot_play_random.py` - Bot player example
- `test_recording.py` - Quick 5-second recording test

---

## 🎯 Key Features

✅ **Non-blocking threaded input** - Game stays @ 60 FPS even if AI takes 200ms  
✅ **60 FPS game rendering** - Smooth gameplay maintained constantly  
✅ **10 FPS dataset recording** - Every 6th frame captured  
✅ **20+ text annotation variations** - Auto-generated from game state  
✅ **Smart state-change detection** - Only records on annotation change  
✅ **Discrete 4-action space** - Accelerate, Brake, Left, Right  
✅ **JSONL metadata format** - Easy for ML model training  
✅ **Modular, extensible design** - Easy to add custom controllers  
✅ **Production-ready code** - Thread-safe and well-tested  
✅ **Timestamped recordings** - Auto-organized by date/time  

---

## 🏗️ Architecture

### Threading Model

The VLA module uses a non-blocking threading architecture to maintain 60 FPS rendering regardless of controller latency:

```
Main Thread (60 FPS)              Background Input Thread
├─ game.render() ✓                ├─ controller.get_action()
├─ action = queue.get() ✓         │  (may take 200ms+)
├─ game.update()  ✓               └─ queue.put(action) ✓
└─ record if needed ✓

Result: Smooth 60 FPS + 16-32ms input latency
```

### Core Components

#### 1. **Controller Interface** (`controller.py`)
- Abstract base class for all input sources
- Defines `get_action(observation) → dict` contract
- Enables swapping human/bot/model inputs seamlessly

#### 2. **Observation** (`observation.py`)
- Encapsulates single-frame game state
- Provides: frame surface, speed, lap, position, road status
- Tracks state changes for annotation generation

#### 3. **Human Controller** (`human_controller.py`)
- Maps keyboard input to 4-action discrete space
- Non-blocking event processing
- Arrow keys: ↑ accelerate, ↓ brake, ← left, → right, ESC quit

#### 4. **Bot Controller** (`bot_controller.py`)
- Random action selection (current)
- Extensible for learned policies
- Deterministic via seed for reproducibility

#### 5. **Recorder** (`recorder.py`)
- Captures frames at 10 FPS (every 6th frame at 60 FPS)
- Generates rich text annotations (20+ variations)
- **Default path**: `/vla/data/recordings/YYYYMMDD_HHMMSS/`
- **Smart recording**: Only saves on annotation state change

#### 6. **GameEnvironment** (`environment.py`)
- Main orchestrator tying everything together
- Threading-based non-blocking input
- Provides `step()` and `run()` interfaces
- Handles game initialization and cleanup

---

## 📝 Text Annotation System

The system generates rich descriptions by automatically combining multiple aspects of game state:

### Speed Analysis (8 variations - 0-1 normalized scale)
Stationary → Barely moving → Crawling → Slow → Moderate pace → Good speed → High speed → Max speed

### Action Analysis (8 combinations)
- Full throttle, Braking, Hard braking
- Turning left/right
- Accelerating left/right
- Coasting
- Conflicting inputs

### Road Position (5 variations)
- Centered on track
- Drifting left/right
- Hugging left/right edge
- Off road
- Approaching edge

### Speed Trends (2 variations)
- Accelerating
- Decelerating

### Track Curvature (4 variations)
- On straight
- Gentle curve
- Curving
- Sharp turn

### Speed vs Curve Safety (3 variations)
- Too fast for turn
- Pushing hard
- Room to accelerate

### Additional States
- Holding left / Holding right (steering consistency)
- Recovering to track (returning from off-road)

### Example Outputs
```
"Max speed, Full throttle, Centered on track, On straight, Pushing hard"
"Good speed, Accelerating left, Drifting left, Sharp turn, Too fast for turn"
"Moderate pace, Braking while turning right, Off road on left, Recovering to track"
"High speed, Holding right, Centered on track, Gentle curve, Room to accelerate"
```

**Note**: All lap-related text has been removed. Track is endless with frame-by-frame timestamps instead.

---

## 📊 Recording Format

### Directory Structure
```
/vla/data/recordings/
├── 20260307_180017/          # Auto-timestamped directory
│   ├── images/
│   │   ├── frame_00001.jpg
│   │   ├── frame_00002.jpg
│   │   └── ...
│   └── metadata.jsonl
├── 20260307_180245/          # Next recording
│   ├── images/
│   └── metadata.jsonl
└── ...
```

### JSONL Format (metadata.jsonl)
```json
{"frame": "frame_00001.jpg", "timestamp": 0.0, "speed": 5.5, "action": [1.0, 0.0, 1.0, 0.0], "text": "Very high speed, Accelerating left"}
{"frame": "frame_00002.jpg", "timestamp": 0.6, "speed": 6.2, "action": [1.0, 0.0, 0.0, 0.0], "text": "High speed, Full throttle, Centered on track"}
{"frame": "frame_00003.jpg", "timestamp": 1.2, "speed": 4.8, "action": [0.0, 1.0, 0.0, 0.0], "text": "Moderate pace, Braking"}
```

### Metadata Fields
- `frame`: Image filename (e.g., `frame_00001.jpg`)
- `timestamp`: Seconds elapsed since recording started (float)
- `speed`: Current car speed (float)
- `action`: [accel, brake, left, right] normalized to 0.0 or 1.0
- `text`: Auto-generated description of game state

### Action Vector Encoding
- Format: `[accel, brake, left, right]`
- Values: 0.0 or 1.0 (discrete)
- Examples:
  - `[1.0, 0.0, 0.0, 0.0]` = Accelerating straight
  - `[1.0, 0.0, 1.0, 0.0]` = Accelerating left
  - `[0.0, 1.0, 0.0, 0.0]` = Braking
  - `[0.0, 0.0, 1.0, 0.0]` = Steering left
  - `[0.0, 0.0, 0.0, 1.0]` = Steering right
  - `[0.0, 0.0, 0.0, 0.0]` = Coasting

---

## 🎮 Usage Examples

### Pattern 1: Human Play + Recording
```python
from vla import GameEnvironment, HumanController, Recorder

# Uses default: /vla/data/recordings/YYYYMMDD_HHMMSS/
recorder = Recorder(enabled=True)
controller = HumanController()
env = GameEnvironment(controller, recorder)
env.run(max_laps=2)

# Or with custom base directory (still creates timestamped subdir)
recorder = Recorder("/my/dataset", enabled=True)
env = GameEnvironment(controller, recorder)
env.run(max_laps=2)
```

### Pattern 2: Bot Play (Random)
```python
from vla import GameEnvironment, BotController

controller = BotController(seed=42)
env = GameEnvironment(controller)
env.run(max_laps=2)
```

### Pattern 3: Bot + Recording
```python
from vla import GameEnvironment, BotController, Recorder

recorder = Recorder("/dataset", enabled=True)
controller = BotController()
env = GameEnvironment(controller, recorder)
env.run(max_laps=3)
```

### Pattern 4: Custom Controller
```python
from vla import Controller, Observation, GameEnvironment

class MyCustomController(Controller):
    def get_action(self, observation: Observation) -> dict:
        # Your logic here
        # observation provides: frame, speed, lap, position, etc.
        return {
            "accel": observation.speed < 5,
            "brake": False,
            "left": False,
            "right": False,
        }

# Use it
env = GameEnvironment(MyCustomController())
env.run()
```

---

## 📊 Observation Object

The observation passed to controllers contains:

```python
observation.frame               # pygame.Surface (480x270)
observation.speed               # float
observation.lap                 # int
observation.lap_time            # float (seconds)
observation.position            # float (distance traveled)
observation.car_x               # float (screen x)
observation.car_y               # float (screen y)
observation.on_road             # bool
observation.car_offset_from_center  # float (from center)
observation.to_dict()           # dict representation
```

---

## ⚡ Performance Metrics

| Metric | Value |
|--------|-------|
| Game FPS | 60 (constant) |
| Recording FPS | 10 (every 6th frame) |
| Input latency | 16-32ms (1-2 frames, imperceptible) |
| AI impact on FPS | 0 (non-blocking) |
| Max AI inference time | 200ms+ (no effect on rendering) |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Game frozen/unresponsive | Check controller.get_action() isn't blocking (no I/O) |
| No recording created | Ensure recorder.enabled=True and env.run() called |
| Annotations not changing | Normal - only records on state change |
| Slow performance | Check for blocking I/O in get_action() |
| Directory not created | Ensure /vla/data/ exists with write permissions |

---

## 🎯 Next Steps

### For Human Data Collection
1. Run `python3 human_play_record.py`
2. Play multiple racing sessions
3. Each run creates: `/vla/data/recordings/YYYYMMDD_HHMMSS/metadata.jsonl`
4. Use accumulated data for training VLA models

### For Bot Testing
1. Create custom controller inheriting from `Controller`
2. Implement `get_action()` with your policy
3. Use `GameEnvironment(YourController())`

### For Model Integration
1. Create `LearnedPolicyController(Controller)` class
2. Load model weights in `__init__()`
3. Use `model.predict(observation.frame)` in `get_action()`
4. Run with `GameEnvironment(LearnedPolicyController())`

---

## 🔗 Future Extensions

### 1. Learned Policy Controller
```python
class LearnedPolicyController(Controller):
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def get_action(self, observation):
        return self.model.predict(observation.frame)
```

### 2. Continuous Actions
- Current: Discrete 4-action space (buttons)
- Future: Continuous [-1, 1] steering/throttle values

### 3. Advanced Observations
- Road sensor arrays
- Lap progress percentage
- Predicted track ahead

### 4. Metrics Collection
- Lap time comparison
- Racing line deviation
- Collision/off-road detection

---

## 📚 Module Imports

```python
from vla import (
    GameEnvironment,      # Main orchestrator
    Controller,           # Base class for controllers
    HumanController,      # Keyboard input
    BotController,        # Random/learned policies
    Observation,          # Game state wrapper
    Recorder,            # Frame + metadata logging
)
```

---

## ✅ Implementation Status

- ✅ Core architecture implemented and tested
- ✅ Non-blocking threading working correctly
- ✅ 10 FPS recording verified
- ✅ Default `/vla/data/recordings/` configured
- ✅ Timestamped directories auto-created
- ✅ No lap-related text in annotations
- ✅ Rich 20+ text variation system
- ✅ JSONL metadata format working
- ✅ Documentation complete
- ✅ Production-ready code

---

## 🎮 Key Controls

| Key | Action |
|-----|--------|
| ↑ UP | Accelerate |
| ↓ DOWN | Brake |
| ← LEFT | Steer left |
| → RIGHT | Steer right |
| ESC | Quit |

---

## 📦 Dependencies

- `pygame` - Game rendering and input
- `threading` - Non-blocking input (Python stdlib)
- `queue` - Thread-safe queues (Python stdlib)
- `pathlib` - Path operations (Python stdlib)
- `json` - JSONL metadata (Python stdlib)
- `datetime` - Timestamped recording directories (Python stdlib)

---

## 🖥️ VLA Training Setup

### Working Configuration (Verified)

| Component | Version | Notes |
|-----------|---------|-------|
| **PyTorch** | 2.10.0+cu126 | Latest |
| **flash-attn** | 2.8.3 | For torch 2.9+ |
| **transformers** | Latest | Auto-downloads model |
| **Attention** | `flash_attention_2` | Works now |

### Installation

```bash
# Create venv and install PyTorch
python -m venv .venv
.venv/bin/pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install flash-attn==2.8.3 --no-build-isolation
.venv/bin/pip install transformers accelerate peft
```

### Alternative: Use SDPA (Faster for small models)

For small models with short sequences, SDPA can be faster than flash_attention_2:

```python
model = SmolVLMForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Use this for speed
).to("cuda")
```

### Inference Performance

- **SDPA**: ~0.3s per inference
- **flash_attention_2**: ~0.4s per inference

### Notes

- Use `_attn_implementation="flash_attention_2"` or `attn_implementation="sdpa"` when loading models
- Warmup run recommended: first `generate()` call is slower due to CUDA kernel compilation

---

## 💡 Architecture Highlights

The VLA module is designed with several key principles:

1. **Modularity** - Each component has a single responsibility
2. **Extensibility** - Easy to add new controllers and features
3. **Thread-Safety** - All threading primitives properly synchronized
4. **Non-Blocking** - Main game loop never waits for controller
5. **Production-Ready** - Well-tested and documented

---

**Ready to collect VLA training data!** 🚀

For questions or issues, refer to the PyRacer main documentation or the AGENTS.md development guide.
