# AGENTS.md - PyRacer Development Guide

This document provides guidance for agentic coding agents working in the PyRacer repository.

## Project Overview

PyRacer is a retro racing game built with Pygame (Python 3.12). The project uses object-oriented design with separate modules for game logic, menus, UI, and utilities.

**Key technologies:** Python 3.12, Pygame, OS/file operations

## Running and Testing

### Start the Game
```bash
python main.py
```

### Project Structure
- `main.py` - Entry point; initializes Game and runs menu/game loop
- `data/gameFiles/` - Core game logic (Game, Car, Map classes)
- `data/menus/` - Menu system (MainMenu, OptionsMenu, ControlsMenu, CreditsMenu)
- `data/util/` - Utilities (FPS counter, controls, spritesheet handling)
- `data/images/` - Image assets and spritesheet management

### Current Testing Status
**No automated tests exist.** The project relies on manual testing via `python main.py`. Agents should:
- Test changes manually by running the game
- Verify visual output, gameplay mechanics, and menu navigation
- Use print statements or logging for debugging if needed

## Code Style Guidelines

### Imports
- **Order:** Standard library → Third-party → Local imports
- **Format:** One import per line is preferred; multi-import on same line only for standard library (e.g., `import os, time`)
- **Examples:**
  - ✓ `import pygame, time, os` (standard library)
  - ✓ `from data.gameFiles.map import Map` (local import)
  - ✓ Use `import os` then `os.path.join()` for path operations

### Naming Conventions
- **Classes:** PascalCase (e.g., `Game`, `Car`, `Map`, `Menu`)
- **Functions/Methods:** snake_case (e.g., `game_loop()`, `clamp_speed()`, `load_controls()`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `DISPLAY_W`, `DISPLAY_H`, `SCREEN_WIDTH`)
- **Instance Variables:** snake_case (e.g., `self.running`, `self.position`, `self.speed`)
- **Descriptive names:** Use full words; avoid abbreviations (e.g., `countdown` not `cd`)

### Code Formatting
- **Indentation:** 4 spaces
- **Line length:** No strict limit, but keep reasonable (<100 chars preferred)
- **Spacing:** 
  - One blank line between methods in a class
  - One blank line between top-level function/class definitions
- **Comments:** Use `#` for single-line comments; explain the "why", not the "what"
- **Class definitions:** `class ClassName():` (parentheses even if no inheritance)

### Type Hints
- **Status:** Not currently used in the codebase
- **Recommendation:** Agents should NOT add type hints (would break code style consistency)
- **Future consideration:** Could be added project-wide if agreed upon

### Error Handling
- **Current approach:** Minimal explicit error handling; relies on Pygame and OS exceptions
- **File operations:** Use `os.path.join()` for cross-platform path handling
- **Asset loading:** Always use `os.path.join()` when loading images/sounds
- **Example:** `pygame.image.load(os.path.join(self.game.img_dir, "car.png"))`

### Method Patterns
- **`__init__`:** Initialize all instance variables; load assets
- **`update()`:** Game state updates; called every frame
- **`draw()`:** Render to display surface; called every frame
- **`reset()`:** Reset level/state between game sessions

### Common Patterns
- **Pygame initialization:** Call `pygame.init()` early in Game setup
- **Display surfaces:** Use `pygame.Surface()` for drawing, `pygame.display.set_mode()` for actual screen
- **Asset loading:** Preload images/sounds in `__init__` or dedicated load methods
- **Delta time:** Use `self.game.dt` for frame-independent movement
- **Keyboard input:** Access via `self.game.actions` dict (see controls_menu.py)

## Directory Structure for Agents

When modifying or adding code:
- **Game logic changes:** Edit `data/gameFiles/*.py`
- **Menu changes:** Edit `data/menus/*.py`
- **Utility functions:** Add to `data/util/*.py`
- **Asset management:** Keep images in `data/images/` directory

## Development Workflow

1. **Identify the relevant file(s)** in the structure above
2. **Read existing code** to understand patterns and style
3. **Make changes** following the style guidelines
4. **Test manually** by running `python main.py`
5. **Verify output** - check visual rendering, gameplay, and menu navigation

## Quick Reference - Common Tasks

### Adding a new menu:
1. Create new class inheriting from `Menu` in `data/menus/`
2. Implement required methods: `__init__()`, `display_menu()`, `check_input()`, `draw()`
3. Register in Game class

### Adding a new game feature:
1. Add logic to `data/gameFiles/game.py` or create new class if substantial
2. Call update and draw methods in game loop
3. Use `self.game` reference to access global state

### Modifying car behavior:
1. Edit `data/gameFiles/car.py` in `update()` method
2. Adjust constants (speed multipliers, etc.) at top of method
3. Test by playing and verifying movement feel

### Changing display/rendering:
1. Edit draw methods in relevant classes
2. Use `self.game.display.blit()` for rendering to game surface
3. Adjust position/size values if needed

## Notes for Agents

- **Consistency first:** Follow existing patterns, even if they seem non-standard
- **Test thoroughly:** Since no automated tests exist, manual testing is critical
- **Preserve functionality:** Changes should not break existing menus or gameplay
- **Keep it simple:** The codebase is straightforward; avoid over-engineering
- **Reference:** Check `main.py` to understand initialization and execution flow
