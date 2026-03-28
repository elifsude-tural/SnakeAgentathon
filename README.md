# Snake Arena — 1v1 AI Snake Competition

A competitive 1v1 snake game environment built for hackathons. Teams build AI agents that compete head-to-head on a toroidal grid, with a real-time GUI for spectating and a league runner for round-robin tournaments.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Pygame](https://img.shields.io/badge/pygame-2.5+-green)
![NumPy](https://img.shields.io/badge/numpy-1.24+-orange)

## Quick Start

```bash
# 1. Create and activate the environment
conda env create -f environment.yml
conda activate snake-arena

# 2. Watch a match
python -m src.runner --p1 agents/random_agent.py --p2 agents/greedy_agent.py --gui

# 3. Run headless
python -m src.runner --p1 agents/random_agent.py --p2 agents/greedy_agent.py
```

## Game Rules

- **Grid:** 50x50 toroidal (edges wrap around)
- **Snakes:** 2 players, each starting at length 3
- **Apples:** Always exactly 3 on the grid — eating one grows the snake by 1
- **Actions:** `0=UP`, `1=RIGHT`, `2=DOWN`, `3=LEFT`
- **Simultaneous movement:** Both snakes move at the same time each tick
- **No reversals:** A snake cannot turn 180 degrees into itself

### Death Conditions

| Condition | Result |
|-----------|--------|
| Head enters own body | That snake dies |
| Head enters opponent's body | That snake dies |
| Both heads enter the same cell | Both die (draw), regardless of length |

### Game End

- A snake dies → opponent wins immediately
- Both die → draw
- 1000 steps reached → longer snake wins; equal = draw

## Project Structure

```
snake-arena/
├── environment.yml          # Conda environment definition
├── benchmark.py             # Machine speed benchmark
├── src/
│   ├── game.py              # Core game logic (no rendering)
│   ├── renderer.py          # Pygame GUI renderer
│   ├── agent.py             # Base SnakeAgent class
│   ├── runner.py            # Single match runner
│   └── league.py            # Round-robin tournament runner
├── agents/
│   ├── random_agent.py      # Example: random valid moves
│   └── greedy_agent.py      # Example: nearest apple chaser
├── teams/                   # Drop agent submissions here
└── results/                 # League output (leaderboard, match logs)
```

## Usage

### Single Match

```bash
# With GUI
python -m src.runner --p1 agents/random_agent.py --p2 agents/greedy_agent.py --gui

# With options
python -m src.runner --p1 agents/random_agent.py --p2 agents/greedy_agent.py --gui --speed 7 --seed 42

# Headless (prints result only)
python -m src.runner --p1 agents/random_agent.py --p2 agents/greedy_agent.py --seed 42
```

| Flag | Description |
|------|-------------|
| `--p1` | Path to player 1 agent file |
| `--p2` | Path to player 2 agent file |
| `--gui` | Enable pygame window |
| `--speed 1-9` | Game speed (GUI only, default: 5) |
| `--seed N` | Random seed for reproducibility |

### League (Round-Robin Tournament)

```bash
# Run all teams against each other (default: 1000 games per matchup)
python -m src.league --teams-dir teams/ --output results/

# Watch games live
python -m src.league --teams-dir teams/ --output results/ --gui
```

Outputs `results/leaderboard.json` and `results/match_log.json`.

**Scoring (per matchup):** Each win = 1.0 pts, each draw = 0.25 pts. Dominance bonus: if an agent wins >500 games in a matchup, each win = 1.25 pts. Tiebreaker: total apples eaten.

### Benchmark

```bash
python benchmark.py
```

Estimates your machine's speed relative to the tournament machine so teams can gauge whether their agent fits within the 200ms time budget.

## GUI Controls

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| R | Restart current match |
| F | Fast-forward (skip rendering) |
| 1-9 | Set speed (1 = 5fps, 5 = 15fps, 9 = 60fps) |
| ESC / Q | Quit |

## Writing an Agent

### 1. Subclass `SnakeAgent`

```python
from src.agent import SnakeAgent

class MyAgent(SnakeAgent):
    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        # Load models, precompute tables, etc. (no time limit)

    def get_action(self, observation: dict) -> int:
        # Must return within 200ms
        # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        return 1
```

### 2. Observation Format

```python
{
    "grid": np.ndarray(50, 50),   # 0=empty, 1=my_head, 2=my_body,
                                   # 3=enemy_head, 4=enemy_body, 5=apple
    "step": int,
    "my_head": (x, y),
    "my_length": int,
    "enemy_head": (x, y),
    "enemy_length": int,
    "apples": [(x, y), ...],
}
```

The grid is always from **your perspective** — you are always `1`/`2`, opponent is always `3`/`4`.

### 3. Test Your Agent

```bash
python -m src.runner --p1 my_agent.py --p2 agents/greedy_agent.py --gui
```

### 4. Submit

Drop your `.py` file into `teams/` on the tournament machine.

**Requirements:**
- Single file, one class inheriting `SnakeAgent`
- Only Python stdlib + numpy (no torch/tensorflow on tournament machine unless announced)
- `get_action` must return within **200ms** (timeout = continue straight)
- Exceptions are caught and logged (action defaults to current direction)

## License

Built for hackathon use. MIT License.
