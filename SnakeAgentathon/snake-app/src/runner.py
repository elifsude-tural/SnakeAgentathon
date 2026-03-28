"""Single match runner — connects agents, game, and optional renderer."""

import argparse
import importlib.util
import os
import sys
import time
from typing import Optional

from src.agent import SnakeAgent
from src.game import SnakeGame


def load_agent(filepath: str, player_id: int) -> SnakeAgent:
    """Dynamically load an agent from a Python file.

    Args:
        filepath: Path to the agent .py file.
        player_id: 1 or 2.

    Returns:
        Instantiated SnakeAgent subclass.

    Raises:
        RuntimeError: If no SnakeAgent subclass is found in the file.
    """
    spec = importlib.util.spec_from_file_location("agent_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the SnakeAgent subclass
    agent_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, SnakeAgent)
            and attr is not SnakeAgent
        ):
            agent_class = attr
            break

    if agent_class is None:
        raise RuntimeError(f"No SnakeAgent subclass found in {filepath}")

    return agent_class(player_id)


def get_agent_action(agent: SnakeAgent, observation: dict, default_direction: int,
                     timeout_ms: float = 200.0) -> int:
    """Get an action from an agent with timeout and exception handling.

    Args:
        agent: The agent to query.
        observation: Game observation dict.
        default_direction: Fallback direction on timeout/error.
        timeout_ms: Maximum time in milliseconds.

    Returns:
        Action integer.
    """
    try:
        start = time.perf_counter()
        action = agent.get_action(observation)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > timeout_ms:
            print(f"  [TIMEOUT] Player {agent.player_id}: {elapsed_ms:.1f}ms > {timeout_ms}ms")
            return default_direction
        return action
    except Exception as e:
        print(f"  [ERROR] Player {agent.player_id}: {e}")
        return default_direction


def run_match(agent1: SnakeAgent, agent2: SnakeAgent, seed: int = 0,
              gui: bool = False, speed: int = 5,
              name1: str = "Player 1", name2: str = "Player 2") -> dict:
    """Run a single match between two agents.

    Args:
        agent1: Player 1 agent.
        agent2: Player 2 agent.
        seed: Random seed for the game.
        gui: Whether to render with pygame.
        speed: Game speed 1-9 (only used with gui).
        name1: Display name for player 1.
        name2: Display name for player 2.

    Returns:
        Final game status dict.
    """
    game = SnakeGame(seed=seed)
    renderer = None

    if gui:
        from src.renderer import Renderer
        renderer = Renderer(name1=name1, name2=name2)
        renderer.set_speed(speed)

    result = None
    running = True

    while running:
        if gui:
            action = renderer.handle_events()
            if action == "quit":
                running = False
                break
            if action == "restart":
                game = SnakeGame(seed=seed)
                continue
            if action == "fast_forward":
                # Run to completion without rendering
                while not game.game_over:
                    obs1 = game.get_observation(1)
                    obs2 = game.get_observation(2)
                    a1 = get_agent_action(agent1, obs1, game.snake1.direction)
                    a2 = get_agent_action(agent2, obs2, game.snake2.direction)
                    result = game.step(a1, a2)
                renderer.render(game)
                renderer.show_game_over(result)
                # Wait for keypress or timeout
                wait_start = time.time()
                waiting = True
                while waiting and (time.time() - wait_start) < 5.0:
                    ev = renderer.handle_events()
                    if ev == "quit":
                        running = False
                        waiting = False
                    elif ev is not None:
                        waiting = False
                    renderer.clock.tick(30)
                running = False
                break
            if renderer.paused:
                renderer.render(game)
                renderer.clock.tick(30)
                continue

        if game.game_over:
            if gui:
                renderer.render(game)
                renderer.show_game_over(result)
                wait_start = time.time()
                waiting = True
                while waiting and (time.time() - wait_start) < 5.0:
                    ev = renderer.handle_events()
                    if ev == "quit":
                        waiting = False
                        running = False
                    elif ev is not None:
                        waiting = False
                    renderer.clock.tick(30)
            running = False
            break

        obs1 = game.get_observation(1)
        obs2 = game.get_observation(2)
        a1 = get_agent_action(agent1, obs1, game.snake1.direction)
        a2 = get_agent_action(agent2, obs2, game.snake2.direction)
        result = game.step(a1, a2)

        if gui:
            renderer.render(game)
            renderer.clock.tick(renderer.fps)

    if gui and renderer:
        renderer.quit()

    return result if result else game._status()


def agent_name_from_path(filepath: str) -> str:
    """Extract a display name from an agent file path.

    Args:
        filepath: Path to agent file.

    Returns:
        Clean agent name string.
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def main() -> None:
    """CLI entry point for running a single match."""
    parser = argparse.ArgumentParser(description="Run a 1v1 snake match")
    parser.add_argument("--p1", required=True, help="Path to player 1 agent file")
    parser.add_argument("--p2", required=True, help="Path to player 2 agent file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gui", action="store_true", help="Enable GUI rendering")
    parser.add_argument("--speed", type=int, default=5, choices=range(1, 10),
                        help="Game speed 1-9 (GUI only)")
    args = parser.parse_args()

    name1 = agent_name_from_path(args.p1)
    name2 = agent_name_from_path(args.p2)

    print(f"Match: {name1} vs {name2} (seed={args.seed})")

    agent1 = load_agent(args.p1, 1)
    agent2 = load_agent(args.p2, 2)

    result = run_match(agent1, agent2, seed=args.seed, gui=args.gui,
                       speed=args.speed, name1=name1, name2=name2)

    winner = result["winner"]
    if winner is None:
        outcome = "DRAW"
    elif winner == 1:
        outcome = f"{name1} WINS"
    else:
        outcome = f"{name2} WINS"

    print(f"Result: {outcome}")
    print(f"  Steps: {result['step']}")
    print(f"  {name1}: length={result['p1_length']}, apples={result['p1_apples']}")
    print(f"  {name2}: length={result['p2_length']}, apples={result['p2_apples']}")


if __name__ == "__main__":
    main()
