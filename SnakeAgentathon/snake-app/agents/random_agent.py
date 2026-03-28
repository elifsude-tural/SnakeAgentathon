"""Example agent that makes random valid moves."""

import numpy as np

from src.agent import SnakeAgent
from src.game import UP, RIGHT, DOWN, LEFT, DIRECTION_VECTORS, GRID_SIZE, OPPOSITES


class RandomAgent(SnakeAgent):
    """Agent that picks a random valid direction each step."""

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.rng = np.random.RandomState(player_id)
        self.last_direction = RIGHT if player_id == 1 else LEFT

    def get_action(self, observation: dict) -> int:
        """Return a random action that doesn't reverse into the snake's body.

        Args:
            observation: Game observation dict.

        Returns:
            Random valid action.
        """
        head = observation["my_head"]
        grid = observation["grid"]

        # Find valid moves (not reversing, not hitting own body)
        valid = []
        for action in [UP, RIGHT, DOWN, LEFT]:
            if action == OPPOSITES.get(self.last_direction):
                continue
            dx, dy = DIRECTION_VECTORS[action]
            nx = (head[0] + dx) % GRID_SIZE
            ny = (head[1] + dy) % GRID_SIZE
            cell = grid[ny, nx]
            if cell != 2:  # not my own body
                valid.append(action)

        if not valid:
            # All moves lead to body — just go straight
            chosen = self.last_direction
        else:
            chosen = valid[self.rng.randint(len(valid))]

        self.last_direction = chosen
        return chosen
