"""Example agent that moves toward the nearest apple."""

import numpy as np

from src.agent import SnakeAgent
from src.game import UP, RIGHT, DOWN, LEFT, DIRECTION_VECTORS, GRID_SIZE, OPPOSITES


class GreedyAgent(SnakeAgent):
    """Agent that greedily moves toward the closest apple (toroidal distance)."""

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.last_direction = RIGHT if player_id == 1 else LEFT

    def get_action(self, observation: dict) -> int:
        """Pick the action that moves closest to the nearest apple.

        Args:
            observation: Game observation dict.

        Returns:
            Action toward nearest apple.
        """
        head = observation["my_head"]
        grid = observation["grid"]
        apples = observation["apples"]

        if not apples:
            self.last_direction = self.last_direction
            return self.last_direction

        # Find nearest apple by toroidal Manhattan distance
        def toroidal_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
            dx = min(abs(a[0] - b[0]), GRID_SIZE - abs(a[0] - b[0]))
            dy = min(abs(a[1] - b[1]), GRID_SIZE - abs(a[1] - b[1]))
            return dx + dy

        nearest = min(apples, key=lambda a: toroidal_dist(head, a))

        # Evaluate each valid direction
        best_action = self.last_direction
        best_dist = float("inf")

        for action in [UP, RIGHT, DOWN, LEFT]:
            if action == OPPOSITES.get(self.last_direction):
                continue
            dx, dy = DIRECTION_VECTORS[action]
            nx = (head[0] + dx) % GRID_SIZE
            ny = (head[1] + dy) % GRID_SIZE
            cell = grid[ny, nx]
            if cell == 2:  # own body — avoid
                continue
            dist = toroidal_dist((nx, ny), nearest)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        self.last_direction = best_action
        return best_action
