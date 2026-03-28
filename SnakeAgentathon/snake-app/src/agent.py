"""Base agent class for the snake arena."""

import numpy as np


class SnakeAgent:
    """Base class every team must subclass."""

    def __init__(self, player_id: int) -> None:
        """Called once before the match starts. No time limit.

        Use for model loading, precomputation, etc.

        Args:
            player_id: 1 or 2
        """
        self.player_id = player_id

    def get_action(self, observation: dict) -> int:
        """Called every game step. Must return within 200ms.

        Args:
            observation: {
                "grid": np.ndarray (50,50) int,
                    # 0=empty, 1=my_head, 2=my_body, 3=enemy_head, 4=enemy_body, 5=apple
                "step": int,
                "my_head": tuple(int, int),
                "my_length": int,
                "enemy_head": tuple(int, int),
                "enemy_length": int,
                "apples": list[tuple(int, int)],
            }

        Returns:
            int — 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        """
        raise NotImplementedError
