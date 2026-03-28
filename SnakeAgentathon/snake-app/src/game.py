"""SnakeGame — core game logic for 1v1 snake arena. No rendering."""

from collections import deque
from typing import Optional

import numpy as np

# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Direction vectors: (dx, dy) where x=col, y=row
DIRECTION_VECTORS = {
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}

# Opposite directions for reverse-guard
OPPOSITES = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# Grid cell values (from player 1's perspective in internal state)
EMPTY = 0
P1_HEAD = 1
P1_BODY = 2
P2_HEAD = 3
P2_BODY = 4
APPLE = 5

GRID_SIZE = 50
MAX_STEPS = 1000
NUM_APPLES = 3


class Snake:
    """Represents a single snake on the grid."""

    def __init__(self, head: tuple[int, int], direction: int) -> None:
        """Initialize a snake with head position and direction.

        Args:
            head: (x, y) position of the head.
            direction: Initial direction (UP/RIGHT/DOWN/LEFT).
        """
        self.direction = direction
        dx, dy = DIRECTION_VECTORS[direction]
        # Body is a deque: front=head, back=tail
        self.body: deque[tuple[int, int]] = deque()
        for i in range(3):
            x = (head[0] - dx * i) % GRID_SIZE
            y = (head[1] - dy * i) % GRID_SIZE
            self.body.append((x, y))
        self.alive = True
        self.apples_eaten = 0

    @property
    def head(self) -> tuple[int, int]:
        """Return the head position."""
        return self.body[0]

    @property
    def length(self) -> int:
        """Return current length."""
        return len(self.body)


class SnakeGame:
    """Core game logic for a 1v1 snake match.

    Fully deterministic given the same seed. No rendering dependencies.
    """

    def __init__(self, seed: int = 0) -> None:
        """Initialize a new game.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.game_over = False
        self.winner: Optional[int] = None  # 1, 2, or None (draw)

        # Create snakes
        self.snake1 = Snake((12, 25), RIGHT)
        self.snake2 = Snake((37, 25), LEFT)

        # Build the grid
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self._place_snake_on_grid(self.snake1, P1_HEAD, P1_BODY)
        self._place_snake_on_grid(self.snake2, P2_HEAD, P2_BODY)

        # Spawn apples
        self.apples: list[tuple[int, int]] = []
        for _ in range(NUM_APPLES):
            self._spawn_apple()

    def _place_snake_on_grid(self, snake: Snake, head_val: int, body_val: int) -> None:
        """Place a snake's segments on the grid.

        Args:
            snake: The snake to place.
            head_val: Grid value for the head cell.
            body_val: Grid value for body cells.
        """
        for i, (x, y) in enumerate(snake.body):
            self.grid[y, x] = head_val if i == 0 else body_val

    def _spawn_apple(self) -> None:
        """Spawn one apple at a random empty cell."""
        empty_cells = list(zip(*np.where(self.grid == EMPTY)))
        if not empty_cells:
            return
        idx = self.rng.randint(len(empty_cells))
        row, col = empty_cells[idx]
        self.grid[row, col] = APPLE
        self.apples.append((col, row))

    def get_observation(self, player_id: int) -> dict:
        """Build observation dict for a player.

        The observation is always from the agent's own perspective:
        1=my_head, 2=my_body, 3=enemy_head, 4=enemy_body, 5=apple.

        Args:
            player_id: 1 or 2.

        Returns:
            Observation dictionary with grid, positions, and metadata.
        """
        if player_id == 1:
            my_snake = self.snake1
            enemy_snake = self.snake2
            obs_grid = self.grid.copy()
        else:
            my_snake = self.snake2
            enemy_snake = self.snake1
            # Swap player cell values
            obs_grid = np.zeros_like(self.grid)
            obs_grid[self.grid == EMPTY] = EMPTY
            obs_grid[self.grid == P1_HEAD] = 3  # enemy head
            obs_grid[self.grid == P1_BODY] = 4  # enemy body
            obs_grid[self.grid == P2_HEAD] = 1  # my head
            obs_grid[self.grid == P2_BODY] = 2  # my body
            obs_grid[self.grid == APPLE] = APPLE

        return {
            "grid": obs_grid,
            "step": self.step_count,
            "my_head": my_snake.head,
            "my_length": my_snake.length,
            "enemy_head": enemy_snake.head,
            "enemy_length": enemy_snake.length,
            "apples": list(self.apples),
        }

    def _validate_action(self, action: int, snake: Snake) -> int:
        """Validate an action, preventing reversal.

        Args:
            action: Proposed action.
            snake: The snake taking the action.

        Returns:
            Valid action (original or current direction if invalid).
        """
        if action not in DIRECTION_VECTORS:
            return snake.direction
        if action == OPPOSITES[snake.direction]:
            return snake.direction
        return action

    def step(self, action1: int, action2: int) -> dict:
        """Advance the game by one step.

        Both actions resolve simultaneously.

        Args:
            action1: Action for player 1.
            action2: Action for player 2.

        Returns:
            Dict with game state info: winner, game_over, step, lengths.
        """
        if self.game_over:
            return self._status()

        # Validate actions
        action1 = self._validate_action(action1, self.snake1)
        action2 = self._validate_action(action2, self.snake2)

        # Update directions
        self.snake1.direction = action1
        self.snake2.direction = action2

        # Compute new head positions
        dx1, dy1 = DIRECTION_VECTORS[action1]
        new_head1 = (
            (self.snake1.head[0] + dx1) % GRID_SIZE,
            (self.snake1.head[1] + dy1) % GRID_SIZE,
        )
        dx2, dy2 = DIRECTION_VECTORS[action2]
        new_head2 = (
            (self.snake2.head[0] + dx2) % GRID_SIZE,
            (self.snake2.head[1] + dy2) % GRID_SIZE,
        )

        # Check if apples are eaten
        apple1 = new_head1 in self.apples
        apple2 = new_head2 in self.apples
        # Handle case where both heads target the same apple
        same_apple = apple1 and apple2 and new_head1 == new_head2

        # Move snakes: add new head
        self.snake1.body.appendleft(new_head1)
        self.snake2.body.appendleft(new_head2)

        # Remove tails (unless apple eaten)
        if not apple1:
            tail1 = self.snake1.body.pop()
        else:
            tail1 = None
            self.snake1.apples_eaten += 1

        if not apple2:
            tail2 = self.snake2.body.pop()
        else:
            tail2 = None
            self.snake2.apples_eaten += 1

        # Check collisions
        # Build body sets (excluding heads for body collision checks)
        body1_set = set(list(self.snake1.body)[1:])
        body2_set = set(list(self.snake2.body)[1:])

        dead1 = False
        dead2 = False

        # Self-collision
        if new_head1 in body1_set:
            dead1 = True
        if new_head2 in body2_set:
            dead2 = True

        # Body collision (head into opponent's body)
        if new_head1 in body2_set:
            dead1 = True
        if new_head2 in body1_set:
            dead2 = True

        # Head-on collision — always both die (draw)
        if new_head1 == new_head2:
            dead1 = True
            dead2 = True

        # Apply deaths
        if dead1:
            self.snake1.alive = False
        if dead2:
            self.snake2.alive = False

        # Determine winner
        if dead1 and dead2:
            self.game_over = True
            self.winner = None  # draw
        elif dead1:
            self.game_over = True
            self.winner = 2
        elif dead2:
            self.game_over = True
            self.winner = 1

        # Rebuild grid
        self.grid.fill(EMPTY)

        if self.snake1.alive:
            self._place_snake_on_grid(self.snake1, P1_HEAD, P1_BODY)
        if self.snake2.alive:
            self._place_snake_on_grid(self.snake2, P2_HEAD, P2_BODY)

        # Handle apples
        if apple1 and not same_apple:
            self.apples.remove(new_head1)
            self._place_apples_on_grid()
            self._spawn_apple()
        if apple2 and (not same_apple or not apple1):
            if new_head2 in self.apples:
                self.apples.remove(new_head2)
            self._place_apples_on_grid()
            self._spawn_apple()
        if same_apple:
            # Both targeted same apple — it's eaten by whoever survives,
            # or removed if both ate it
            if new_head1 in self.apples:
                self.apples.remove(new_head1)
            self._place_apples_on_grid()
            self._spawn_apple()

        self._place_apples_on_grid()

        self.step_count += 1

        # Check max steps
        if not self.game_over and self.step_count >= MAX_STEPS:
            self.game_over = True
            if self.snake1.length > self.snake2.length:
                self.winner = 1
            elif self.snake2.length > self.snake1.length:
                self.winner = 2
            else:
                self.winner = None  # draw

        return self._status()

    def _place_apples_on_grid(self) -> None:
        """Place all current apples on the grid."""
        for ax, ay in self.apples:
            self.grid[ay, ax] = APPLE

    def _status(self) -> dict:
        """Return current game status.

        Returns:
            Dict with game_over, winner, step, and snake lengths/apples.
        """
        return {
            "game_over": self.game_over,
            "winner": self.winner,
            "step": self.step_count,
            "p1_length": self.snake1.length,
            "p2_length": self.snake2.length,
            "p1_apples": self.snake1.apples_eaten,
            "p2_apples": self.snake2.apples_eaten,
        }
