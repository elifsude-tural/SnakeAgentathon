"""Hybrid competitive snake agent.

Strategy:
  Phase 1 — Safe move filter: reverse guard, self/enemy body collision, immediate death.
  Phase 2 — Flood fill: measure reachable area after each candidate move.
  Phase 3 — Scoring: area + apple proximity + enemy head danger + phase-based risk.
  Phase 4 — Phase control: early=elma agresif, late/ahead=survival, behind=agresif elma.
"""

from collections import deque

import numpy as np

from src.agent import SnakeAgent
from src.game import DOWN, LEFT, RIGHT, UP, DIRECTION_VECTORS, GRID_SIZE, OPPOSITES

ACTIONS = [UP, RIGHT, DOWN, LEFT]

# Grid cell values (from agent's own perspective)
EMPTY = 0
MY_HEAD = 1
MY_BODY = 2
ENEMY_HEAD = 3
ENEMY_BODY = 4
APPLE = 5


class HybridAgent(SnakeAgent):
    """Safe greedy + flood fill + enemy danger + phase-based risk control."""

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.last_direction = RIGHT if player_id == 1 else LEFT

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _toroidal_dist(a: tuple, b: tuple) -> int:
        """Toroidal (wrap-around) Manhattan distance between two (x,y) points."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return min(dx, GRID_SIZE - dx) + min(dy, GRID_SIZE - dy)

    @staticmethod
    def _flood_fill(start: tuple, blocked: set) -> int:
        """BFS flood fill. Returns count of reachable cells from start.

        Args:
            start: (x, y) starting cell.
            blocked: Set of (x, y) cells that cannot be entered.

        Returns:
            Number of reachable cells (including start if not blocked).
        """
        if start in blocked:
            return 0
        visited = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nb = ((cx + ddx) % GRID_SIZE, (cy + ddy) % GRID_SIZE)
                if nb not in visited and nb not in blocked:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited)

    @staticmethod
    def _build_blocked(grid: np.ndarray) -> set:
        """Return set of (x, y) cells that are body segments (walls for pathfinding).

        My body (2), enemy body (4), and enemy head (3) are walls.
        My head (1) is NOT a wall — it's where we currently are and will vacate.
        Apples (5) and empty (0) are passable.
        """
        rows, cols = np.where((grid == MY_BODY) | (grid == ENEMY_BODY) | (grid == ENEMY_HEAD))
        return set(zip(cols.tolist(), rows.tolist()))  # (x, y)

    @staticmethod
    def _enemy_reach_cells(enemy_head: tuple, grid: np.ndarray) -> set:
        """All cells the enemy head could move to next step (all 4 dirs).

        We don't know the enemy's last direction precisely, so we conservatively
        consider all 4 neighbours of the enemy head as potentially dangerous.
        """
        cells = set()
        for action in ACTIONS:
            dx, dy = DIRECTION_VECTORS[action]
            nx = (enemy_head[0] + dx) % GRID_SIZE
            ny = (enemy_head[1] + dy) % GRID_SIZE
            cells.add((nx, ny))
        return cells

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def get_action(self, observation: dict) -> int:
        head: tuple = observation["my_head"]
        grid: np.ndarray = observation["grid"]
        apples: list = observation["apples"]
        step: int = observation["step"]
        my_length: int = observation["my_length"]
        enemy_head: tuple = observation["enemy_head"]
        enemy_length: int = observation["enemy_length"]

        # --- Game phase flags ---
        ahead = my_length > enemy_length
        behind = my_length < enemy_length
        early_game = step < 200
        late_game = step > 750

        # --- Precompute danger zones ---
        base_blocked = self._build_blocked(grid)
        enemy_reach = self._enemy_reach_cells(enemy_head, grid)

        best_score = float("-inf")
        best_action = self.last_direction
        has_safe_action = False

        for action in ACTIONS:
            # Phase 1: reverse guard
            if action == OPPOSITES.get(self.last_direction):
                continue

            dx, dy = DIRECTION_VECTORS[action]
            nx = (head[0] + dx) % GRID_SIZE
            ny = (head[1] + dy) % GRID_SIZE
            cell = grid[ny, nx]

            # Phase 1: hard death — own body or enemy body
            if cell == MY_BODY or cell == ENEMY_BODY:
                continue

            score = 0.0

            # -------------------------------------------------------
            # Enemy head contest scoring
            # -------------------------------------------------------
            if (nx, ny) == enemy_head:
                # Head-on collision = draw
                if behind:
                    score -= 9000  # draw when behind is very bad
                elif ahead:
                    score -= 3000  # draw when ahead is wasteful
                else:
                    score -= 6000
            elif (nx, ny) in enemy_reach:
                # Cell the enemy might also move to: dangerous
                if enemy_length >= my_length:
                    score -= 2500  # enemy could kill or tie us
                else:
                    score -= 400  # less dangerous when we're longer

            # -------------------------------------------------------
            # Phase 2: Flood fill — measure surviving space after move
            # -------------------------------------------------------
            # After moving: old head vacates (becomes body), new head at (nx, ny).
            # Build blocked set for new state: add old head as body, keep rest.
            new_blocked = base_blocked | {head}  # old head now body
            # New head is NOT added to blocked (we're standing there).

            area = self._flood_fill((nx, ny), new_blocked)

            # Severely penalise trapping ourselves
            if area == 0:
                score -= 1_000_000  # certain death
            elif area < 5:
                score -= 50_000
            elif area < 10:
                score -= 10_000
            elif area < my_length // 2:
                score -= 3_000
            elif area < my_length:
                score -= 800

            score += area * 4

            # -------------------------------------------------------
            # Phase 3: Apple attraction
            # -------------------------------------------------------
            if apples:
                nearest = min(apples, key=lambda a: self._toroidal_dist((nx, ny), a))
                apple_dist = self._toroidal_dist((nx, ny), nearest)

                # Apple weight depends on phase
                if early_game:
                    apple_weight = 18  # grow fast early
                elif late_game and ahead:
                    apple_weight = 4   # survival first when winning late
                elif behind:
                    apple_weight = 16  # chase apples when losing
                else:
                    apple_weight = 11

                score -= apple_dist * apple_weight

                # Direct apple pickup bonus
                if (nx, ny) in apples:
                    score += 600

            # -------------------------------------------------------
            # Phase 4: Situational bonuses / penalties
            # -------------------------------------------------------
            if late_game and ahead and area > my_length:
                # Safe and winning — reward conservative moves
                score += 150

            if behind and not late_game:
                # Losing — slight push toward aggression
                score += 80

            # Prefer moving toward the centre (avoid hugging walls uselessly
            # in a toroidal map — not critical, very small bonus)
            centre = GRID_SIZE // 2
            centre_dist = self._toroidal_dist((nx, ny), (centre, centre))
            score -= centre_dist * 0.2

            # Track if any safe action exists
            has_safe_action = True

            if score > best_score:
                best_score = score
                best_action = action

        # Fallback: if somehow no action scored (all were instant death),
        # just continue current direction to avoid a ValueError.
        if not has_safe_action:
            best_action = self.last_direction

        self.last_direction = best_action
        return best_action
