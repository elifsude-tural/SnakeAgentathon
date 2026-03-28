"""Edi — evrimsel optimizasyonla tuning yapilan hibrit snake ajani.

Temel mimari hybrid_agent ile ayni. Fark: skorlama agirliklarini
dis dosyadan (edi_weights.npy) yukler. Dosya yoksa DEFAULT_WEIGHTS kullanir.
Egitim icin train_edi.py'yi kullan.

Performans optimizasyonlari:
  - Flood fill: numpy boolean array (set yerine) + erken cikis
  - Blocked cells: grid'den direkt numpy where ile
"""

import os
from collections import deque

import numpy as np

from src.agent import SnakeAgent
from src.game import DOWN, LEFT, RIGHT, UP, DIRECTION_VECTORS, GRID_SIZE, OPPOSITES

ACTIONS = [UP, RIGHT, DOWN, LEFT]

EMPTY = 0
MY_HEAD = 1
MY_BODY = 2
ENEMY_HEAD = 3
ENEMY_BODY = 4
APPLE = 5

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "edi_weights.npy")

DEFAULT_WEIGHTS = {
    # Alan skorlamasi
    "area_multiplier":   4.0,
    "area_trap_5":      -50_000.0,
    "area_trap_10":     -10_000.0,
    "area_trap_half":    -3_000.0,
    "area_trap_length":    -800.0,
    # Elma agirliklar (faza gore)
    "apple_early":        18.0,
    "apple_late_ahead":    4.0,
    "apple_behind":       16.0,
    "apple_normal":       11.0,
    "apple_pickup_bonus": 600.0,
    # Dusman kafa tehlikesi
    "enemy_head_behind":  -9_000.0,
    "enemy_head_ahead":   -3_000.0,
    "enemy_head_equal":   -6_000.0,
    "enemy_reach_danger": -2_500.0,
    "enemy_reach_safe":    -400.0,
    # Faz bonuslari
    "late_ahead_bonus":    150.0,
    "behind_bonus":         80.0,
    "centre_weight":         0.2,
}


def load_weights() -> dict:
    if os.path.exists(WEIGHTS_FILE):
        data = np.load(WEIGHTS_FILE, allow_pickle=True).item()
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(data)
        return weights
    return dict(DEFAULT_WEIGHTS)


class EdiAgent(SnakeAgent):
    """Evrimsel optimizasyonla tuning yapilan hibrit ajan."""

    def __init__(self, player_id: int, weights: dict | None = None) -> None:
        super().__init__(player_id)
        self.last_direction = RIGHT if player_id == 1 else LEFT
        self.w = weights if weights is not None else load_weights()
        # Flood fill icin kalici visited array — her adimda sifirlanir
        self._visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    # ------------------------------------------------------------------
    # Yardimci fonksiyonlar
    # ------------------------------------------------------------------

    @staticmethod
    def _toroidal_dist(a: tuple, b: tuple) -> int:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return min(dx, GRID_SIZE - dx) + min(dy, GRID_SIZE - dy)

    def _flood_fill(self, start: tuple, blocked: np.ndarray, limit: int) -> int:
        """BFS flood fill — numpy boolean array + erken cikis.

        Args:
            start:   (x, y) baslangic noktasi.
            blocked: (GRID_SIZE, GRID_SIZE) bool array; True = girilmez.
            limit:   Bu kadara ulasinca dur (erken cikis).

        Returns:
            Ulasilan hucre sayisi (limit ile sinirli).
        """
        sx, sy = start
        if blocked[sy, sx]:
            return 0

        visited = self._visited
        visited[:] = False
        visited[sy, sx] = True

        queue = deque([(sx, sy)])
        count = 0

        while queue:
            cx, cy = queue.popleft()
            count += 1
            if count >= limit:
                return count  # yeterince genis — erken cik

            for ddx, ddy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx = (cx + ddx) % GRID_SIZE
                ny = (cy + ddy) % GRID_SIZE
                if not visited[ny, nx] and not blocked[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

        return count

    @staticmethod
    def _enemy_reach_cells(enemy_head: tuple) -> set:
        cells = set()
        for action in ACTIONS:
            dx, dy = DIRECTION_VECTORS[action]
            cells.add(((enemy_head[0] + dx) % GRID_SIZE,
                        (enemy_head[1] + dy) % GRID_SIZE))
        return cells

    # ------------------------------------------------------------------
    # Ana karar
    # ------------------------------------------------------------------

    def get_action(self, observation: dict) -> int:
        head: tuple        = observation["my_head"]
        grid: np.ndarray   = observation["grid"]
        apples: list       = observation["apples"]
        step: int          = observation["step"]
        my_length: int     = observation["my_length"]
        enemy_head: tuple  = observation["enemy_head"]
        enemy_length: int  = observation["enemy_length"]

        w = self.w
        ahead      = my_length > enemy_length
        behind     = my_length < enemy_length
        early_game = step < 200
        late_game  = step > 750

        # Engel haritasi: MY_BODY, ENEMY_BODY, ENEMY_HEAD = True
        blocked_base = (grid == MY_BODY) | (grid == ENEMY_BODY) | (grid == ENEMY_HEAD)

        # Mevcut kafayi da engele ekle (tasindiktan sonra bos kalir ama
        # flood fill icin muhafazakar: blokluyoruz)
        hx, hy = head
        blocked_base[hy, hx] = True

        enemy_reach = self._enemy_reach_cells(enemy_head)

        # Flood fill erken cikis limiti: my_length'in 2 kati yeterli
        ff_limit = my_length * 2 + 20

        best_score  = float("-inf")
        best_action = self.last_direction
        has_safe    = False

        for action in ACTIONS:
            if action == OPPOSITES.get(self.last_direction):
                continue

            dx, dy = DIRECTION_VECTORS[action]
            nx = (head[0] + dx) % GRID_SIZE
            ny = (head[1] + dy) % GRID_SIZE
            cell = grid[ny, nx]

            if cell == MY_BODY or cell == ENEMY_BODY:
                continue

            score = 0.0

            # Dusman kafa contest
            if (nx, ny) == enemy_head:
                if behind:
                    score += w["enemy_head_behind"]
                elif ahead:
                    score += w["enemy_head_ahead"]
                else:
                    score += w["enemy_head_equal"]
            elif (nx, ny) in enemy_reach:
                if enemy_length >= my_length:
                    score += w["enemy_reach_danger"]
                else:
                    score += w["enemy_reach_safe"]

            # Flood fill
            area = self._flood_fill((nx, ny), blocked_base, ff_limit)

            if area == 0:
                score -= 1_000_000
            elif area < 5:
                score += w["area_trap_5"]
            elif area < 10:
                score += w["area_trap_10"]
            elif area < my_length // 2:
                score += w["area_trap_half"]
            elif area < my_length:
                score += w["area_trap_length"]

            score += area * w["area_multiplier"]

            # Elma cazibesi
            if apples:
                nearest    = min(apples, key=lambda a: self._toroidal_dist((nx, ny), a))
                apple_dist = self._toroidal_dist((nx, ny), nearest)

                if early_game:
                    aw = w["apple_early"]
                elif late_game and ahead:
                    aw = w["apple_late_ahead"]
                elif behind:
                    aw = w["apple_behind"]
                else:
                    aw = w["apple_normal"]

                score -= apple_dist * aw
                if (nx, ny) in apples:
                    score += w["apple_pickup_bonus"]

            # Faz bonuslari
            if late_game and ahead and area > my_length:
                score += w["late_ahead_bonus"]
            if behind and not late_game:
                score += w["behind_bonus"]

            centre = GRID_SIZE // 2
            score -= self._toroidal_dist((nx, ny), (centre, centre)) * w["centre_weight"]

            has_safe = True
            if score > best_score:
                best_score  = score
                best_action = action

        if not has_safe:
            best_action = self.last_direction

        self.last_direction = best_action
        return best_action
