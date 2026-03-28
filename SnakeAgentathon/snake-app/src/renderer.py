"""PyGame renderer for the snake arena. Observes game state and draws."""

import pygame

from src.game import (
    SnakeGame, GRID_SIZE, MAX_STEPS,
    EMPTY, P1_HEAD, P1_BODY, P2_HEAD, P2_BODY, APPLE,
)

# Layout
CELL_SIZE = 12
GRID_PX = GRID_SIZE * CELL_SIZE  # 600
PANEL_WIDTH = 220
WINDOW_WIDTH = GRID_PX + PANEL_WIDTH
WINDOW_HEIGHT = GRID_PX + 40  # 640

# Colors
COLOR_BG = (0x1a, 0x1a, 0x2e)
COLOR_GRID_LINE = (0x16, 0x21, 0x3e)
COLOR_P1_HEAD = (0x00, 0xff, 0x88)
COLOR_P1_BODY = (0x00, 0xcc, 0x66)
COLOR_P2_HEAD = (0x00, 0xaa, 0xff)
COLOR_P2_BODY = (0x00, 0x88, 0xcc)
COLOR_APPLE = (0xff, 0x44, 0x44)
COLOR_PANEL_BG = (0x0f, 0x0f, 0x23)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (170, 170, 170)
COLOR_OVERLAY_BG = (0, 0, 0, 180)

# Speed presets: key 1-9 → fps
SPEED_FPS = {1: 5, 2: 7, 3: 10, 4: 12, 5: 15, 6: 20, 7: 30, 8: 45, 9: 60}

CELL_COLORS = {
    EMPTY: COLOR_BG,
    P1_HEAD: COLOR_P1_HEAD,
    P1_BODY: COLOR_P1_BODY,
    P2_HEAD: COLOR_P2_HEAD,
    P2_BODY: COLOR_P2_BODY,
    APPLE: COLOR_APPLE,
}


class Renderer:
    """PyGame-based renderer for the snake game."""

    def __init__(self, name1: str = "Player 1", name2: str = "Player 2") -> None:
        """Initialize the renderer.

        Args:
            name1: Display name for player 1.
            name2: Display name for player 2.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake Arena — 1v1")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18)
        self.font_large = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.name1 = name1
        self.name2 = name2
        self.paused = False
        self.fps = 10  # default speed

    def set_speed(self, speed: int) -> None:
        """Set game speed from key 1-9.

        Args:
            speed: Speed level 1-9.
        """
        self.fps = SPEED_FPS.get(speed, 15)

    def handle_events(self) -> str | None:
        """Process pygame events and return control action if any.

        Returns:
            "quit", "restart", "fast_forward", or None.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "quit"
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_r:
                    return "restart"
                if event.key == pygame.K_f:
                    return "fast_forward"
                # Speed keys 1-9
                if pygame.K_1 <= event.key <= pygame.K_9:
                    speed = event.key - pygame.K_1 + 1
                    self.set_speed(speed)
        return None

    def render(self, game: SnakeGame) -> None:
        """Render the current game state.

        Args:
            game: The game to render.
        """
        self.screen.fill(COLOR_BG)
        self._draw_grid(game)
        self._draw_panel(game)
        pygame.display.flip()

    def _draw_grid(self, game: SnakeGame) -> None:
        """Draw the game grid.

        Args:
            game: The game to render.
        """
        # Draw cells
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell = game.grid[row, col]
                if cell == EMPTY:
                    continue
                color = CELL_COLORS.get(cell, COLOR_BG)
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            # Vertical
            pygame.draw.line(self.screen, COLOR_GRID_LINE,
                             (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_PX))
            # Horizontal
            pygame.draw.line(self.screen, COLOR_GRID_LINE,
                             (0, i * CELL_SIZE), (GRID_PX, i * CELL_SIZE))

    def _draw_panel(self, game: SnakeGame) -> None:
        """Draw the info panel on the right.

        Args:
            game: The game to render.
        """
        panel_x = GRID_PX
        panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)

        x = panel_x + 15
        y = 20

        # Title
        title = self.font_large.render("SNAKE ARENA", True, COLOR_WHITE)
        self.screen.blit(title, (x, y))
        y += 45

        # Divider
        pygame.draw.line(self.screen, COLOR_GRAY, (x, y), (panel_x + PANEL_WIDTH - 15, y))
        y += 15

        # Player 1
        pygame.draw.rect(self.screen, COLOR_P1_HEAD, (x, y, 14, 14))
        p1_label = self.font.render(f" {self.name1}", True, COLOR_P1_HEAD)
        self.screen.blit(p1_label, (x + 16, y - 2))
        y += 25
        p1_info = self.font_small.render(
            f"  Length: {game.snake1.length}  Apples: {game.snake1.apples_eaten}", True, COLOR_GRAY)
        self.screen.blit(p1_info, (x, y))
        y += 30

        # Player 2
        pygame.draw.rect(self.screen, COLOR_P2_HEAD, (x, y, 14, 14))
        p2_label = self.font.render(f" {self.name2}", True, COLOR_P2_HEAD)
        self.screen.blit(p2_label, (x + 16, y - 2))
        y += 25
        p2_info = self.font_small.render(
            f"  Length: {game.snake2.length}  Apples: {game.snake2.apples_eaten}", True, COLOR_GRAY)
        self.screen.blit(p2_info, (x, y))
        y += 40

        # Divider
        pygame.draw.line(self.screen, COLOR_GRAY, (x, y), (panel_x + PANEL_WIDTH - 15, y))
        y += 15

        # Step counter
        step_text = self.font.render(f"Step: {game.step_count}/{MAX_STEPS}", True, COLOR_WHITE)
        self.screen.blit(step_text, (x, y))
        y += 30

        # Speed
        speed_text = self.font_small.render(f"Speed: {self.fps} fps", True, COLOR_GRAY)
        self.screen.blit(speed_text, (x, y))
        y += 25

        # Paused
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 255, 0))
            self.screen.blit(pause_text, (x, y))
        y += 30

        # Status
        if game.game_over:
            if game.winner == 1:
                status = f"{self.name1} WINS!"
                color = COLOR_P1_HEAD
            elif game.winner == 2:
                status = f"{self.name2} WINS!"
                color = COLOR_P2_HEAD
            else:
                status = "DRAW!"
                color = COLOR_WHITE
            status_text = self.font_large.render(status, True, color)
            self.screen.blit(status_text, (x, y))
        else:
            status_text = self.font_small.render("Playing...", True, COLOR_GRAY)
            self.screen.blit(status_text, (x, y))

        # Controls help at bottom
        y = WINDOW_HEIGHT - 120
        pygame.draw.line(self.screen, COLOR_GRAY, (x, y), (panel_x + PANEL_WIDTH - 15, y))
        y += 10
        controls = [
            "SPC  Pause/Resume",
            "R    Restart",
            "F    Fast-forward",
            "1-9  Speed",
            "ESC  Quit",
        ]
        for line in controls:
            ctrl_text = self.font_small.render(line, True, COLOR_GRAY)
            self.screen.blit(ctrl_text, (x, y))
            y += 18

    def show_game_over(self, result: dict) -> None:
        """Show a game-over overlay on the screen.

        Args:
            result: Game result dict with winner info.
        """
        # Semi-transparent overlay
        overlay = pygame.Surface((GRID_PX, GRID_PX), pygame.SRCALPHA)
        overlay.fill(COLOR_OVERLAY_BG)
        self.screen.blit(overlay, (0, 0))

        # Result text
        winner = result.get("winner")
        if winner == 1:
            text = f"{self.name1} WINS!"
            color = COLOR_P1_HEAD
        elif winner == 2:
            text = f"{self.name2} WINS!"
            color = COLOR_P2_HEAD
        else:
            text = "DRAW!"
            color = COLOR_WHITE

        rendered = self.font_large.render(text, True, color)
        rect = rendered.get_rect(center=(GRID_PX // 2, GRID_PX // 2 - 20))
        self.screen.blit(rendered, rect)

        # Step info
        step_info = self.font.render(f"Step {result['step']}/{MAX_STEPS}", True, COLOR_GRAY)
        step_rect = step_info.get_rect(center=(GRID_PX // 2, GRID_PX // 2 + 20))
        self.screen.blit(step_info, step_rect)

        pygame.display.flip()

    def quit(self) -> None:
        """Clean up pygame."""
        pygame.quit()
