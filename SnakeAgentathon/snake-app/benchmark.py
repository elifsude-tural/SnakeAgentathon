"""Machine speed benchmark for teams to estimate 200ms time budget."""

import time

import numpy as np

# Set this to the tournament machine's average ms/step after running there.
TOURNAMENT_MS = 1.0  # placeholder — update with actual value

NUM_ITERATIONS = 5000
GRID_SIZE = 50


def benchmark() -> None:
    """Run a CPU benchmark simulating game-like numpy operations.

    Prints average time per step and speed ratio vs tournament machine.
    """
    rng = np.random.RandomState(42)

    # Simulate typical agent workload: grid operations, pathfinding-like work
    grid = rng.randint(0, 6, size=(GRID_SIZE, GRID_SIZE), dtype=np.int32)

    print("Running benchmark...")
    print(f"  {NUM_ITERATIONS} iterations of game-like numpy operations")
    print()

    start = time.perf_counter()

    for _ in range(NUM_ITERATIONS):
        # Typical agent operations
        empty_mask = grid == 0
        apple_mask = grid == 5
        body_mask = (grid == 2) | (grid == 4)

        # Distance computations (BFS-like)
        distances = np.full((GRID_SIZE, GRID_SIZE), np.inf)
        head = (rng.randint(GRID_SIZE), rng.randint(GRID_SIZE))
        distances[head] = 0

        # Simulate flood-fill-like computation
        for _ in range(10):
            shifted = np.roll(distances, 1, axis=0)
            shifted = np.minimum(shifted, np.roll(distances, -1, axis=0))
            shifted = np.minimum(shifted, np.roll(distances, 1, axis=1))
            shifted = np.minimum(shifted, np.roll(distances, -1, axis=1))
            distances = np.minimum(distances, shifted + 1)
            distances[body_mask] = np.inf

        # Evaluate moves
        scores = np.zeros(4)
        for i, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            nx = (head[1] + dx) % GRID_SIZE
            ny = (head[0] + dy) % GRID_SIZE
            scores[i] = -distances[ny, nx] if distances[ny, nx] < np.inf else -1000

        _ = int(np.argmax(scores))

        # Regenerate grid occasionally
        if rng.random() < 0.1:
            grid = rng.randint(0, 6, size=(GRID_SIZE, GRID_SIZE), dtype=np.int32)

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / NUM_ITERATIONS) * 1000

    print(f"Results:")
    print(f"  Total time:    {elapsed:.2f}s")
    print(f"  Avg per step:  {avg_ms:.3f}ms")
    print(f"  200ms budget:  {200 / avg_ms:.0f}x headroom")
    print()

    if TOURNAMENT_MS > 0:
        ratio = avg_ms / TOURNAMENT_MS
        print(f"Tournament machine reference: {TOURNAMENT_MS:.3f}ms/step")
        print(f"Your machine speed ratio:     {ratio:.2f}x")
        if ratio > 1.5:
            print("  Your machine is SLOWER — your agent may timeout on tournament machine!")
        elif ratio < 0.7:
            print("  Your machine is FASTER — you have extra headroom on tournament machine.")
        else:
            print("  Your machine is similar speed to the tournament machine.")
    else:
        print("Set TOURNAMENT_MS after running this on the tournament machine.")

    print()
    print("Tip: If your agent uses heavy computation, aim for <100ms on this")
    print("machine to leave margin for the 200ms tournament limit.")


if __name__ == "__main__":
    benchmark()
