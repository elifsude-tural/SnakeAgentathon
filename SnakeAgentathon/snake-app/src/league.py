"""Round-robin tournament runner with logging."""

import argparse
import json
import os
from itertools import combinations
from typing import Any

from src.runner import load_agent, run_match, agent_name_from_path


def discover_agents(teams_dir: str) -> list[str]:
    """Find all agent .py files in the teams directory.

    Args:
        teams_dir: Path to teams directory.

    Returns:
        Sorted list of agent file paths.
    """
    agents = []
    for filename in sorted(os.listdir(teams_dir)):
        if filename.endswith(".py") and not filename.startswith("_"):
            agents.append(os.path.join(teams_dir, filename))
    return agents


def run_league(teams_dir: str, games_per_match: int, output_dir: str,
               gui: bool = False, speed: int = 5) -> None:
    """Run a full round-robin league.

    Args:
        teams_dir: Path to directory containing agent files.
        games_per_match: Number of games per matchup.
        output_dir: Directory for output files.
        gui: Whether to render matches.
        speed: Game speed for GUI.
    """
    agent_paths = discover_agents(teams_dir)

    if len(agent_paths) < 2:
        print(f"Need at least 2 agents in {teams_dir}, found {len(agent_paths)}")
        return

    agent_names = [agent_name_from_path(p) for p in agent_paths]
    print(f"League: {len(agent_paths)} agents found")
    for name in agent_names:
        print(f"  - {name}")
    print()

    # Initialize standings
    standings: dict[str, dict[str, Any]] = {}
    for name in agent_names:
        standings[name] = {
            "points": 0.0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "total_apples": 0,
            "games_played": 0,
        }

    match_log: list[dict] = []

    # Round-robin: every pair plays N games
    matchups = list(combinations(range(len(agent_paths)), 2))
    total_games = len(matchups) * games_per_match

    print(f"Total matchups: {len(matchups)}, games: {total_games}")
    print("=" * 60)

    # Dominance bonus threshold: if wins > this in a matchup, each win = 1.25 instead of 1.0
    DOMINANCE_THRESHOLD = games_per_match // 2

    game_count = 0
    for i, j in matchups:
        path_i, path_j = agent_paths[i], agent_paths[j]
        name_i, name_j = agent_names[i], agent_names[j]

        print(f"\n{name_i} vs {name_j} ({games_per_match} games)")

        # Track per-matchup results for scoring
        matchup_wins_i = 0
        matchup_wins_j = 0
        matchup_draws = 0
        matchup_games: list[dict] = []

        for game_num in range(games_per_match):
            game_count += 1
            # Alternate spawn sides
            if game_num % 2 == 0:
                p1_path, p2_path = path_i, path_j
                p1_name, p2_name = name_i, name_j
            else:
                p1_path, p2_path = path_j, path_i
                p1_name, p2_name = name_j, name_i

            seed = game_count * 1000 + game_num

            # Load agents fresh each game
            try:
                agent1 = load_agent(p1_path, 1)
            except Exception as e:
                print(f"  Game {game_num + 1}: {p1_name} CRASHED on load: {e}")
                _record_crash(standings, match_log, p1_name, p2_name, game_num, seed)
                # Count crash as a loss/win for matchup scoring
                if p1_name == name_i:
                    matchup_wins_j += 1
                else:
                    matchup_wins_i += 1
                continue

            try:
                agent2 = load_agent(p2_path, 2)
            except Exception as e:
                print(f"  Game {game_num + 1}: {p2_name} CRASHED on load: {e}")
                _record_crash(standings, match_log, p2_name, p1_name, game_num, seed)
                if p2_name == name_j:
                    matchup_wins_i += 1
                else:
                    matchup_wins_j += 1
                continue

            try:
                result = run_match(agent1, agent2, seed=seed, gui=gui,
                                   speed=speed, name1=p1_name, name2=p2_name)
            except Exception as e:
                print(f"  Game {game_num + 1}: MATCH ERROR: {e}")
                continue

            # Map result back to original names
            winner_id = result.get("winner")
            if winner_id == 1:
                winner_name = p1_name
            elif winner_id == 2:
                winner_name = p2_name
            else:
                winner_name = None

            # Track per-matchup wins/draws
            if winner_name == name_i:
                matchup_wins_i += 1
                standings[name_i]["wins"] += 1
                standings[name_j]["losses"] += 1
            elif winner_name == name_j:
                matchup_wins_j += 1
                standings[name_j]["wins"] += 1
                standings[name_i]["losses"] += 1
            else:
                matchup_draws += 1
                standings[name_i]["draws"] += 1
                standings[name_j]["draws"] += 1

            # Track apples
            if game_num % 2 == 0:
                standings[name_i]["total_apples"] += result["p1_apples"]
                standings[name_j]["total_apples"] += result["p2_apples"]
            else:
                standings[name_j]["total_apples"] += result["p1_apples"]
                standings[name_i]["total_apples"] += result["p2_apples"]

            standings[name_i]["games_played"] += 1
            standings[name_j]["games_played"] += 1

            # Log
            game_entry = {
                "matchup": f"{name_i} vs {name_j}",
                "game_num": game_num + 1,
                "p1": p1_name,
                "p2": p2_name,
                "seed": seed,
                "winner": winner_name,
                "steps": result["step"],
                "p1_length": result["p1_length"],
                "p2_length": result["p2_length"],
                "p1_apples": result["p1_apples"],
                "p2_apples": result["p2_apples"],
            }
            match_log.append(game_entry)

            outcome = winner_name if winner_name else "Draw"
            print(f"  Game {game_num + 1}: {outcome} "
                  f"(step {result['step']}, "
                  f"{p1_name}={result['p1_length']}, "
                  f"{p2_name}={result['p2_length']})")

        # Calculate matchup points using the new scoring system
        # Each win = 1.0 (or 1.25 if dominant), each draw = 0.25
        win_multiplier_i = 1.25 if matchup_wins_i > DOMINANCE_THRESHOLD else 1.0
        win_multiplier_j = 1.25 if matchup_wins_j > DOMINANCE_THRESHOLD else 1.0

        points_i = matchup_wins_i * win_multiplier_i + matchup_draws * 0.25
        points_j = matchup_wins_j * win_multiplier_j + matchup_draws * 0.25

        standings[name_i]["points"] += points_i
        standings[name_j]["points"] += points_j

        print(f"  Matchup result: {name_i}={matchup_wins_i}W/{matchup_draws}D/{matchup_wins_j}L "
              f"→ {points_i:.1f}pts | {name_j}={matchup_wins_j}W/{matchup_draws}D/{matchup_wins_i}L "
              f"→ {points_j:.1f}pts")
        if matchup_wins_i > DOMINANCE_THRESHOLD:
            print(f"  ** {name_i} dominance bonus active (>{DOMINANCE_THRESHOLD} wins → 1.25x)")
        if matchup_wins_j > DOMINANCE_THRESHOLD:
            print(f"  ** {name_j} dominance bonus active (>{DOMINANCE_THRESHOLD} wins → 1.25x)")

    # Build leaderboard sorted by points, then total_apples
    leaderboard = sorted(
        [{"name": name, **stats} for name, stats in standings.items()],
        key=lambda x: (x["points"], x["total_apples"]),
        reverse=True,
    )

    # Add rank
    for rank, entry in enumerate(leaderboard, 1):
        entry["rank"] = rank

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "leaderboard.json"), "w") as f:
        json.dump(leaderboard, f, indent=2)

    with open(os.path.join(output_dir, "match_log.json"), "w") as f:
        json.dump(match_log, f, indent=2)

    # Print leaderboard
    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD")
    print("=" * 70)
    print(f"{'Rank':<6}{'Name':<20}{'Points':<10}{'W':<6}{'D':<6}{'L':<6}{'Apples':<8}")
    print("-" * 70)
    for entry in leaderboard:
        print(f"{entry['rank']:<6}{entry['name']:<20}{entry['points']:<10.2f}"
              f"{entry['wins']:<6}{entry['draws']:<6}{entry['losses']:<6}"
              f"{entry['total_apples']:<8}")
    print()
    print(f"Results saved to {output_dir}/")


def _record_crash(standings: dict, match_log: list, crashed_name: str,
                  other_name: str, game_num: int, seed: int) -> None:
    """Record a crash as a loss for the crashed agent.

    Args:
        standings: Standings dict to update.
        match_log: Match log list to append to.
        crashed_name: Name of agent that crashed.
        other_name: Name of the opponent.
        game_num: Game number in the matchup.
        seed: Random seed used.
    """
    standings[crashed_name]["losses"] += 1
    standings[other_name]["wins"] += 1
    standings[crashed_name]["games_played"] += 1
    standings[other_name]["games_played"] += 1

    match_log.append({
        "matchup": f"{crashed_name} vs {other_name}",
        "game_num": game_num + 1,
        "p1": crashed_name,
        "p2": other_name,
        "seed": seed,
        "winner": other_name,
        "steps": 0,
        "p1_length": 0,
        "p2_length": 0,
        "p1_apples": 0,
        "p2_apples": 0,
        "note": f"{crashed_name} crashed on load",
    })


def main() -> None:
    """CLI entry point for the league runner."""
    parser = argparse.ArgumentParser(description="Run a round-robin snake league")
    parser.add_argument("--teams-dir", required=True, help="Directory with agent files")
    parser.add_argument("--games-per-match", type=int, default=1000,
                        help="Games per matchup")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--gui", action="store_true", help="Watch games live")
    parser.add_argument("--speed", type=int, default=5, choices=range(1, 10),
                        help="Game speed 1-9 (GUI only)")
    args = parser.parse_args()

    run_league(args.teams_dir, args.games_per_match, args.output,
               gui=args.gui, speed=args.speed)


if __name__ == "__main__":
    main()
