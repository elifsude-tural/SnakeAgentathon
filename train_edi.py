"""Evrimsel optimizasyon — EdiAgent agirliklarini optimize et.

Algoritma (1+lambda Evrim Stratejisi):
  1. Baslangic: DEFAULT_WEIGHTS
  2. Her nesilde lambda adet mutant uret (Gaussian gurultu)
  3. Mutantlar multiprocessing ile paralel degerlendirilir
  4. En iyi mutanti yeni ebeveyn yap
  5. En iyi agirliklar edi_weights.npy'ye kaydedilir

Kullanim:
  python train_edi.py
  python train_edi.py --games 40 --generations 30 --offspring 8 --workers 4
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from agents.edi_agent import DEFAULT_WEIGHTS, WEIGHTS_FILE, EdiAgent
from agents.hybrid_agent import HybridAgent
from src.game import SnakeGame


# ------------------------------------------------------------------
# Tek oyun
# ------------------------------------------------------------------

def play_game(agent1, agent2, seed: int):
    game = SnakeGame(seed=seed)
    while not game.game_over:
        game.step(
            agent1.get_action(game.get_observation(1)),
            agent2.get_action(game.get_observation(2)),
        )
    return game.winner


# ------------------------------------------------------------------
# Degerlendirme (multiprocessing worker icin top-level fonksiyon)
# ------------------------------------------------------------------

def _eval_worker(args):
    """Multiprocessing worker — (weights, n_games, seed_offset) alir."""
    weights, n_games, seed_offset = args
    wins = 0.0
    for i in range(n_games):
        seed = seed_offset + i
        if i % 2 == 0:
            edi   = EdiAgent(player_id=1, weights=weights)
            rival = HybridAgent(player_id=2)
            w = play_game(edi, rival, seed)
            if w == 1:
                wins += 1.0
            elif w is None:
                wins += 0.5
        else:
            rival = HybridAgent(player_id=1)
            edi   = EdiAgent(player_id=2, weights=weights)
            w = play_game(rival, edi, seed)
            if w == 2:
                wins += 1.0
            elif w is None:
                wins += 0.5
    return wins / n_games


# ------------------------------------------------------------------
# Mutasyon
# ------------------------------------------------------------------

def mutate(weights: dict, sigma: float, rng: np.random.RandomState) -> dict:
    mutant = {}
    for key, val in weights.items():
        noise = rng.normal(0, abs(val) * sigma + 1e-6)
        mutant[key] = val + noise
    return mutant


# ------------------------------------------------------------------
# Ana egitim dongusu
# ------------------------------------------------------------------

def train(generations: int, offspring: int, games: int,
          sigma: float, seed: int, workers: int) -> dict:

    rng = np.random.RandomState(seed)
    parent_weights = dict(DEFAULT_WEIGHTS)

    # Baslangic skoru
    parent_score = _eval_worker((parent_weights, games, 0))
    print(f"Baslangic skoru: {parent_score:.3f}  ({games} oyun)")
    print("=" * 60)

    best_ever_score   = parent_score
    best_ever_weights = dict(parent_weights)
    history = []

    for gen in range(1, generations + 1):
        gen_start = time.time()

        # offspring uret
        children = [mutate(parent_weights, sigma, rng) for _ in range(offspring)]
        args = [
            (child, games, gen * 10_000 + k * 100)
            for k, child in enumerate(children)
        ]

        # Paralel degerlendirme
        if workers > 1:
            with Pool(processes=workers) as pool:
                scores = pool.map(_eval_worker, args)
        else:
            scores = [_eval_worker(a) for a in args]

        best_idx   = int(np.argmax(scores))
        best_child_score   = scores[best_idx]
        best_child_weights = children[best_idx]

        improved = best_child_score > parent_score
        if improved:
            parent_weights = best_child_weights
            parent_score   = best_child_score

        if parent_score > best_ever_score:
            best_ever_score   = parent_score
            best_ever_weights = dict(parent_weights)
            np.save(WEIGHTS_FILE, best_ever_weights)

        elapsed = time.time() - gen_start
        marker  = " (*)" if improved else "    "
        print(f"Gen {gen:3d}/{generations}{marker} | "
              f"ebeveyn={parent_score:.3f} | "
              f"en_iyi_cocuk={best_child_score:.3f} | "
              f"all_time={best_ever_score:.3f} | "
              f"{elapsed:.1f}s")

        history.append({
            "generation":    gen,
            "parent_score":  parent_score,
            "best_child":    best_child_score,
            "all_time_best": best_ever_score,
            "improved":      improved,
            "elapsed_s":     round(elapsed, 2),
        })

    # Son kayit
    np.save(WEIGHTS_FILE, best_ever_weights)
    print(f"\nEgitim tamamlandi. En iyi skor: {best_ever_score:.3f}")
    print(f"Agirliklar: {WEIGHTS_FILE}")

    log_path = os.path.join(os.path.dirname(WEIGHTS_FILE), "edi_train_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": dict(generations=generations, offspring=offspring,
                           games=games, sigma=sigma, seed=seed, workers=workers),
            "history":      history,
            "best_score":   best_ever_score,
            "best_weights": best_ever_weights,
        }, f, indent=2)
    print(f"Log:        {log_path}")

    return best_ever_weights


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(description="EdiAgent evrimsel egitimi")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--offspring",   type=int, default=8)
    parser.add_argument("--games",       type=int, default=40)
    parser.add_argument("--sigma",       type=float, default=0.15)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--workers",     type=int, default=min(cpu_count, 8),
                        help=f"Paralel worker sayisi (varsayilan: {min(cpu_count,8)}, "
                             f"mevcut CPU: {cpu_count})")
    args = parser.parse_args()

    total_games = args.generations * args.offspring * args.games
    print("EdiAgent Evrimsel Optimizasyon")
    print(f"  Nesil={args.generations} | Cocuk/nesil={args.offspring} | "
          f"Oyun/eval={args.games} | Sigma={args.sigma} | Workers={args.workers}")
    print(f"  Toplam oyun: ~{total_games:,}")
    print()

    train(
        generations=args.generations,
        offspring=args.offspring,
        games=args.games,
        sigma=args.sigma,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
