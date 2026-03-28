"""100 maclik GUI izleme — EdiAgent vs secilen rakip.

Kullanim:
  python watch_edi.py                          # edi vs greedy, 100 mac
  python watch_edi.py --n 50 --speed 8        # 50 mac, hizli
  python watch_edi.py --rival agents/hybrid_agent.py
"""

import argparse

from src.runner import load_agent, run_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1",    default="agents/edi_agent.py",   help="P1 ajan dosyasi")
    parser.add_argument("--rival", default="agents/greedy_agent.py", help="Rakip ajan dosyasi")
    parser.add_argument("--n",     type=int,   default=100,  help="Mac sayisi")
    parser.add_argument("--speed", type=int,   default=7,    choices=range(1, 10))
    args = parser.parse_args()

    wins, losses, draws = 0, 0, 0

    for i in range(args.n):
        # Taraflar donusumlu degisir
        if i % 2 == 0:
            a1 = load_agent(args.p1,    player_id=1)
            a2 = load_agent(args.rival, player_id=2)
            result = run_match(a1, a2, seed=i, gui=True, speed=args.speed,
                               name1="EDI", name2="RAKIP")
            w = result["winner"]
            if w == 1:   wins   += 1
            elif w == 2: losses += 1
            else:        draws  += 1
        else:
            a1 = load_agent(args.rival, player_id=1)
            a2 = load_agent(args.p1,    player_id=2)
            result = run_match(a1, a2, seed=i, gui=True, speed=args.speed,
                               name1="RAKIP", name2="EDI")
            w = result["winner"]
            if w == 2:   wins   += 1
            elif w == 1: losses += 1
            else:        draws  += 1

        total = i + 1
        print(f"Mac {total:3d}/{args.n} | EDI: {wins}W {losses}L {draws}D | "
              f"Oran: {wins/total:.2f}")

    print("\n=== FINAL ===")
    print(f"EDI:  {wins}W / {losses}L / {draws}D  ({wins/args.n:.1%} kazanma)")


if __name__ == "__main__":
    main()
