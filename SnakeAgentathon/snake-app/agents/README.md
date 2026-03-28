# Example Agents

## random_agent.py
Picks a random valid direction each step. Avoids reversing into itself.

## greedy_agent.py
Moves toward the nearest apple using toroidal Manhattan distance.

## Writing Your Own Agent
1. Copy one of these files as a starting point
2. Subclass `SnakeAgent` from `src.agent`
3. Implement `get_action(observation) -> int`
4. Actions: `0=UP, 1=RIGHT, 2=DOWN, 3=LEFT`
5. Drop your file into `teams/` for the tournament
