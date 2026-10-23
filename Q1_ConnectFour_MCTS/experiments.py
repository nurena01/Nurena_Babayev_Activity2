"""Experiments for the report: vary the utility function, the number of
cycles, the number of expansion nodes (x) and the height of the tree.

Every setting is run RUNS times on three test positions, each run with a
different random seed, and we record which move is generated. Some settings
also play full games. Results are written to results/*.csv.

Run:  python3 experiments.py            (about 4 minutes)
      python3 experiments.py --runs 5   (quicker, less data)
"""

import argparse
import csv
import os
import random
import time
from collections import Counter

from games import ConnectFour, state_from_rows
from mcts import build_tree, best_move, monte_carlo_tree_search, play_game, random_player

GAME = ConnectFour()

POSITIONS = {
    # Opening move: no tactics, every move is legal.
    'empty': GAME.initial,
    # Middle game with no immediate win or loss for either player.
    'midgame': state_from_rows(GAME, ['.......', '.......', '.......',
                                      '...O...', '..XX...', '.OXOX.O'], 'X'),
    # O threatens to win on the diagonal at (4, 4); X must block there.
    'block': state_from_rows(GAME, ['.......', '.......', '.......',
                                    '..OX...', '.OXO...', 'OXOX..X'], 'X'),
}
BLOCK_MOVE = (4, 4)

DEFAULT = dict(cycles=500, expand_x=7, height=3, C=1.4, W=1.0)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def count_nodes(root):
    stack, n = [root], 0
    while stack:
        node = stack.pop()
        n += 1
        stack.extend(node.children)
    return n


def run_setting(config, runs):
    """Generate a move RUNS times (seeds 0..RUNS-1) for each position."""
    row = {}
    for name, state in POSITIONS.items():
        moves, seconds, nodes = [], 0.0, 0
        for seed in range(runs):
            start = time.perf_counter()
            root = build_tree(state, GAME, seed=seed, **config)
            seconds += time.perf_counter() - start
            nodes += count_nodes(root)
            moves.append(best_move(root))
        counts = Counter(moves)
        top, top_count = counts.most_common(1)[0]
        row[f'{name}_top_move'] = f'{top[0]}'  # column of the most common move
        row[f'{name}_same_%'] = round(100 * top_count / runs)
        row[f'{name}_distinct'] = len(counts)
        if name == 'block':
            row['block_correct_%'] = round(100 * counts[BLOCK_MOVE] / runs)
        if name == 'midgame':
            row['midgame_moves'] = ' '.join(f'{m[0]}:{c}' for m, c in sorted(counts.items()))
        if name == 'empty':
            row['ms_per_move'] = round(1000 * seconds / runs, 1)
            row['tree_nodes'] = round(nodes / runs)
    return row


def match(config_a, config_b, games):
    """config_a plays config_b (or the random player if config_b is None),
    alternating who moves first. Returns (wins, draws, losses) for config_a."""
    w = d = l = 0
    for g in range(games):
        rng = random.Random(1000 + g)
        a = lambda game, s, g=g: monte_carlo_tree_search(s, game, seed=g, **config_a)
        if config_b is None:
            b = lambda game, s, rng=rng: random_player(game, s, rng)
        else:
            b = lambda game, s, g=g: monte_carlo_tree_search(s, game, seed=10_000 + g, **config_b)
        if g % 2 == 0:
            final, me = play_game(GAME, a, b), 'X'
        else:
            final, me = play_game(GAME, b, a), 'O'
        u = GAME.utility(final, me)
        w, d, l = w + (u == 1), d + (u == 0), l + (u == -1)
    return w, d, l


def save(name, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + '.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n== {name} ({path})')
    for r in rows:
        print(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=20, help='runs (seeds) per setting')
    parser.add_argument('--games', type=int, default=20, help='games per match')
    args = parser.parse_args()

    # 1. Utility function: weight of the objective part (W) and of exploration (C)
    rows = []
    for C, W in [(1.4, 0.0), (1.4, 0.5), (1.4, 1.0), (1.4, 2.0), (1.4, 5.0),
                 (0.5, 1.0), (3.0, 1.0)]:
        config = dict(DEFAULT, C=C, W=W)
        rows.append(dict(C=C, W=W, **run_setting(config, args.runs)))
    save('exp1_utility', rows)

    # 2. Number of cycles
    rows = []
    for cycles in [10, 50, 100, 250, 500, 1000, 2000]:
        config = dict(DEFAULT, cycles=cycles)
        rows.append(dict(cycles=cycles, **run_setting(config, args.runs)))
    save('exp2_cycles', rows)

    # 3. Number of nodes expanded at each level (x)
    rows = []
    for x in [1, 2, 3, 4, 5, 6, 7]:
        config = dict(DEFAULT, expand_x=x)
        rows.append(dict(x=x, **run_setting(config, args.runs)))
    save('exp3_expansion', rows)

    # 4. Height of the tree
    rows = []
    for height in [1, 2, 3, 4, 6, 8]:
        config = dict(DEFAULT, height=height)
        rows.append(dict(height=height, **run_setting(config, args.runs)))
    save('exp4_height', rows)

    # 5. Same seed vs different seeds (is the same move always generated?)
    rows = []
    for cycles in [50, 500]:
        for seeded in (True, False):
            config = dict(DEFAULT, cycles=cycles)
            state = POSITIONS['midgame']
            moves = [monte_carlo_tree_search(state, GAME, seed=(0 if seeded else s), **config)
                     for s in range(args.runs)]
            counts = Counter(moves)
            rows.append(dict(cycles=cycles, seed='same (0)' if seeded else 'different',
                             same_move_pct=round(100 * counts.most_common(1)[0][1] / args.runs),
                             moves=' '.join(f'{m[0]}:{c}' for m, c in sorted(counts.items()))))
    save('exp5_seed', rows)

    # 6. Playing strength in full games (fewer cycles to keep the run time down)
    base = dict(DEFAULT, cycles=200)
    matches = [
        ('W=1 vs random player', base, None),
        ('W=0 vs random player', dict(base, W=0.0), None),
        ('W=1 vs W=0', base, dict(base, W=0.0)),
        ('cycles=1000 vs cycles=200', dict(base, cycles=1000), base),
        ('height=4 vs height=1', dict(base, height=4), dict(base, height=1)),
        ('x=3 vs x=7', dict(base, expand_x=3), base),
    ]
    rows = []
    for name, a, b in matches:
        w, d, l = match(a, b, args.games)
        rows.append(dict(match=name, games=args.games, wins=w, draws=d, losses=l,
                         win_pct=round(100 * w / args.games)))
    save('exp6_games', rows)


if __name__ == '__main__':
    main()
