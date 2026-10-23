# Q1: Connect Four move generator using Monte Carlo tree search

Nurena Babayev, Activity 2

A move generator for Connect Four (6 rows by 7 columns) based on
MONTE-CARLO-TREE-SEARCH (textbook Figure 5.11) and the textbook's code
(`Game`, `TicTacToe` and `ConnectFour` from aima-python `games.py`, and
`MCT_Node`, `ucb` and `monte_carlo_tree_search` from `games4e.py`).

Requirements: Python 3.8 or newer. Only the standard library is used, so
nothing needs to be installed.

## Files

| File | Contents |
|---|---|
| `games.py` | Connect Four rules: `GameState`, `Game`, `TicTacToe`, `ConnectFour` |
| `mcts.py` | The MCTS move generator (selection, expansion, simulation, back-propagation) |
| `experiments.py` | Experiments for the report; writes `results/*.csv` |
| `results/` | Data collected by `experiments.py` |
| `REPORT.md` | Report: design, data and analysis |
| `tests/test_games.py` | Tests for the game rules |
| `tests/test_mcts.py` | Tests for the move generator |
| `tests/test_data/positions.json` | Test data: board positions with the expected results |

## Running the tests

From inside the `Q1_ConnectFour_MCTS` folder:

```
python3 -m unittest discover -s tests -v
```

All 27 tests should pass in a few seconds. The tests check:

- the rules: board size, gravity, legal moves, full columns, wins in all four
  directions, draws (positions in `tests/test_data/positions.json`)
- the objective part of the utility (win = +1, loss = -1, symmetric, range)
- the selection utility formula
- the move generator: legal moves, winning and blocking moves in the test
  positions (5 seeds each), the number of cycles, the fixed tree height, at most
  x children per node, the x best moves are expanded, same seed gives the same
  move, and it beats a random player

The tests also run automatically on GitHub with GitHub Actions
(`.github/workflows/tests.yml` in the top folder) on every push.

## Running the move generator

```
python3 mcts.py                                   # opening move, default settings
python3 mcts.py --cycles 2000 --x 4 --height 5 --C 1.4 --W 1.0 --seed 3
python3 mcts.py --rows ....... ....... ....... ..OX... .OXO... OXOX..X
```

| Option | Meaning | Default |
|---|---|---|
| `--cycles` | number of MCTS cycles (repeats) | 1000 |
| `--x` | nodes expanded at each level of the tree | 7 |
| `--height` | fixed level of the tree where the random playout starts | 3 |
| `--C` | exploration constant (playout part) | 1.4 |
| `--W` | weight of the objective part | 1.0 |
| `--seed` | random seed; leave it out for a different result each run | none |
| `--rows` | a board as 6 strings, top row first (`X`, `O`, `.`) | empty board |

It prints the board, the statistics of each move at the root (N, U/N, objective)
and the move chosen: the one with the most playouts.

From Python:

```python
from games import ConnectFour
from mcts import monte_carlo_tree_search

game = ConnectFour()
move = monte_carlo_tree_search(game.initial, game, cycles=1000, expand_x=7,
                               height=3, C=1.4, W=1.0)
```

Moves are `(column, row)`, with column 1 to 7 from the left and row 1 to 6 from the bottom.

## Running the experiments

```
python3 experiments.py
```

This takes about 4 minutes and rewrites the CSV files in `results/`.
`python3 experiments.py --runs 5 --games 6` is a quicker version.
