# Q1 Report: Connect Four Move Generator Using Monte Carlo Tree Search

Nurena Babayev, Activity 2

## 1. Design

The move generator follows MONTE-CARLO-TREE-SEARCH from Figure 5.11 of the text:

```
function MONTE-CARLO-TREE-SEARCH(state) returns an action
    tree <- NODE(state)
    while IS-TIME-REMAINING() do
        leaf   <- SELECT(tree)
        child  <- EXPAND(leaf)
        result <- SIMULATE(child)
        BACK-PROPAGATE(result, child)
    return the move in ACTIONS(state) whose node has highest number of playouts
```

The code is built from the textbook's code snippets. `games.py` uses the `Game`,
`TicTacToe` and `ConnectFour` classes (initial state, TO-MOVE, ACTIONS, RESULT,
TERMINAL-TEST, UTILITY). `mcts.py` uses `MCT_Node`, `ucb` and the
select / expand / simulate / backprop structure of `monte_carlo_tree_search`.
The board is 7 columns by 6 rows. A move is (column, row) and must be on the
bottom row or directly above an occupied square.

Each part of the assignment maps to the code like this:

| Assignment | Implementation (`mcts.py`) |
|---|---|
| **Selection** with a utility that has a **playout** part (stored from previous results) and an **objective** part (from the rules of the game) | `utility(n) = U(n)/N(n) + C·sqrt(ln N(parent)/N(n)) + W·objective(n)`. The first two terms are UCB1 (Section 5.5) and use the U and N stored in each node by earlier playouts. The last term is an evaluation function (Section 5.4) computed from the board. An unvisited node gets ∞, as in UCB1. SELECT moves down the tree to the child with the highest utility. |
| **Expansion**: at each level, select **x** nodes to expand | When a node is expanded, all legal moves are scored by the objective and only the **x best** are added as children (`expand_x`). This is the forward-pruning idea from Section 5.4. |
| **Simulation** when you reach the **fixed level** of the tree, with a **random playout policy** | Each cycle keeps expanding (x children per level) and descending until the node is at depth `height` (or the game is over). From there, SIMULATE plays random legal moves until the game ends. |
| **Repeat a configurable number of times** | `cycles` replaces IS-TIME-REMAINING(). |
| Back-propagation | Every node on the path gets N + 1, and U + 1 for a win, 0.5 for a draw, 0 for a loss, all from the point of view of the player who made the move into that node. |
| Move returned | The child of the root with the most playouts (highest N), as in Figure 5.11. |

**Objective part.** The objective is a weighted linear function of features
(Section 5.4), scaled to the range [-1, 1] and computed for the player who made
the move:

- A won state scores +1, a lost state −1 and a drawn state 0.
- Otherwise, the function looks at every line of four cells (horizontal,
  vertical and both diagonals) that only one player can still complete. That
  player gets 1, 3 or 9 points when the line holds 1, 2 or 3 of their pieces.
- Each piece in the centre column is worth 1 more point.
- The opponent's points are subtracted, and the total is divided by 50.

**Settings** (all configurable): `cycles`, `expand_x` (x), `height`, `C` (weight of
the exploration term), `W` (weight of the objective part) and `seed` (random seed).

## 2. Experiments

Every setting was run **20 times with different random seeds (0–19)** on three
positions, recording the move generated each time. Columns are numbered 1–7 from
the left, and a move is reported by its column.

```
empty                midgame (X to move)     block (X to move)
. . . . . . .        . . . . . . .           . . . . . . .
. . . . . . .        . . . . . . .           . . . . . . .
. . . . . . .        . . . . . . .           . . . . . . .
. . . . . . .        . . . O . . .           . . O X . . .
. . . . . . .        . . X X . . .           . O X O . . .
. . . . . . .        . O X O X . O           O X O X . . X
1 2 3 4 5 6 7        1 2 3 4 5 6 7           1 2 3 4 5 6 7
```

- **empty**: the opening move.
- **midgame**: no immediate win or loss for either player, so several moves are reasonable.
- **block**: O threatens to win on the diagonal at column 4, so **column 4 is the only correct move**.

In the tables, **same %** is how often the most common move was chosen across
the 20 runs (100% = always the same move). **Moves** lists `column:count`.
Unless a table says otherwise, the settings are cycles = 500, x = 7, height = 3,
C = 1.4 and W = 1.0. The raw data is in `results/*.csv` and is produced by
`python3 experiments.py`.

### 2.1 Utility function (C and W)

| C | W | empty: move, same % | midgame: same %, moves | block: correct % |
|---|---|---|---|---|
| 1.4 | 0.0 (playout only) | 4, 70% | 45%: 2:8 3:3 5:9 | 100% |
| 1.4 | 0.5 | 4, 95% | 60%: 2:2 3:6 5:12 | 100% |
| 1.4 | 1.0 | 4, 85% | 60%: 3:12 5:8 | 100% |
| 1.4 | 2.0 | 4, 100% | 100%: 3:20 | 100% |
| 1.4 | 5.0 | 4, 100% | 100%: 3:20 | 100% |
| 0.5 | 1.0 | 4, 85% | 70%: 3:14 5:6 | 100% |
| 3.0 | 1.0 | 4, 95% | 75%: 3:15 5:5 | 100% |

### 2.2 Number of cycles

| cycles | ms per move | empty: same %, different moves | midgame: same %, moves | block: correct % |
|---|---|---|---|---|
| 10 | 6 | 60%, 5 | 95%: 3:19 5:1 | 95% |
| 50 | 20 | 40%, 6 | 55%: 2:2 3:11 5:7 | 95% |
| 100 | 30 | 50%, 6 | 50%: 2:2 3:10 5:8 | 100% |
| 250 | 50 | 65%, 5 | 55%: 2:1 3:11 5:8 | 100% |
| 500 | 86 | 85%, 4 | 60%: 3:12 5:8 | 100% |
| 1000 | 164 | 100%, 1 | 70%: 3:14 5:6 | 100% |
| 2000 | 300 | 100%, 1 | 75%: 3:15 5:5 | 100% |

On the empty board, the most common move was always column 4 (the centre).

### 2.3 Number of nodes expanded at each level (x)

| x | tree nodes | ms per move | empty: same % | midgame: same %, moves | block: correct % |
|---|---|---|---|---|---|
| 1 | 4 | 69 | 100% | 100%: 3:20 | 100% |
| 2 | 15 | 67 | 100% | 100%: 5:20 | 100% |
| 3 | 40 | 70 | 100% | 80%: 3:16 5:4 | 100% |
| 4 | 85 | 76 | 90% | 85%: 3:17 5:3 | 100% |
| 5 | 156 | 76 | 100% | 60%: 3:12 5:8 | 100% |
| 6 | 259 | 80 | 100% | 80%: 3:16 5:4 | 100% |
| 7 | 400 | 85 | 85% | 60%: 3:12 5:8 | 100% |

(Tree nodes are counted on the empty board. With height 3 the full tree has
1 + x + x² + x³ nodes, so 500 cycles builds all of it.)

### 2.4 Height of the tree

| height | tree nodes | ms per move | empty: same % | midgame: same %, moves | block: correct % |
|---|---|---|---|---|---|
| 1 | 8 | 88 | 100% | 90%: 3:18 5:2 | 100% |
| 2 | 57 | 77 | 90% | 60%: 3:12 5:8 | 100% |
| 3 | 400 | 86 | 85% | 60%: 3:12 5:8 | 100% |
| 4 | 2,513 | 167 | 85% | 50%: 3:10 5:10 | 100% |
| 6 | 9,467 | 446 | 100% | 95%: 3:1 5:19 | 100% |
| 8 | 16,024 | 708 | 95% | 80%: 3:4 5:16 | 100% |

### 2.5 Same seed vs different seeds (midgame position)

| cycles | seed | same % | moves |
|---|---|---|---|
| 50 | same seed (0) every run | 100% | 3:20 |
| 50 | different seed every run | 55% | 2:2 3:11 5:7 |
| 500 | same seed (0) every run | 100% | 3:20 |
| 500 | different seed every run | 60% | 3:12 5:8 |

### 2.6 Full games (20 games per match, each side moves first in 10)

Both players use cycles = 200, x = 7, height = 3, C = 1.4 and W = 1.0, except for
the setting named in the match.

| match | wins | draws | losses | win % |
|---|---|---|---|---|
| W = 1 vs random player | 20 | 0 | 0 | 100% |
| W = 0 vs random player | 20 | 0 | 0 | 100% |
| W = 1 vs W = 0 | 17 | 0 | 3 | 85% |
| cycles = 1000 vs cycles = 200 | 16 | 1 | 3 | 80% |
| height = 4 vs height = 1 | 14 | 0 | 6 | 70% |
| x = 3 vs x = 7 | 9 | 3 | 8 | 45% |

## 3. Analysis and reflection

### Does the simulator always generate the same move?

**No.** With a different random seed on each run, the same position can give
different moves. In the midgame position with the default settings, 20 runs chose
column 3 twelve times and column 5 eight times (2.5). With the playout part alone
(W = 0), three different moves came up and none was chosen more than 45% of the
time (2.1).

**With the same seed, it always generates the same move** (100% in 2.5). The
algorithm itself is deterministic. All of the variation comes from the random
numbers.

**In positions with one clearly correct move, it generates that move.** In the
block position it found column 4 in 100% of runs with 100 or more cycles, and
in 95% of runs with only 10–50 cycles.

### What causes the differences?

1. **The random playout policy.** SIMULATE plays random moves, so the U/N value
   of a node is an estimate from a sample of games, and every seed gives a
   different sample. When two moves have similar true values (columns 3 and 5 in
   the midgame), small sampling differences change which one ends up with the
   most playouts.
2. **Too few cycles.** With few cycles each node has only a few playouts, so the
   estimates are noisy. On the empty board, 50–100 cycles gave 6 different
   opening moves, while 1000 and 2000 cycles always gave column 4 (2.2).
3. **The weight of the playout part compared with the objective part.** When
   W = 0, selection depends only on the noisy playout results (45% agreement in
   the midgame). The objective part is computed from the rules and has no
   randomness, so it pulls every run toward the same moves.
4. **Random tie-breaking.** Ties between moves (for example unvisited children
   with the same objective value) are broken at random.
5. **The height of the tree.** Changing the height changes *which* move is
   preferred, not only how often. In the midgame, height 1–4 mostly chose
   column 3 and height 6–8 mostly chose column 5 (2.4). A taller tree looks
   further ahead with nodes chosen by the utility, including the objective part,
   before the random playout takes over. That gives different value estimates
   than short trees, where the playout starts after one move.

### What causes the similarities?

1. **The objective part.** With W ≥ 2, all 20 runs agreed on every position (2.1),
   because the objective differences between moves outweigh the noise in U/N.
2. **More cycles.** The playout averages settle as N grows. Agreement rose to
   100% on the empty board and to 75% on the midgame at 2000 cycles (2.2), and the
   most common move did not change. This matches Section 5.5: as the number of
   playouts increases, UCB1 spends them on the best moves.
3. **Forced moves.** When one move wins or blocks a loss, its playout results
   are very different from the other moves, so every run finds it.
4. **Small x.** With x = 1 or 2, the root has only 1 or 2 children, picked by
   the objective. The move is then decided almost entirely by the objective, so
   every run agrees (2.3). The weakness is that the best move can be pruned
   before it is ever simulated. With x = 2 the generator always played column 5
   in the midgame, because column 3 was not among the two moves the objective
   ranked highest. This is the risk of forward pruning described in Section 5.4.
5. **Very few cycles.** At 10 cycles the midgame result was 95% the same, but only
   because nearly every root child had been visited once, so the choice mostly
   came from the objective tie-breaking, not from real playout information. A
   high agreement rate therefore does not always mean a good move.

### Effect of each setting on playing strength and cost

- **Utility function.** Adding the objective part made the generator stronger:
  W = 1 beat W = 0 in 17 of 20 games (2.6). Both versions beat a random player
  20–0. Changing C between 0.5 and 3.0 had only a small effect at 500 cycles
  (70–85% agreement). A large C spreads playouts across more moves (exploration)
  and a small C focuses on the current best moves (exploitation).
- **Cycles.** More cycles gave better and more consistent moves (1000 cycles beat
  200 cycles 16–1–3), but time grows linearly: about 0.15 ms per cycle,
  6 ms for 10 cycles and 300 ms for 2000.
- **Expansion nodes x.** A smaller x builds a much smaller tree (4 nodes for
  x = 1 vs 400 for x = 7) and is a little faster, but risks pruning the best move.
  x = 3 vs x = 7 was roughly even (9 wins, 3 draws, 8 losses), so keeping the 3 best
  moves lost little strength here.
- **Height.** A taller tree costs much more time (88 ms at height 1, 708 ms at
  height 8, with 16,024 nodes) because every cycle expands x nodes at each level
  down to the fixed height. It also plays better: height 4 beat height 1 in 14 of
  20 games.

### Summary

The generator is not deterministic: with different random seeds it can choose
different moves. The differences come from the random playout policy, and they
are largest when there are few cycles, when the objective weight is small, and
when two moves have similar values. Results become more alike with more cycles, a
larger objective weight, a smaller x, or a position with a forced move, and they are
identical with the same seed. More cycles, a taller tree and the objective part
all made the generator stronger, at the cost of more computation. The value of
x is a trade-off between speed and the risk of pruning the best move.

**Limitations.** Each setting was run 20 times and each match was 20 games, so
differences of a few percent are within random variation. The midgame position
has no known single best move, so for it we measure consistency, not correctness.
